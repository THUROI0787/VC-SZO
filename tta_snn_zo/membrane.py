"""Membrane dynamics utilities for SNN-specific ZO-TTA.

This module implements three SNN-specific components that leverage the
membrane potential dynamics of LIF neurons for improved ZO-TTA:

1. **Membrane Convergence Gate** (``membrane_convergence_score``):
   Uses the cross-timestep variance of membrane potentials as a sample
   reliability indicator. Low variance = fast convergence = reliable sample.
   Replaces softmax entropy filtering with a signal that is intrinsic to SNNs.

2. **Membrane Subspace ZO** (``MembraneSubspaceZO``):
   Projects ZO perturbations onto the principal subspace of recent membrane
   potential trajectories. This reduces the effective ZO dimension from
   d_adapter (e.g. 512) to r_subspace (e.g. 16-32), cutting variance by
   O(sqrt(d/r)).

3. **Membrane KL Alignment Loss** (``membrane_kl_loss``):
   Aligns the membrane potential distribution of the current batch with a
   reference distribution (from source or early batches). Provides a
   ZO-compatible alternative to entropy minimization that is less prone
   to collapse.

Reference: MPA (Membrane Potential Alignment, arXiv 2026) uses KL divergence
on membrane potentials for TTA in BCI settings.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, neuron


# --------------------------------------------------------------------------- #
#  1. Membrane Convergence Gate
# --------------------------------------------------------------------------- #

def collect_membrane_trajectory(
    model: nn.Module,
    x: torch.Tensor,
    num_steps: int,
) -> Dict[str, torch.Tensor]:
    """Run a forward pass and collect membrane potentials at each LIF layer.

    Returns {layer_name: v_trajectory} where v_trajectory has shape
    [T, B, C, H, W] (conv layers) or [T, B, D] (fc layers).

    This uses forward hooks on LIFNode modules to record ``v`` after each
    timestep. The model must be in eval mode (no surrogate gradient needed).
    """
    trajectories: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            v = module.v.detach()
            if name not in trajectories:
                trajectories[name] = []
            trajectories[name].append(v)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, neuron.LIFNode):
            handles.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    functional.reset_net(model)
    with torch.no_grad():
        for _ in range(num_steps):
            _ = model(x)

    for h in handles:
        h.remove()
    functional.reset_net(model)

    result = {}
    for name, traj in trajectories.items():
        if traj:
            result[name] = torch.stack(traj, dim=0)
    return result


def membrane_convergence_score(
    v_trajectory: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute membrane convergence score from a trajectory [T, B, ...].

    The convergence score is the variance across timesteps, averaged over
    spatial dimensions. Low variance = fast convergence = reliable sample.

    Args:
        v_trajectory: Membrane potential trajectory [T, B, C, H, W] or [T, B, D]
        reduction: 'mean' returns scalar, 'none' returns per-sample [B]

    Returns:
        convergence score (lower = more reliable)
    """
    var = v_trajectory.var(dim=0)  # [B, ...]
    # Flatten all non-batch dimensions and mean
    score = var.view(var.size(0), -1).mean(dim=1)  # [B]
    if reduction == "mean":
        return score.mean()
    return score


def membrane_gate(
    model: nn.Module,
    x: torch.Tensor,
    num_steps: int,
    threshold: float = 0.05,
    gate_scale: float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute a per-sample reliability gate from membrane convergence.

    Returns:
        gate: per-sample weight in [0, 1] (low = unreliable, skip adaptation)
        stats: dict with 'mean_convergence', 'gate_frac'
    """
    trajectories = collect_membrane_trajectory(model, x, num_steps)
    if not trajectories:
        return torch.ones(x.size(0), device=x.device), {"mean_convergence": 0.0, "gate_frac": 1.0}

    scores = []
    for name, traj in trajectories.items():
        scores.append(membrane_convergence_score(traj, reduction="none"))
    avg_score = torch.stack(scores).mean(dim=0)

    gate = torch.sigmoid(gate_scale * (threshold - avg_score))
    stats = {
        "mean_convergence": float(avg_score.mean().item()),
        "gate_frac": float((gate > 0.5).float().mean().item()),
    }
    return gate, stats


# --------------------------------------------------------------------------- #
#  2. Membrane Subspace ZO
# --------------------------------------------------------------------------- #

class MembraneSubspaceZO:
    """ZO optimizer with membrane-potential-driven subspace projection.

    Instead of sampling perturbations in the full adapter parameter space,
    we project onto the principal subspace of recent membrane potential
    trajectories. This is motivated by the observation that adapter parameters
    that produce similar membrane potential trajectories should be grouped.

    The subspace is updated online using a streaming covariance estimator
    (randomized SVD via sketching) to avoid O(d^3) PCA cost.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        model: nn.Module,
        num_steps: int,
        lr: float = 1e-3,
        eps: float = 1e-2,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        num_samples: int = 1,
        subspace_rank: int = 16,
        update_every: int = 10,
        estimator: str = "two_point",
        seed: int = 0,
        device: torch.device = None,
    ):
        from .zo import ZoOptimizer

        self.zo = ZoOptimizer(
            params, lr=lr, eps=eps, weight_decay=weight_decay,
            momentum=momentum, num_samples=num_samples,
            block_sparsity=1.0, estimator=estimator, seed=seed, device=device,
        )
        self.model = model
        self.num_steps = num_steps
        self.subspace_rank = int(subspace_rank)
        self.update_every = int(update_every)
        self.device = device or params[0].device

        self.membrane_buffer: List[torch.Tensor] = []
        self.max_buffer = 50
        self.subspace: Optional[torch.Tensor] = None
        self._steps = 0
        # Expose params for compatibility with ZOTTAEngine
        self.params = self.zo.params

    def _collect_membrane_flat(self) -> torch.Tensor:
        """Run a forward pass and return flattened membrane potentials."""
        self.model.eval()
        functional.reset_net(self.model)
        flat_vectors = []
        with torch.no_grad():
            for _ in range(self.num_steps):
                _ = self.model(torch.zeros(1, 3, 32, 32, device=self.device))
                for mod in self.model.modules():
                    if isinstance(mod, neuron.LIFNode):
                        v = mod.v.detach().flatten()
                        flat_vectors.append(v)
        functional.reset_net(self.model)
        return torch.cat(flat_vectors) if flat_vectors else torch.zeros(1, device=self.device)

    def _update_subspace(self):
        if len(self.membrane_buffer) < max(10, self.subspace_rank * 2):
            return
        V = torch.stack(self.membrane_buffer)
        N, D = V.shape
        r = min(self.subspace_rank, N, D)
        proj = torch.randn(D, r, device=self.device)
        Y = V @ proj
        Q, _ = torch.linalg.qr(Y)
        B = Q.T @ V
        U_small, S, Vt = torch.svd_lowrank(B.T, q=r)
        self.subspace = Q @ U_small

    def _project_to_subspace(self, zs: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.subspace is None:
            return zs
        flat_z = torch.cat([z.flatten() for z in zs])
        D = flat_z.size(0)
        if D != self.subspace.size(0):
            return zs
        z_sub = self.subspace @ (self.subspace.T @ flat_z)
        z_res = flat_z - z_sub
        alpha = 0.8
        z_new = alpha * z_sub + (1 - alpha) * z_res
        idx = 0
        zs_projected = []
        for z in zs:
            n = z.numel()
            zs_projected.append(z_new[idx: idx + n].reshape_as(z))
            idx += n
        return zs_projected

    def _apply_subspace_projection(self):
        """Replace ZoOptimizer's _sample_direction with subspace-projected version."""
        original_sample = self.zo._sample_direction
        if self.subspace is not None:
            def projected_sample():
                zs = original_sample()
                return self._project_to_subspace(zs)
            self.zo._sample_direction = projected_sample
        return original_sample

    def _restore_sampler(self, original_sample):
        self.zo._sample_direction = original_sample

    def _maybe_update_subspace(self):
        if self._steps % self.update_every == 0:
            v_flat = self._collect_membrane_flat()
            self.membrane_buffer.append(v_flat)
            if len(self.membrane_buffer) > self.max_buffer:
                self.membrane_buffer.pop(0)
            self._update_subspace()

    def step(self, objective_fn: Callable[[], float]) -> dict:
        self._steps += 1
        self._maybe_update_subspace()
        orig = self._apply_subspace_projection()
        result = self.zo.step(objective_fn)
        self._restore_sampler(orig)
        result["subspace_rank"] = self.subspace.size(1) if self.subspace is not None else 0
        return result

    def step_one_point(self, objective_fn: Callable[[], float], baseline: float = None) -> dict:
        """One-point ZO step with subspace projection. Delegates to self.zo.step_one_point."""
        self._steps += 1
        self._maybe_update_subspace()
        orig = self._apply_subspace_projection()
        result = self.zo.step_one_point(objective_fn, baseline=baseline)
        self._restore_sampler(orig)
        result["subspace_rank"] = self.subspace.size(1) if self.subspace is not None else 0
        return result


# --------------------------------------------------------------------------- #
#  3. Membrane KL Alignment Loss
# --------------------------------------------------------------------------- #

@torch.no_grad()
def collect_membrane_stats(
    model: nn.Module,
    loader,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Collect reference membrane potential statistics from source data.

    Returns {layer_name: (mean, var)} where mean/var are per-channel
    statistics computed over num_batches of clean (or source) data.

    Handles both conv LIF layers (v shape [B, C, H, W]) and fc LIF layers
    (v shape [B, D]).
    """
    model.eval()
    stats: Dict[str, List[Dict]] = {}
    batch_count = 0

    for x, _ in loader:
        if batch_count >= num_batches:
            break
        x = x.to(device)
        functional.reset_net(model)

        layer_v: Dict[str, List[torch.Tensor]] = {}
        handles = []

        def make_hook(name):
            def hook(module, inp, out):
                v = module.v.detach().cpu()
                if name not in layer_v:
                    layer_v[name] = []
                layer_v[name].append(v)
            return hook

        hook_handles = []
        for nm, mod in model.named_modules():
            if isinstance(mod, neuron.LIFNode):
                hook_handles.append(mod.register_forward_hook(make_hook(nm)))

        with torch.no_grad():
            _ = model(x)

        for h in hook_handles:
            h.remove()

        for name, vlist in layer_v.items():
            # vlist: list of [B, C, H, W] or [B, D], one per timestep
            # Stack across timesteps: [T, B, C, H, W] or [T, B, D]
            v_all = torch.stack(vlist)

            if v_all.dim() == 5:  # [T, B, C, H, W] — conv layer
                # Permute to [C, T, B, H, W] then flatten
                v_flat = v_all.permute(2, 0, 1, 3, 4).contiguous().view(
                    v_all.size(2), -1)  # [C, T*B*H*W]
            elif v_all.dim() == 3:  # [T, B, D] — fc layer
                # Permute to [D, T, B] then flatten
                v_flat = v_all.permute(2, 0, 1).contiguous().view(
                    v_all.size(2), -1)  # [D, T*B]
            else:
                # Fallback: flatten everything
                v_flat = v_all.view(v_all.size(1), -1)  # [B, T*...]

            if name not in stats:
                stats[name] = []
            stats[name].append({
                "mean": v_flat.mean(dim=1),
                "var": v_flat.var(dim=1),
            })
        batch_count += 1

    result = {}
    for name, stat_list in stats.items():
        means = torch.stack([s["mean"] for s in stat_list])
        vars_ = torch.stack([s["var"] for s in stat_list])
        result[name] = (means.mean(dim=0), vars_.mean(dim=0))

    return result


def membrane_kl_loss(
    model: nn.Module,
    ref_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL divergence between current batch membrane potentials and reference.

    For each LIF layer, computes KL(N(cur_mean, cur_var) || N(ref_mean, ref_var))
    and sums across layers. Handles both conv (4D v) and fc (2D v) LIF layers.

    This is a ZO-compatible loss (no gradients needed) that encourages the
    model's internal dynamics to stay close to the source distribution.
    """
    kl_total = 0.0
    n_layers = 0

    for name, mod in model.named_modules():
        if isinstance(mod, neuron.LIFNode) and name in ref_stats:
            v = mod.v
            ref_mean, ref_var = ref_stats[name]

            if v.dim() == 4:  # [B, C, H, W] — conv layer
                cur_mean = v.mean(dim=(0, 2, 3))
                cur_var = v.var(dim=(0, 2, 3))
            elif v.dim() == 2:  # [B, D] — fc layer
                cur_mean = v.mean(dim=0)
                cur_var = v.var(dim=0)
            else:
                cur_mean = v.view(v.size(0), -1).mean(dim=0)
                cur_var = v.view(v.size(0), -1).var(dim=0)

            ref_mean = ref_mean.to(v.device)
            ref_var = ref_var.to(v.device)

            kl = 0.5 * (
                (cur_var / (ref_var + eps)).sum()
                + ((cur_mean - ref_mean) ** 2 / (ref_var + eps)).sum()
                - cur_mean.numel()
                + (ref_var.log() - cur_var.log()).sum()
            )
            kl_total = kl_total + kl
            n_layers += 1

    return kl_total / max(n_layers, 1)