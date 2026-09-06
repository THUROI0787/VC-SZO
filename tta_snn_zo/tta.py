"""Test-time adaptation engines for SNNs.

* ``SourceOnlyEngine`` - no adaptation (baseline).
* ``TentEngine``       - BP-based entropy minimization on BN affine params
                         (TENT adapted to SNN; BN running stats frozen, like the
                         reference SPACE/PAR comparison protocol).  Made robust
                         with reliability gating (entropy window + per-sample
                         filter + grad clip) -- plain TENT collapses to chance
                         on corrupted inputs (observed: acc 10%, entropy -> 0).
* ``ZOTTAEngine``      - the proposed method: frozen backbone + MeZO-style
                         zeroth-order entropy minimization on a tiny adapter
                         (or BN affine params).

Anti-collapse machinery (merged from two rounds of analysis):
  1. batch-level entropy window gate [ent_min_batch, ent_max_batch]:
     skip batches that are already collapsed (entropy ~ 0) or garbage
     (entropy ~ log C) -- prevents error accumulation;
  2. per-sample confidence filter (sample_ent_thresh): only reliable
     (low-entropy) samples contribute to the loss (selective TTA);
  3. adapter parameter clamp (clamp_abs): keep scale/bias within a small box;
  4. collapse rollback (rollback_collapse): if a batch collapses after
     adaptation, restore the pre-adaptation parameters;
  plus optional PAR-style temporal stability gate, KL-to-source anti-forgetting,
  weight-decay-toward-init, per-param gradient clip, and per-batch reset
  (candidate-sweep knobs; see phase1/candidates.py).

SNN-specific extensions (phase 1b / ODI):
  5. membrane convergence gate (membrane_gate): uses LIF membrane potential
     cross-timestep variance as a sample reliability indicator;
  6. membrane subspace ZO (MembraneSubspaceZO): projects ZO perturbations
     onto the principal subspace of membrane potential trajectories;
  7. membrane KL alignment loss (membrane_kl_loss): aligns current membrane
     distribution with a source reference distribution.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from .adapters import AdapterHooks, build_adapter
from .losses import (
    filtered_entropy,
    kl_to_source,
    logsumexp_loss,
    margin_loss,
    mean_spike_rate,
    pseudo_label_ce,
    reliability_gate,
    softmax_entropy,
    temp_entropy,
    temporal_state_loss,
)
from .membrane import (
    MembraneSubspaceZO,
    collect_membrane_stats,
    membrane_convergence_score,
    membrane_gate,
    membrane_kl_loss,
)
from .models import SNN_VGG9, SNN_ResNet19
from .utils import accuracy
from .zo import ZoOptimizer, collect_bn_affine_params, freeze_model


def _infer_layer_channels(model: nn.Module, target_layers: Tuple[str, ...],
                          img_size: int = 32, device: torch.device = None) -> Dict[str, int]:
    """Infer output channel count of target layers via a dummy forward."""
    shapes: Dict[str, int] = {}
    handles = []
    module_dict = dict(model.named_modules())
    for name in target_layers:
        if name not in module_dict:
            raise ValueError(f"target layer '{name}' not found in model")
        def make_hook(nm):
            def hook(m, inp, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                shapes[nm] = out.shape[1]
            return hook
        handles.append(module_dict[name].register_forward_hook(make_hook(name)))
    device = device or next(model.parameters()).device
    model.eval()
    was_collect = getattr(model, "_collect_features", False)
    model._collect_features = False
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        functional.reset_net(model)
        _ = model(dummy)
        functional.reset_net(model)
    model._collect_features = was_collect
    for h in handles:
        h.remove()
    return shapes


# --------------------------------------------------------------------------- #
#  Source only
# --------------------------------------------------------------------------- #
class SourceOnlyEngine:
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device

    def predict_and_adapt(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        functional.reset_net(self.model)
        with torch.no_grad():
            logits = self.model(x)
        return logits, {"loss": float(softmax_entropy(logits).mean().item())}


# --------------------------------------------------------------------------- #
#  TENT (BP) - robust version with reliability gating
# --------------------------------------------------------------------------- #
class TentEngine:
    """TENT: Test-time entropy minimization on BN affine parameters.

    Follows the original TENT paper (Wang et al., ICLR 2021):
    - Freezes all model parameters except BN affine (gamma/beta)
    - BN runs in train mode so running statistics are updated from test batches
    - Minimizes prediction entropy via Adam (lr=1e-3 for CIFAR datasets)

    For SNN models, LIF neurons are set to train mode so the surrogate gradient
    path is available. For ANN models, all layers are in train mode.

    Optional reliability gating (entropy window + per-sample filter + grad clip)
    can be enabled for robustness on heavily corrupted inputs.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, bn: str = "all",
                 steps: int = 1, gate: bool = True,
                 ent_min_batch: float = 0.01, ent_max_batch: float = 2.40,
                 sample_ent_thresh: float = 1.2, max_grad_norm: float = 1.0,
                 device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.steps = steps
        self.gate = gate
        self.ent_min_batch = float(ent_min_batch)
        self.ent_max_batch = float(ent_max_batch)
        self.sample_ent_thresh = float(sample_ent_thresh)
        self.max_grad_norm = float(max_grad_norm)
        freeze_model(model, keep_bn_affine=True)
        model.train()
        # BN stays in train mode so running statistics are updated
        # (this matches the original TENT implementation)
        params, names = collect_bn_affine_params(model, last_only=(bn == "last"))
        for p in params:
            p.requires_grad_(True)
        self.names = names
        self.param_names = names
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    def predict_and_adapt(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        stats = {"loss": 0.0, "entropy": 0.0, "gate": 1.0, "adapted": 0.0}

        # batch=1 workaround: BN layers in train mode need ≥2 samples.
        # Temporarily switch BN to eval mode so forward passes don't crash.
        batch_size = x.size(0)
        bn_eval_mode = batch_size == 1
        if bn_eval_mode:
            _bn_states = {}
            for nm, m in self.model.named_modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    _bn_states[nm] = m.training
                    m.eval()

        for _ in range(self.steps):
            functional.reset_net(self.model)
            logits = self.model(x)
            ent = softmax_entropy(logits)
            mean_ent = float(ent.mean().item())
            stats.update({"loss": mean_ent, "entropy": mean_ent})

            if self.gate and not (self.ent_min_batch < mean_ent < self.ent_max_batch):
                functional.reset_net(self.model)
                if bn_eval_mode:
                    for nm, m in self.model.named_modules():
                        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                            if nm in _bn_states and _bn_states[nm]:
                                m.train()
                return logits.detach(), stats

            mask = ent < self.sample_ent_thresh
            loss = ent[mask].mean() if mask.any() else ent.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.optimizer.param_groups[0]["params"]],
                self.max_grad_norm)
            self.optimizer.step()
            stats["adapted"] = 1.0
        functional.reset_net(self.model)

        # Restore BN train mode after adaptation (for next batch)
        if bn_eval_mode:
            for nm, m in self.model.named_modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    if nm in _bn_states and _bn_states[nm]:
                        m.train()

        return logits.detach(), stats


# --------------------------------------------------------------------------- #
#  ZO TTA (proposed)
# --------------------------------------------------------------------------- #
class ZOTTAEngine:
    def __init__(
        self,
        model: nn.Module,
        adapt: str = "adapter",
        adapter_kind: str = "channel",
        target_layers: Tuple[str, ...] = ("pool3",),
        img_size: int = 32,
        lr: float = 1e-3,
        eps: float = 1e-2,
        momentum: float = 0.0,
        num_samples: int = 1,
        block_sparsity: float = 1.0,
        weight_decay: float = 1e-3,
        zo_steps: int = 1,
        estimator: str = "two_point",
        # --- reliability gating / anti-collapse -----------------------------
        gate: bool = True,
        ent_min_batch: float = 0.01,
        ent_max_batch: float = 2.40,
        sample_ent_thresh: float = 1.2,
        clamp_abs: float = 0.2,
        rollback_collapse: bool = True,
        use_temporal_gate: bool = False,
        stability_thresh: float = 0.65,
        gate_scale: float = 8.0,
        adapt_gate_min: float = 0.35,
        lambda_rate: float = 0.0,
        lambda_tda: float = 0.0,
        # --- objective & candidate-sweep knobs ------------------------------
        objective: str = "entropy",
        entropy_thresh: float = 0.6,
        conf_thresh: float = 0.8,
        temp: float = 1.0,
        wd_init: float = 0.0,
        kl_lambda: float = 0.0,
        clip_norm: float = 0.0,
        reset_per_batch: bool = False,
        # --- SNN-specific extensions (phase 1b) -----------------------------
        use_membrane_gate: bool = False,
        membrane_gate_threshold: float = 0.05,
        membrane_gate_scale: float = 10.0,
        use_membrane_subspace: bool = False,
        membrane_subspace_rank: int = 16,
        membrane_subspace_update_every: int = 10,
        membrane_kl_lambda: float = 0.0,
        ref_stats: Optional[Dict] = None,
        seed: int = 0,
        device: torch.device = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.adapt = adapt
        self.target_layers = tuple(target_layers)
        self.zo_steps = int(zo_steps)
        self.estimator = estimator
        self.gate = gate
        self.ent_min_batch = float(ent_min_batch)
        self.ent_max_batch = float(ent_max_batch)
        self.sample_ent_thresh = float(sample_ent_thresh)
        self.clamp_abs = float(clamp_abs)
        self.rollback_collapse = bool(rollback_collapse)
        self.use_temporal_gate = bool(use_temporal_gate)
        self.stability_thresh = float(stability_thresh)
        self.gate_scale = float(gate_scale)
        self.adapt_gate_min = float(adapt_gate_min)
        self.lambda_rate = float(lambda_rate)
        self.lambda_tda = float(lambda_tda)
        self.objective = objective
        assert objective in ("entropy", "filtered_entropy", "margin",
                             "logsumexp", "pseudo_label", "temp_entropy",
                             "membrane_kl"), objective
        self.entropy_thresh = float(entropy_thresh)
        self.conf_thresh = float(conf_thresh)
        self.temp = float(temp)
        self.wd_init = float(wd_init)
        self.kl_lambda = float(kl_lambda)
        self.clip_norm = float(clip_norm)
        self.reset_per_batch = bool(reset_per_batch)
        self.num_steps = int(getattr(model, "num_steps", 1))

        # SNN-specific extensions
        self.use_membrane_gate = bool(use_membrane_gate)
        self.membrane_gate_threshold = float(membrane_gate_threshold)
        self.membrane_gate_scale = float(membrane_gate_scale)
        self.use_membrane_subspace = bool(use_membrane_subspace)
        self.membrane_subspace_rank = int(membrane_subspace_rank)
        self.membrane_subspace_update_every = int(membrane_subspace_update_every)
        self.membrane_kl_lambda = float(membrane_kl_lambda)
        self.ref_stats = ref_stats

        freeze_model(model)
        self.hooks: Optional[AdapterHooks] = None

        if adapt == "adapter":
            channels = _infer_layer_channels(model, target_layers, img_size, self.device)
            adapters = {
                name: build_adapter(adapter_kind, channels[name], self.num_steps).to(self.device)
                for name in target_layers
            }
            self.hooks = AdapterHooks(model, adapters)
            self.hooks.set_collect(False)
            params = self.hooks.parameters()
            self.param_names = [n for n, _ in self.hooks.named_parameters()]
        elif adapt in ("bn_all", "bn_last"):
            params, names = collect_bn_affine_params(model, last_only=(adapt == "bn_last"))
            for p in params:
                p.requires_grad_(True)
            self.param_names = names
        else:
            raise ValueError(f"unknown adapt mode {adapt}")

        if not params:
            raise ValueError("no parameters selected for adaptation")

        # Membrane subspace ZO or standard ZO
        if self.use_membrane_subspace:
            self.zo = MembraneSubspaceZO(
                params, model, self.num_steps,
                lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum,
                num_samples=num_samples, subspace_rank=self.membrane_subspace_rank,
                update_every=self.membrane_subspace_update_every,
                estimator=estimator, seed=seed, device=self.device,
            )
        else:
            self.zo = ZoOptimizer(
                params, lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum,
                num_samples=num_samples, block_sparsity=block_sparsity,
                estimator=estimator, max_grad_norm=self.clip_norm,
                seed=seed, device=self.device,
            )
        self._init_params = [p.detach().clone() for p in self.zo.params]
        self._src_probs: Optional[torch.Tensor] = None
        self._enc_seed = int(seed) + 1_000_000
        self._batch_count = 0

    # ------------------------------------------------------------------ #
    def _forward(self, x: torch.Tensor, enc_seed: Optional[int] = None,
                 collect: bool = False) -> torch.Tensor:
        functional.reset_net(self.model)
        if enc_seed is not None:
            torch.manual_seed(enc_seed)
        if self.hooks is not None:
            self.hooks.begin_sequence()
            self.hooks.set_collect(collect)
        self.model._collect_features = collect
        with torch.no_grad():
            logits = self.model(x)
        self.model._collect_features = False
        if self.hooks is not None:
            self.hooks.set_collect(False)
        return logits

    def _clamp_adapter(self):
        if self.hooks is None:
            return
        with torch.no_grad():
            for p in self.hooks.parameters():
                p.clamp_(-self.clamp_abs, self.clamp_abs)

    def _objective_loss(self, logits: torch.Tensor) -> torch.Tensor:
        obj = self.objective
        if obj in ("entropy", "temp_entropy"):
            e = temp_entropy(logits, self.temp) if obj == "temp_entropy" else softmax_entropy(logits)
            if self.sample_ent_thresh > 0:
                mask = e < self.sample_ent_thresh
                loss = e[mask].mean() if mask.any() else e.mean()
            else:
                loss = e.mean()
            if self.kl_lambda > 0 and self._src_probs is not None:
                loss = loss + self.kl_lambda * kl_to_source(logits, self._src_probs)
            return loss
        if obj == "filtered_entropy":
            loss, _ = filtered_entropy(logits, thresh=self.entropy_thresh)
            return loss
        if obj == "margin":
            return margin_loss(logits)
        if obj == "logsumexp":
            return logsumexp_loss(logits)
        if obj == "pseudo_label":
            loss, _ = pseudo_label_ce(logits, conf_thresh=self.conf_thresh)
            return loss
        if obj == "membrane_kl":
            # Membrane KL alignment loss
            if self.ref_stats is not None:
                return membrane_kl_loss(self.model, self.ref_stats)
            # Fallback to entropy if no ref_stats
            return softmax_entropy(logits).mean()
        raise ValueError(f"unknown objective {obj}")

    def _objective(self, x: torch.Tensor, enc_seed: int) -> float:
        functional.reset_net(self.model)
        torch.manual_seed(enc_seed)
        collect = self.lambda_rate > 0 or self.lambda_tda > 0 or self.objective == "membrane_kl"
        if self.hooks is not None:
            self.hooks.begin_sequence()
            self.hooks.set_collect(collect)
        self.model._collect_features = collect
        with torch.no_grad():
            logits = self.model(x)
        self.model._collect_features = False
        if self.hooks is not None:
            self.hooks.set_collect(False)
        loss = self._objective_loss(logits)
        if collect and self.hooks is not None:
            feats = self.hooks.collect()
            last = self.target_layers[-1]
            post = feats[last]["post"]
            pre = feats[last]["pre"]
            if self.lambda_rate > 0 and post is not None:
                loss = loss + self.lambda_rate * mean_spike_rate(post)
            if self.lambda_tda > 0 and pre is not None and post is not None:
                loss = loss + self.lambda_tda * temporal_state_loss(pre, post)
        return float(loss.item())

    # ------------------------------------------------------------------ #
    def predict_and_adapt(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x = x.to(self.device)
        enc_seed = self._enc_seed + self._batch_count

        # 1) prediction forward (pre-adaptation) + source snapshot
        collect = self.lambda_tda > 0 or (self.gate and self.use_temporal_gate) or self.use_membrane_gate
        logits = self._forward(x, enc_seed=enc_seed, collect=collect)
        ent = softmax_entropy(logits)
        mean_ent = float(ent.mean().item())
        self._src_probs = logits.detach().softmax(1)
        stats: Dict[str, float] = {
            "loss": mean_ent, "entropy": mean_ent, "gate": 1.0,
            "stability": 1.0, "projected_grad": 0.0, "adapted": 0.0,
            "rolled_back": 0.0, "membrane_gate": 1.0, "membrane_conv": 0.0,
        }

        # 2) batch-level reliability gates
        adapt = True
        if self.gate and not (self.ent_min_batch < mean_ent < self.ent_max_batch):
            adapt = False

        # 2b) Membrane convergence gate (SNN-specific)
        if adapt and self.use_membrane_gate:
            mem_gate, mem_stats = membrane_gate(
                self.model, x, self.num_steps,
                threshold=self.membrane_gate_threshold,
                gate_scale=self.membrane_gate_scale,
            )
            stats["membrane_gate"] = float(mem_gate.mean().item())
            stats["membrane_conv"] = mem_stats["mean_convergence"]
            if mem_stats["gate_frac"] < 0.3:  # less than 30% samples reliable
                adapt = False

        # 2c) Temporal stability gate (PAR-style)
        if adapt and self.hooks is not None and collect and self.use_temporal_gate:
            feats = self.hooks.collect()
            post = feats[self.target_layers[-1]]["post"]
            if post is not None:
                _, gstats = reliability_gate(
                    post, stability_thresh=self.stability_thresh,
                    gate_scale=self.gate_scale)
                stats.update(gstats)
                if gstats["gate"] < self.adapt_gate_min:
                    adapt = False

        # 3) ZO adaptation
        if adapt:
            if self.reset_per_batch:
                with torch.no_grad():
                    for p, p0 in zip(self.zo.params, self._init_params):
                        p.data.copy_(p0)
            saved = None
            if self.rollback_collapse:
                saved = [p.detach().clone() for p in self.zo.params]
            baseline = None
            if self.estimator == "one_point" and self.lambda_rate == 0 and self.lambda_tda == 0 and not self.use_membrane_gate:
                baseline = float(self._objective_loss(logits).item())
            obj = lambda: self._objective(x, enc_seed)
            for _ in range(self.zo_steps):
                if self.estimator == "two_point":
                    r = self.zo.step(obj)
                else:
                    r = self.zo.step_one_point(obj, baseline=baseline)
                    baseline = r["loss_plus"]
                stats["projected_grad"] = abs(r["projected_grad"])
            self._clamp_adapter()
            if self.wd_init > 0:
                with torch.no_grad():
                    for p, p0 in zip(self.zo.params, self._init_params):
                        p.data.sub_(self.wd_init * (p.data - p0))
            stats["adapted"] = 1.0

            if self.rollback_collapse:
                logits_post = self._forward(x, enc_seed=enc_seed, collect=False)
                ent_post = float(softmax_entropy(logits_post).mean().item())
                if ent_post < 0.01:  # collapse detection: near-zero entropy
                    for p, s in zip(self.zo.params, saved):
                        p.data.copy_(s)
                    stats["rolled_back"] = 1.0
                    stats["adapted"] = 0.0
        self._batch_count += 1
        return logits, stats


# --------------------------------------------------------------------------- #
#  Evaluation runners
# --------------------------------------------------------------------------- #
def evaluate_on_loader(engine, loader, device, logger=None,
                       max_batches: Optional[int] = None,
                       profile_memory: bool = False) -> dict:
    """Run TTA engine over a loader; returns accuracy + adaptation stats.

    If profile_memory=True, records peak CUDA memory (requires torch.cuda).
    """
    accs, losses, gates, projs = [], [], [], []
    adapted, rolled_back = [], []
    mem_gates, mem_convs = [], []
    peak_memory_mb = 0.0

    if profile_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, stats = engine.predict_and_adapt(x)
        accs.append(accuracy(logits, y)[0])
        losses.append(stats["loss"])
        gates.append(stats.get("gate", 1.0))
        projs.append(stats.get("projected_grad", 0.0))
        adapted.append(stats.get("adapted", 0.0))
        rolled_back.append(stats.get("rolled_back", 0.0))
        mem_gates.append(stats.get("membrane_gate", 1.0))
        mem_convs.append(stats.get("membrane_conv", 0.0))

        if profile_memory and torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_memory_mb = max(peak_memory_mb, current_peak)

        if logger is not None and (i + 1) % 20 == 0:
            logger.info(f"  batch {i+1}/{len(loader)} acc={accs[-1]:.2f}% "
                        f"loss={losses[-1]:.4f} gate={gates[-1]:.3f} "
                        f"|g|={projs[-1]:.2e} adapted={adapted[-1]:.0f} "
                        f"rb={rolled_back[-1]:.0f}")
    n = len(accs)
    result = {
        "acc": sum(accs) / n if n else 0.0,
        "loss": sum(losses) / n if n else 0.0,
        "gate": sum(gates) / n if n else 0.0,
        "mean_abs_proj_grad": sum(projs) / n if n else 0.0,
        "adapted_frac": sum(adapted) / n if n else 0.0,
        "rolled_back_frac": sum(rolled_back) / n if n else 0.0,
        "membrane_gate": sum(mem_gates) / n if n else 0.0,
        "membrane_convergence": sum(mem_convs) / n if n else 0.0,
        "num_batches": n,
        "peak_memory_mb": round(peak_memory_mb, 1),
    }
    return result


def run_tta_corruptions(
    engine_factory: Callable[[], nn.Module],
    cifar10c_root: str,
    corruptions: List[str],
    batch_size: int = 64,
    num_workers: int = 2,
    level: int = 5,
    device=None,
    logger=None,
    max_batches: Optional[int] = None,
    profile_memory: bool = False,
) -> dict:
    """Run an engine over all corruptions; returns {corruption: acc} + mean."""
    from .data import get_cifar10c_loader

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    peak_mem = 0.0
    for name in corruptions:
        loader = get_cifar10c_loader(cifar10c_root, name, batch_size, num_workers, level)
        engine = engine_factory()
        r = evaluate_on_loader(engine, loader, device, logger=logger,
                               max_batches=max_batches, profile_memory=profile_memory)
        results[name] = r["acc"]
        peak_mem = max(peak_mem, r.get("peak_memory_mb", 0))
        if logger is not None:
            logger.info(f"[{name}] acc={r['acc']:.2f}% loss={r['loss']:.4f} "
                        f"gate={r['gate']:.3f} adapted={r['adapted_frac']:.2f} "
                        f"rolled_back={r['rolled_back_frac']:.2f}")
    results["mean"] = sum(results.values()) / len(results) if results else 0.0
    results["peak_memory_mb"] = peak_mem
    return results