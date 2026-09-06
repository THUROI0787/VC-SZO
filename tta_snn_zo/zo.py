"""Zeroth-order (ZO) optimization engine.

Implements the MeZO two-point estimator (Malladi et al., NeurIPS 2023):

    z ~ N(0, I) per parameter
    g_hat = ( L(theta + eps*z) - L(theta - eps*z) ) / (2*eps) * z
    theta <- theta - lr * g_hat        (optional weight decay on non-bias params)

plus practical extensions needed for SNN test-time adaptation:

* ``num_samples``   : average over several independent z draws (variance reduction)
* ``momentum``      : EMA of the per-coordinate gradient estimate g_hat
* ``block_sparsity``: random-subspace / blockwise ZO - only a fraction of
                      coordinates are perturbed per step (SZO-inspired; cuts
                      variance ~ d -> d_eff and lowers compute)
* ``one_point``     : one-point estimator with baseline (OPZO-inspired); used in
                      phase 2.  Reuses the prediction-forward loss as baseline to
                      keep a single extra forward pass per step.

The engine is agnostic to the model: the caller supplies ``objective_fn()``
(a callable returning a scalar loss) that handles SNN membrane resetting, and the
parameter list to adapt.
"""
from __future__ import annotations

import math
from typing import Callable, List

import torch
import torch.nn as nn


class ZoOptimizer:
    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-2,
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        num_samples: int = 1,
        block_sparsity: float = 1.0,
        estimator: str = "two_point",  # 'two_point' | 'one_point'
        max_grad_norm: float = 0.0,    # per-parameter L2 clip on g_hat (0 = off)
        seed: int = 0,
        device: torch.device = None,
    ):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("ZoOptimizer needs at least one trainable parameter")
        self.lr = float(lr)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.num_samples = int(num_samples)
        self.block_sparsity = float(block_sparsity)
        assert 0.0 < self.block_sparsity <= 1.0
        self.estimator = estimator
        assert estimator in ("two_point", "one_point")
        self.max_grad_norm = float(max_grad_norm)

        dev = device or self.params[0].device
        self.gen = torch.Generator(device=dev).manual_seed(int(seed))
        self._z: List[torch.Tensor] | None = None
        self._buf: List[torch.Tensor] | None = None
        if self.momentum > 0:
            self._buf = [torch.zeros_like(p.data) for p in self.params]
        self.num_params = sum(p.numel() for p in self.params)
        self.projected_grad = 0.0
        self.last_loss_plus = 0.0
        self.last_loss_minus = 0.0
        self.steps = 0

    # ------------------------------------------------------------------ #
    def _sample_direction(self) -> List[torch.Tensor]: 
        zs = [torch.empty_like(p).normal_(generator=self.gen) for p in self.params]
        if self.block_sparsity < 1.0:
            # random-subspace: keep a fraction of coordinates per parameter
            keep = torch.rand(self.num_params, generator=self.gen,
                              device=self.params[0].device) < self.block_sparsity
            idx = 0
            for z, p in zip(zs, self.params):
                n = p.numel()
                mask = keep[idx : idx + n].reshape_as(p).to(p.dtype)
                z.mul_(mask)
                idx += n
        return zs

    def _apply(self, zs: List[torch.Tensor], alpha: float) -> None:
        for p, z in zip(self.params, zs):
            p.data.add_(z, alpha=alpha)

    # ------------------------------------------------------------------ #
    def _evaluate(self, objective_fn: Callable[[], float]) -> float:
        return float(objective_fn())

    def _projected_gradient(self, loss_p: float, loss_m: float) -> float:
        return (loss_p - loss_m) / (2.0 * self.eps)

    def _update(self, scale: float, ghat_list: List[torch.Tensor]) -> None:
        """Apply update: theta <- theta - lr * scale * ghat, with optional momentum & clipping.

        Args:
            scale: multiplier for ghat (typically 1.0 when ghat is already averaged,
                   or proj when ghat is a single-sample direction z)
            ghat_list: list of gradient tensors, one per parameter
        """
        with torch.no_grad():
            for i, (p, g_hat) in enumerate(zip(self.params, ghat_list)):
                g_hat = g_hat * scale
                if self._buf is not None:
                    self._buf[i].mul_(self.momentum).add_(g_hat, alpha=1.0 - self.momentum)
                    g_hat = self._buf[i]
                if self.max_grad_norm > 0:
                    gn = g_hat.norm()
                    if gn > self.max_grad_norm:
                        g_hat = g_hat * (self.max_grad_norm / (gn + 1e-12))
                update = self.lr * g_hat
                if self.weight_decay > 0:
                    update = update + self.lr * self.weight_decay * p.data
                p.data.sub_(update)

    # ------------------------------------------------------------------ #
    def step(self, objective_fn: Callable[[], float]) -> dict:
        """One two-point (antithetic) ZO step. Returns stats dict.

        FIXED: Now correctly averages the per-parameter gradient g_hat across
        multiple samples, instead of averaging only the scalar projected gradient
        and applying it with the last sample's direction.
        """
        proj_acc = 0.0
        lp_acc, lm_acc = 0.0, 0.0
        ghat_acc = None  # accumulated per-parameter gradient

        with torch.no_grad():
            for _ in range(self.num_samples):
                zs = self._sample_direction()
                self._apply(zs, +self.eps)
                loss_p = self._evaluate(objective_fn)
                self._apply(zs, -2.0 * self.eps)
                loss_m = self._evaluate(objective_fn)
                self._apply(zs, +self.eps)  # restore theta

                proj = self._projected_gradient(loss_p, loss_m)
                proj_acc += proj

                # Accumulate per-parameter gradient: g_hat = proj * z
                if ghat_acc is None:
                    ghat_acc = [proj * z for z in zs]
                else:
                    for i, z in enumerate(zs):
                        ghat_acc[i] += proj * z

                lp_acc += loss_p
                lm_acc += loss_m

        self.projected_grad = proj_acc / self.num_samples
        self.last_loss_plus = lp_acc / self.num_samples
        self.last_loss_minus = lm_acc / self.num_samples

        # Average the accumulated per-parameter gradient
        avg_ghat = [g / self.num_samples for g in ghat_acc]
        # Apply update with scale=1.0 (ghat is already the averaged gradient)
        self._update(1.0, avg_ghat)
        self.steps += 1

        return {
            "loss_plus": self.last_loss_plus,
            "loss_minus": self.last_loss_minus,
            "projected_grad": self.projected_grad,
            "num_forward": 2 * self.num_samples,
        }

    def step_one_point(self, objective_fn: Callable[[], float], baseline: float = None) -> dict:
        """One-point ZO step: L(theta+eps z) with optional baseline L(theta).

        If ``baseline`` is None, an extra unperturbed forward is made to estimate
        it (costs 2 forwards total); if provided (e.g. the prediction-forward
        loss already computed for TTA), it costs a single extra forward.
        """
        with torch.no_grad():
            zs = self._sample_direction()
            self._apply(zs, +self.eps)
            loss_p = self._evaluate(objective_fn)
            self._apply(zs, -self.eps)  # restore
            if baseline is None:
                baseline = self._evaluate(objective_fn)
            proj = (loss_p - baseline) / self.eps
            # For one-point, g_hat = proj * z (single sample)
            ghat = [proj * z for z in zs]
        self.projected_grad = proj
        self.last_loss_plus = loss_p
        self.last_loss_minus = baseline
        self._update(1.0, ghat)
        self.steps += 1
        return {"loss_plus": loss_p, "loss_minus": baseline,
                "projected_grad": proj, "num_forward": 2 if baseline is None else 1}

    def estimate(self, objective_fn: Callable[[], float], num_samples: int = None) -> tuple:
        """Averaged two-point gradient estimate *without* updating parameters.

        Returns (est_params, projected_grad).  Used by the sanity check to
        measure gradient fidelity (cosine similarity with the surrogate gradient)
        and to demonstrate that the estimate aligns with the true gradient as
        ``num_samples`` grows (variance reduction).
        """
        num_samples = num_samples or self.num_samples
        acc = [torch.zeros_like(p) for p in self.params]
        proj = 0.0
        with torch.no_grad():
            for _ in range(num_samples):
                zs = self._sample_direction()
                self._apply(zs, +self.eps)
                lp = self._evaluate(objective_fn)
                self._apply(zs, -2.0 * self.eps)
                lm = self._evaluate(objective_fn)
                self._apply(zs, +self.eps)
                g = (lp - lm) / (2.0 * self.eps)
                proj += g
                for p, z, a in zip(self.params, zs, acc):
                    a.add_(z, alpha=g)
        for a in acc:
            a.div_(num_samples)
        return acc, proj / num_samples

    def zero_grad(self):
        """ZO has no autograd graph; provided for interface parity."""
        pass


def collect_bn_affine_params(model: nn.Module, last_only: bool = False) -> tuple:
    """Collect BN affine (gamma/weight) parameters, TENT-style.

    Returns (params, names). With BNTT, every per-timestep BN is included.
    ``last_only`` keeps only the final conv block + FC block BNs (smaller dim,
    better for ZO variance).
    """
    params, names = [], []
    blocks = []
    for nm, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            blocks.append((nm, m))
    if last_only and blocks:
        # keep the last ~2 blocks (by name order, conv blocks then fc)
        blocks = blocks[-2:]
    for nm, m in blocks:
        for np_, p in m.named_parameters():
            if np_ in ("weight", "bias") and p is not None:
                params.append(p)
                names.append(f"{nm}.{np_}")
    return params, names


def freeze_model(model: nn.Module, keep_bn_affine: bool = False) -> None:
    """Freeze all parameters; optionally keep BN affine trainable; BN runs in
    eval mode so running statistics stay frozen (safer for SNN TTA)."""
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()
            if keep_bn_affine:
                for p in m.parameters():
                    p.requires_grad_(True)