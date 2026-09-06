"""Losses and temporal-reliability helpers.

* ``tet_loss``          - TET loss: (1-lambda)*mean_t CE + lambda*mean_t MSE(O_t, phi)
                          with phi = v_threshold (TET, ICLR'22; lambda=5e-2 for CIFAR).
* ``softmax_entropy``   - entropy of softmax over logits (TENT-style objective).
* temporal stability / reliability gate (PAR-style spike-energy trace).
* temporal-dynamics alignment (TDA) loss used by PAR's PHSA, re-used here as an
  optional spike-aware ZO objective in phase 2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  TET loss (source training)
# --------------------------------------------------------------------------- #
def tet_loss(
    logits_t: torch.Tensor,      # [T, B, C]
    targets: torch.Tensor,       # [B]
    lambda_mse: float = 0.05,    # weight of the MSE regularizer (TET: 5e-2)
    v_threshold: float = 1.0,    # phi constant (TET: phi = V_th)
    mse_target: str = "const",   # 'const' -> MSE(O_t, phi); 'onehot' -> MSE(O_t, Vth*onehot)
) -> torch.Tensor:
    T = logits_t.size(0)
    loss_ce = sum(F.cross_entropy(logits_t[t], targets) for t in range(T)) / T
    if mse_target == "const":
        phi = torch.full_like(logits_t, v_threshold)
        loss_mse = F.mse_loss(logits_t, phi)
    elif mse_target == "onehot":
        onehot = F.one_hot(targets, num_classes=logits_t.size(-1)).float() * v_threshold
        loss_mse = sum(F.mse_loss(logits_t[t], onehot) for t in range(T)) / T
    else:
        raise ValueError(mse_target)
    return (1.0 - lambda_mse) * loss_ce + lambda_mse * loss_mse


# --------------------------------------------------------------------------- #
#  Entropy objective (TTA)
# --------------------------------------------------------------------------- #
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample softmax entropy; input [B, C]."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


# --------------------------------------------------------------------------- #
#  PAR-style temporal reliability gate
# --------------------------------------------------------------------------- #
def spike_energy_trace(feat_seq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """feat_seq [B,T,C,H,W] -> [B,T] mean |activation| per timestep."""
    if feat_seq.dim() != 5:
        raise ValueError(f"expected [B,T,C,H,W], got {tuple(feat_seq.shape)}")
    return feat_seq.abs().mean(dim=(2, 3, 4)) + eps


def compute_temporal_stability(trace: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Temporal stability S(X) = 1 - std(e)/mean(e) via exp(-mean|diff|/mean)."""
    if trace.size(1) <= 1:
        return trace.new_tensor(1.0)
    step_diff = (trace[:, 1:] - trace[:, :-1]).abs().mean(dim=1)
    trace_mean = trace.mean(dim=1) + eps
    return torch.exp(-step_diff / trace_mean).mean()


def reliability_gate(
    post_seq: torch.Tensor,          # [B,T,C,H,W] adapted features
    stability_thresh: float = 0.65,
    gate_scale: float = 8.0,
) -> tuple:
    """Returns (gate, stats). gate in [0,1]; skip adaptation when gate is low."""
    trace = spike_energy_trace(post_seq)
    stability = compute_temporal_stability(trace)
    gate = torch.sigmoid(gate_scale * (stability - stability_thresh))
    stats = {"stability": float(stability.detach()), "gate": float(gate.detach())}
    return gate, stats


# --------------------------------------------------------------------------- #
#  Temporal-Dynamics Alignment (TDA) loss (PAR's PHSA objective)
# --------------------------------------------------------------------------- #
def normalized_cumulative_trace(trace: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    cum = trace.cumsum(dim=1)
    denom = cum[:, -1:] + eps
    return cum / denom.detach()


def temporal_state_loss(
    pre_seq: torch.Tensor,            # [B,T,C,H,W]
    post_seq: torch.Tensor,           # [B,T,C,H,W]
    lambda_curve: float = 1.0,
    lambda_diff: float = 0.5,
) -> torch.Tensor:
    """PAR's TDA loss: align normalized cumulative evidence and its temporal
    differences between pre- and post-adaptation hidden states."""
    pre_trace = spike_energy_trace(pre_seq)
    post_trace = spike_energy_trace(post_seq)
    pre_cum = normalized_cumulative_trace(pre_trace)
    post_cum = normalized_cumulative_trace(post_trace)
    loss_curve = F.l1_loss(post_cum, pre_cum.detach())
    if pre_trace.size(1) > 1:
        pre_diff = pre_trace[:, 1:] - pre_trace[:, :-1]
        post_diff = post_trace[:, 1:] - post_trace[:, :-1]
        loss_diff = F.l1_loss(post_diff, pre_diff.detach())
    else:
        loss_diff = pre_trace.new_zeros(())
    return lambda_curve * loss_curve + lambda_diff * loss_diff


def mean_spike_rate(feat_seq: torch.Tensor) -> torch.Tensor:
    """Mean absolute spike rate over [B,T,C,H,W]."""
    return feat_seq.abs().mean()


# --------------------------------------------------------------------------- #
#  Anti-collapse TTA objectives (phase-1 candidate sweep)
#  Raw entropy minimization collapses to a single class on weak SNNs under
#  strong corruptions (observed: TENT-BP 9.95%, ZO-TTA 16.6% vs source 33.8%).
#  These objectives trade raw entropy for stability; each returns a scalar
#  loss and (where relevant) the reliability mask for logging.
# --------------------------------------------------------------------------- #
def filtered_entropy(logits: torch.Tensor, thresh: float = 0.6) -> tuple:
    """EATA-style: mean entropy of reliable (low-entropy) samples only.

    Samples whose softmax entropy exceeds ``thresh`` are masked out.  When no
    sample is reliable the loss is exactly 0 (with a valid grad_fn, so BP-based
    engines can call ``backward()`` safely) -- i.e. the batch is skipped.
    """
    e = softmax_entropy(logits)
    mask = e < thresh
    denom = mask.sum().clamp(min=1).float()
    return (e * mask.float()).sum() / denom, mask


def margin_loss(logits: torch.Tensor) -> torch.Tensor:
    """Negative mean top-1/top-2 logit margin: maximize the decision gap.

    Smoother and much less collapse-prone than entropy: pushing the top-2 gap
    wider does not require all samples to become one-hot toward a single class.
    """
    top2 = logits.topk(2, dim=1).values
    return -(top2[:, 0] - top2[:, 1]).mean()


def logsumexp_loss(logits: torch.Tensor) -> torch.Tensor:
    """-mean(logsumexp(logits)): smooth, convex-in-logits confidence objective."""
    return -logits.logsumexp(1).mean()


def pseudo_label_ce(logits: torch.Tensor, conf_thresh: float = 0.8) -> tuple:
    """Self-training with confidence-gated pseudo labels (CoTTA-style).

    Only samples whose max softmax confidence exceeds ``conf_thresh`` contribute;
    their argmax is used as a fixed pseudo label.  Returns (loss, mask); when no
    sample qualifies the loss is exactly 0 (valid grad_fn).
    """
    probs = logits.softmax(1)
    conf, pseudo = probs.max(1)
    mask = conf > conf_thresh
    denom = mask.sum().clamp(min=1).float()
    loss = (F.cross_entropy(logits, pseudo, reduction="none") * mask.float()).sum() / denom
    return loss, mask


def temp_entropy(logits: torch.Tensor, temp: float = 2.0) -> torch.Tensor:
    """Entropy of the temperature-scaled softmax (softens the objective)."""
    scaled = logits / temp
    p = scaled.softmax(1)
    return -(p * scaled.log_softmax(1)).sum(1)


def kl_to_source(logits: torch.Tensor, src_probs: torch.Tensor) -> torch.Tensor:
    """KL(softmax(logits) || src_probs), batchmean.  Anti-forgetting regularizer
    (EATA-style Fisher prior): keeps adapted predictions close to the source
    model's predictions on the same batch."""
    return F.kl_div(logits.log_softmax(1), src_probs, reduction="batchmean")
