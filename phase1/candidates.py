"""Phase-1 candidate definitions for the ZO-TTA sweep — Refined Set.

DESIGN PHILOSOPHY (v3 — after Phase 1 analysis):
Phase 1 showed that on T=6 SNN-VGG9, ALL methods (including TENT-BP and MEMO)
barely beat source-only. The bottleneck is NOT ZO method design but the weak
source model. For the refined sweep we:

1. Keep ~20 diverse candidates covering fundamentally different approaches
2. Include multiple hyperparameter variants for the most promising methods
3. Ensure coverage of known ZO variance-reduction techniques
4. Add stronger ZO settings (higher lr, more steps) for T=25 source

We organize candidates into 5 METHOD FAMILIES (A-E):

  A. Vanilla ZO-TTA (3 candidates — reference baselines)
  B. Variance-Reduced ZO (5 candidates — SZO-inspired subspace + momentum)
  C. SNN-Specific Objectives (4 candidates — membrane/spike-aware losses)
  D. Structured Perturbation (4 candidates — BN affine, coordinate/block)
  E. Strong ZO (4 candidates — higher lr/steps for T=25 source)

Total: ~20 candidates + 4 baselines = 24 configurations.
"""
from __future__ import annotations

from typing import Dict, List

from tta_snn_zo.tta import ZOTTAEngine

# --------------------------------------------------------------------------- #
#  Default engine kwargs (safe defaults)
# --------------------------------------------------------------------------- #
DEFAULT_ENGINE: Dict = dict(
    adapt="adapter",
    adapter_kind="channel",
    target_layers=("pool3",),
    img_size=32,
    lr=1e-3,
    eps=1e-2,
    momentum=0.0,
    num_samples=1,
    block_sparsity=1.0,
    weight_decay=1e-3,
    zo_steps=1,
    estimator="two_point",
    gate=True,
    ent_min_batch=0.30,
    ent_max_batch=2.30,
    sample_ent_thresh=1.2,
    clamp_abs=0.2,
    rollback_collapse=True,
    use_temporal_gate=False,
    stability_thresh=0.65,
    gate_scale=8.0,
    adapt_gate_min=0.35,
    lambda_rate=0.0,
    lambda_tda=0.0,
    objective="entropy",
    entropy_thresh=0.6,
    conf_thresh=0.8,
    temp=1.0,
    wd_init=0.0,
    kl_lambda=0.0,
    clip_norm=0.0,
    reset_per_batch=False,
    # SNN-specific defaults (all off)
    use_membrane_gate=False,
    membrane_gate_threshold=0.05,
    membrane_gate_scale=10.0,
    use_membrane_subspace=False,
    membrane_subspace_rank=16,
    membrane_subspace_update_every=10,
    membrane_kl_lambda=0.0,
    ref_stats=None,
)

ALLOWED_KEYS = set(DEFAULT_ENGINE.keys())
OBJECTIVES = ("entropy", "filtered_entropy", "margin", "logsumexp",
              "pseudo_label", "temp_entropy", "membrane_kl")


def _c(cid: str, family: str, method_name: str,
       question: str, expected: str, **overrides) -> Dict:
    """Build a candidate dict from DEFAULT_ENGINE + overrides."""
    cfg = dict(DEFAULT_ENGINE)
    cfg.update(overrides)
    return {
        "id": cid,
        "family": family,
        "method_name": method_name,
        "question": question,
        "expected": expected,
        "engine": cfg,
    }


# =========================================================================== #
#  FAMILY A: Vanilla ZO-TTA (reference baselines)
#  Core question: Does basic ZO-TTA work at all on SNNs?
# =========================================================================== #
FAMILY_A: List[Dict] = [
    # A1: Standard MeZO two-point (reference)
    _c("a01", "A_vanilla", "MeZO-2pt (lr=1e-3)",
       "Does standard MeZO two-point work on SNN TTA?",
       "Reference baseline, may struggle with spike variance",
       lr=1e-3),

    # A2: Higher lr (stronger signal)
    _c("a02", "A_vanilla", "MeZO-2pt (lr=3e-3)",
       "Higher lr to overcome small gradient signal?",
       "May collapse faster but stronger update",
       lr=3e-3),

    # A3: Multi-step ZO (accumulate updates)
    _c("a03", "A_vanilla", "MeZO-2pt K=3",
       "Multiple ZO steps per batch — more aggressive adaptation",
       "Risk of over-adaptation on single batch",
       lr=1e-3, zo_steps=3),
]

# =========================================================================== #
#  FAMILY B: Variance-Reduced ZO (SZO-inspired)
#  Core idea: Reduce ZO variance by restricting/structuring the perturbation
#  space. SZO (ICML 2026) shows spike activation amplifies ZO variance.
# =========================================================================== #
FAMILY_B: List[Dict] = [
    # B1: Random subspace 50% — simplest subspace method
    _c("b01", "B_subspace", "Random subspace 50%",
       "Perturb only 50% of coordinates each step — SZO-lite",
       "Lower effective dimension -> lower variance",
       lr=1e-3, block_sparsity=0.5),

    # B2: Subspace + momentum (SZO's practical recipe)
    _c("b02", "B_subspace", "Subspace 50% + momentum 0.9",
       "Combine subspace with momentum EMA — SZO's practical recipe",
       "Smooth gradient trajectory in subspace",
       lr=1e-3, block_sparsity=0.5, momentum=0.9),

    # B3: Multi-sample variance reduction (ns=8)
    _c("b03", "B_subspace", "ZO ns=8 variance reduction",
       "8-sample averaging — strong variance reduction",
       "Should give cleanest gradient estimate at 8x forward cost",
       lr=1e-3, num_samples=8),

    # B4: Momentum + multi-sample (combined)
    _c("b04", "B_subspace", "ZO momentum=0.9 ns=4",
       "Momentum EMA + 4-sample averaging",
       "Momentum smooths trajectory, samples reduce variance",
       lr=1e-3, momentum=0.9, num_samples=4),

    # B5: Membrane subspace ZO (our SNN-specific SZO)
    _c("b05", "B_subspace", "Membrane subspace ZO rank=16",
       "Project ZO onto membrane potential principal subspace",
       "SNN-specific: leverages LIF dynamics for subspace",
       lr=1e-3, use_membrane_subspace=True, membrane_subspace_rank=16),
]

# =========================================================================== #
#  FAMILY C: SNN-Specific Objectives
#  Core idea: Replace entropy with objectives that leverage SNN signals.
# =========================================================================== #
FAMILY_C: List[Dict] = [
    # C1: Membrane KL alignment (our SNN-specific objective)
    _c("c01", "C_snn_objective", "Membrane KL alignment λ=1.0",
       "Align membrane distribution with source — no entropy needed",
       "SNN-specific: leverages LIF dynamics, very stable",
       objective="membrane_kl", membrane_kl_lambda=1.0, lr=1e-3),

    # C2: Membrane KL + higher weight
    _c("c02", "C_snn_objective", "Membrane KL alignment λ=2.0",
       "Stronger membrane KL alignment",
       "May over-regularize but more stable",
       objective="membrane_kl", membrane_kl_lambda=2.0, lr=1e-3),

    # C3: Margin loss (smoother than entropy, less collapse-prone)
    _c("c03", "C_snn_objective", "Margin loss (top1-top2)",
       "Maximize decision gap instead of minimizing entropy",
       "Smoother landscape -> less collapse",
       objective="margin", lr=1e-3),

    # C4: Pseudo-label CE (self-training)
    _c("c04", "C_snn_objective", "Pseudo-label CE conf=0.8",
       "Self-training with confidence-gated pseudo-labels",
       "More stable than entropy, needs good initial predictions",
       objective="pseudo_label", conf_thresh=0.8, lr=1e-3),
]

# =========================================================================== #
#  FAMILY D: Structured Perturbation
#  Core idea: Change WHAT/WHERE we perturb, not just how.
# =========================================================================== #
FAMILY_D: List[Dict] = [
    # D1: BN affine ZO (all layers) — different parameterization
    _c("d01", "D_structured", "ZO on BN affine (all layers)",
       "Adapt all BN affine params — more parameters but more flexible",
       "Higher dimension -> higher ZO variance, needs lower lr",
       adapt="bn_all", lr=1e-4),

    # D2: BN affine (last block only) — fewer params
    _c("d02", "D_structured", "ZO on BN affine (last block)",
       "Adapt BN affine params instead of adapters — different geometry",
       "BN params have different loss landscape than adapters",
       adapt="bn_last", lr=1e-4),

    # D3: KL-to-source regularization (EATA-style anti-drift)
    _c("d03", "D_structured", "MeZO-2pt + KL=0.1",
       "KL divergence to source predictions — anti-forgetting",
       "Should prevent collapse on strong corruptions",
       lr=1e-3, kl_lambda=0.1),

    # D4: Weight decay toward init (anti-drift)
    _c("d04", "D_structured", "MeZO-2pt + wd-init=0.1",
       "Regularize adapter toward identity — prevent drift",
       "Conservative but stable",
       lr=1e-3, wd_init=0.1),
]

# =========================================================================== #
#  FAMILY E: Strong ZO (for T=25 source)
#  Core idea: T=25 source has more capacity, can tolerate stronger updates.
# =========================================================================== #
FAMILY_E: List[Dict] = [
    # E1: Higher lr + multi-sample (strong but stable)
    _c("e01", "E_strong_zo", "ZO lr=1e-2 ns=4 strong",
       "Higher lr with multi-sample variance reduction",
       "Stronger updates for T=25 source",
       lr=1e-2, num_samples=4),

    # E2: Higher eps (larger perturbation, better SNR for SNN)
    _c("e02", "E_strong_zo", "ZO eps=5e-2 (large perturbation)",
       "Larger perturbation for better gradient SNR through LIF",
       "SZO shows larger eps helps with spike discontinuity",
       lr=1e-3, eps=5e-2),

    # E3: More ZO steps (K=5)
    _c("e03", "E_strong_zo", "MeZO-2pt K=5 aggressive",
       "5 ZO steps per batch — strong multi-step adaptation",
       "Risk of over-adaptation but more adaptation capacity",
       lr=1e-3, zo_steps=5),

    # E4: One-point estimator (halves forwards, OPZO-inspired)
    _c("e04", "E_strong_zo", "One-point + momentum 0.9",
       "One-point with momentum — OPZO-style momentum feedback",
       "Momentum smooths one-point's higher variance",
       lr=1e-3, estimator="one_point", momentum=0.9),
]

# =========================================================================== #
#  ASSEMBLE ALL CANDIDATES
# =========================================================================== #
ALL_CANDIDATES: List[Dict] = FAMILY_A + FAMILY_B + FAMILY_C + FAMILY_D + FAMILY_E
CANDIDATE_IDS = [c["id"] for c in ALL_CANDIDATES]

# Family summary for reporting
FAMILY_SUMMARY = {
    "A_vanilla": "Vanilla ZO-TTA (MeZO baselines)",
    "B_subspace": "Variance-Reduced ZO (SZO-inspired subspace + momentum)",
    "C_snn_objective": "SNN-Specific Objectives (membrane/spike-aware losses)",
    "D_structured": "Structured Perturbation (BN affine, anti-drift)",
    "E_strong_zo": "Strong ZO (higher lr/steps for T=25 source)",
}

# =========================================================================== #
#  Baselines
# =========================================================================== #
def baseline_source_only():
    return "B0", None


def baseline_tent_fixed():
    """Robust TENT-BP baseline (expert defaults)."""
    return "B1", dict(lr=1e-4, bn="all", steps=1, gate=True,
                      ent_min_batch=0.30, ent_max_batch=2.30,
                      sample_ent_thresh=1.2, max_grad_norm=1.0)


def baseline_tent_adapter():
    """TENT-BP on adapter params (fairer comparison to ZO)."""
    return "B2", dict(lr=1e-4, bn="adapter", steps=1, gate=True,
                      ent_min_batch=0.30, ent_max_batch=2.30,
                      sample_ent_thresh=1.2, max_grad_norm=1.0)


# =========================================================================== #
#  Validation / engine construction
# =========================================================================== #
def validate_candidates(candidates: List[Dict]) -> List[str]:
    """Return list of error strings; empty list means all valid."""
    errors = []
    seen = set()
    for c in candidates:
        cid = c.get("id", "<no-id>")
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)
        eng = c.get("engine", {})
        bad_keys = set(eng.keys()) - ALLOWED_KEYS
        if bad_keys:
            errors.append(f"{cid}: unknown engine keys {sorted(bad_keys)}")
        if "lr" in eng and eng["lr"] <= 0:
            errors.append(f"{cid}: lr must be > 0")
        if "eps" in eng and eng["eps"] <= 0:
            errors.append(f"{cid}: eps must be > 0")
        if "objective" in eng and eng["objective"] not in OBJECTIVES:
            errors.append(f"{cid}: bad objective {eng['objective']}")
        if "block_sparsity" in eng and not (0 < eng["block_sparsity"] <= 1):
            errors.append(f"{cid}: block_sparsity out of range")
        if "zo_steps" in eng and eng["zo_steps"] < 1:
            errors.append(f"{cid}: zo_steps must be >= 1")
        if "conf_thresh" in eng and not (0 <= eng["conf_thresh"] <= 1):
            errors.append(f"{cid}: conf_thresh out of range")
        if "adapt" in eng and eng["adapt"] not in ("adapter", "bn_all", "bn_last"):
            errors.append(f"{cid}: bad adapt {eng['adapt']}")
    return errors


def build_engine_from_cfg(candidate: Dict, model, device, seed: int) -> ZOTTAEngine:
    """Construct a ZOTTAEngine for a candidate dict."""
    eng = dict(candidate["engine"])
    return ZOTTAEngine(model, device=device, seed=int(seed), **eng)


def select_candidates(ids: str = "", families: str = "") -> List[Dict]:
    """Filter candidates by comma-separated --ids / --families."""
    id_set = {x.strip() for x in ids.split(",") if x.strip()}
    fam_set = {x.strip() for x in families.split(",") if x.strip()}
    out = []
    for c in ALL_CANDIDATES:
        if id_set and c["id"] not in id_set:
            continue
        if fam_set and c["family"] not in fam_set:
            continue
        out.append(c)
    return out