"""Phase-2 candidates: deep dive on c03 (Margin Loss ZO).

Design philosophy:
  Phase 1 found that c03 (ZO + margin loss) consistently outperforms all methods
  at batch=8 across levels 1/3/5. Phase 2 explores the full potential of this
  finding through:

  1. Hyperparameter sweeps (lr, eps, zo_steps, num_samples, momentum)
  2. Margin loss variants (temperature-scaled margin, clamped margin, etc.)
  3. Combined with variance reduction (subspace, momentum, multi-sample)
  4. Combined with anti-drift (KL, wd_init)
  5. Different adapter configurations (BN affine, multiple layers)

  All candidates use objective="margin" unless otherwise noted.
  Setting: level3, batch=8, 15 corruptions (sweet spot from Phase 1).
"""
from __future__ import annotations

from typing import Dict, List

from tta_snn_zo.tta import ZOTTAEngine

# --------------------------------------------------------------------------- #
#  Default engine kwargs (margin loss base)
# --------------------------------------------------------------------------- #
MARGIN_BASE: Dict = dict(
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
    ent_min_batch=0.01,
    ent_max_batch=2.40,
    sample_ent_thresh=1.2,
    clamp_abs=0.2,
    rollback_collapse=True,
    use_temporal_gate=False,
    stability_thresh=0.65,
    gate_scale=8.0,
    adapt_gate_min=0.35,
    lambda_rate=0.0,
    lambda_tda=0.0,
    objective="margin",
    entropy_thresh=0.6,
    conf_thresh=0.8,
    temp=1.0,
    wd_init=0.0,
    kl_lambda=0.0,
    clip_norm=0.0,
    reset_per_batch=False,
    use_membrane_gate=False,
    membrane_gate_threshold=0.05,
    membrane_gate_scale=10.0,
    use_membrane_subspace=False,
    membrane_subspace_rank=16,
    membrane_subspace_update_every=10,
    membrane_kl_lambda=0.0,
    ref_stats=None,
)

ALLOWED_KEYS = set(MARGIN_BASE.keys())


def _c(cid: str, family: str, method_name: str,
       question: str, expected: str, **overrides) -> Dict:
    cfg = dict(MARGIN_BASE)
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
#  FAMILY M1: Learning Rate Sweep
#  Core question: What's the optimal lr for margin loss ZO?
# =========================================================================== #
FAMILY_M1: List[Dict] = [
    _c("m101", "M1_lr", "Margin ZO lr=3e-4",
       "Lower lr — more conservative updates",
       "May underfit but more stable",
       lr=3e-4),
    _c("m102", "M1_lr", "Margin ZO lr=1e-3",
       "Default lr from Phase 1",
       "Reference",
       lr=1e-3),
    _c("m103", "M1_lr", "Margin ZO lr=3e-3",
       "Higher lr — stronger updates",
       "May overshoot but faster adaptation",
       lr=3e-3),
    _c("m104", "M1_lr", "Margin ZO lr=1e-2",
       "Aggressive lr",
       "Risk of collapse but max adaptation",
       lr=1e-2),
    _c("m105", "M1_lr", "Margin ZO lr=3e-4 K=3",
       "Low lr + multi-step — safe but effective",
       "Conservative multi-step",
       lr=3e-4, zo_steps=3),
    _c("m106", "M1_lr", "Margin ZO lr=1e-3 K=3",
       "Default lr + multi-step",
       "More adaptation per batch",
       lr=1e-3, zo_steps=3),
    _c("m107", "M1_lr", "Margin ZO lr=3e-3 K=3",
       "High lr + multi-step",
       "Strongest per-batch adaptation",
       lr=3e-3, zo_steps=3),
]

# =========================================================================== #
#  FAMILY M2: Epsilon (Perturbation Scale) Sweep
#  Core question: What eps works best for margin loss ZO on SNN?
# =========================================================================== #
FAMILY_M2: List[Dict] = [
    _c("m201", "M2_eps", "Margin ZO eps=1e-3",
       "Fine perturbation",
       "May not perturb enough through LIF",
       eps=1e-3),
    _c("m202", "M2_eps", "Margin ZO eps=3e-2 (medium perturbation)",
       "Medium perturbation — between default and coarse",
       "Intermediate SNR through LIF",
       eps=3e-2),
    _c("m203", "M2_eps", "Margin ZO eps=5e-2",
       "Coarse perturbation (e02 style)",
       "Better SNR through LIF discontinuity",
       eps=5e-2),
    _c("m204", "M2_eps", "Margin ZO eps=1e-1",
       "Very coarse perturbation",
       "May be too coarse but max SNR",
       eps=1e-1),
    _c("m205", "M2_eps", "Margin ZO eps=5e-2 lr=3e-3",
       "Coarse eps + higher lr",
       "Combined strong perturbation + update",
       eps=5e-2, lr=3e-3),
    _c("m206", "M2_eps", "Margin ZO eps=1e-1 lr=3e-3",
       "Very coarse + higher lr",
       "Maximum signal strength",
       eps=1e-1, lr=3e-3),
]

# =========================================================================== #
#  FAMILY M3: Variance Reduction
#  Core question: Can variance reduction make margin ZO even better?
# =========================================================================== #
FAMILY_M3: List[Dict] = [
    _c("m301", "M3_var", "Margin ZO ns=2",
       "2-sample averaging — mild variance reduction",
       "2x compute, cleaner gradient",
       num_samples=2),
    _c("m302", "M3_var", "Margin ZO ns=4",
       "4-sample averaging",
       "4x compute, much cleaner gradient",
       num_samples=4),
    _c("m303", "M3_var", "Margin ZO ns=8",
       "8-sample averaging — strong variance reduction",
       "8x compute, cleanest gradient",
       num_samples=8),
    _c("m304", "M3_var", "Margin ZO momentum=0.9",
       "Momentum EMA — smooths gradient trajectory",
       "Free variance reduction (no extra forwards)",
       momentum=0.9),
    _c("m305", "M3_var", "Margin ZO momentum=0.99",
       "Strong momentum — very smooth trajectory",
       "May lag behind distribution changes",
       momentum=0.99),
    _c("m306", "M3_var", "Margin ZO momentum=0.9 ns=4",
       "Momentum + multi-sample — combined",
       "Best variance reduction",
       momentum=0.9, num_samples=4),
    _c("m307", "M3_var", "Margin ZO subspace 50%",
       "Random subspace — SZO-lite",
       "Lower effective dimension",
       block_sparsity=0.5),
    _c("m308", "M3_var", "Margin ZO subspace 50% + momentum",
       "Subspace + momentum",
       "SZO's practical recipe",
       block_sparsity=0.5, momentum=0.9),
]

# =========================================================================== #
#  FAMILY M4: Anti-Drift / Regularization
#  Core question: Can we prevent margin ZO from drifting on long sequences?
# =========================================================================== #
FAMILY_M4: List[Dict] = [
    _c("m401", "M4_drift", "Margin ZO + KL=0.1",
       "KL-to-source — prevents prediction drift",
       "Conservative but stable",
       kl_lambda=0.1),
    _c("m402", "M4_drift", "Margin ZO + KL=0.5",
       "Stronger KL regularization",
       "Very conservative",
       kl_lambda=0.5),
    _c("m403", "M4_drift", "Margin ZO + wd_init=0.1",
       "Weight decay toward init adapter",
       "Prevents adapter parameter drift",
       wd_init=0.1),
    _c("m404", "M4_drift", "Margin ZO + wd_init=0.5",
       "Stronger weight decay toward init",
       "Very conservative adapter",
       wd_init=0.5),
    _c("m405", "M4_drift", "Margin ZO + clip_norm=0.1",
       "Gradient clipping — prevents outlier steps",
       "Safe but may limit adaptation",
       clip_norm=0.1),
    _c("m406", "M4_drift", "Margin ZO + clip_norm=0.01",
       "Strong gradient clipping",
       "Very conservative updates",
       clip_norm=0.01),
    _c("m407", "M4_drift", "Margin ZO episodic reset",
       "Reset adapter per batch — no memory",
       "Safe but no cross-batch improvement",
       reset_per_batch=True),
]

# =========================================================================== #
#  FAMILY M5: Margin Loss Variants
#  Core question: Can we improve the margin loss itself?
# =========================================================================== #
FAMILY_M5: List[Dict] = [
    _c("m501", "M5_variant", "Margin ZO + temp_entropy T=2",
       "Temperature-scaled entropy — softer than margin",
       "Hybrid: margin-like but with distribution info",
       objective="temp_entropy", temp=2.0),
    _c("m502", "M5_variant", "Margin ZO + filtered_entropy thr=0.6",
       "Filtered entropy — only confident samples",
       "Anti-collapse entropy variant",
       objective="filtered_entropy", entropy_thresh=0.6),
    _c("m503", "M5_variant", "Margin ZO + logsumexp",
       "LogSumExp loss — smooth confidence max",
       "Convex-in-logits, anti-collapse",
       objective="logsumexp"),
    _c("m504", "M5_variant", "Margin ZO + pseudo_label conf=0.8",
       "Pseudo-label CE — self-training variant",
       "Different objective family",
       objective="pseudo_label", conf_thresh=0.8),
    _c("m505", "M5_variant", "Margin ZO + margin + KL=0.1 + eps=5e-2",
       "Margin + KL with large eps — best of both",
       "Margin's stability + KL's anti-drift + better SNR",
       kl_lambda=0.1, eps=5e-2),
    _c("m506", "M5_variant", "Margin ZO + margin + wd_init=0.1 + eps=5e-2",
       "Margin + wd_init with large eps",
       "Margin's stability + parameter regularization + better SNR",
       wd_init=0.1, eps=5e-2),
]

# =========================================================================== #
#  FAMILY M6: Adapter / Architecture Variants
#  Core question: Where and how should we apply the adapter?
# =========================================================================== #
FAMILY_M6: List[Dict] = [
    _c("m601", "M6_adapter", "Margin ZO BN affine (all)",
       "Adapt BN affine params instead of adapter",
       "Different parameterization",
       adapt="bn_all", lr=1e-3),
    _c("m602", "M6_adapter", "Margin ZO BN affine (last)",
       "Adapt only last BN block",
       "Fewer params, lower ZO variance",
       adapt="bn_last", lr=1e-3),
    _c("m603", "M6_adapter", "Margin ZO pool2+pool3",
       "Adapt two layers simultaneously",
       "More adaptation capacity",
       target_layers=("pool3", "pool2")),
    _c("m604", "M6_adapter", "Margin ZO temporal adapter",
       "Timestep-dependent adapter",
       "More expressive but more params",
       adapter_kind="temporal"),
    _c("m605", "M6_adapter", "Margin ZO no clamp",
       "Remove adapter parameter clamping",
       "More freedom but risk of drift",
       clamp_abs=0.0),
    _c("m606", "M6_adapter", "Margin ZO clamp=0.5",
       "Looser clamping",
       "More adaptation range",
       clamp_abs=0.5),
]

# =========================================================================== #
#  FAMILY M7: Combined Best (from Phase 1 insights)
#  Core question: What's the best combination of all techniques?
# =========================================================================== #
FAMILY_M7: List[Dict] = [
    # Best from Phase 1: e02 (large eps) + c03 (margin)
    _c("m701", "M7_best", "Margin ZO eps=5e-2 lr=3e-4 (best from P1)",
       "Phase 1's best ZO setting + margin loss, conservative lr",
       "Expected to be strongest",
       eps=5e-2, lr=3e-4),
    # Margin + large eps + momentum
    _c("m702", "M7_best", "Margin ZO eps=5e-2 momentum=0.9",
       "Large eps + momentum smoothing",
       "Combined SNR boost + trajectory smoothing",
       eps=5e-2, momentum=0.9),
    # Margin + large eps + multi-sample
    _c("m703", "M7_best", "Margin ZO eps=5e-2 ns=4",
       "Large eps + multi-sample variance reduction",
       "Cleanest gradient estimate",
       eps=5e-2, num_samples=4),
    # Margin + large eps + KL
    _c("m704", "M7_best", "Margin ZO eps=5e-2 KL=0.5",
       "Large eps + strong anti-drift",
       "Strong signal + strong regularization",
       eps=5e-2, kl_lambda=0.5),
    # Margin + large eps + higher lr
    _c("m705", "M7_best", "Margin ZO eps=5e-2 lr=1e-2",
       "Large eps + much higher lr",
       "Strongest update",
       eps=5e-2, lr=1e-2),
    # Margin + large eps + K=3
    _c("m706", "M7_best", "Margin ZO eps=5e-2 K=3",
       "Large eps + multi-step",
       "Strong per-batch adaptation",
       eps=5e-2, zo_steps=3),
    # Margin + everything
    _c("m707", "M7_best", "Margin ZO eps=5e-2 momentum ns=4 KL",
       "All techniques combined",
       "Maximum everything",
       eps=5e-2, momentum=0.9, num_samples=4, kl_lambda=0.1),
    # One-point estimator (compute efficient)
    _c("m708", "M7_best", "Margin ZO one-point momentum=0.9",
       "One-point estimator — halves forwards",
       "Compute efficient variant",
       estimator="one_point", momentum=0.9),
    # One-point + large eps
    _c("m709", "M7_best", "Margin ZO one-point eps=5e-2",
       "One-point + large eps",
       "Compute efficient + strong signal",
       estimator="one_point", eps=5e-2),
]

# =========================================================================== #
#  ASSEMBLE ALL CANDIDATES
# =========================================================================== #
ALL_CANDIDATES: List[Dict] = (
    FAMILY_M1 + FAMILY_M2 + FAMILY_M3 + FAMILY_M4 +
    FAMILY_M5 + FAMILY_M6 + FAMILY_M7
)
CANDIDATE_IDS = [c["id"] for c in ALL_CANDIDATES]

FAMILY_SUMMARY = {
    "M1_lr": "Learning Rate Sweep (7 candidates)",
    "M2_eps": "Epsilon/Perturbation Sweep (6 candidates)",
    "M3_var": "Variance Reduction (8 candidates)",
    "M4_drift": "Anti-Drift / Regularization (7 candidates)",
    "M5_variant": "Margin Loss Variants (6 candidates)",
    "M6_adapter": "Adapter / Architecture Variants (6 candidates)",
    "M7_best": "Combined Best Configurations (9 candidates)",
}


def validate_candidates(candidates: List[Dict]) -> List[str]:
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
        if "block_sparsity" in eng and not (0 < eng["block_sparsity"] <= 1):
            errors.append(f"{cid}: block_sparsity out of range")
        if "zo_steps" in eng and eng["zo_steps"] < 1:
            errors.append(f"{cid}: zo_steps must be >= 1")
        if "adapt" in eng and eng["adapt"] not in ("adapter", "bn_all", "bn_last"):
            errors.append(f"{cid}: bad adapt {eng['adapt']}")
    return errors


def build_engine_from_cfg(candidate: Dict, model, device, seed: int) -> ZOTTAEngine:
    eng = dict(candidate["engine"])
    return ZOTTAEngine(model, device=device, seed=int(seed), **eng)


def select_candidates(ids: str = "", families: str = "") -> List[Dict]:
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