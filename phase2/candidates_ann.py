"""Phase-2 candidates for ANN VGG9 baseline.

Same 49 candidates as phase2/candidates.py, but with ANN-compatible defaults:
- target_layers=("features.23",) instead of ("pool3",)
- No membrane-related features (all off by default anyway)

Candidate IDs are IDENTICAL to the SNN version for cross-comparison.
"""
from __future__ import annotations

from typing import Dict, List

from tta_snn_zo.tta import ZOTTAEngine

# --------------------------------------------------------------------------- #
#  Default engine kwargs (margin loss base, ANN-compatible)
# --------------------------------------------------------------------------- #
MARGIN_BASE_ANN: Dict = dict(
    adapt="adapter",
    adapter_kind="channel",
    target_layers=("features.23",),  # ANN VGG9's last pool layer
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

ALLOWED_KEYS = set(MARGIN_BASE_ANN.keys())


def _c(cid: str, family: str, method_name: str,
       question: str, expected: str, **overrides) -> Dict:
    cfg = dict(MARGIN_BASE_ANN)
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
# =========================================================================== #
FAMILY_M1: List[Dict] = [
    _c("m101", "M1_lr", "Margin ZO lr=3e-4", "", "", lr=3e-4),
    _c("m102", "M1_lr", "Margin ZO lr=1e-3", "", "", lr=1e-3),
    _c("m103", "M1_lr", "Margin ZO lr=3e-3", "", "", lr=3e-3),
    _c("m104", "M1_lr", "Margin ZO lr=1e-2", "", "", lr=1e-2),
    _c("m105", "M1_lr", "Margin ZO lr=3e-4 K=3", "", "", lr=3e-4, zo_steps=3),
    _c("m106", "M1_lr", "Margin ZO lr=1e-3 K=3", "", "", lr=1e-3, zo_steps=3),
    _c("m107", "M1_lr", "Margin ZO lr=3e-3 K=3", "", "", lr=3e-3, zo_steps=3),
]

# =========================================================================== #
#  FAMILY M2: Epsilon Sweep
# =========================================================================== #
FAMILY_M2: List[Dict] = [
    _c("m201", "M2_eps", "Margin ZO eps=1e-3", "", "", eps=1e-3),
    _c("m202", "M2_eps", "Margin ZO eps=3e-2", "", "", eps=3e-2),
    _c("m203", "M2_eps", "Margin ZO eps=5e-2", "", "", eps=5e-2),
    _c("m204", "M2_eps", "Margin ZO eps=1e-1", "", "", eps=1e-1),
    _c("m205", "M2_eps", "Margin ZO eps=5e-2 lr=3e-3", "", "", eps=5e-2, lr=3e-3),
    _c("m206", "M2_eps", "Margin ZO eps=1e-1 lr=3e-3", "", "", eps=1e-1, lr=3e-3),
]

# =========================================================================== #
#  FAMILY M3: Variance Reduction
# =========================================================================== #
FAMILY_M3: List[Dict] = [
    _c("m301", "M3_var", "Margin ZO ns=2", "", "", num_samples=2),
    _c("m302", "M3_var", "Margin ZO ns=4", "", "", num_samples=4),
    _c("m303", "M3_var", "Margin ZO ns=8", "", "", num_samples=8),
    _c("m304", "M3_var", "Margin ZO momentum=0.9", "", "", momentum=0.9),
    _c("m305", "M3_var", "Margin ZO momentum=0.99", "", "", momentum=0.99),
    _c("m306", "M3_var", "Margin ZO momentum=0.9 ns=4", "", "", momentum=0.9, num_samples=4),
    _c("m307", "M3_var", "Margin ZO subspace 50%", "", "", block_sparsity=0.5),
    _c("m308", "M3_var", "Margin ZO subspace 50% + momentum", "", "", block_sparsity=0.5, momentum=0.9),
]

# =========================================================================== #
#  FAMILY M4: Anti-Drift
# =========================================================================== #
FAMILY_M4: List[Dict] = [
    _c("m401", "M4_drift", "Margin ZO + KL=0.1", "", "", kl_lambda=0.1),
    _c("m402", "M4_drift", "Margin ZO + KL=0.5", "", "", kl_lambda=0.5),
    _c("m403", "M4_drift", "Margin ZO + wd_init=0.1", "", "", wd_init=0.1),
    _c("m404", "M4_drift", "Margin ZO + wd_init=0.5", "", "", wd_init=0.5),
    _c("m405", "M4_drift", "Margin ZO + clip_norm=0.1", "", "", clip_norm=0.1),
    _c("m406", "M4_drift", "Margin ZO + clip_norm=0.01", "", "", clip_norm=0.01),
    _c("m407", "M4_drift", "Margin ZO episodic reset", "", "", reset_per_batch=True),
]

# =========================================================================== #
#  FAMILY M5: Margin Loss Variants
# =========================================================================== #
FAMILY_M5: List[Dict] = [
    _c("m501", "M5_variant", "Margin ZO + temp_entropy T=2", "", "", objective="temp_entropy", temp=2.0),
    _c("m502", "M5_variant", "Margin ZO + filtered_entropy thr=0.6", "", "", objective="filtered_entropy", entropy_thresh=0.6),
    _c("m503", "M5_variant", "Margin ZO + logsumexp", "", "", objective="logsumexp"),
    _c("m504", "M5_variant", "Margin ZO + pseudo_label conf=0.8", "", "", objective="pseudo_label", conf_thresh=0.8),
    _c("m505", "M5_variant", "Margin ZO + margin + KL=0.1 + eps=5e-2", "", "", kl_lambda=0.1, eps=5e-2),
    _c("m506", "M5_variant", "Margin ZO + margin + wd_init=0.1 + eps=5e-2", "", "", wd_init=0.1, eps=5e-2),
]

# =========================================================================== #
#  FAMILY M6: Adapter Variants
# =========================================================================== #
FAMILY_M6: List[Dict] = [
    _c("m601", "M6_adapter", "Margin ZO BN affine (all)", "", "", adapt="bn_all", lr=1e-3),
    _c("m602", "M6_adapter", "Margin ZO BN affine (last)", "", "", adapt="bn_last", lr=1e-3),
    _c("m603", "M6_adapter", "Margin ZO features.22+features.23", "", "", target_layers=("features.23", "features.22")),
    _c("m604", "M6_adapter", "Margin ZO temporal adapter", "", "", adapter_kind="temporal"),
    _c("m605", "M6_adapter", "Margin ZO no clamp", "", "", clamp_abs=0.0),
    _c("m606", "M6_adapter", "Margin ZO clamp=0.5", "", "", clamp_abs=0.5),
]

# =========================================================================== #
#  FAMILY M7: Combined Best
# =========================================================================== #
FAMILY_M7: List[Dict] = [
    _c("m701", "M7_best", "Margin ZO eps=5e-2 lr=3e-4", "", "", eps=5e-2, lr=3e-4),
    _c("m702", "M7_best", "Margin ZO eps=5e-2 momentum=0.9", "", "", eps=5e-2, momentum=0.9),
    _c("m703", "M7_best", "Margin ZO eps=5e-2 ns=4", "", "", eps=5e-2, num_samples=4),
    _c("m704", "M7_best", "Margin ZO eps=5e-2 KL=0.5", "", "", eps=5e-2, kl_lambda=0.5),
    _c("m705", "M7_best", "Margin ZO eps=5e-2 lr=1e-2", "", "", eps=5e-2, lr=1e-2),
    _c("m706", "M7_best", "Margin ZO eps=5e-2 K=3", "", "", eps=5e-2, zo_steps=3),
    _c("m707", "M7_best", "Margin ZO eps=5e-2 momentum ns=4 KL", "", "", eps=5e-2, momentum=0.9, num_samples=4, kl_lambda=0.1),
    _c("m708", "M7_best", "Margin ZO one-point momentum=0.9", "", "", estimator="one_point", momentum=0.9),
    _c("m709", "M7_best", "Margin ZO one-point eps=5e-2", "", "", estimator="one_point", eps=5e-2),
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