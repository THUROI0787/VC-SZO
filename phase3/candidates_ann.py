"""Phase 3 candidates for ANN VGG9 baseline (T=0).

Same 8 candidates as candidates_phase3.py, but:
- target_layers=("features.23",) instead of ("pool3",)
- m604 (temporal adapter) is SKIPPED — it is SNN-specific (timestep dimension)
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


def _c(cid: str, family: str, method_name: str, **overrides) -> Dict:
    cfg = dict(MARGIN_BASE_ANN)
    cfg.update(overrides)
    return {
        "id": cid,
        "family": family,
        "method_name": method_name,
        "engine": cfg,
    }


# =========================================================================== #
#  7 Selected Candidates (m604 temporal adapter is SNN-specific, skipped)
# =========================================================================== #
PHASE3_ANN_CANDIDATES: List[Dict] = [
    # m301: Multi-sample ns=2 — generic ZO, works for ANN
    _c("m301", "M3_var", "Margin ZO ns=2",
       num_samples=2),

    # m307: Subspace 50% — generic ZO, works for ANN
    _c("m307", "M3_var", "Margin ZO subspace 50%",
       block_sparsity=0.5),

    # m306: Momentum=0.9 + ns=4 — generic ZO, works for ANN
    _c("m306", "M3_var", "Margin ZO momentum=0.9 ns=4",
       momentum=0.9, num_samples=4),

    # m407: Episodic reset — generic ZO, works for ANN
    _c("m407", "M4_drift", "Margin ZO episodic reset",
       reset_per_batch=True),

    # m502: Filtered entropy — different objective, works for ANN
    _c("m502", "M5_variant", "Margin ZO + filtered_entropy thr=0.6",
       objective="filtered_entropy", entropy_thresh=0.6),

    # m709: One-point eps=5e-2 — efficient, works for ANN
    _c("m709", "M7_best", "Margin ZO one-point eps=5e-2",
       estimator="one_point", eps=5e-2),

    # m202: Medium eps=3e-2 — generic ZO, works for ANN
    _c("m202", "M2_eps", "Margin ZO eps=3e-2 (medium perturbation)",
       eps=3e-2),
]

ALL_CANDIDATES = PHASE3_ANN_CANDIDATES
CANDIDATE_IDS = [c["id"] for c in ALL_CANDIDATES]


def build_engine_from_cfg(candidate: Dict, model, device, seed: int) -> ZOTTAEngine:
    eng = dict(candidate["engine"])
    return ZOTTAEngine(model, device=device, seed=int(seed), **eng)


def select_candidates(ids: str = "") -> List[Dict]:
    id_set = {x.strip() for x in ids.split(",") if x.strip()}
    if not id_set:
        return ALL_CANDIDATES
    return [c for c in ALL_CANDIDATES if c["id"] in id_set]