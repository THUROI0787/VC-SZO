"""Configurations reported in the VC-SZO paper."""
from __future__ import annotations

BASE_CONFIG = {
    "adapt": "adapter",
    "adapter_kind": "channel",
    "target_layers": ("pool3",),
    "img_size": 32,
    "lr": 1e-3,
    "eps": 1e-2,
    "momentum": 0.0,
    "num_samples": 1,
    "block_sparsity": 1.0,
    "weight_decay": 1e-3,
    "zo_steps": 1,
    "estimator": "two_point",
    "gate": True,
    "ent_min_batch": 0.01,
    "ent_max_batch": 2.40,
    "sample_ent_thresh": 1.2,
    "clamp_abs": 0.2,
    "rollback_collapse": True,
    "objective": "margin",
}


def _config(**overrides):
    config = dict(BASE_CONFIG)
    config.update(overrides)
    return config


PAPER_CONFIGS = {
    # Sparse direction: an unrescaled Bernoulli mask with density q=0.5.
    "vcszo_sd": _config(block_sparsity=0.5),
    # Multi-direction: average K=2 dense Gaussian directions.
    "vcszo_md": _config(num_samples=2),
}
