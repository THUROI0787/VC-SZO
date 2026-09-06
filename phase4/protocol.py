"""Evaluation protocol utilities for paper-ready Phase-4 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ComputeBudget:
    prediction_forwards: int = 0
    objective_forwards: int = 0
    backward_calls: int = 0
    samples_seen: int = 0
    updates: int = 0


def _add_budget(total: ComputeBudget, stats: dict, batch_size: int) -> ComputeBudget:
    return ComputeBudget(
        prediction_forwards=total.prediction_forwards + int(stats.get("prediction_forwards", 1)),
        objective_forwards=total.objective_forwards + int(stats.get("objective_forwards", 0)),
        backward_calls=total.backward_calls + int(stats.get("backward_calls", 0)),
        samples_seen=total.samples_seen + batch_size,
        updates=total.updates + int(stats.get("updates", stats.get("adapted", 0) > 0)),
    )


class CommonRandomNumbers:
    """Give every SNN method the same per-batch Poisson encoding seed.

    ZOTTAEngine internally uses ``seed + 1_000_000 + batch_index``.  Construct
    ZO engines with ``seed=run_seed`` and wrap all non-ZO engines with this
    class using the same ``run_seed``.  Direction sampling remains isolated in
    ZoOptimizer's private torch.Generator.
    """

    def __init__(self, engine, run_seed: int, zo_internal_seeding: bool = False):
        self.engine = engine
        self.seed_base = int(run_seed) + 1_000_000
        self.zo_internal_seeding = bool(zo_internal_seeding)
        self.batch_index = 0

    def predict_and_adapt(self, x: torch.Tensor):
        if not self.zo_internal_seeding:
            torch.manual_seed(self.seed_base + self.batch_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed_base + self.batch_index)
        logits, stats = self.engine.predict_and_adapt(x)
        self.batch_index += 1
        return logits, stats


def evaluate_fixed_samples(
    engine,
    loader,
    device: torch.device,
    max_samples: Optional[int],
    profile_memory: bool = False,
    checkpoints: Optional[set[int]] = None,
) -> dict:
    """Evaluate exactly ``max_samples`` examples with sample-weighted metrics.

    The final batch is sliced when needed.  This makes batch-size comparisons
    use the same images and stream length, unlike Phase 3's fixed-batch limit.
    """

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or None")
    if profile_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    correct = 0
    seen = 0
    weighted_stats: dict[str, float] = {}
    budget = ComputeBudget()
    checkpoint_set = set(checkpoints or ())
    trajectory: list[dict] = []

    for x, y in loader:
        if max_samples is not None:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            if x.size(0) > remaining:
                x, y = x[:remaining], y[:remaining]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, stats = engine.predict_and_adapt(x)
        batch_size = y.numel()
        correct += int((logits.argmax(1) == y).sum().item())
        seen += batch_size
        budget = _add_budget(budget, stats, batch_size)
        for key in ("loss", "entropy", "adapted", "rolled_back", "projected_grad"):
            weighted_stats[key] = weighted_stats.get(key, 0.0) + float(stats.get(key, 0.0)) * batch_size
        if seen in checkpoint_set:
            trajectory.append({"samples": seen, "acc": 100.0 * correct / seen})

    result = {
        "acc": 100.0 * correct / seen if seen else 0.0,
        "correct": correct,
        "num_samples": seen,
        **({key: value / seen for key, value in weighted_stats.items()} if seen else {}),
        "prediction_forwards": budget.prediction_forwards,
        "objective_forwards": budget.objective_forwards,
        "backward_calls": budget.backward_calls,
        "updates": budget.updates,
    }
    if trajectory:
        result["trajectory"] = trajectory
    if profile_memory and torch.cuda.is_available():
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        result["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
    return result

