"""BN Stats Adaptation: BP-free baseline that updates BN running statistics.

This is the simplest BP-free TTA method:
- For each test batch, recompute BN mean/var from the batch itself
- No parameter updates, no backpropagation
- Only works for models with BatchNorm layers

Reference: "Robustness via Test-Time Normalization" (Schneider et al., 2020)
Also used as a baseline in SPACE and PAR papers.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .utils import accuracy


class BNStatsAdaptEngine:
    """BP-free TTA by updating BN running statistics.

    For each test batch, sets BN layers to train mode so they update their
    running mean/var using the current batch statistics. No gradient computation
    or parameter updates needed.

    This is the simplest and most lightweight TTA baseline.
    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.1,
        device: torch.device = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.momentum = momentum

        # Store original BN modes
        self._orig_modes = {}
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self._orig_modes[name] = m.training

    def predict_and_adapt(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward with BN in train mode (updates running stats).

        Returns prediction and stats.
        """
        functional.reset_net(self.model)

        batch_size = x.size(0)
        # batch=1: BN train mode needs ≥2 samples, skip BN stats update
        if batch_size == 1:
            # Model may be in train() mode (build_model_fn default).
            # Force eval to avoid BN crash, then restore.
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
            if was_training:
                self.model.train()
            stats = {
                "loss": 0.0,
                "gate": 1.0,
                "adapted": 0.0,
                "rolled_back": 0.0,
            }
            return logits, stats

        # Set BN layers to train mode (updates running stats)
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()

        # Forward (BN will update running stats)
        with torch.no_grad():
            logits = self.model(x)

        # Restore BN modes
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

        stats = {
            "loss": 0.0,
            "gate": 1.0,
            "adapted": 1.0,
            "rolled_back": 0.0,
        }
        return logits, stats


def evaluate_bn_stats_adapt(
    engine_factory: Callable[[], BNStatsAdaptEngine],
    loader,
    device,
    logger=None,
    max_batches: Optional[int] = None,
) -> dict:
    """Run BN stats adaptation over a loader.

    Creates a fresh engine via engine_factory() for each call.
    """
    engine = engine_factory()
    accs = []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, stats = engine.predict_and_adapt(x)
        accs.append(accuracy(logits, y)[0])
        if logger is not None and (i + 1) % 20 == 0:
            logger.info(f"  batch {i+1}/{len(loader)} acc={accs[-1]:.2f}%")
    n = len(accs)
    return {
        "acc": sum(accs) / n if n else 0.0,
        "loss": 0.0,
        "num_batches": n,
    }