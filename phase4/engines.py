"""Controlled Phase-4 baselines that share the proposed method's adapter."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from tta_snn_zo.adapters import AdapterHooks, build_adapter
from tta_snn_zo.losses import margin_loss, softmax_entropy
from tta_snn_zo.tta import _infer_layer_channels
from tta_snn_zo.zo import freeze_model


class AdapterBPEngine:
    """BP ablation with the exact same adapter location and objective as ZO.

    Predictions follow the repository's online protocol: predict current batch,
    then update parameters for the next batch.  BN remains in eval mode, so a
    comparison against ZO changes the optimizer only.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Tuple[str, ...] = ("pool3",),
        adapter_kind: str = "channel",
        objective: str = "margin",
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        clamp_abs: float = 0.2,
        img_size: int = 32,
        seed: int = 0,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.target_layers = tuple(target_layers)
        self.objective = objective
        if objective not in ("margin", "entropy"):
            raise ValueError(f"unsupported objective: {objective}")
        self.clamp_abs = float(clamp_abs)
        self._enc_seed = int(seed) + 1_000_000
        self._batch_count = 0
        freeze_model(model)
        channels = _infer_layer_channels(model, self.target_layers, img_size, self.device)
        num_steps = int(getattr(model, "num_steps", 1))
        adapters = {
            name: build_adapter(adapter_kind, channels[name], num_steps).to(self.device)
            for name in self.target_layers
        }
        self.hooks = AdapterHooks(model, adapters)
        model.train()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
        self.params = self.hooks.parameters()
        self.optimizer = torch.optim.Adam(self.params, lr=lr, weight_decay=weight_decay)

    def _begin_forward(self) -> None:

        functional.reset_net(self.model)
        self.hooks.begin_sequence()
        self.hooks.set_collect(False)

    def _loss(self, logits: torch.Tensor) -> torch.Tensor:
        if self.objective == "margin":
            return margin_loss(logits)
        return softmax_entropy(logits).mean()

    def predict_and_adapt(self, x: torch.Tensor):
        x = x.to(self.device)
        enc_seed = self._enc_seed + self._batch_count
        torch.manual_seed(enc_seed)
        self._begin_forward()
        with torch.no_grad():
            logits = self.model(x)

        torch.manual_seed(enc_seed)
        self._begin_forward()
        logits_update = self.model(x)
        loss = self._loss(logits_update)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for param in self.params:
                param.clamp_(-self.clamp_abs, self.clamp_abs)

        functional.reset_net(self.model)
        self._batch_count += 1
        return logits.detach(), {
            "loss": float(loss.detach().item()),
            "adapted": 1.0,
            "rolled_back": 0.0,
            "prediction_forwards": 1,
            "objective_forwards": 1,
            "backward_calls": 1,
            "updates": 1,
        }


def zo_compute_stats(engine, adapted: bool) -> dict:
    """Infer ZO query counts from a configured ZOTTAEngine."""
    if not adapted:
        return {"prediction_forwards": 1, "objective_forwards": 0, "updates": 0}
    samples = int(engine.zo.num_samples)
    per_step = 2 * samples if engine.estimator == "two_point" else 1
    rollback = int(bool(engine.rollback_collapse))
    return {
        "prediction_forwards": 1,
        "objective_forwards": engine.zo_steps * per_step + rollback,
        "backward_calls": 0,
        "updates": engine.zo_steps,
    }

