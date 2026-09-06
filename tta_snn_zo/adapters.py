"""Lightweight adapters inserted into the frozen SNN backbone at test time.

Two variants:

* ``ChannelAffine``          - channel-wise scale & shift (residual, identity
                               initialization).  Tiny parameter count (2*C) makes
                               the ZO gradient estimate low-variance.
* ``TemporalChannelAffine``  - channel *and* timestep-wise scale & shift
                               (PAR's PHSA-style adapter geometry, but optimized
                               by ZO with a *global* loss signal instead of BP,
                               and without the Hebbian THP branch).

The adapter is installed on a target module (e.g. ``pool3``) via a forward hook:
the hook replaces the module output with the adapted representation, so the rest
of the frozen network sees the adapted features.  Pre/post features are recorded
for the reliability gate and spike-aware objectives.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import poisson_gen  # noqa: F401  (re-exported for convenience)


class ChannelAffine(nn.Module):
    """y = x * (1 + scale) + bias, identity at init."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.scale) + self.bias


class TemporalChannelAffine(nn.Module):
    """Channel + timestep affine, residual form:

    y_t = x_t + x_t * (scale_c + scale_t[t]) + (bias_c + bias_t[t])
    """

    def __init__(self, num_channels: int, num_steps: int):
        super().__init__()
        self.num_channels = num_channels
        self.num_steps = num_steps
        self.scale_c = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias_c = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale_t = nn.Parameter(torch.zeros(num_steps, num_channels, 1, 1))
        self.bias_t = nn.Parameter(torch.zeros(num_steps, num_channels, 1, 1))
        self._t = 0

    def begin_sequence(self):
        self._t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = min(self._t, self.num_steps - 1)
        y = x + x * (self.scale_c + self.scale_t[t]) + (self.bias_c + self.bias_t[t])
        self._t += 1
        return y


class AdapterHooks:
    """Installs adapters on target layers of a model via forward hooks.

    Usage::

        hooks = AdapterHooks(model, target_layers={"pool3": adapter})
        logits = model(x)          # adapter applied inside
        feats = hooks.collect()    # {name: {"pre": [T,B,C,H,W], "post": [...]}}

    The ``_collect`` flag controls whether pre/post features are recorded
    (needed for the gate / spike-aware losses; disable during ZO objective
    evaluations if only entropy is used).
    """

    def __init__(self, model: nn.Module, target_layers: dict):
        """
        target_layers: {module_name: adapter_module}
        """
        self.model = model
        self.target_layers = dict(target_layers)
        self._collect = False
        self._current = {name: {"pre": [], "post": []} for name in self.target_layers}
        self._handles = []

        AdapterHooks.cleanup(model)  # remove hooks from any previous engine
        module_dict = dict(model.named_modules())
        for name, adapter in self.target_layers.items():
            if name not in module_dict:
                raise ValueError(f"target layer '{name}' not found in model")
            h = module_dict[name].register_forward_hook(self._make_hook(name, adapter))
            self._handles.append(h)
            setattr(model, f"_adapter_{name.replace('.', '_')}", adapter)
        model._adapter_hooks_handles = self._handles

    @staticmethod
    def cleanup(model: nn.Module):
        """Remove hooks installed by any previous AdapterHooks on this model."""
        handles = getattr(model, "_adapter_hooks_handles", None)
        if handles:
            for h in handles:
                h.remove()
            model._adapter_hooks_handles = []

    def _make_hook(self, layer_name: str, adapter: nn.Module):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            pre = out.detach()
            post = adapter(pre)
            if self._collect:
                self._current[layer_name]["pre"].append(pre)
                self._current[layer_name]["post"].append(post)
            return post
        return hook

    def set_collect(self, flag: bool):
        self._collect = bool(flag)

    def begin_sequence(self):
        for name in self.target_layers:
            self._current[name] = {"pre": [], "post": []}
            adapter = getattr(self.model, f"_adapter_{name.replace('.', '_')}")
            if hasattr(adapter, "begin_sequence"):
                adapter.begin_sequence()

    def collect(self) -> dict:
        out = {}
        for name, d in self._current.items():
            out[name] = {
                "pre": torch.stack(d["pre"], 0) if d["pre"] else None,   # [T,B,C,H,W]
                "post": torch.stack(d["post"], 0) if d["post"] else None,
            }
        return out

    def parameters(self) -> list:
        params = []
        for name in self.target_layers:
            adapter = getattr(self.model, f"_adapter_{name.replace('.', '_')}")
            params.extend(p for p in adapter.parameters() if p.requires_grad)
        return params

    def named_parameters(self) -> list:
        out = []
        for name in self.target_layers:
            adapter = getattr(self.model, f"_adapter_{name.replace('.', '_')}")
            for n, p in adapter.named_parameters():
                if p.requires_grad:
                    out.append((f"adapter.{name}.{n}", p))
        return out

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def build_adapter(kind: str, num_channels: int, num_steps: int) -> nn.Module:
    if kind == "channel":
        return ChannelAffine(num_channels)
    if kind == "temporal":
        return TemporalChannelAffine(num_channels, num_steps)
    raise ValueError(f"unknown adapter kind {kind}")
