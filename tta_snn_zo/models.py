"""Spiking neural network backbones built on SpikingJelly.

* ``SNN_VGG9``   - VGG-9 with BNTT (BatchNorm-Through-Time, one BN per timestep),
                   LIF neurons from ``spikingjelly.activation_based.neuron``.
                   Architecture / hyperparameters follow the TET / BNTT recipe
                   (also used by SPACE and PAR).
* ``SNN_ResNet19`` - compact spiking ResNet with standard (shared) BN, used in
                   phase 3 to show ZO-TTA generality across architectures.

Both models use *Poisson rate encoding* (TET-style signed Bernoulli generator)
and a non-spiking membrane-accumulation readout, matching the reference recipe.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate


# --------------------------------------------------------------------------- #
#  Poisson encoding (TET / PAR style): signed Bernoulli spikes in {-1, 0, +1}
# --------------------------------------------------------------------------- #
def poisson_gen(inp: torch.Tensor, rescale_fac: float = 2.0) -> torch.Tensor:
    rand_inp = torch.rand_like(inp)
    return torch.mul(
        torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp)
    )


# --------------------------------------------------------------------------- #
#  BNTT: BatchNorm Through Time (per-timestep BN, bias disabled like the
#  reference recipe; matches PAR / BNTT checkpoint naming)
# --------------------------------------------------------------------------- #
class BNTT(nn.Module):
    def __init__(self, num_features: int, num_steps: int, ndim: int = 2,
                 eps: float = 1e-4, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        bn_cls = nn.BatchNorm2d if ndim == 2 else nn.BatchNorm1d
        self.bns = nn.ModuleList(
            [bn_cls(num_features, eps=eps, momentum=momentum, affine=affine)
             for _ in range(num_steps)]
        )
        if affine:
            for bn in self.bns:
                bn.bias = None  # PAR/BNTT style: no BN bias

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return self.bns[min(t, len(self.bns) - 1)](x)


# --------------------------------------------------------------------------- #
#  SNN VGG-9 (BNTT)
# --------------------------------------------------------------------------- #
class SNN_VGG9(nn.Module):
    """VGG-9 for CIFAR-10 with BNTT and SpikingJelly LIF neurons.

    Forward: internally loops ``num_steps`` timesteps; at each step the input is
    Poisson-encoded, propagated through conv blocks (BNTT + LIF + pooling) and the
    FC head; the final linear layer accumulates membrane potential (non-spiking)
    and logits are temporally averaged.

    ``forward`` optionally returns the per-timestep logits and, when requested,
    per-timestep spike features at the named "feature layers" (used by the TTA
    reliability gate and the ZO adapters).
    """

    def __init__(
        self,
        num_steps: int = 25,
        num_classes: int = 10,
        img_size: int = 32,
        tau: float = 20.0,
        v_threshold: float = 1.0,
        surrogate_fn=None,
        bn_affine: bool = True,
        feature_layers: tuple = ("pool3",),
        width_scale: float = 1.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.img_size = img_size
        self.tau = tau
        self.v_threshold = v_threshold
        self.feature_layers = tuple(feature_layers)
        self.use_checkpoint = bool(use_checkpoint)
        w = float(width_scale)
        c1, c2, c3 = int(64 * w), int(128 * w), int(256 * w)

        surrogate_fn = surrogate_fn or surrogate.ATan()
        lif_kw = dict(tau=tau, v_threshold=v_threshold, v_reset=0.0,
                      surrogate_function=surrogate_fn)

        self.conv1 = nn.Conv2d(3, c1, 3, 1, 1, bias=False)
        self.bntt1 = BNTT(c1, num_steps, affine=bn_affine)
        self.lif1 = neuron.LIFNode(**lif_kw)
        self.conv2 = nn.Conv2d(c1, c1, 3, 1, 1, bias=False)
        self.bntt2 = BNTT(c1, num_steps, affine=bn_affine)
        self.lif2 = neuron.LIFNode(**lif_kw)
        self.pool1 = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(c1, c2, 3, 1, 1, bias=False)
        self.bntt3 = BNTT(c2, num_steps, affine=bn_affine)
        self.lif3 = neuron.LIFNode(**lif_kw)
        self.conv4 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bntt4 = BNTT(c2, num_steps, affine=bn_affine)
        self.lif4 = neuron.LIFNode(**lif_kw)
        self.pool2 = nn.AvgPool2d(2)

        self.conv5 = nn.Conv2d(c2, c3, 3, 1, 1, bias=False)
        self.bntt5 = BNTT(c3, num_steps, affine=bn_affine)
        self.lif5 = neuron.LIFNode(**lif_kw)
        self.conv6 = nn.Conv2d(c3, c3, 3, 1, 1, bias=False)
        self.bntt6 = BNTT(c3, num_steps, affine=bn_affine)
        self.lif6 = neuron.LIFNode(**lif_kw)
        self.conv7 = nn.Conv2d(c3, c3, 3, 1, 1, bias=False)
        self.bntt7 = BNTT(c3, num_steps, affine=bn_affine)
        self.lif7 = neuron.LIFNode(**lif_kw)
        self.pool3 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear((img_size // 8) ** 2 * c3, 1024, bias=False)
        self.bntt_fc = BNTT(1024, num_steps, ndim=1, affine=bn_affine)
        self.lif_fc = neuron.LIFNode(**lif_kw)
        self.fc2 = nn.Linear(1024, num_classes, bias=False)

        # feature bookkeeping (used by TTA engine / hooks)
        self._collect_features = False
        self._feats: dict = {}
        self._logits_t: list = []

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=2)

    def _reset_feature_buffers(self):
        self._feats = {name: [] for name in self.feature_layers}
        self._logits_t = []

    def _record_features(self, name: str, tensor: torch.Tensor):
        if self._collect_features and name in self._feats:
            self._feats[name].append(tensor.detach())

    # -- stateless conv+BNTT segments (used by gradient checkpointing) ------ #
    def _c1(self, out, t): return self.bntt1(self.conv1(out), t) * self.tau
    def _c2(self, out, t): return self.bntt2(self.conv2(out), t) * self.tau
    def _c3(self, out, t): return self.bntt3(self.conv3(out), t) * self.tau
    def _c4(self, out, t): return self.bntt4(self.conv4(out), t) * self.tau
    def _c5(self, out, t): return self.bntt5(self.conv5(out), t) * self.tau
    def _c6(self, out, t): return self.bntt6(self.conv6(out), t) * self.tau
    def _c7(self, out, t): return self.bntt7(self.conv7(out), t) * self.tau
    def _cf(self, out, t): return self.bntt_fc(self.fc1(out), t) * self.tau

    def _seg(self, fn, out, t):
        """Apply a stateless conv segment, optionally under gradient checkpointing."""
        if self.use_checkpoint and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, out, t, use_reentrant=False)
        return fn(out, t)

    # -- one timestep ------------------------------------------------------- #
    def forward_timestep(self, x: torch.Tensor, t: int) -> torch.Tensor:
        # NOTE: SpikingJelly LIFNode charges as  V <- (1-1/tau)V + X/tau.
        # To reproduce the reference (TET/BNTT/PAR) recurrence  V <- leak*V + X
        # with leak = 1-1/tau, we feed tau * X into the neuron.
        out = poisson_gen(x)
        out = self.lif1(self._seg(self._c1, out, t))
        out = self.lif2(self._seg(self._c2, out, t))
        out = self.pool1(out)

        out = self.lif3(self._seg(self._c3, out, t))
        out = self.lif4(self._seg(self._c4, out, t))
        out = self.pool2(out)

        out = self.lif5(self._seg(self._c5, out, t))
        out = self.lif6(self._seg(self._c6, out, t))
        out = self.lif7(self._seg(self._c7, out, t))
        out = self.pool3(out)
        self._record_features("pool3", out)

        out = out.flatten(1)
        out = self.lif_fc(self._seg(self._cf, out, t))
        out = self.fc2(out)  # non-spiking membrane accumulation
        return out

    def forward(self, x: torch.Tensor, return_logits_t: bool = False):
        self._reset_feature_buffers()
        logits_t = []
        for t in range(self.num_steps):
            logits_t.append(self.forward_timestep(x, t))
        logits = torch.stack(logits_t).mean(0)
        if return_logits_t:
            return logits, torch.stack(logits_t)
        return logits

    def features(self) -> dict:
        """Return collected per-timestep features {name: [T,B,C,H,W]}."""
        return {name: torch.stack(seq, dim=0) if seq else None
                for name, seq in self._feats.items()}


# --------------------------------------------------------------------------- #
#  SNN ResNet-19 (standard shared BN; used for phase-3 architecture generality)
# --------------------------------------------------------------------------- #
class _SpikeBasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 lif_kw: dict = None, input_scale: float = 1.0, use_checkpoint: bool = False):
        super().__init__()
        lif_kw = lif_kw or dict(tau=2.0, v_threshold=1.0, v_reset=0.0,
                                surrogate_function=surrogate.ATan())
        self.input_scale = float(input_scale)
        self.use_checkpoint = bool(use_checkpoint)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = neuron.LIFNode(**lif_kw)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = neuron.LIFNode(**lif_kw)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def _seg1(self, x): return self.bn1(self.conv1(x)) * self.input_scale
    def _seg2(self, x): return self.bn2(self.conv2(x))

    def _ck(self, fn, x):
        if self.use_checkpoint and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x):
        out = self.lif1(self._ck(self._seg1, x))
        out = self._ck(self._seg2, out)
        if self.shortcut is not None:
            x = self.shortcut(x)
        out = out + x
        out = self.lif2(out)
        return out


class SNN_ResNet19(nn.Module):
    """Compact spiking ResNet for CIFAR (channels 16/32/64, 3 stages).

    Standard shared BN (no BNTT) keeps the model light; last layer is a
    non-spiking accumulator like the VGG.  Suitable for T=4..8 direct training.
    """

    def __init__(self, num_steps: int = 6, num_classes: int = 10,
                 tau: float = 2.0, v_threshold: float = 1.0, surrogate_fn=None,
                 use_checkpoint: bool = False):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.use_checkpoint = bool(use_checkpoint)
        lif_kw = dict(tau=tau, v_threshold=v_threshold, v_reset=0.0,
                      surrogate_function=surrogate_fn or surrogate.ATan())
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = neuron.LIFNode(**lif_kw)
        self.layer1 = self._make_layer(16, 16, 3, 1, lif_kw, input_scale=tau, use_checkpoint=self.use_checkpoint)
        self.layer2 = self._make_layer(16, 32, 3, 2, lif_kw, input_scale=tau, use_checkpoint=self.use_checkpoint)
        self.layer3 = self._make_layer(32, 64, 3, 2, lif_kw, input_scale=tau, use_checkpoint=self.use_checkpoint)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes, bias=False)
        self.feature_layers = ("layer3",)
        self._collect_features = False
        self._feats = {}
        self._logits_t = []
        self._init_weights()

    @staticmethod
    def _make_layer(in_planes, planes, blocks, stride, lif_kw, input_scale=1.0, use_checkpoint=False):
        layers = [_SpikeBasicBlock(in_planes, planes, stride, lif_kw, input_scale, use_checkpoint)]
        for _ in range(1, blocks):
            layers.append(_SpikeBasicBlock(planes, planes, 1, lif_kw, input_scale, use_checkpoint))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _reset_feature_buffers(self):
        self._feats = {name: [] for name in self.feature_layers}
        self._logits_t = []

    def _record_features(self, name: str, tensor: torch.Tensor):
        if self._collect_features and name in self._feats:
            self._feats[name].append(tensor.detach())

    def forward_timestep(self, x: torch.Tensor, t: int) -> torch.Tensor:
        out = poisson_gen(x)
        out = self.lif1(self.bn1(self.conv1(out)) * self.tau)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self._record_features("layer3", out)
        out = self.pool(out).flatten(1)
        out = self.fc(out)
        return out

    def forward(self, x: torch.Tensor, return_logits_t: bool = False):
        self._reset_feature_buffers()
        logits_t = []
        for t in range(self.num_steps):
            logits_t.append(self.forward_timestep(x, t))
        logits = torch.stack(logits_t).mean(0)
        if return_logits_t:
            return logits, torch.stack(logits_t)
        return logits

    def features(self) -> dict:
        return {name: torch.stack(seq, dim=0) if seq else None
                for name, seq in self._feats.items()}


def build_model(name: str = "vgg9", num_steps: int = 25, num_classes: int = 10,
                tau: float = 20.0, v_threshold: float = 1.0, width_scale: float = 1.0,
                use_checkpoint: bool = False, **kwargs) -> nn.Module:
    if name == "vgg9":
        return SNN_VGG9(num_steps=num_steps, num_classes=num_classes, tau=tau,
                        v_threshold=v_threshold, width_scale=width_scale,
                        use_checkpoint=use_checkpoint, **kwargs)
    if name == "resnet19":
        # width_scale is VGG-specific; drop it for ResNet
        kwargs.pop("width_scale", None)
        return SNN_ResNet19(num_steps=num_steps, num_classes=num_classes, tau=tau,
                            v_threshold=v_threshold, use_checkpoint=use_checkpoint, **kwargs)
    raise ValueError(f"unknown model {name}")
