"""MEMO: Test Time Robustness via Adaptation and Augmentation.

MEMO (Zhang et al., NeurIPS 2022) is a BP-free TTA method that:
1. For each test sample, creates N augmentations
2. Computes predictions for all augmentations
3. Minimizes entropy of the *average* (marginal) prediction

This requires NO backpropagation and NO parameter updates — only forward passes.
It is the ideal BP-free baseline for our ZO-TTA comparison.

Reference: https://arxiv.org/abs/2110.09506

NOTE: The original MEMO uses torchvision.transforms for augmentations.
We replicate the exact same pipeline: RandomResizedCrop, RandomHorizontalFlip,
ColorJitter, and RandomGrayscale.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torchvision import transforms

from .utils import accuracy


# --------------------------------------------------------------------------- #
#  MEMO augmentations (matching the original paper exactly)
# --------------------------------------------------------------------------- #
# The original MEMO uses:
#   RandomResizedCrop(32, scale=(0.8, 1.0))
#   RandomHorizontalFlip()
#   ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
#   RandomGrayscale(p=0.1)
#
# These are applied to each test sample to create N diverse views.
# All operations are BP-free (no gradients needed).

MEMO_AUGMENT = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
])


def _memo_augment_tensor(x: torch.Tensor, n_aug: int = 32) -> torch.Tensor:
    """Create N augmentations of each sample using torchvision transforms.

    Each sample is independently augmented n_aug times. The input tensor is
    denormalized back to [0,1] range, augmented via PIL-compatible transforms,
    then re-normalized.

    Args:
        x: Input tensor [B, C, H, W] in normalized format.
        n_aug: Number of augmentations per sample.

    Returns:
        Augmented tensor [B*n_aug, C, H, W].
    """
    B, C, H, W = x.shape
    device = x.device

    # Denormalize: the input is normalized with CIFAR-10 mean/std
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
    x_denorm = x * std + mean  # [B, C, H, W] in [0, 1] range
    x_denorm = x_denorm.clamp(0, 1)

    # Convert to PIL, augment, convert back
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    augs = []
    for i in range(B):
        # Get single sample as PIL
        img = x_denorm[i].cpu()  # [C, H, W]
        img_pil = to_pil(img)

        for _ in range(n_aug):
            aug_pil = MEMO_AUGMENT(img_pil)
            aug_t = to_tensor(aug_pil).to(device)  # [C, H, W] in [0,1]
            augs.append(aug_t)

    # Stack and renormalize
    augs = torch.stack(augs, dim=0)  # [B*n_aug, C, H, W]
    augs = (augs - mean) / std
    return augs


class MEMOEngine:
    """MEMO: TTA via marginal entropy minimization with parameter updates.

    For each batch, creates N augmentations per sample, computes predictions,
    and minimizes entropy of the marginal (average) prediction via
    backpropagation through BN affine parameters (same as TENT).

    Reference: Zhang et al., "MEMO: Test Time Robustness via Adaptation and
    Augmentation", NeurIPS 2022.

    NOTE: MEMO is NOT BP-free — it updates model parameters via SGD on the
    marginal entropy loss. The original paper uses per-sample optimization
    with SGD; here we optimize BN affine parameters batch-wise (matching the
    TENT-style parameterization for fair comparison).
    """

    def __init__(
        self,
        model: nn.Module,
        n_aug: int = 32,
        lr: float = 1e-3,
        device: torch.device = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.n_aug = n_aug

        # Freeze all params, keep BN affine trainable (same as TENT)
        from .zo import collect_bn_affine_params, freeze_model
        freeze_model(model, keep_bn_affine=True)
        model.train()
        # BN runs in train mode so running stats are updated (same as TENT)
        params, names = collect_bn_affine_params(model, last_only=False)
        for p in params:
            p.requires_grad_(True)
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    def predict_and_adapt(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """MEMO adaptation: augment, forward, marginal entropy minimization.

        Returns the original (unaugmented) prediction and stats.
        """
        B = x.size(0)
        functional.reset_net(self.model)

        # BN must be in eval mode: per-sample/per-aug micro-batches use
        # [1, C, H, W] or small inputs that crash BN in train mode.
        # BN affine params still get gradients via loss.backward().
        for _nm, _m in self.model.named_modules():
            if isinstance(_m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                _m.eval()

        # 1) Get original prediction (before adaptation)
        with torch.no_grad():
            logits_orig = self.model(x)

        # 2) Create augmentations and get predictions (with gradients)
        x_aug = _memo_augment_tensor(x, self.n_aug)  # [B*n_aug, C, H, W]
        functional.reset_net(self.model)
        logits_all = self.model(x_aug)  # [B*n_aug, C] — with grad

        # 3) Reshape to [B, n_aug, C] and compute marginal entropy
        logits_all = logits_all.view(B, self.n_aug, -1)
        probs_all = logits_all.softmax(dim=-1)  # [B, n_aug, C]
        marginal_probs = probs_all.mean(dim=1)   # [B, C]

        # MEMO loss: entropy of marginal distribution
        ent = -(marginal_probs * marginal_probs.log()).sum(dim=1)  # [B]
        loss = ent.mean()

        # 4) Backprop and update BN affine params
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "entropy": float(ent.mean().item()),
            "gate": 1.0,
            "adapted": 1.0,
            "rolled_back": 0.0,
        }
        return logits_orig, stats


def evaluate_memo(
    engine_factory: Callable[[], MEMOEngine],
    loader,
    device,
    logger=None,
    max_batches: Optional[int] = None,
) -> dict:
    """Run MEMO engine over a loader; returns accuracy + adaptation stats.

    Creates a fresh engine via engine_factory() for each call.
    """
    engine = engine_factory()
    accs, losses = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, stats = engine.predict_and_adapt(x)
        accs.append(accuracy(logits, y)[0])
        losses.append(stats["loss"])
        if logger is not None and (i + 1) % 20 == 0:
            logger.info(f"  batch {i+1}/{len(loader)} acc={accs[-1]:.2f}% "
                        f"loss={losses[-1]:.4f}")
    n = len(accs)
    return {
        "acc": sum(accs) / n if n else 0.0,
        "loss": sum(losses) / n if n else 0.0,
        "num_batches": n,
    }