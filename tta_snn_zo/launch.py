"""Shared launch helpers used by the phase scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from .data import auto_resolve_cifar10c_root
from .models import build_model
from .utils import load_checkpoint, resolve_device


def _read_config(ckpt: dict, key: str, default):
    """Read a config value from a checkpoint, tolerating several layouts."""
    for holder in ("args", "config", "cfg"):
        d = ckpt.get(holder)
        if isinstance(d, dict) and key in d:
            return d[key]
    return ckpt.get(key, default)


def build_pretrained_model(model_name: str, num_steps: int, num_classes: int,
                           tau: float, v_threshold: float, pretrained: str,
                           device: str = "auto", logger=None, width_scale=None):
    """Build a model and load a pretrained checkpoint (tolerating key diffs).

    ``width_scale`` (and any other architectural config) is read from the
    checkpoint when not provided explicitly, so TTA scripts only need the
    checkpoint path.
    """
    device = resolve_device(device)
    ckpt = load_checkpoint(pretrained, str(device))
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    # the checkpoint is authoritative for architecture config (num_steps/tau/...)
    ck_num_steps = _read_config(ckpt, "num_steps", None)
    if ck_num_steps is not None:
        num_steps = int(ck_num_steps)
    if width_scale is None:
        width_scale = float(_read_config(ckpt, "width_scale", 1.0))
    use_checkpoint = bool(_read_config(ckpt, "use_checkpoint", False))
    model = build_model(model_name, num_steps=num_steps, num_classes=num_classes,
                        tau=tau, v_threshold=v_threshold, width_scale=width_scale,
                        use_checkpoint=use_checkpoint)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if logger is not None:
        logger.info(f"loaded pretrained {pretrained} (missing={len(missing)}, "
                    f"unexpected={len(unexpected)})")
        if missing:
            logger.info(f"  missing keys: {missing[:10]}")
        if unexpected:
            logger.info(f"  unexpected keys: {unexpected[:10]}")
        if "test_acc" in ckpt:
            logger.info(f"  reported clean test acc: {ckpt['test_acc']:.2f}%")
    return model.to(device), device


def resolve_cifar10c(args, clean_test_set=None, logger=None):
    """Return (cifar10c_root, data_mode) honoring --data_mode."""
    mode = getattr(args, "data_mode", "auto")
    use_synthetic = mode == "synthetic"
    if mode == "auto":
        root = Path(args.data_root)
        official = root / "CIFAR-10-C"
        use_synthetic = not (official / "gaussian_noise.npy").exists()
    root_path, data_mode = auto_resolve_cifar10c_root(
        args.data_root,
        use_synthetic=use_synthetic,
        prepare_synthetic=True,
        level=getattr(args, "level", 5),
        clean_test_set=clean_test_set,
    )
    if logger is not None:
        logger.info(f"CIFAR-10-C root: {root_path} (mode={data_mode})")
    return root_path, data_mode
