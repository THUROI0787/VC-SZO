"""Shared utilities: seeding, logging, metrics, checkpointing, CSV export."""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Seed & device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 0) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (not trainable_only or p.requires_grad))


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #
def setup_logger(
    name: str,
    log_dir: str,
    run_name: str,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """File + console logger. Returns logger; creates log_dir/run_name.log."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"

    logger = logging.getLogger(f"{name}.{run_name}")
    logger.setLevel(log_level)
    logger.handlers = []
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Iterable[int] = (1,)) -> list:
    """Top-k accuracy in percent. output: [B, C] or [B]; target: [B]."""
    maxk = max(topk)
    batch_size = target.size(0)
    if output.ndim == 1:
        pred = output.view(-1, 1)
    else:
        _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(float(correct_k.mul_(100.0 / batch_size).item()))
    return res


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


# --------------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    state_dict: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"state_dict": state_dict}
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))
    return str(path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


# --------------------------------------------------------------------------- #
#  CSV export
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: list) -> str:
    """rows: list of dicts with identical keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return str(path)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return str(path)


def load_csv(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dump_json(path: str, obj: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return str(path)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
