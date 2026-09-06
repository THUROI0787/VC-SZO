"""Phase 0 - prepare CIFAR-10-C (synthetic fallback).

Generates the 15 CIFAR-10-C corruption .npy files (level ``--level``, official
layout) into ``<data_root>/CIFAR-10-C`` using the official Hendrycks corruption
algorithm (see tta_snn_zo/corruptions.py).  Used when the official 2.7GB tar
cannot be downloaded.  If the official files already exist, this script skips.

Usage:
    python phase0/prepare_cifar10c.py --data_root data --level 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tta_snn_zo.corruptions import CORRUPTIONS, prepare_level5_dir  # noqa: E402
from tta_snn_zo.utils import setup_logger  # noqa: E402
from torchvision import datasets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--n_per_severity", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="outputs/phase0")
    args = p.parse_args()

    logger = setup_logger("prepare_cifar10c", args.out_dir, f"level{args.level}")
    out_dir = Path(args.data_root) / "CIFAR-10-C"
    all_present = all((out_dir / f"{c}.npy").exists() for c in CORRUPTIONS) \
        and (out_dir / "labels.npy").exists()
    if all_present:
        logger.info(f"official-style files already present at {out_dir}; nothing to do")
        return 0

    logger.info(f"loading CIFAR-10 test set from {args.data_root} ...")
    ds = datasets.CIFAR10(root=args.data_root, train=False, download=True)
    images = np.asarray(ds.data)
    labels = np.asarray(ds.targets)
    logger.info(f"generating {len(CORRUPTIONS)} corruptions at level {args.level} "
                f"(n={args.n_per_severity}) -> {out_dir}")
    summary = prepare_level5_dir(images, labels, str(out_dir),
                                 severities=(args.level,),
                                 n_per_severity=args.n_per_severity,
                                 seed=args.seed)
    logger.info(f"done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
