"""Phase 2 - Continual Test-Time Adaptation (CTTA): ZO-TTA on a continuous stream.

Follows the CTTA protocol: the 15 corruptions arrive *sequentially* (level 5),
the model adapts continuously and is never reset between corruptions.  This
exposes error accumulation / source forgetting - the core challenge PAR targets
with its reliability filtering - and lets us measure how much the ZO adapter
drifts over the long horizon.

Two modes:
  * ``--mode continual``  (default): single engine across the whole stream.
  * ``--mode episodic``   : reset adapter between corruptions (reference).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tta_snn_zo.corruptions import CORRUPTIONS  # noqa: E402
from tta_snn_zo.data import get_cifar10c_loader  # noqa: E402
from tta_snn_zo.launch import build_pretrained_model, resolve_cifar10c  # noqa: E402
from tta_snn_zo.tta import ZOTTAEngine  # noqa: E402
from tta_snn_zo.utils import (  # noqa: E402
    accuracy, dump_json, set_seed, setup_logger, write_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/phase2")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--model", type=str, default="vgg9")
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--v_threshold", type=float, default=1.0)
    p.add_argument("--adapt", type=str, default="adapter")
    p.add_argument("--adapter_kind", type=str, default="channel")
    p.add_argument("--target_layer", type=str, default="pool3")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--zo_steps", type=int, default=1)
    p.add_argument("--mode", type=str, default="continual", choices=["continual", "episodic"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--data_mode", type=str, default="auto")
    p.add_argument("--corruptions", type=str, default="")
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or f"zo_ctta_{args.mode}_lr{args.lr}_eps{args.eps}_s{args.seed}"
    logger = setup_logger("zo_ctta", args.out_dir, run_name)
    logger.info(f"args: {vars(args)}")

    model, device = build_pretrained_model(args.model, args.num_steps,
                                           args.num_classes, args.tau,
                                           args.v_threshold, args.pretrained,
                                           args.device, logger)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()] or CORRUPTIONS
    cifar10c_root, mode = resolve_cifar10c(args, logger=logger)

    rows, per_corr = [], {}
    stream_batch = 0
    total_acc, total_n = 0.0, 0
    engine = None

    for name in corruptions:
        loader = get_cifar10c_loader(cifar10c_root, name, args.batch_size,
                                     args.num_workers, args.level)
        if engine is None or args.mode == "episodic":
            engine = ZOTTAEngine(
                model, adapt=args.adapt, adapter_kind=args.adapter_kind,
                target_layers=(args.target_layer,), img_size=32, lr=args.lr,
                eps=args.eps, momentum=args.momentum, zo_steps=args.zo_steps,
                gate=True, seed=args.seed, device=device,
            )
            if args.mode == "episodic":
                logger.info(f"  [episodic] reset engine before {name}")
        acc_sum, gate_sum, n = 0.0, 0.0, 0
        for i, (x, y) in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, stats = engine.predict_and_adapt(x)
            acc_sum += accuracy(logits, y)[0]
            gate_sum += stats["gate"]
            n += 1
            total_acc += accuracy(logits, y)[0]
            total_n += 1
            stream_batch += 1
        acc = acc_sum / max(n, 1)
        per_corr[name] = round(acc, 2)
        rows.append({"method": f"zo_ctta_{args.mode}", "corruption": name,
                     "acc": round(acc, 3), "gate": round(gate_sum / max(n, 1), 4)})
        logger.info(f"[{name}] acc={acc:.2f}% (stream batch {stream_batch})")

    mean_acc = sum(per_corr.values()) / len(per_corr)
    per_corr["mean"] = round(mean_acc, 2)
    rows.append({"method": f"zo_ctta_{args.mode}", "corruption": "mean",
                 "acc": round(mean_acc, 3)})
    csv_path = write_csv(f"{args.out_dir}/{run_name}_results.csv", rows)
    dump_json(f"{args.out_dir}/{run_name}_per_corruption.json", per_corr)
    logger.info("=" * 50)
    logger.info(f"ZO-CTTA ({args.mode}) mean acc over {len(corruptions)} corruptions: {mean_acc:.2f}%")
    logger.info(f"results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
