"""Phase 1 - baseline: TENT (BP-based entropy minimization) on CIFAR-10-C.

Adapts BN affine parameters (gamma only, BN bias disabled in the BNTT recipe)
via entropy minimization with real backpropagation (surrogate gradient) and an
Adam optimizer - the standard ANN TTA baseline adapted to SNNs, exactly as the
SPACE / PAR papers compare against.

Reports per-corruption accuracy + mean, saves CSV + JSON.
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
from tta_snn_zo.tta import TentEngine, evaluate_on_loader  # noqa: E402
from tta_snn_zo.utils import dump_json, set_seed, setup_logger, timestamp, write_csv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/phase1")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--model", type=str, default="vgg9")
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--v_threshold", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4, help="Adam lr (robust TENT default)")
    p.add_argument("--bn", type=str, default="all", choices=["all", "last"])
    p.add_argument("--steps", type=int, default=1, help="adaptation steps per batch")
    p.add_argument("--no_gate", action="store_true", help="disable batch entropy-window gate")
    p.add_argument("--ent_min_batch", type=float, default=0.30)
    p.add_argument("--ent_max_batch", type=float, default=2.30)
    p.add_argument("--sample_ent_thresh", type=float, default=1.2,
                   help="per-sample entropy filter (0 = off)")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
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
    run_name = args.run_name or f"tent_bp_lr{args.lr}_bn{args.bn}_s{args.seed}"
    logger = setup_logger("tent_bp", args.out_dir, run_name)
    logger.info(f"args: {vars(args)}")

    model, device = build_pretrained_model(args.model, args.num_steps,
                                           args.num_classes, args.tau,
                                           args.v_threshold, args.pretrained,
                                           args.device, logger)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()] or CORRUPTIONS
    cifar10c_root, mode = resolve_cifar10c(args, logger=logger)

    def factory():
        m = model  # NOTE: TentEngine mutates the shared model in place
        eng = TentEngine(m, lr=args.lr, bn=args.bn, steps=args.steps,
                         gate=not args.no_gate,
                         ent_min_batch=args.ent_min_batch,
                         ent_max_batch=args.ent_max_batch,
                         sample_ent_thresh=args.sample_ent_thresh,
                         max_grad_norm=args.max_grad_norm,
                         device=device)
        return eng

    rows, per_corr = [], {}
    for name in corruptions:
        loader = get_cifar10c_loader(cifar10c_root, name, args.batch_size,
                                     args.num_workers, args.level)
        engine = factory()
        r = evaluate_on_loader(engine, loader, device, logger=logger,
                               max_batches=args.max_batches or None)
        per_corr[name] = round(r["acc"], 2)
        rows.append({"method": "tent_bp", "corruption": name, "acc": round(r["acc"], 3),
                     "loss": round(r["loss"], 4)})
        logger.info(f"[{name}] acc={r['acc']:.2f}%")

    mean_acc = sum(per_corr.values()) / len(per_corr)
    per_corr["mean"] = round(mean_acc, 2)
    rows.append({"method": "tent_bp", "corruption": "mean", "acc": round(mean_acc, 3)})
    csv_path = write_csv(f"{args.out_dir}/{run_name}_results.csv", rows)
    dump_json(f"{args.out_dir}/{run_name}_per_corruption.json", per_corr)
    logger.info("=" * 50)
    logger.info(f"TENT-BP mean CIFAR-10-C acc: {mean_acc:.2f}%")
    logger.info(f"results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
