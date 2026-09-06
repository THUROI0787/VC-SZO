"""Phase 1 - baseline: source-only (no adaptation) on CIFAR-10-C.

Evaluates the pretrained SNN on clean CIFAR-10 and all 15 corruptions (level 5)
without any adaptation.  Reports per-corruption accuracy, mean, and a CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tta_snn_zo.corruptions import CORRUPTIONS  # noqa: E402
from tta_snn_zo.data import get_clean_test_accuracy_loader, get_cifar10c_loader  # noqa: E402
from tta_snn_zo.launch import build_pretrained_model, resolve_cifar10c  # noqa: E402
from tta_snn_zo.tta import SourceOnlyEngine, evaluate_on_loader  # noqa: E402
from tta_snn_zo.utils import accuracy, dump_json, set_seed, setup_logger, timestamp, write_csv  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


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
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--data_mode", type=str, default="auto", choices=["auto", "official", "synthetic"])
    p.add_argument("--corruptions", type=str, default="", help="comma list; default all 15")
    p.add_argument("--max_batches", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or f"source_only_s{args.seed}"
    logger = setup_logger("source_only", args.out_dir, run_name)
    logger.info(f"args: {vars(args)}")

    model, device = build_pretrained_model(args.model, args.num_steps,
                                           args.num_classes, args.tau,
                                           args.v_threshold, args.pretrained,
                                           args.device, logger)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()] or CORRUPTIONS
    cifar10c_root, mode = resolve_cifar10c(args, clean_test_set=None, logger=logger)

    # clean test
    clean_loader = get_clean_test_accuracy_loader(args.data_root, args.batch_size,
                                                  args.num_workers)
    engine = SourceOnlyEngine(model, device)
    clean = evaluate_on_loader(engine, clean_loader, device, logger=logger,
                               max_batches=args.max_batches or None)
    logger.info(f"[clean] acc={clean['acc']:.2f}%")

    rows = []
    per_corr = {}
    for name in corruptions:
        loader = get_cifar10c_loader(cifar10c_root, name, args.batch_size,
                                     args.num_workers, args.level)
        r = evaluate_on_loader(engine, loader, device, logger=logger,
                               max_batches=args.max_batches or None)
        per_corr[name] = round(r["acc"], 2)
        rows.append({"method": "source_only", "corruption": name,
                     "acc": round(r["acc"], 3), "loss": round(r["loss"], 4)})
        logger.info(f"[{name}] acc={r['acc']:.2f}%")

    mean_acc = sum(per_corr.values()) / len(per_corr)
    per_corr["mean"] = round(mean_acc, 2)
    per_corr["clean"] = round(clean["acc"], 2)
    rows.append({"method": "source_only", "corruption": "mean", "acc": round(mean_acc, 3)})
    rows.append({"method": "source_only", "corruption": "clean", "acc": round(clean["acc"], 3)})
    csv_path = write_csv(f"{args.out_dir}/{run_name}_results.csv", rows)
    dump_json(f"{args.out_dir}/{run_name}_per_corruption.json", per_corr)
    logger.info("=" * 50)
    logger.info(f"source-only mean CIFAR-10-C acc: {mean_acc:.2f}%  (clean {clean['acc']:.2f}%)")
    logger.info(f"results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
