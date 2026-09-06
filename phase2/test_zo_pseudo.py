"""Phase 2 - pseudo / one-point ZO (OPZO-inspired, arXiv 2407.12516).

Instead of two perturbed forward passes per step, the one-point estimator uses
the loss of the *unperturbed prediction forward* as the baseline:

    g_hat = ( L(theta + eps z) - L0 ) / eps * z

so each adaptation step costs a single extra forward pass (halving the ZO
latency in the streaming setting).  This script sweeps eps/lr for the one-point
estimator on CIFAR-10-C and compares against the two-point default.
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
from tta_snn_zo.tta import ZOTTAEngine, evaluate_on_loader  # noqa: E402
from tta_snn_zo.utils import set_seed, setup_logger, write_csv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/phase2")
    p.add_argument("--model", type=str, default="vgg9")
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--v_threshold", type=float, default=1.0)
    p.add_argument("--adapt", type=str, default="adapter")
    p.add_argument("--target_layer", type=str, default="pool3")
    p.add_argument("--zo_steps", type=int, default=2)
    p.add_argument("--eps_list", type=str, default="1e-2,5e-2")
    p.add_argument("--lr_list", type=str, default="1e-2,1e-1")
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
    logger = setup_logger("zo_pseudo", args.out_dir, f"sweep_s{args.seed}")
    logger.info(f"args: {vars(args)}")

    model, device = build_pretrained_model(args.model, args.num_steps,
                                           args.num_classes, args.tau,
                                           args.v_threshold, args.pretrained,
                                           args.device, logger)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()] or CORRUPTIONS
    cifar10c_root, mode = resolve_cifar10c(args, logger=logger)

    eps_list = [float(x) for x in args.eps_list.split(",")]
    lr_list = [float(x) for x in args.lr_list.split(",")]

    rows = []
    for eps in eps_list:
        for lr in lr_list:
            tag = f"onepoint_eps{eps}_lr{lr}"
            logger.info("=" * 60)
            logger.info(f"config: estimator=one_point eps={eps} lr={lr}")
            accs = {}
            for name in corruptions:
                loader = get_cifar10c_loader(cifar10c_root, name, args.batch_size,
                                             args.num_workers, args.level)
                engine = ZOTTAEngine(
                    model, adapt=args.adapt, target_layers=(args.target_layer,),
                    img_size=32, lr=lr, eps=eps, zo_steps=args.zo_steps,
                    estimator="one_point", gate=True, seed=args.seed, device=device,
                )
                r = evaluate_on_loader(engine, loader, device, logger=logger,
                                       max_batches=args.max_batches or None)
                accs[name] = r["acc"]
                logger.info(f"  [{name}] acc={r['acc']:.2f}%")
            mean_acc = sum(accs.values()) / len(accs)
            rows.append({"config": tag, "estimator": "one_point", "eps": eps, "lr": lr,
                         "mean_acc": round(mean_acc, 3),
                         **{k: round(v, 3) for k, v in accs.items()}})
            logger.info(f"  ==> mean acc = {mean_acc:.2f}%  ({tag})")

    csv_path = write_csv(f"{args.out_dir}/zo_pseudo_sweep_s{args.seed}.csv", rows)
    logger.info("=" * 60)
    logger.info("One-point ZO sweep summary:")
    for r in rows:
        logger.info(f"  {r['config']:<24} {r['mean_acc']:.2f}%")
    logger.info(f"saved to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
