"""Phase 2 - variance-reduced ZO: random-subspace (blockwise) + momentum +
multi-sample averaging.

Motivation (report section 2.3/2.5 & SZO): the spiking nonlinearity amplifies ZO
variance; the two standard cures are (a) perturb only a random low-dimensional
subspace of the adapter each step (``--block_sparsity``), and (b) average /
smooth the gradient estimate (``--num_samples``, ``--momentum``).  This script
sweeps those knobs on the CIFAR-10-C benchmark and prints a compact comparison.
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
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--zo_steps", type=int, default=2)
    # sweep knobs
    p.add_argument("--block_sparsity_list", type=str, default="1.0,0.5,0.25")
    p.add_argument("--momentum_list", type=str, default="0.0,0.9")
    p.add_argument("--num_samples_list", type=str, default="1,2")
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
    logger = setup_logger("zo_subspace", args.out_dir, f"sweep_s{args.seed}")
    logger.info(f"args: {vars(args)}")

    model, device = build_pretrained_model(args.model, args.num_steps,
                                           args.num_classes, args.tau,
                                           args.v_threshold, args.pretrained,
                                           args.device, logger)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()] or CORRUPTIONS
    cifar10c_root, mode = resolve_cifar10c(args, logger=logger)

    blocks = [float(x) for x in args.block_sparsity_list.split(",")]
    momenta = [float(x) for x in args.momentum_list.split(",")]
    samples = [int(x) for x in args.num_samples_list.split(",")]

    rows = []
    for bs in blocks:
        for mom in momenta:
            for ns in samples:
                tag = f"bs{bs}_mom{mom}_ns{ns}"
                logger.info("=" * 60)
                logger.info(f"config: block_sparsity={bs} momentum={mom} num_samples={ns}")
                accs = {}
                for name in corruptions:
                    loader = get_cifar10c_loader(cifar10c_root, name, args.batch_size,
                                                 args.num_workers, args.level)
                    engine = ZOTTAEngine(
                        model, adapt=args.adapt, target_layers=(args.target_layer,),
                        img_size=32, lr=args.lr, eps=args.eps, momentum=mom,
                        num_samples=ns, block_sparsity=bs, zo_steps=args.zo_steps,
                        gate=True, seed=args.seed, device=device,
                    )
                    r = evaluate_on_loader(engine, loader, device, logger=logger,
                                           max_batches=args.max_batches or None)
                    accs[name] = r["acc"]
                    logger.info(f"  [{name}] acc={r['acc']:.2f}%")
                mean_acc = sum(accs.values()) / len(accs)
                rows.append({"config": tag, "block_sparsity": bs, "momentum": mom,
                             "num_samples": ns, "mean_acc": round(mean_acc, 3),
                             **{k: round(v, 3) for k, v in accs.items()}})
                logger.info(f"  ==> mean acc = {mean_acc:.2f}%  ({tag})")

    csv_path = write_csv(f"{args.out_dir}/zo_subspace_sweep_s{args.seed}.csv", rows)
    logger.info("=" * 60)
    logger.info("Sweep summary (config -> mean acc):")
    for r in rows:
        logger.info(f"  {r['config']:<16} {r['mean_acc']:.2f}%")
    logger.info(f"saved to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
