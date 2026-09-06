"""Phase 1 - proposed method: ZO-TTA (zeroth-order test-time adaptation).

Frozen pretrained SNN + tiny channel adapter (or BN affine) adapted by the
MeZO two-point estimator with the entropy objective, optionally gated by the
PAR-style temporal reliability filter, K ZO steps per batch.

This is the direct instantiation of the report's Phase-1 minimal viable setup:

  * freeze backbone, adapt only the adapter (2*C params) or BN affine,
  * objective = softmax entropy (same as TENT),
  * K = 1..5 ZO steps per batch,
  * two-point Gaussian estimator, fixed input encoding per batch.

Run several configs / seeds to observe the ZO behavior (see run_all.sh).
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
from tta_snn_zo.utils import dump_json, set_seed, setup_logger, write_csv  # noqa: E402


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

    # ZO-TTA knobs
    p.add_argument("--adapt", type=str, default="adapter",
                   choices=["adapter", "bn_all", "bn_last"])
    p.add_argument("--adapter_kind", type=str, default="channel",
                   choices=["channel", "temporal"])
    p.add_argument("--target_layer", type=str, default="pool3")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--block_sparsity", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-3,
                   help="decay adapter scale/bias toward 0 (identity)")
    p.add_argument("--zo_steps", type=int, default=1)
    p.add_argument("--estimator", type=str, default="two_point",
                   choices=["two_point", "one_point"])
    p.add_argument("--no_gate", action="store_true", help="disable batch entropy-window gate")
    p.add_argument("--ent_min_batch", type=float, default=0.30,
                   help="skip batches with mean entropy <= this (collapse guard)")
    p.add_argument("--ent_max_batch", type=float, default=2.30,
                   help="skip batches with mean entropy >= this (garbage guard)")
    p.add_argument("--sample_ent_thresh", type=float, default=1.2,
                   help="per-sample entropy filter for the loss (0 = off)")
    p.add_argument("--clamp_abs", type=float, default=0.2,
                   help="max |scale|,|bias| of adapter params (0 = off)")
    p.add_argument("--no_rollback", action="store_true",
                   help="disable collapse rollback")
    p.add_argument("--use_temporal_gate", action="store_true",
                   help="also apply PAR-style temporal stability gate")
    p.add_argument("--stability_thresh", type=float, default=0.65)
    p.add_argument("--adapt_gate_min", type=float, default=0.35)

    # anti-collapse objective knobs (phase-1 sweep)
    p.add_argument("--objective", type=str, default="entropy",
                   choices=["entropy", "filtered_entropy", "margin",
                            "logsumexp", "pseudo_label", "temp_entropy"])
    p.add_argument("--entropy_thresh", type=float, default=0.6)
    p.add_argument("--conf_thresh", type=float, default=0.8)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--wd_init", type=float, default=0.0,
                   help="weight decay toward init (0 = off)")
    p.add_argument("--kl_lambda", type=float, default=0.0,
                   help="KL-to-source regularizer weight (0 = off)")
    p.add_argument("--clip_norm", type=float, default=0.0,
                   help="per-param ZO gradient L2 clip (0 = off)")
    p.add_argument("--reset_per_batch", action="store_true",
                   help="episodic: reset adapter params each batch")

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
    run_name = args.run_name or (
        f"zo_tta_{args.adapt}_lr{args.lr}_eps{args.eps}_K{args.zo_steps}"
        f"_{args.objective}"
        f"_ent{args.ent_min_batch}_samp{args.sample_ent_thresh}"
        f"_clamp{args.clamp_abs}"
        f"{'_tgate' if args.use_temporal_gate else ''}"
        f"{'_wd' + str(args.wd_init) if args.wd_init > 0 else ''}"
        f"{'_kl' + str(args.kl_lambda) if args.kl_lambda > 0 else ''}"
        f"{'_clip' + str(args.clip_norm) if args.clip_norm > 0 else ''}"
        f"{'_reset' if args.reset_per_batch else ''}"
        f"{'' if args.no_gate else '_gate'}"
        f"{'_norollback' if args.no_rollback else ''}_s{args.seed}"
    )
    logger = setup_logger("zo_tta", args.out_dir, run_name)
    logger.info(f"args: {vars(args)}")

    model, device = build_pretrained_model(args.model, args.num_steps,
                                           args.num_classes, args.tau,
                                           args.v_threshold, args.pretrained,
                                           args.device, logger)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()] or CORRUPTIONS
    cifar10c_root, mode = resolve_cifar10c(args, logger=logger)

    def factory():
        return ZOTTAEngine(
            model,
            adapt=args.adapt,
            adapter_kind=args.adapter_kind,
            target_layers=(args.target_layer,),
            img_size=32,
            lr=args.lr,
            eps=args.eps,
            momentum=args.momentum,
            num_samples=args.num_samples,
            block_sparsity=args.block_sparsity,
            weight_decay=args.weight_decay,
            zo_steps=args.zo_steps,
            estimator=args.estimator,
            gate=not args.no_gate,
            ent_min_batch=args.ent_min_batch,
            ent_max_batch=args.ent_max_batch,
            sample_ent_thresh=args.sample_ent_thresh,
            clamp_abs=args.clamp_abs,
            rollback_collapse=not args.no_rollback,
            use_temporal_gate=args.use_temporal_gate,
            stability_thresh=args.stability_thresh,
            adapt_gate_min=args.adapt_gate_min,
            objective=args.objective,
            entropy_thresh=args.entropy_thresh,
            conf_thresh=args.conf_thresh,
            temp=args.temp,
            wd_init=args.wd_init,
            kl_lambda=args.kl_lambda,
            clip_norm=args.clip_norm,
            reset_per_batch=args.reset_per_batch,
            seed=args.seed,
            device=device,
        )

    rows, per_corr = [], {}
    gate_sum = 0.0
    for name in corruptions:
        loader = get_cifar10c_loader(cifar10c_root, name, args.batch_size,
                                     args.num_workers, args.level)
        engine = factory()
        r = evaluate_on_loader(engine, loader, device, logger=logger,
                               max_batches=args.max_batches or None)
        per_corr[name] = round(r["acc"], 2)
        rows.append({"method": "zo_tta", "corruption": name, "acc": round(r["acc"], 3),
                     "loss": round(r["loss"], 4), "gate": round(r["gate"], 4),
                     "mean_abs_proj_grad": round(r["mean_abs_proj_grad"], 6)})
        gate_sum += r["gate"]
        logger.info(f"[{name}] acc={r['acc']:.2f}% loss={r['loss']:.4f} "
                    f"gate={r['gate']:.3f}")

    mean_acc = sum(per_corr.values()) / len(per_corr)
    per_corr["mean"] = round(mean_acc, 2)
    rows.append({"method": "zo_tta", "corruption": "mean", "acc": round(mean_acc, 3),
                 "gate": round(gate_sum / len(corruptions), 4)})
    csv_path = write_csv(f"{args.out_dir}/{run_name}_results.csv", rows)
    dump_json(f"{args.out_dir}/{run_name}_per_corruption.json", per_corr)
    logger.info("=" * 50)
    logger.info(f"ZO-TTA mean CIFAR-10-C acc: {mean_acc:.2f}%")
    logger.info(f"results: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
