"""Phase 2 sweep runner for ANN VGG9 baseline.

Same 49 candidates as the SNN version, but:
- Uses ANN VGG9 model (loaded from ANN checkpoint)
- Uses candidates_ann.py (ANN-compatible target_layers)
- All candidate IDs are IDENTICAL to SNN version for cross-comparison

Usage:
    python phase2/run_phase2_sweep_ann.py --pretrained <ckpt> --data_root <data>
        [--gpu 0] [--quick] [--families M1_lr,M2_eps]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=int, default=-1)
_args0, _ = _ap.parse_known_args()
if _args0.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args0.gpu)

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for candidates.py

from tta_snn_zo.corruptions import CORRUPTIONS
from tta_snn_zo.data import get_cifar10c_loader
from tta_snn_zo.launch import build_pretrained_model, resolve_cifar10c
from tta_snn_zo.tta import (
    SourceOnlyEngine, TentEngine, ZOTTAEngine, evaluate_on_loader,
)
from tta_snn_zo.memo import MEMOEngine, evaluate_memo
from tta_snn_zo.bn_stats import BNStatsAdaptEngine, evaluate_bn_stats_adapt
from tta_snn_zo.utils import dump_json, load_checkpoint, set_seed, setup_logger, write_csv

from candidates_ann import (
    ALL_CANDIDATES, FAMILY_SUMMARY,
    build_engine_from_cfg, select_candidates, validate_candidates,
)

# Phase 2 default setting: level3, batch=8, 15 corruptions
PHASE2_LEVEL = 3
PHASE2_BATCH = 8
PHASE2_CORRUPTIONS = CORRUPTIONS  # all 15
PHASE2_MAX_BATCHES = 60


def parse_args():
    p = argparse.ArgumentParser(parents=[_ap])
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/phase2")
    p.add_argument("--model", type=str, default="vgg9")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--v_threshold", type=float, default=1.0)
    p.add_argument("--level", type=int, default=PHASE2_LEVEL)
    p.add_argument("--batch_size", type=int, default=PHASE2_BATCH)
    p.add_argument("--corruptions", type=str, default="")
    p.add_argument("--max_batches", type=int, default=PHASE2_MAX_BATCHES)
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--ids", type=str, default="")
    p.add_argument("--families", type=str, default="")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="3 corruptions x 30 batches")
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _load_ckpt_meta(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def build_ann_model(pretrained: str, device):
    """Load ANN VGG9 from checkpoint."""
    from phase0.train_ann_vgg9 import ANN_VGG9
    ckpt = load_checkpoint(pretrained, str(device))
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = ANN_VGG9(num_classes=10)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  ANN model missing keys: {missing[:5]}")
    if unexpected:
        print(f"  ANN model unexpected keys: {unexpected[:5]}")
    # if "test_acc" in ckpt:
    #     print(f"  ANN checkpoint test_acc: {ckpt['test_acc']:.2f}%")
    return model.to(device), device


def main():
    args = parse_args()
    if args.quick:
        args.corruptions = "gaussian_noise,motion_blur,contrast"
        args.max_batches = 30

    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()] \
        or PHASE2_CORRUPTIONS
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [0]
    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    candidates = select_candidates(args.ids, args.families) or ALL_CANDIDATES
    errors = validate_candidates(candidates)
    if errors:
        for e in errors:
            print(f"[VALIDATION ERROR] {e}")
        if not args.dry_run:
            sys.exit(1)

    logger = setup_logger("phase2", str(out_dir / "logs"), "overview")
    logger.info(f"args: {vars(args)}")
    logger.info(f"Phase 2: {len(candidates)} candidates, "
                f"level={args.level}, batch={args.batch_size}, "
                f"{len(corruptions)} corruptions")

    if args.dry_run:
        from collections import defaultdict
        by_fam = defaultdict(list)
        for c in candidates:
            by_fam[c["family"]].append(c)
        for fam, clist in sorted(by_fam.items()):
            print(f"\n  [{fam}] ({FAMILY_SUMMARY.get(fam, '')})")
            for c in clist:
                print(f"    {c['id']:5s} {c['method_name']:45s}")
        print(f"\nTotal: {len(candidates)} candidates")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    cifar10c_root, mode = resolve_cifar10c(
        SimpleNamespace(data_root=args.data_root, data_mode="auto",
                        level=args.level), logger=logger)

    def fresh_model():
        return build_ann_model(args.pretrained, device)

    # ------------------------------------------------------------------ #
    #  Baselines (B0: source, B1: TENT, B2: MEMO, B3: BN Stats)
    #  Each baseline logs per-corruption accuracy and writes to summary_rows
    #  so the CSV has consistent format with candidate rows.
    # ------------------------------------------------------------------ #
    summary_rows = []
    source_acc = {}
    if not args.skip_baselines:
        logger.info("--- B0: Source-only ---")
        per_seed = {corr: [] for corr in corruptions}
        for s in seeds:
            set_seed(s)
            model, dev = fresh_model()
            engine = SourceOnlyEngine(model.to(dev), device=dev)
            for corr in corruptions:
                loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                             args.num_workers, args.level)
                r = evaluate_on_loader(engine, loader, dev,
                                       max_batches=args.max_batches or None)
                per_seed[corr].append(r["acc"])
                logger.info(f"  B0 [{corr}] acc={r['acc']:.2f}%")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for corr in corruptions:
            source_acc[corr] = sum(per_seed[corr]) / len(per_seed[corr])
        src_mean = sum(source_acc.values()) / len(source_acc)
        per_corr = {corr: sum(v) / len(v) for corr, v in per_seed.items()}
        summary_rows.append({
            "id": "B0_source", "family": "baseline", "method": "Source-only",
            "question": "", "mean_acc": round(src_mean, 3), "mean_loss": 0,
            "adapted_frac": 0, "rolled_back_frac": 0, "time_s": 0,
            **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
        })
        logger.info(f"  B0 source-only: mean={src_mean:.2f}%")

        # B1: TENT-BP
        logger.info("--- B1: TENT-BP ---")
        per_seed = {corr: [] for corr in corruptions}
        for s in seeds:
            for corr in corruptions:
                set_seed(s)
                model, dev = fresh_model()
                engine = TentEngine(model.to(dev), device=dev,
                                    lr=1e-3, bn="all", steps=1, gate=True,
                                    ent_min_batch=0.01, ent_max_batch=2.40,
                                    sample_ent_thresh=1.2, max_grad_norm=1.0)
                loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                             args.num_workers, args.level)
                r = evaluate_on_loader(engine, loader, dev,
                                       max_batches=args.max_batches or None)
                per_seed[corr].append(r["acc"])
                logger.info(f"  B1 [{corr}] acc={r['acc']:.2f}%")
                del model, engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        tent_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        per_corr = {corr: sum(v) / len(v) for corr, v in per_seed.items()}
        summary_rows.append({
            "id": "B1_tent_bp", "family": "baseline", "method": "TENT-BP",
            "question": "", "mean_acc": round(tent_mean, 3), "mean_loss": 0,
            "adapted_frac": 1, "rolled_back_frac": 0, "time_s": 0,
            **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
        })
        logger.info(f"  B1 TENT-BP: mean={tent_mean:.2f}%")

        # B2: MEMO
        logger.info("--- B2: MEMO ---")
        per_seed = {corr: [] for corr in corruptions}
        for s in seeds:
            for corr in corruptions:
                set_seed(s)
                model, dev = fresh_model()
                loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                             args.num_workers, args.level)

                def make_memo(m=model, d=dev):
                    return MEMOEngine(m.to(d), n_aug=32, device=d)
                r = evaluate_memo(make_memo, loader, dev,
                                  max_batches=args.max_batches or None)
                per_seed[corr].append(r["acc"])
                logger.info(f"  B2 [{corr}] acc={r['acc']:.2f}%")
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        memo_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        per_corr = {corr: sum(v) / len(v) for corr, v in per_seed.items()}
        summary_rows.append({
            "id": "B2_memo", "family": "baseline", "method": "MEMO (BP-free)",
            "question": "", "mean_acc": round(memo_mean, 3), "mean_loss": 0,
            "adapted_frac": 1, "rolled_back_frac": 0, "time_s": 0,
            **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
        })
        logger.info(f"  B2 MEMO: mean={memo_mean:.2f}%")

        # B3: BN Stats
        logger.info("--- B3: BN Stats ---")
        per_seed = {corr: [] for corr in corruptions}
        for s in seeds:
            for corr in corruptions:
                set_seed(s)
                model, dev = fresh_model()
                loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                             args.num_workers, args.level)

                def make_bn(m=model, d=dev):
                    return BNStatsAdaptEngine(m.to(d), device=d)
                r = evaluate_bn_stats_adapt(make_bn, loader, dev,
                                            max_batches=args.max_batches or None)
                per_seed[corr].append(r["acc"])
                logger.info(f"  B3 [{corr}] acc={r['acc']:.2f}%")
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        bn_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        per_corr = {corr: sum(v) / len(v) for corr, v in per_seed.items()}
        summary_rows.append({
            "id": "B3_bn_stats", "family": "baseline", "method": "BN Stats Adapt",
            "question": "", "mean_acc": round(bn_mean, 3), "mean_loss": 0,
            "adapted_frac": 1, "rolled_back_frac": 0, "time_s": 0,
            **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
        })
        logger.info(f"  B3 BN Stats: mean={bn_mean:.2f}%")

    # ------------------------------------------------------------------ #
    #  Candidates
    # ------------------------------------------------------------------ #
    # NOTE: summary_rows already contains B0-B3 baselines from above.
    # Do NOT re-initialize here — that would discard baseline rows.
    for cand in candidates:
        cid = cand["id"]
        t0 = time.time()
        clog = setup_logger("phase2", str(out_dir / "logs"), cid)
        clog.info(f"[{cand['family']}] {cid}: {cand['method_name']} — {cand['question']}")

        try:
            per_seed_corr = {corr: [] for corr in corruptions}
            losses, adapted_f, rolled_f = [], [], []

            for s in seeds:
                for corr in corruptions:
                    set_seed(s)
                    model, dev = fresh_model()
                    model = model.to(dev)
                    engine = build_engine_from_cfg(cand, model, dev, seed=s)
                    loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                                 args.num_workers, args.level)
                    r = evaluate_on_loader(engine, loader, dev,
                                           max_batches=args.max_batches or None)
                    per_seed_corr[corr].append(r["acc"])
                    losses.append(r["loss"])
                    adapted_f.append(r["adapted_frac"])
                    rolled_f.append(r["rolled_back_frac"])
                    clog.info(f"  s{s} [{corr}] acc={r['acc']:.2f}%")
                    del model, engine
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            mean_acc = sum(sum(v) for v in per_seed_corr.values()) / \
                (len(corruptions) * len(seeds))
            per_corr = {corr: sum(v) / len(v) for corr, v in per_seed_corr.items()}
            summary_rows.append({
                "id": cid, "family": cand["family"],
                "method": cand.get("method_name", ""),
                "question": cand["question"],
                "mean_acc": round(mean_acc, 3),
                "mean_loss": round(sum(losses) / len(losses), 4) if losses else 0,
                "adapted_frac": round(sum(adapted_f) / len(adapted_f), 3) if adapted_f else 0,
                "rolled_back_frac": round(sum(rolled_f) / len(rolled_f), 3) if rolled_f else 0,
                "time_s": round(time.time() - t0, 1),
                **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
            })
            clog.info(f"=== {cid}: mean_acc={mean_acc:.2f}% ({time.time()-t0:.0f}s)")

        except Exception as e:
            import traceback
            clog.error(f"FAILED: {e}\n{traceback.format_exc()}")
            summary_rows.append({
                "id": cid, "family": cand.get("family", ""),
                "method": cand.get("method_name", ""),
                "question": cand["question"], "mean_acc": -1,
                "mean_loss": 0, "adapted_frac": 0, "rolled_back_frac": 0,
                "time_s": round(time.time() - t0, 1), "error": str(e)[:200],
            })

    summary_rows.sort(key=lambda r: r["mean_acc"], reverse=True)
    csv_path = write_csv(str(out_dir / "summary.csv"), summary_rows)
    dump_json(str(out_dir / "summary.json"), {
        "args": vars(args),
        "source_only_mean": round(sum(source_acc.values()) / len(source_acc), 3) if source_acc else None,
        "rows": summary_rows,
    })
    logger.info("=" * 60)
    logger.info("PHASE 2 ANN SUMMARY (ranked by mean acc, BASELINES INCLUDED):")
    for r in summary_rows[:15]:
        logger.info(f"  {r['id']:5s} acc={r['mean_acc']:6.2f}% {r.get('error','')[:60]}")
    logger.info(f"Results in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())