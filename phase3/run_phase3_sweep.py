"""Phase 3:大规模验证 SNN+ZO candidates。

在多种 setting(level×batch×seed)下系统评测 4 个 baseline + 8 个 SNN+ZO 方法。

Usage:
    python phase3/run_phase3_sweep.py --pretrained <ckpt> --data_root <data>
        [--gpu 0] [--T 6] [--level 1,3,5] [--batch 1,4,8,32,128]
        [--seeds 0,1,2,3] [--quick]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=int, default=-1)
_args0, _ = _ap.parse_known_args()
if _args0.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args0.gpu)

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tta_snn_zo.corruptions import CORRUPTIONS
from tta_snn_zo.data import get_cifar10c_loader
from tta_snn_zo.launch import build_pretrained_model, resolve_cifar10c
from tta_snn_zo.tta import (
    SourceOnlyEngine, TentEngine, ZOTTAEngine, evaluate_on_loader,
)
from tta_snn_zo.memo import MEMOEngine, evaluate_memo
from tta_snn_zo.bn_stats import BNStatsAdaptEngine, evaluate_bn_stats_adapt
from tta_snn_zo.utils import dump_json, set_seed, setup_logger, write_csv

from candidates_phase3 import (
    ALL_CANDIDATES, build_engine_from_cfg, select_candidates,
)

# Default sweep space
# DEFAULT_LEVELS = [1, 3, 5]
# DEFAULT_BATCHES = [1, 2, 4, 8, 32, 128]
DEFAULT_LEVELS = [5]
DEFAULT_BATCHES = [8, 32, 128] # ! TODO
DEFAULT_SEEDS = [42, 3407, 215]
MAX_BATCHES = 60



def parse_args():
    p = argparse.ArgumentParser(parents=[_ap])
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/phase3")
    p.add_argument("--model", type=str, default="vgg9")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--v_threshold", type=float, default=1.0)
    p.add_argument("--T", type=int, default=0,
                   help="Override num_steps (0=use checkpoint default)")
    p.add_argument("--level", type=str, default="",
                   help="Comma-separated levels, e.g. '1,3,5'")
    p.add_argument("--batch", type=str, default="",
                   help="Comma-separated batch sizes, e.g. '1,4,8,32,128'")
    p.add_argument("--seeds", type=str, default="",
                   help="Comma-separated seeds, e.g. '0,1,2,3'")
    p.add_argument("--ids", type=str, default="",
                   help="Comma-separated candidate IDs")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Single setting: level=3, batch=8, seed=0")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--reverse", action="store_true",
               help="Reverse sweep order: seed 215→3407→42, batch large→small, level 5→3→1")
    return p.parse_args()


def build_model_fn(pretrained: str, T: int, device):
    """Build model and load pretrained checkpoint."""
    model, dev = build_pretrained_model(
        "vgg9", T if T > 0 else 25, 10, 20.0, 1.0,
        pretrained, "auto", logger=None)
    return model.to(dev), dev


def run_setting(args, setting, device, logger):
    """Run one setting (level, batch, seed). Returns summary row dict."""
    level, batch_size, seed = setting["level"], setting["batch_size"], setting["seed"]
    out_dir = Path(args.out_dir) / setting["rel_path"]
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"

    # Skip if already completed
    if csv_path.exists() and json_path.exists():
        try:
            with open(json_path) as f:
                existing = json.load(f)
            if len(existing.get("rows", [])) >= 12:  # 4 baselines + 8 candidates
                logger.info(f"  [SKIP] already completed: {setting['rel_path']}")
                return existing
        except Exception:
            pass

    logger.info(f"\n{'='*60}")
    logger.info(f"Setting: level={level}, batch={batch_size}, seed={seed}")
    logger.info(f"{'='*60}")

    # Resolve CIFAR-10-C
    cifar10c_root, mode = resolve_cifar10c(
        SimpleNamespace(data_root=args.data_root, data_mode="auto", level=level),
        logger=logger)

    # Select candidates
    candidates = select_candidates(args.ids) or ALL_CANDIDATES

    summary_rows = []
    source_acc = {}

    # ---- Helper: run one method on all corruptions ----
    def eval_method(engine_fn, method_id, method_name, family, is_baseline=False):
        nonlocal source_acc
        per_corr = {}
        accs = []
        adapted_fracs = []
        rolled_back_fracs = []
        losses = []
        t0 = time.time()

        for corr in CORRUPTIONS:
            try:
                set_seed(seed)
                model, dev = build_model_fn(args.pretrained, args.T, device)
                engine = engine_fn(model.to(dev), dev)
                loader = get_cifar10c_loader(
                    cifar10c_root, corr, batch_size, args.num_workers, level)
                r = evaluate_on_loader(engine, loader, dev,
                                       max_batches=MAX_BATCHES)
                per_corr[corr] = round(r["acc"], 3)
                accs.append(r["acc"])
                adapted_fracs.append(r.get("adapted_frac", 0))
                rolled_back_fracs.append(r.get("rolled_back_frac", 0))
                losses.append(r.get("loss", 0))
                logger.info(f"    [{corr:20s}] acc={r['acc']:.2f}%")
            except Exception as e:
                logger.error(f"    [{corr:20s}] FAILED: {e}")
                per_corr[corr] = -1.0
                accs.append(-1.0)
            finally:
                del model, engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        valid_accs = [a for a in accs if a >= 0]
        mean_acc = sum(valid_accs) / len(valid_accs) if valid_accs else -1.0
        row = {
            "id": method_id, "family": family, "method": method_name,
            "mean_acc": round(mean_acc, 3),
            "mean_loss": round(sum(losses)/len(losses), 4) if losses else 0,
            "adapted_frac": round(sum(adapted_fracs)/len(adapted_fracs), 3) if adapted_fracs else 0,
            "rolled_back_frac": round(sum(rolled_back_fracs)/len(rolled_back_fracs), 3) if rolled_back_fracs else 0,
            "time_s": round(time.time() - t0, 1),
            **{f"acc_{corr}": per_corr.get(corr, -1.0) for corr in CORRUPTIONS},
        }
        if is_baseline and method_id == "B0_source":
            source_acc = per_corr
        return row

    # ---- Baselines ----
    logger.info("--- B0: Source-only ---")
    summary_rows.append(eval_method(
        lambda m, d: SourceOnlyEngine(m, device=d),
        "B0_source", "Source-only", "baseline", is_baseline=True))

    logger.info("--- B1: TENT-BP ---")
    summary_rows.append(eval_method(
        lambda m, d: TentEngine(m, device=d, lr=1e-3, bn="all", steps=1,
                                gate=True, ent_min_batch=0.01, ent_max_batch=2.40,
                                sample_ent_thresh=1.2, max_grad_norm=1.0),
        "B1_tent_bp", "TENT-BP", "baseline"))

    logger.info("--- B2: MEMO ---")
    # MEMO n_aug tuned for memory constraints (V100 32GB, ~22GB available):
    #   T=25: batch=128→n_aug=1, batch=32→n_aug=4
    #   T=12: batch=128→n_aug=2
    #   T=4/6: batch=128→n_aug=4
    #   all other settings: n_aug=8
    T = getattr(args, "T", 0)
    if T == 25:
        if batch_size >= 128:
            memo_n_aug = 1
        elif batch_size >= 32:
            memo_n_aug = 4
        else:
            memo_n_aug = 8
    elif T == 12:
        if batch_size >= 128:
            memo_n_aug = 2
        else:
            memo_n_aug = 8
    elif T in (4, 6):
        if batch_size >= 128:
            memo_n_aug = 4
        else:
            memo_n_aug = 8
    else:
        memo_n_aug = 8
    summary_rows.append(eval_method(
        lambda m, d: MEMOEngine(m, n_aug=memo_n_aug, lr=1e-3, device=d),
        "B2_memo", f"MEMO (n_aug={memo_n_aug})", "baseline"))

    logger.info("--- B3: BN Stats ---")
    summary_rows.append(eval_method(
        lambda m, d: BNStatsAdaptEngine(m, device=d),
        "B3_bn_stats", "BN Stats Adapt", "baseline"))

    # ---- SNN+ZO Candidates ----
    for cand in candidates:
        cid = cand["id"]
        logger.info(f"--- {cid}: {cand['method_name']} ---")
        summary_rows.append(eval_method(
            lambda m, d, c=cand, s=seed: build_engine_from_cfg(c, m, d, s),
            cid, cand["method_name"], cand["family"]))

    # Sort by mean_acc descending
    summary_rows.sort(key=lambda r: r["mean_acc"], reverse=True)

    # Save
    csv_path = str(csv_path)
    write_csv(csv_path, summary_rows)
    dump_json(str(json_path), {
        "setting": setting,
        "args": vars(args),
        "rows": summary_rows,
    })

    logger.info(f"  Results saved to {out_dir}")
    for r in summary_rows[:5]:
        logger.info(f"    {r['id']:6s} acc={r['mean_acc']:.2f}%")

    return {"rows": summary_rows}


def main():
    args = parse_args()

    # Resolve sweep space
    if args.quick:
        levels = [3]
        batches = [8]
        seeds = [0]
    else:
        levels = [int(x) for x in args.level.split(",") if x.strip()] or DEFAULT_LEVELS
        batches = [int(x) for x in args.batch.split(",") if x.strip()] or DEFAULT_BATCHES
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or DEFAULT_SEEDS


    # Build all settings
    all_settings = []
    for level in levels:
        for batch_size in batches:
            for seed in seeds:
                all_settings.append({
                    "level": level,
                    "batch_size": batch_size,
                    "seed": seed,
                    "rel_path": f"level{level}_batch{batch_size}_seed{seed}",
                })

    # Sort: default easy-first (level↑ batch↑ seed↑), --reverse flips all
    reverse = getattr(args, "reverse", False)
    all_settings.sort(key=lambda s: (s["level"], s["batch_size"], s["seed"]),
                      reverse=reverse)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("phase3", str(Path(args.out_dir) / "logs"), "overview")
    logger.info(f"args: {vars(args)}")
    logger.info(f"Phase 3: {len(all_settings)} settings, "
                f"{len(ALL_CANDIDATES)} candidates + 4 baselines")
    logger.info(f"Device: {device}")

    if args.dry_run:
        print(f"\nPhase 3 settings ({len(all_settings)} total):")
        for s in all_settings:
            print(f"  {s['rel_path']}")
        print(f"\nCandidates ({len(ALL_CANDIDATES)}):")
        for c in ALL_CANDIDATES:
            print(f"  {c['id']:6s} {c['method_name']}")
        return 0

    # Run all settings
    results = {}
    for i, setting in enumerate(all_settings):
        logger.info(f"\n[{i+1}/{len(all_settings)}] {setting['rel_path']}")
        try:
            r = run_setting(args, setting, device, logger)
            results[setting["rel_path"]] = r
        except Exception as e:
            logger.error(f"SETTING FAILED: {setting['rel_path']}: {e}\n{traceback.format_exc()}")
            results[setting["rel_path"]] = {"error": str(e)}

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("="*60)
    completed = sum(1 for v in results.values() if "error" not in v)
    failed = sum(1 for v in results.values() if "error" in v)
    logger.info(f"Completed: {completed}/{len(all_settings)}, Failed: {failed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())