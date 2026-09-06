"""Phase 1 Multi-Setting Sweep Runner.

Runs ALL candidates + baselines across MULTIPLE settings (level/batch/corruption
combinations) in a single automated pass. Designed for overnight unattended runs.

Settings are defined as a grid of:
  - levels: [3, 4, 5]
  - batch_sizes: [64, 200]
  - corruption_sets: ["5_representative", "all_15"]

Each setting runs:
  - B0: Source-only (no adaptation)
  - B1: TENT-BP (with anti-collapse gates)
  - B2: MEMO (BP-free, augmentation-based)
  - B3: BN Stats Adaptation (BP-free)
  - All 46 ZO candidates

Output: outputs/sweep_v2/<setting>/summary.csv

Usage:
    python phase1/run_multi_setting_sweep.py --pretrained <ckpt> --data_root <data>
        [--gpu 0] [--quick] [--settings level5_b64_c15]

Two-GPU mode (recommended):
    # GPU 0 runs settings 1-3, GPU 1 runs settings 4-6
    CUDA_VISIBLE_DEVICES=0 python phase1/run_multi_setting_sweep.py ... --gpu 0 --settings half1
    CUDA_VISIBLE_DEVICES=1 python phase1/run_multi_setting_sweep.py ... --gpu 1 --settings half2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

# GPU selection before torch import
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=int, default=-1)
_args0, _ = _ap.parse_known_args()
if _args0.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args0.gpu)

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tta_snn_zo.data import get_cifar10c_loader
from tta_snn_zo.launch import build_pretrained_model, resolve_cifar10c
from tta_snn_zo.tta import (
    SourceOnlyEngine, TentEngine, ZOTTAEngine, evaluate_on_loader,
)
from tta_snn_zo.memo import MEMOEngine, evaluate_memo
from tta_snn_zo.bn_stats import BNStatsAdaptEngine, evaluate_bn_stats_adapt
from tta_snn_zo.utils import dump_json, set_seed, setup_logger, write_csv

from candidates import (
    ALL_CANDIDATES, FAMILY_SUMMARY,
    build_engine_from_cfg, select_candidates, validate_candidates,
)

# --------------------------------------------------------------------------- #
#  Settings definition
# --------------------------------------------------------------------------- #
ALL_15_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

REP_5_CORRUPTIONS = ["gaussian_noise", "motion_blur", "snow",
                     "contrast", "jpeg_compression"]

SETTINGS = {
    # (name, level, batch_size, corruptions, max_batches, description)
    "level5_b64_c15": (5, 64, ALL_15_CORRUPTIONS, 60,
                       "Level 5, batch=64, 15 corr (hardest, standard)"),
    "level5_b8_c15": (5, 8, ALL_15_CORRUPTIONS, 60,
                      "Level 5, batch=8, 15 corr (small batch, high variance)"),
    "level5_b160_c15": (5, 160, ALL_15_CORRUPTIONS, 30,
                        "Level 5, batch=160, 15 corr (large batch, TENT standard)"),
    "level3_b64_c15": (3, 64, ALL_15_CORRUPTIONS, 60,
                       "Level 3, batch=64, 15 corr (medium difficulty)"),
    "level3_b8_c15": (3, 8, ALL_15_CORRUPTIONS, 60,
                      "Level 3, batch=8, 15 corr (medium, small batch)"),
    "level3_b160_c15": (3, 160, ALL_15_CORRUPTIONS, 30,
                        "Level 3, batch=160, 15 corr (medium, large batch)"),
    "level1_b64_c15": (1, 64, ALL_15_CORRUPTIONS, 60,
                       "Level 1, batch=64, 15 corr (easiest)"),
    "level1_b8_c15": (1, 8, ALL_15_CORRUPTIONS, 60,
                      "Level 1, batch=8, 15 corr (easiest, small batch)"),
    "level1_b160_c15": (1, 160, ALL_15_CORRUPTIONS, 30,
                        "Level 1, batch=160, 15 corr (easiest, large batch)"),
}

SETTING_GROUPS = {
    "all": list(SETTINGS.keys()),
    "half1": ["level5_b8_c15", "level5_b64_c15", "level1_b8_c15", "level3_b8_c15", "level5_b160_c15"],
    "half2": ["level1_b64_c15", "level3_b64_c15", "level3_b160_c15", "level1_b160_c15",],
    "quick": ["level5_b64_c15"],
    "sweep_fast": ["level5_b64_c15", "level3_b64_c15"],
    "level5_focus": ["level5_b64_c15", "level5_b8_c15", "level5_b160_c15"],
    "level3_focus": ["level3_b64_c15", "level3_b8_c15", "level3_b160_c15"],
}


def parse_args():
    p = argparse.ArgumentParser(parents=[_ap])
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/sweep_v2")
    p.add_argument("--model", type=str, default="vgg9", choices=["vgg9", "resnet19"])
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--v_threshold", type=float, default=1.0)
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--ids", type=str, default="", help="comma list of candidate ids")
    p.add_argument("--families", type=str, default="", help="comma list of families")
    p.add_argument("--settings", type=str, default="all",
                   help="comma list or group name (all/half1/half2/quick/sweep_fast)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="single quick setting (level5_b64_c15) when --settings is 'all'; "
                        "respects explicit --settings if provided")
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--skip_candidates", action="store_true")
    p.add_argument("--profile_memory", action="store_true",
                   help="record peak CUDA memory")
    p.add_argument("--dry_run", action="store_true",
                   help="print plan, run nothing")
    return p.parse_args()


def _load_ckpt_meta(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def run_setting(
    setting_name: str,
    level: int,
    batch_size: int,
    corruptions: list,
    max_batches: int,
    args,
    seeds: list,
    out_dir: Path,
    logger,
):
    """Run all baselines + candidates for one setting."""
    setting_dir = out_dir / setting_name
    (setting_dir / "logs").mkdir(parents=True, exist_ok=True)
    slog = setup_logger("setting", str(setting_dir / "logs"), "overview")
    slog.info(f"=== Setting: {setting_name} (level={level}, batch={batch_size}, "
              f"{len(corruptions)} corr, max_batches={max_batches}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar10c_root, mode = resolve_cifar10c(
        SimpleNamespace(data_root=args.data_root, data_mode="auto", level=level),
        logger=slog)
    slog.info(f"CIFAR-10-C root: {cifar10c_root} (mode={mode})")

    # Collect membrane reference stats if needed (for membrane_kl candidates)
    ref_stats = None
    active_candidates = select_candidates(args.ids, args.families) or ALL_CANDIDATES
    needs_ref = any(
        c["engine"].get("objective") == "membrane_kl" or
        c["engine"].get("use_membrane_gate", False)
        for c in active_candidates
    )
    if needs_ref:
        slog.info("Collecting membrane reference statistics from source data...")
        from tta_snn_zo.membrane import collect_membrane_stats
        from tta_snn_zo.data import get_cifar10_loaders
        m_ref, d_ref = build_pretrained_model(
            args.model, 25, args.num_classes, args.tau, args.v_threshold,
            args.pretrained, "auto", logger=None)
        _, clean_loader = get_cifar10_loaders(
            args.data_root, batch_size=min(batch_size, 128), num_workers=args.num_workers)
        ref_stats = collect_membrane_stats(m_ref, clean_loader, d_ref, num_batches=10)
        slog.info(f"Collected membrane stats for {len(ref_stats)} LIF layers")
        del m_ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fresh_model():
        return build_pretrained_model(
            args.model, 25, args.num_classes, args.tau, args.v_threshold,
            args.pretrained, "auto", logger=None)

    summary_rows = []

    # ================================================================== #
    #  B0: Source-only
    # ================================================================== #
    if not args.skip_baselines:
        slog.info("--- B0: Source-only ---")
        per_seed = {corr: [] for corr in corruptions}
        for s in seeds:
            set_seed(s)
            model, dev = fresh_model()
            engine = SourceOnlyEngine(model.to(dev), device=dev)
            for corr in corruptions:
                loader = get_cifar10c_loader(cifar10c_root, corr, batch_size,
                                             args.num_workers, level)
                r = evaluate_on_loader(engine, loader, dev,
                                       max_batches=max_batches or None,
                                       profile_memory=args.profile_memory)
                per_seed[corr].append(r["acc"])
                slog.info(f"  B0 [{corr}] acc={r['acc']:.2f}%")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        source_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        per_corr = {corr: sum(v) / len(v) for corr, v in per_seed.items()}
        summary_rows.append({
            "id": "B0_source", "family": "baseline", "method": "Source-only",
            "mean_acc": round(source_mean, 3), "mean_loss": 0,
            "adapted_frac": 0, "rolled_back_frac": 0,
            "peak_memory_mb": 0,
            **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
        })
        slog.info(f"  B0 source-only: mean={source_mean:.2f}%")

    # ================================================================== #
    #  B1: TENT-BP
    # ================================================================== #
    if not args.skip_baselines:
        slog.info("--- B1: TENT-BP ---")
        per_seed = {corr: [] for corr in corruptions}
        peak_mem = 0
        for s in seeds:
            for corr in corruptions:
                set_seed(s)
                model, dev = fresh_model()
                engine = TentEngine(model.to(dev), device=dev,
                                    lr=1e-4, bn="all", steps=1, gate=True,
                                    ent_min_batch=0.30, ent_max_batch=2.30,
                                    sample_ent_thresh=1.2, max_grad_norm=1.0)
                loader = get_cifar10c_loader(cifar10c_root, corr, batch_size,
                                             args.num_workers, level)
                r = evaluate_on_loader(engine, loader, dev,
                                       max_batches=max_batches or None,
                                       profile_memory=args.profile_memory)
                per_seed[corr].append(r["acc"])
                peak_mem = max(peak_mem, r.get("peak_memory_mb", 0))
                slog.info(f"  B1 [{corr}] acc={r['acc']:.2f}%")
                del model, engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        tent_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        per_corr = {corr: sum(v) / len(v) for corr, v in per_seed.items()}
        summary_rows.append({
            "id": "B1_tent_bp", "family": "baseline", "method": "TENT-BP",
            "mean_acc": round(tent_mean, 3), "mean_loss": 0,
            "adapted_frac": 1, "rolled_back_frac": 0,
            "peak_memory_mb": round(peak_mem, 1),
            **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
        })
        slog.info(f"  B1 TENT-BP: mean={tent_mean:.2f}%")

    # ================================================================== #
    #  B2: MEMO (BP-free)
    # ================================================================== #
    if not args.skip_baselines:
        slog.info("--- B2: MEMO (BP-free) ---")
        per_seed = {corr: [] for corr in corruptions}
        peak_mem = 0
        for s in seeds:
            for corr in corruptions:
                set_seed(s)
                model, dev = fresh_model()
                loader = get_cifar10c_loader(cifar10c_root, corr, batch_size,
                                             args.num_workers, level)
                def make_memo(m=model, d=dev):
                    return MEMOEngine(m.to(d), n_aug=32, device=d)
                r = evaluate_memo(make_memo, loader, dev,
                                  max_batches=max_batches or None)
                per_seed[corr].append(r["acc"])
                slog.info(f"  B2 [{corr}] acc={r['acc']:.2f}%")
                if args.profile_memory and torch.cuda.is_available():
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024**2))
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        memo_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        per_corr = {corr: sum(v) / len(v) for corr, v in per_seed.items()}
        summary_rows.append({
            "id": "B2_memo", "family": "baseline", "method": "MEMO (BP-free)",
            "mean_acc": round(memo_mean, 3), "mean_loss": 0,
            "adapted_frac": 1, "rolled_back_frac": 0,
            "peak_memory_mb": round(peak_mem, 1),
            **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
        })
        slog.info(f"  B2 MEMO: mean={memo_mean:.2f}%")

    # ================================================================== #
    #  B3: BN Stats Adaptation (BP-free)
    # ================================================================== #
    if not args.skip_baselines:
        slog.info("--- B3: BN Stats Adaptation (BP-free) ---")
        per_seed = {corr: [] for corr in corruptions}
        peak_mem = 0
        for s in seeds:
            for corr in corruptions:
                set_seed(s)
                model, dev = fresh_model()
                loader = get_cifar10c_loader(cifar10c_root, corr, batch_size,
                                             args.num_workers, level)
                def make_bn(m=model, d=dev):
                    return BNStatsAdaptEngine(m.to(d), device=d)
                r = evaluate_bn_stats_adapt(make_bn, loader, dev,
                                            max_batches=max_batches or None)
                per_seed[corr].append(r["acc"])
                slog.info(f"  B3 [{corr}] acc={r['acc']:.2f}%")
                if args.profile_memory and torch.cuda.is_available():
                    peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024**2))
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        bn_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        per_corr = {corr: sum(v) / len(v) for corr, v in per_seed.items()}
        summary_rows.append({
            "id": "B3_bn_stats", "family": "baseline", "method": "BN Stats Adapt",
            "mean_acc": round(bn_mean, 3), "mean_loss": 0,
            "adapted_frac": 1, "rolled_back_frac": 0,
            "peak_memory_mb": round(peak_mem, 1),
            **{f"acc_{corr}": round(per_corr[corr], 3) for corr in corruptions},
        })
        slog.info(f"  B3 BN Stats: mean={bn_mean:.2f}%")

    # ================================================================== #
    #  ZO Candidates
    # ================================================================== #
    if not args.skip_candidates:
        candidates = select_candidates(args.ids, args.families) or ALL_CANDIDATES
        errors = validate_candidates(candidates)
        if errors:
            for e in errors:
                slog.error(f"[VALIDATION] {e}")
            return summary_rows

        slog.info(f"--- Running {len(candidates)} ZO candidates ---")
        for cand in candidates:
            cid = cand["id"]
            t0 = time.time()
            clog = setup_logger("sweep", str(setting_dir / "logs"), cid)
            clog.info(f"[{setting_name}] {cid} [{cand['family']}] {cand['question']}")

            try:
                per_seed_corr = {corr: [] for corr in corruptions}
                losses, adapted_f, rolled_f = [], [], []
                peak_mem = 0

                for s in seeds:
                    for corr in corruptions:
                        set_seed(s)
                        model, dev = fresh_model()
                        model = model.to(dev)
                        # Inject ref_stats for membrane_kl candidates
                        if ref_stats is not None and cand["engine"].get("objective") == "membrane_kl":
                            cand["engine"]["ref_stats"] = ref_stats
                        engine = build_engine_from_cfg(cand, model, dev, seed=s)
                        loader = get_cifar10c_loader(cifar10c_root, corr, batch_size,
                                                     args.num_workers, level)
                        r = evaluate_on_loader(engine, loader, dev,
                                               max_batches=max_batches or None,
                                               profile_memory=args.profile_memory)
                        per_seed_corr[corr].append(r["acc"])
                        losses.append(r["loss"])
                        adapted_f.append(r["adapted_frac"])
                        rolled_f.append(r["rolled_back_frac"])
                        peak_mem = max(peak_mem, r.get("peak_memory_mb", 0))
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
                    "peak_memory_mb": round(peak_mem, 1),
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
                    "peak_memory_mb": 0, "time_s": round(time.time() - t0, 1),
                    "error": str(e)[:200],
                })

    # Sort and save
    summary_rows.sort(key=lambda r: r["mean_acc"], reverse=True)
    csv_path = write_csv(str(setting_dir / "summary.csv"), summary_rows)
    dump_json(str(setting_dir / "summary.json"), {
        "setting": setting_name, "level": level, "batch_size": batch_size,
        "corruptions": corruptions, "max_batches": max_batches,
        "rows": summary_rows,
    })
    slog.info(f"Results saved to {csv_path}")
    return summary_rows


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve settings
    # --smoke overrides to a single quick setting, but respects explicit --settings
    if args.smoke and args.settings == "all":
        setting_names = ["level5_b64_c15"]
    else:
        setting_names = []
        for s in args.settings.split(","):
            s = s.strip()
            if s in SETTING_GROUPS:
                setting_names.extend(SETTING_GROUPS[s])
            elif s in SETTINGS:
                setting_names.append(s)
            else:
                print(f"WARNING: unknown setting/group '{s}', skipping")

    if not setting_names:
        print("No valid settings specified. Available:")
        for k, v in SETTINGS.items():
            print(f"  {k}: {v[-1]}")
        print(f"Groups: {list(SETTING_GROUPS.keys())}")
        return 1

    logger = setup_logger("master", str(out_dir / "logs"), "overview")
    logger.info(f"args: {vars(args)}")
    logger.info(f"Settings to run: {setting_names}")
    logger.info(f"Seeds: {seeds}")

    if args.dry_run:
        print(f"Would run {len(setting_names)} settings with seeds={seeds}:")
        for sn in setting_names:
            lv, bs, corr, mb, desc = SETTINGS[sn]
            print(f"  {sn}: {desc} ({len(corr)} corr, {mb} batches)")
        return 0

    all_results = {}
    for sn in setting_names:
        lv, bs, corr, mb, desc = SETTINGS[sn]
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting setting: {sn} - {desc}")
        logger.info(f"{'='*60}")
        t0 = time.time()
        rows = run_setting(sn, lv, bs, corr, mb, args, seeds, out_dir, logger)
        elapsed = time.time() - t0
        all_results[sn] = {
            "rows": rows,
            "elapsed_s": round(elapsed, 1),
            "best_candidate": rows[0]["id"] if rows else None,
            "best_acc": rows[0]["mean_acc"] if rows else None,
        }
        logger.info(f"Setting {sn} done in {elapsed:.0f}s. "
                    f"Best: {all_results[sn]['best_candidate']} "
                    f"({all_results[sn]['best_acc']:.2f}%)")

    # Cross-setting summary
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-SETTING SUMMARY")
    logger.info(f"{'='*60}")
    for sn, res in all_results.items():
        logger.info(f"  {sn:20s}: best={res['best_candidate']:5s} "
                    f"acc={res['best_acc']:.2f}% time={res['elapsed_s']:.0f}s")

    dump_json(str(out_dir / "all_results.json"), all_results)
    logger.info(f"All results in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())