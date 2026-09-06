"""Phase-1 candidate sweep runner (robust, automated).

Runs every candidate from ``candidates.py`` on a screening benchmark
(5 representative corruptions x N batches x seeds), computes source-only (B0)
and fixed-TENT (B1) references, and writes a ranked summary CSV + per-candidate
logs/CSVs to ``--out_dir``.

Design goals (for unattended runs on the user's server):
  * one candidate/seed failure never aborts the sweep (try/except per block);
  * a *fresh* model copy is loaded per candidate so BN-affine candidates cannot
    contaminate each other (adapter candidates mutate hooks only, but BN modes
    mutate weights in place);
  * memory-safe: num_workers=0, torch.cuda.empty_cache() + gc.collect() between
    candidates (2GB-cgroup friendly);
  * ``--quick`` = 3 corruptions x 30 batches x seed 0 (a few minutes);
  * ``--gpu N`` sets CUDA_VISIBLE_DEVICES before torch import.

Usage::

    python phase1/run_sweep.py --pretrained <ckpt> --data_root <data> [--quick]
                               [--ids c01,c09] [--groups obj_filtered] [--gpu 0]

Outputs (in --out_dir):
    summary.csv            one row per candidate, ranked by mean acc
    summary.json           machine-readable copy
    logs/<id>.log          per-candidate console+file log
    <id>_s<seed>.csv       per-candidate per-corruption detail
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

# --------------------------------------------------------------------------- #
#  GPU selection must happen before torch import
# --------------------------------------------------------------------------- #
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=int, default=-1)
_args0, _ = _ap.parse_known_args()
if _args0.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args0.gpu)

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for candidates.py

from tta_snn_zo.data import get_cifar10c_loader  # noqa: E402
from tta_snn_zo.launch import build_pretrained_model, resolve_cifar10c  # noqa: E402
from tta_snn_zo.tta import (  # noqa: E402
    SourceOnlyEngine, TentEngine, ZOTTAEngine, evaluate_on_loader,
)
from tta_snn_zo.utils import (  # noqa: E402
    dump_json, set_seed, setup_logger, write_csv,
)

from candidates import (  # noqa: E402
    ALL_CANDIDATES, FAMILY_SUMMARY, baseline_source_only, baseline_tent_fixed,
    baseline_tent_adapter,
    build_engine_from_cfg, select_candidates, validate_candidates,
)

SCREENING_CORRUPTIONS = ["gaussian_noise", "motion_blur", "snow",
                         "contrast", "jpeg_compression"]


def _load_ckpt_meta(path: str):
    """Read the checkpoint dict (for the 'test_acc' field) without moving to GPU."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(parents=[_ap])
    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/sweep")
    p.add_argument("--model", type=str, default="vgg9", choices=["vgg9", "resnet19"])
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--v_threshold", type=float, default=1.0)
    p.add_argument("--corruptions", type=str, default="")
    p.add_argument("--max_batches", type=int, default=60, help="0 = all")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--ids", type=str, default="", help="comma list of candidate ids")
    p.add_argument("--groups", type=str, default="", help="comma list of axis groups (deprecated, use --families)")
    p.add_argument("--families", type=str, default="", help="comma list of method families (A_vanilla, B_subspace, etc.)")
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="3 corruptions x 30 batches, seed 0 only")
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--skip_snn", action="store_true",
                   help="skip SNN-specific candidates (families D, E)")
    p.add_argument("--collect_ref_stats", action="store_true",
                   help="collect membrane reference stats from source data")
    p.add_argument("--dry_run", action="store_true",
                   help="validate candidates and print the plan, run nothing")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.corruptions = ",".join(SCREENING_CORRUPTIONS[:3])
        args.max_batches = 30
        args.seeds = "0"
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()] \
        or SCREENING_CORRUPTIONS
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [0]
    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Use --families if provided, fall back to --groups for backward compat
    families = args.families or args.groups
    candidates = select_candidates(args.ids, families)
    if args.skip_snn:
        candidates = [c for c in candidates if c["family"] not in ("D_snn_objective", "E_adaptive_gate")]
    errors = validate_candidates(candidates)
    if errors:
        for e in errors:
            print(f"[VALIDATION ERROR] {e}")
        if not args.dry_run:
            sys.exit(1)

    if args.dry_run:
        print(f"pretrained={args.pretrained}  corruptions={corruptions}  "
              f"max_batches={args.max_batches}  seeds={seeds}")
        print(f"{len(candidates)} candidates across {len(set(c['family'] for c in candidates))} families:")
        # Group by family
        from collections import defaultdict
        by_fam = defaultdict(list)
        for c in candidates:
            by_fam[c["family"]].append(c)
        for fam, clist in sorted(by_fam.items()):
            print(f"\n  [{fam}] ({FAMILY_SUMMARY.get(fam, '')})")
            for c in clist:
                print(f"    {c['id']:5s} {c['method_name']:40s} | {c['question']}")
        return 0

    logger = setup_logger("sweep", str(out_dir / "logs"), "overview")
    logger.info(f"args: {vars(args)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} candidates={len(candidates)} "
                f"corruptions={corruptions} seeds={seeds}")

    cifar10c_root, mode = resolve_cifar10c(
        SimpleNamespace(data_root=args.data_root, data_mode="auto",
                        level=args.level), logger=logger)
    logger.info(f"CIFAR-10-C root: {cifar10c_root} (mode={mode})")

    # log the checkpoint's reported clean acc ONCE (diagnostic: pin down which
    # backbone we are actually evaluating -- the round-1 "84%" vs 69%-source
    # mismatch was due to an unknown checkpoint/data inconsistency).
    reported_clean_acc = None
    _clean_logged = {"done": False}

    # Collect membrane reference stats if needed (for membrane_kl candidates)
    ref_stats = None
    if args.collect_ref_stats or any(
        c["engine"].get("objective") == "membrane_kl" or
        c["engine"].get("use_membrane_gate", False)
        for c in candidates
    ):
        logger.info("Collecting membrane reference statistics from source data...")
        from tta_snn_zo.membrane import collect_membrane_stats
        from tta_snn_zo.data import get_cifar10_loaders
        m_ref, d_ref = build_pretrained_model(
            args.model, 25, args.num_classes, args.tau, args.v_threshold,
            args.pretrained, "auto", logger=logger)
        _, clean_loader = get_cifar10_loaders(
            args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
        ref_stats = collect_membrane_stats(m_ref, clean_loader, d_ref, num_batches=10)
        logger.info(f"Collected membrane stats for {len(ref_stats)} LIF layers")
        del m_ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fresh_model():
        nonlocal reported_clean_acc
        if not _clean_logged["done"]:
            m, d = build_pretrained_model(
                args.model, 25, args.num_classes, args.tau, args.v_threshold,
                args.pretrained, "auto", logger=logger)
            ck = _load_ckpt_meta(args.pretrained)
            if ck is not None:
                reported_clean_acc = ck.get("test_acc")
                logger.info(f"[diagnostic] checkpoint test_acc field: {reported_clean_acc}")
            _clean_logged["done"] = True
            return m, d
        return build_pretrained_model(
            args.model, 25, args.num_classes, args.tau, args.v_threshold,
            args.pretrained, "auto", logger=None)

    # ------------------------------------------------------------------ #
    #  B0: source-only reference (needed for collapse detection)
    # ------------------------------------------------------------------ #
    source_acc: dict = {}   # {corr: mean acc over seeds}
    source_rows = []
    if not args.skip_baselines:
        per_seed = {corr: [] for corr in corruptions}
        for s in seeds:
            set_seed(s)
            model, dev = fresh_model()
            engine = SourceOnlyEngine(model.to(dev), device=dev)
            for corr in corruptions:
                loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                             args.num_workers, args.level)
                r = evaluate_on_loader(engine, loader, dev, max_batches=args.max_batches or None)
                per_seed[corr].append(r["acc"])
                source_rows.append({"method": "B0_source", "candidate": "B0",
                                    "seed": s, "corruption": corr,
                                    "acc": round(r["acc"], 3), "loss": round(r["loss"], 4)})
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for corr in corruptions:
            source_acc[corr] = sum(per_seed[corr]) / len(per_seed[corr])
        write_csv(str(out_dir / "B0_source.csv"), source_rows)
        logger.info(f"B0 source-only mean={sum(source_acc.values())/len(source_acc):.2f}% "
                    f"per-corr={ {k: round(v,2) for k,v in source_acc.items()} }")

    # ------------------------------------------------------------------ #
    #  B1: fixed TENT-BP baseline (fresh model+engine per corruption)
    # ------------------------------------------------------------------ #
    tent_rows = []
    tent_mean = None
    if not args.skip_baselines:
        _, tent_cfg = baseline_tent_fixed()
        per_seed = {corr: [] for corr in corruptions}
        for s in seeds:
            for corr in corruptions:
                set_seed(s)
                model, dev = fresh_model()
                engine = TentEngine(model.to(dev), device=dev, **tent_cfg)
                loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                             args.num_workers, args.level)
                r = evaluate_on_loader(engine, loader, dev,
                                       max_batches=args.max_batches or None)
                per_seed[corr].append(r["acc"])
                tent_rows.append({"method": "B1_tent_fixed", "candidate": "B1",
                                  "seed": s, "corruption": corr,
                                  "acc": round(r["acc"], 3), "loss": round(r["loss"], 4)})
                del model, engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        tent_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        write_csv(str(out_dir / "B1_tent_fixed.csv"), tent_rows)
        logger.info(f"B1 tent-fixed mean={tent_mean:.2f}%")

    # ------------------------------------------------------------------ #
    #  B2: TENT-BP on adapter (fairer comparison)
    # ------------------------------------------------------------------ #
    tent_adapter_rows = []
    tent_adapter_mean = None
    if not args.skip_baselines:
        _, tent_adapter_cfg = baseline_tent_adapter()
        per_seed = {corr: [] for corr in corruptions}
        for s in seeds:
            for corr in corruptions:
                set_seed(s)
                model, dev = fresh_model()
                engine = TentEngine(model.to(dev), device=dev, **tent_adapter_cfg)
                loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                             args.num_workers, args.level)
                r = evaluate_on_loader(engine, loader, dev,
                                       max_batches=args.max_batches or None)
                per_seed[corr].append(r["acc"])
                tent_adapter_rows.append({"method": "B2_tent_adapter", "candidate": "B2",
                                          "seed": s, "corruption": corr,
                                          "acc": round(r["acc"], 3), "loss": round(r["loss"], 4)})
                del model, engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        tent_adapter_mean = sum(sum(v) for v in per_seed.values()) / (len(corruptions) * len(seeds))
        write_csv(str(out_dir / "B2_tent_adapter.csv"), tent_adapter_rows)
        logger.info(f"B2 tent-adapter mean={tent_adapter_mean:.2f}%")

    # ------------------------------------------------------------------ #
    #  Candidates
    # ------------------------------------------------------------------ #
    summary_rows = []
    for cand in candidates:
        cid = cand["id"]
        t0 = time.time()
        clog = setup_logger("sweep", str(out_dir / "logs"), cid)
        clog.info(f"candidate {cid} [{cand.get('family', cand.get('group', '?'))}] {cand['question']}")
        clog.info(f"engine={json.dumps(cand['engine'])}")
        try:
            per_seed_corr = {corr: [] for corr in corruptions}
            losses, gates, adapted_f, rolled_f, err = [], [], [], [], ""
            for s in seeds:
                rows = []
                for corr in corruptions:
                    set_seed(s)
                    model, dev = fresh_model()
                    model = model.to(dev)
                    # Inject ref_stats for membrane_kl candidates
                    if ref_stats is not None and cand["engine"].get("objective") == "membrane_kl":
                        cand["engine"]["ref_stats"] = ref_stats
                    engine = build_engine_from_cfg(cand, model, dev, seed=s)
                    loader = get_cifar10c_loader(cifar10c_root, corr, args.batch_size,
                                                 args.num_workers, args.level)
                    r = evaluate_on_loader(engine, loader, dev,
                                           max_batches=args.max_batches or None)
                    per_seed_corr[corr].append(r["acc"])
                    losses.append(r["loss"])
                    gates.append(r["gate"])
                    adapted_f.append(r["adapted_frac"])
                    rolled_f.append(r["rolled_back_frac"])
                    rows.append({"method": "zo_tta", "candidate": cid, "seed": s,
                                 "corruption": corr, "acc": round(r["acc"], 3),
                                 "loss": round(r["loss"], 4), "gate": round(r["gate"], 4),
                                 "adapted_frac": round(r["adapted_frac"], 3),
                                 "rolled_back_frac": round(r["rolled_back_frac"], 3),
                                 "mean_abs_proj_grad": round(r["mean_abs_proj_grad"], 6)})
                    clog.info(f"  s{s} [{corr}] acc={r['acc']:.2f}% loss={r['loss']:.4f} "
                              f"gate={r['gate']:.3f} adapted={r['adapted_frac']:.2f} "
                              f"rb={r['rolled_back_frac']:.2f}")
                    del model, engine
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                write_csv(str(out_dir / f"{cid}_s{s}.csv"), rows)

            mean_acc = sum(sum(v) for v in per_seed_corr.values()) / \
                (len(corruptions) * len(seeds))
            per_corr_mean = {corr: (sum(v) / len(v)) for corr, v in per_seed_corr.items()}
            mean_loss = sum(losses) / len(losses) if losses else float("nan")
            mean_gate = sum(gates) / len(gates) if gates else float("nan")
            mean_adapted = sum(adapted_f) / len(adapted_f) if adapted_f else float("nan")
            mean_rolled = sum(rolled_f) / len(rolled_f) if rolled_f else float("nan")
            # collapse detection vs source-only
            collapsed = 0
            for corr in corruptions:
                if source_acc and per_corr_mean[corr] < source_acc[corr] - 2.0:
                    collapsed += 1
            ent_collapse = bool(mean_loss < 0.15)
            summary_rows.append({
                "id": cid, "family": cand.get("family", ""),
                "method": cand.get("method_name", ""),
                "question": cand["question"],
                "mean_acc": round(mean_acc, 3),
                "mean_loss": round(mean_loss, 4), "mean_gate": round(mean_gate, 4),
                "adapted_frac": round(mean_adapted, 3),
                "rolled_back_frac": round(mean_rolled, 3),
                "collapsed_frac": round(collapsed / len(corruptions), 2),
                "ent_collapse": int(ent_collapse),
                "time_s": round(time.time() - t0, 1), "error": "",
                **{f"acc_{corr}": round(per_corr_mean[corr], 3) for corr in corruptions},
            })
            clog.info(f"=== {cid}: mean_acc={mean_acc:.2f}% mean_loss={mean_loss:.4f} "
                      f"adapted={mean_adapted:.2f} rolled_back={mean_rolled:.2f} "
                      f"collapsed_frac={collapsed}/{len(corruptions)} "
                      f"({time.time()-t0:.0f}s)")
        except Exception as e:  # noqa: BLE001 - robustness is the point
            import traceback
            clog.error(f"FAILED: {e}\n{traceback.format_exc()}")
            summary_rows.append({"id": cid, "family": cand.get("family", ""),
                                 "method": cand.get("method_name", ""),
                                 "question": cand["question"], "mean_acc": -1.0,
                                 "mean_loss": float("nan"), "mean_gate": float("nan"),
                                 "adapted_frac": float("nan"),
                                 "rolled_back_frac": float("nan"),
                                 "collapsed_frac": 1.0, "ent_collapse": 1,
                                 "time_s": round(time.time() - t0, 1), "error": str(e)[:200]})

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #
    summary_rows.sort(key=lambda r: r["mean_acc"], reverse=True)
    write_csv(str(out_dir / "summary.csv"), summary_rows)
    dump_json(str(out_dir / "summary.json"), {
        "args": vars(args),
        "reported_clean_acc": reported_clean_acc,
        "source_only_mean":
            round(sum(source_acc.values()) / len(source_acc), 3) if source_acc else None,
        "tent_fixed_mean": round(tent_mean, 3) if tent_mean is not None else None,
        "tent_adapter_mean": round(tent_adapter_mean, 3) if tent_adapter_mean is not None else None,
        "has_ref_stats": ref_stats is not None,
        "rows": summary_rows,
    })
    logger.info("=" * 60)
    logger.info("SUMMARY (ranked by mean acc):")
    for r in summary_rows:
        logger.info(f"  {r['id']:5s} acc={r['mean_acc']:6.2f}% loss={r['mean_loss']:.4f} "
                    f"collapsed={r['collapsed_frac']:.2f} "
                    f"{r.get('error','')[:60]}")
    logger.info(f"results in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
