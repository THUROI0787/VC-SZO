"""Robust, resumable Phase-4 SNN experiment runner."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", type=int, default=-1)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_args.gpu)

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase3.candidates_phase3 import PHASE3_CANDIDATES
from phase4.engines import AdapterBPEngine, zo_compute_stats
from phase4.protocol import CommonRandomNumbers, evaluate_fixed_samples
from tta_snn_zo.bn_stats import BNStatsAdaptEngine
from tta_snn_zo.corruptions import CORRUPTIONS
from tta_snn_zo.data import get_cifar10c_loader, get_clean_test_accuracy_loader
from tta_snn_zo.launch import build_pretrained_model, resolve_cifar10c
from tta_snn_zo.memo import MEMOEngine
from tta_snn_zo.tta import SourceOnlyEngine, TentEngine, ZOTTAEngine
from tta_snn_zo.utils import set_seed, setup_logger

BLUR = ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur")
CHECKPOINTS = {
    4: "checkpoints/snn/vgg9_t4_best.pt",
    6: "checkpoints/snn/vgg9_t6_best.pt",
    12: "checkpoints/snn/vgg9_t12_best.pt",
    25: "checkpoints/snn/vgg9_t25_best.pt",
}
CANDIDATES = {candidate["id"]: candidate for candidate in PHASE3_CANDIDATES}
ALL_METHODS = (
    "source", "zo_noop", "zo_entropy", "zo_margin", "m307", "m301", "tent", "memo", "bn_stats",
    "bp_margin", "bp_entropy",
)


class ZOStatsWrapper:
    def __init__(self, engine: ZOTTAEngine):
        self.engine = engine

    def predict_and_adapt(self, x):
        logits, stats = self.engine.predict_and_adapt(x)
        stats.update(zo_compute_stats(self.engine, bool(stats.get("adapted", 0))))
        return logits, stats


def csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(parents=[_gpu_parser])
    parser.add_argument("--out-dir", default="outputs/phase4")
    parser.add_argument("--tag", default="p0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--T", default="12", help="comma-separated")
    parser.add_argument("--levels", default="1,3,5")
    parser.add_argument("--batches", default="2")
    parser.add_argument("--seeds", default="42,215,3407")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--corruptions", default="blur", help="blur, all, or comma-separated")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--checkpoints", default="", help="comma-separated cumulative sample checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_model(time_steps: int, device, logger):
    checkpoint = CHECKPOINTS[time_steps]
    model, _ = build_pretrained_model(
        "vgg9", time_steps, 10, 20.0, 1.0, checkpoint, "auto", logger=logger
    )
    return model.to(device)


def memo_n_aug(time_steps: int, batch: int) -> int:
    if time_steps == 25:
        if batch >= 128:
            return 1
        if batch >= 32:
            return 4
    if time_steps == 12 and batch >= 128:
        return 2
    if time_steps in (4, 6) and batch >= 128:
        return 4
    return 8


def make_zo(model, method: str, device, seed: int):
    if method == "zo_noop":
        config = dict(CANDIDATES["m307"]["engine"])
        config["lr"] = 0.0
    elif method in ("zo_entropy", "zo_margin"):
        config = dict(CANDIDATES["m301"]["engine"])
        config["num_samples"] = 1
        config["block_sparsity"] = 1.0
        config["objective"] = "entropy" if method == "zo_entropy" else "margin"
    else:
        config = dict(CANDIDATES[method]["engine"])
    overrides = (("P4_ZO_LR", "lr", float), ("P4_ZO_EPS", "eps", float), ("P4_ZO_NUM_SAMPLES", "num_samples", int), ("P4_ZO_SPARSITY", "block_sparsity", float))
    for env_name, key, cast in overrides:
        if os.environ.get(env_name):
            config[key] = cast(os.environ[env_name])
    if os.environ.get("P4_ZO_RESET") == "1":
        config["reset_per_batch"] = True
    # A no-op control must remain no-op even during global hyperparameter sweeps.
    if method == "zo_noop":
        config["lr"] = 0.0
    engine = ZOTTAEngine(model, device=device, seed=seed, **config)
    return CommonRandomNumbers(ZOStatsWrapper(engine), seed, zo_internal_seeding=True)


def make_engine(model, method: str, time_steps: int, batch: int, device, seed: int):
    if method == "source":
        return CommonRandomNumbers(SourceOnlyEngine(model, device=device), seed)
    if method in ("zo_noop", "zo_entropy", "zo_margin", "m307", "m301"):
        return make_zo(model, method, device, seed)
    if method == "tent":
        engine = TentEngine(
            model, device=device, lr=1e-3, bn="all", steps=1, gate=True,
            ent_min_batch=0.01, ent_max_batch=2.40,
            sample_ent_thresh=1.2, max_grad_norm=1.0,
        )
        return CommonRandomNumbers(engine, seed)
    if method == "memo":
        return CommonRandomNumbers(
            MEMOEngine(model, n_aug=memo_n_aug(time_steps, batch), lr=1e-3, device=device), seed
        )
    if method == "bn_stats":
        return CommonRandomNumbers(BNStatsAdaptEngine(model, device=device), seed)
    if method in ("bp_margin", "bp_entropy"):
        return AdapterBPEngine(
            model, target_layers=("pool3",), objective=method.removeprefix("bp_"),
            lr=1e-3, weight_decay=1e-3, clamp_abs=0.2, seed=seed, device=device,
        )
    raise ValueError(method)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def write_progress(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    args = parse_args()
    checkpoint_samples = set(csv_ints(args.checkpoints))
    times = csv_ints(args.T)
    levels = csv_ints(args.levels)
    batches = csv_ints(args.batches)
    seeds = csv_ints(args.seeds)
    methods = csv_strings(args.methods)
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    corruptions = list(BLUR if args.corruptions == "blur" else CORRUPTIONS)
    if args.corruptions not in ("blur", "all"):
        corruptions = csv_strings(args.corruptions)
    if not set(corruptions).issubset(set(CORRUPTIONS) | {"clean"}):
        raise ValueError("unknown corruption requested")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.out_dir) / args.tag
    root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("phase4", str(root / "logs"), f"gpu{args.gpu}")
    logger.info(f"args={vars(args)} device={device}")
    rows = []

    cifar_root, mode = resolve_cifar10c(
        SimpleNamespace(data_root=args.data_root, data_mode="auto", level=1), logger=logger
    )
    logger.info(f"data_mode={mode}; jobs={len(times)*len(levels)*len(batches)*len(seeds)*len(methods)*len(corruptions)}")

    for time_steps in times:
        for level in levels:
            for batch in batches:
                for seed in seeds:
                    for method in methods:
                        for corruption in corruptions:
                            rel = Path(f"T{time_steps}") / f"level{level}_batch{batch}_seed{seed}" / method
                            result_path = root / rel / f"{corruption}.json"
                            if result_path.exists() and not args.force:
                                try:
                                    with result_path.open() as handle:
                                        rows.append(json.load(handle))
                                    continue
                                except Exception:
                                    pass
                            result_path.parent.mkdir(parents=True, exist_ok=True)
                            started = time.time()
                            base = {
                                "T": time_steps, "level": level, "batch": batch,
                                "seed": seed, "method": method, "corruption": corruption,
                                "max_samples": args.max_samples,
                            }
                            logger.info(f"START {base}")
                            model = engine = None
                            try:
                                set_seed(seed)
                                model = build_model(time_steps, device, logger=None)
                                engine = make_engine(model, method, time_steps, batch, device, seed)
                                if corruption == "clean":
                                    loader = get_clean_test_accuracy_loader(
                                        args.data_root, batch, args.num_workers
                                    )
                                else:
                                    loader = get_cifar10c_loader(
                                        cifar_root, corruption, batch, args.num_workers, level
                                    )
                                result = evaluate_fixed_samples(
                                    engine, loader, device, args.max_samples, profile_memory=True, checkpoints=checkpoint_samples
                                )
                                row = {**base, "status": "ok", "time_s": time.time() - started, **result}
                                atomic_json(result_path, row)
                                rows.append(row)
                                logger.info(
                                    f"DONE {method}/{corruption} acc={row['acc']:.3f} "
                                    f"n={row['num_samples']} time={row['time_s']:.1f}s "
                                    f"mem={row.get('peak_allocated_mb', 0):.0f}MB"
                                )
                            except torch.cuda.OutOfMemoryError as exc:
                                row = {**base, "status": "oom", "time_s": time.time() - started, "error": str(exc)}
                                atomic_json(result_path, row)
                                rows.append(row)
                                logger.error(f"OOM {base}: {exc}")
                            except Exception as exc:
                                row = {**base, "status": "error", "time_s": time.time() - started, "error": str(exc)}
                                atomic_json(result_path, row)
                                rows.append(row)
                                logger.error(f"ERROR {base}: {exc}\n{traceback.format_exc()}")
                            finally:
                                del engine, model
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            write_progress(root / f"progress_gpu{args.gpu}.csv", rows)
    logger.info("COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
