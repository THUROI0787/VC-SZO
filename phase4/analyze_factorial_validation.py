#!/usr/bin/env python3
"""Summarize the frozen factorial setting on fresh windows and seeds."""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "phase4" / "analysis"
METHOD_ORDER = ("source", "zo_noop", "m307", "m301", "bp_margin", "tent", "memo", "bn_stats")
LABELS = {
    "source": "Source", "zo_noop": "ZO-noop", "m307": "VC-SZO-SD",
    "m301": "VC-SZO-MD", "bp_margin": "BP-margin", "tent": "TENT",
    "memo": "MEMO", "bn_stats": "BN",
}
OFFSETS = (2160, 2280, 2400, 2520)


def bootstrap_ci(values: np.ndarray, seed: int = 20260903) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def exact_signflip_p(values: np.ndarray) -> float:
    values = values[np.abs(values) > 1e-12]
    if len(values) == 0:
        return 1.0
    observed = abs(float(values.mean()))
    if len(values) <= 20:
        means = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            means.append(abs(float(np.mean(values * np.asarray(signs)))))
        return float((np.count_nonzero(np.asarray(means) >= observed - 1e-12)) / len(means))
    rng = np.random.default_rng(20260903)
    signs = rng.choice((-1.0, 1.0), size=(100000, len(values)))
    return float(np.mean(np.abs((signs * values).mean(axis=1)) >= observed - 1e-12))


def main() -> None:
    selection = json.loads((ANALYSIS / "factorial_selection.json").read_text())["selected"]
    T, batch, level, lr = (selection[k] for k in ("T", "batch", "level", "lr"))
    lr_key = "lr3e4" if abs(float(lr) - 3e-4) < 1e-12 else "lr1e3"
    setting_key = f"T{int(T)}_b{int(batch)}_L{int(level)}_{lr_key}"

    rows = []
    for offset in OFFSETS:
        tag = ROOT / "outputs" / "phase4" / f"factor_val_{setting_key}_w{offset}"
        files = sorted(tag.glob("T*/level*_batch*_seed*/*/*.json"))
        if len(files) != 96:
            raise SystemExit(f"{tag.name}: expected 96 JSON files, found {len(files)}")
        for path in files:
            row = json.loads(path.read_text())
            if row.get("status") != "ok":
                raise SystemExit(f"non-ok result: {path}")
            row["offset"] = offset
            rows.append(row)

    accuracy = {}
    for r in rows:
        key = (int(r["offset"]), str(r["corruption"]), int(r["seed"]), str(r["method"]))
        accuracy[key] = float(r["acc"])

    # Average adaptation seeds first.  The resulting 16 window-corruption pairs
    # are the units used for intervals and the random-sign test.
    unit_values: dict[str, list[float]] = defaultdict(list)
    unit_acc: dict[str, list[float]] = defaultdict(list)
    for method in METHOD_ORDER:
        for offset in OFFSETS:
            corruptions = sorted({r["corruption"] for r in rows if r["offset"] == offset})
            for corruption in corruptions:
                method_acc = np.mean([
                    accuracy[(offset, corruption, seed, method)] for seed in (23, 61, 109)
                ])
                source_acc = np.mean([
                    accuracy[(offset, corruption, seed, "source")] for seed in (23, 61, 109)
                ])
                unit_acc[method].append(float(method_acc))
                unit_values[method].append(float(method_acc - source_acc))

    output = []
    for method in METHOD_ORDER:
        deltas = np.asarray(unit_values[method])
        lo, hi = bootstrap_ci(deltas)
        output.append({
            "method": method,
            "label": LABELS[method],
            "mean_accuracy": float(np.mean(unit_acc[method])),
            "mean_delta_vs_source_pp": float(deltas.mean()),
            "ci95_low_pp": lo,
            "ci95_high_pp": hi,
            "positive_units": int(np.count_nonzero(deltas > 0)),
            "nonnegative_units": int(np.count_nonzero(deltas >= 0)),
            "num_window_corruption_units": int(len(deltas)),
            "exact_signflip_p_two_sided": exact_signflip_p(deltas),
        })

    # A no-op mismatch means the validation is invalid, not merely noisy.
    noop_max_error = max(abs(a - b) for a, b in zip(unit_acc["source"], unit_acc["zo_noop"]))
    if noop_max_error > 1e-10:
        raise SystemExit(f"Source/ZO-noop mismatch: max unit error {noop_max_error}")

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS / "factorial_validation_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    json_path = ANALYSIS / "factorial_validation_summary.json"
    json_path.write_text(json.dumps({
        "setting": {"T": T, "batch": batch, "level": level, "lr": lr},
        "protocol": {
            "development_offset": 1560,
            "validation_offsets": OFFSETS,
            "development_seeds": (17, 53, 97),
            "validation_seeds": (23, 61, 109),
            "inference_unit": "adaptation-seed mean within window-corruption",
            "num_units": 16,
        },
        "source_noop_max_error": noop_max_error,
        "results": output,
    }, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
