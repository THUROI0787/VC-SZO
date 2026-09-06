#!/usr/bin/env python3
"""Pre-registered selection rule for the Phase-4 factorial development scan.

This script deliberately separates *development* selection from the later,
fresh-offset/fresh-seed confirmatory run.  It must not inspect confirmatory data.

Eligibility rule (fixed before the scan completed): for each (T, batch,
severity, lr) setting, both the worst corruption-level mean and worst seed-level
mean SD-minus-Source uplift must be at least -0.75 percentage points.  Among
eligible settings select the largest overall mean uplift; exact ties prefer the
lower T, then lower batch, severity, and learning rate (lower runtime/budget).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAGS = (
    "factor_dev_T12_lr3e4",
    "factor_dev_T12_lr1e3",
    "factor_dev_T25_lr3e4",
    "factor_dev_T25_lr1e3",
)
MIN_GROUP_UPLIFT = -0.75


def load_records(tags: tuple[str, ...]) -> list[dict]:
    records: list[dict] = []
    for tag in tags:
        tag_dir = ROOT / "outputs" / "phase4" / tag
        if not tag_dir.exists():
            raise SystemExit(f"missing result directory: {tag_dir}")
        lr = 3e-4 if tag.endswith("lr3e4") else 1e-3
        files = sorted(tag_dir.glob("T*/level*_batch*_seed*/*/*.json"))
        if len(files) != 96:
            raise SystemExit(f"{tag}: expected 96 JSON files, found {len(files)}")
        for path in files:
            row = json.loads(path.read_text())
            if row.get("status") != "ok":
                raise SystemExit(f"non-ok cell: {path}: {row.get('status')}")
            row["lr"] = lr
            row["path"] = str(path.relative_to(ROOT))
            records.append(row)
    return records


def summarize(records: list[dict]) -> list[dict]:
    paired: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in records:
        key = (
            int(r["T"]), int(r["batch"]), int(r["level"]), float(r["lr"]),
            int(r["seed"]), str(r["corruption"]), int(r["max_samples"]),
        )
        paired[key][str(r["method"])] = float(r["acc"])

    cells: list[dict] = []
    for key, methods in paired.items():
        if set(methods) != {"source", "m307"}:
            raise SystemExit(f"incomplete pair {key}: {sorted(methods)}")
        T, batch, level, lr, seed, corruption, _ = key
        cells.append({
            "T": T, "batch": batch, "level": level, "lr": lr,
            "seed": seed, "corruption": corruption,
            "delta": methods["m307"] - methods["source"],
        })

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        grouped[(c["T"], c["batch"], c["level"], c["lr"])].append(c)

    output: list[dict] = []
    for (T, batch, level, lr), group in sorted(grouped.items()):
        if len(group) != 12:
            raise SystemExit(f"setting {(T,batch,level,lr)}: expected 12 pairs, found {len(group)}")
        corr_means = {
            corr: sum(x["delta"] for x in group if x["corruption"] == corr) /
                  sum(x["corruption"] == corr for x in group)
            for corr in sorted({x["corruption"] for x in group})
        }
        seed_means = {
            str(seed): sum(x["delta"] for x in group if x["seed"] == seed) /
                       sum(x["seed"] == seed for x in group)
            for seed in sorted({x["seed"] for x in group})
        }
        mean = sum(x["delta"] for x in group) / len(group)
        min_corr = min(corr_means.values())
        min_seed = min(seed_means.values())
        eligible = min_corr >= MIN_GROUP_UPLIFT and min_seed >= MIN_GROUP_UPLIFT
        output.append({
            "T": T, "batch": batch, "level": level, "lr": lr,
            "mean_delta_pp": mean,
            "min_corruption_mean_delta_pp": min_corr,
            "min_seed_mean_delta_pp": min_seed,
            "positive_cells": sum(x["delta"] > 0 for x in group),
            "nonnegative_cells": sum(x["delta"] >= 0 for x in group),
            "num_cells": len(group),
            "eligible": eligible,
            "corruption_means": corr_means,
            "seed_means": seed_means,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", default=list(DEFAULT_TAGS))
    args = parser.parse_args()
    rows = summarize(load_records(tuple(args.tags)))
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise SystemExit("no setting passed the pre-registered stability threshold")
    selected = sorted(
        eligible,
        key=lambda r: (-r["mean_delta_pp"], r["T"], r["batch"], r["level"], r["lr"]),
    )[0]

    out_dir = ROOT / "outputs" / "phase4" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "factorial_dev_summary.csv"
    fields = [
        "T", "batch", "level", "lr", "mean_delta_pp",
        "min_corruption_mean_delta_pp", "min_seed_mean_delta_pp",
        "positive_cells", "nonnegative_cells", "num_cells", "eligible",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    selection_path = out_dir / "factorial_selection.json"
    selection_path.write_text(json.dumps({
        "protocol": {
            "role": "development_only",
            "selection_rule": "max mean uplift among stable settings",
            "minimum_corruption_and_seed_mean_uplift_pp": MIN_GROUP_UPLIFT,
            "tie_break": "lower T, batch, severity, then lr",
            "required_fresh_validation": True,
        },
        "selected": selected,
        "all_settings": rows,
    }, indent=2) + "\n")
    print(json.dumps(selected, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {selection_path}")


if __name__ == "__main__":
    main()
