"""Phase 1 - aggregate the per-corruption CSVs of source-only / TENT-BP / ZO-TTA
into one comparison table (stdout + CSV)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tta_snn_zo.corruptions import CORRUPTIONS  # noqa: E402
from tta_snn_zo.utils import load_csv, write_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="outputs/phase1")
    p.add_argument("--csvs", type=str, default="", help="comma-separated explicit csvs")
    p.add_argument("--out", type=str, default="outputs/phase1/comparison.csv")
    args = p.parse_args()

    if args.csvs:
        files = [f.strip() for f in args.csvs.split(",") if f.strip()]
    else:
        files = sorted(Path(args.results_dir).glob("*_results.csv"))
    if not files:
        print("no result csvs found; run the phase-1 scripts first")
        return 1

    methods = {}
    for f in files:
        for row in load_csv(str(f)):
            if row["corruption"] in CORRUPTIONS or row["corruption"] in ("mean", "clean"):
                methods.setdefault(row["method"], {})[row["corruption"]] = float(row["acc"])

    print(f"{'method':<20}" + "".join(f"{c[:6]:>8}" for c in CORRUPTIONS) + f"{'mean':>8}")
    out_rows = []
    for m, d in methods.items():
        line = f"{m:<20}"
        for c in CORRUPTIONS:
            line += f"{d.get(c, float('nan')):>8.1f}"
        line += f"{d.get('mean', float('nan')):>8.1f}"
        print(line)
        row = {"method": m}
        row.update({c: d.get(c) for c in CORRUPTIONS})
        row["mean"] = d.get("mean")
        out_rows.append(row)
    write_csv(args.out, out_rows)
    print(f"\ncomparison saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
