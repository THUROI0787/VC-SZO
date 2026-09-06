"""Reproducible analysis for the Phase-3 SNN and ANN sweeps.

The script treats each (T, severity, batch size, seed) result as a paired block.
Candidate gains are therefore computed against baselines from the *same* block
before any averaging.  This is important because Poisson encoding and ZO make
raw accuracies seed-dependent.

Usage:
    python phase3/analyze_phase3.py \
        --snn-root outputs/phase3 --ann-root outputs/phase3_ann \
        --out-dir outputs/phase3_analysis
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINES = ("B0_source", "B1_tent_bp", "B2_memo", "B3_bn_stats")
ADAPT_BASELINES = ("B1_tent_bp", "B2_memo", "B3_bn_stats")
SETTING_RE = re.compile(r"level(?P<level>\d+)_batch(?P<batch>\d+)_seed(?P<seed>\d+)")


def _setting_from_json(path: Path, payload: dict) -> dict:
    setting = payload.get("setting", {})
    match = SETTING_RE.fullmatch(path.parent.name)
    if match:
        parsed = {key: int(value) for key, value in match.groupdict().items()}
    else:
        parsed = {}
    return {
        "level": int(setting.get("level", parsed.get("level", -1))),
        "batch": int(setting.get("batch_size", parsed.get("batch", -1))),
        "seed": int(setting.get("seed", parsed.get("seed", -1))),
    }


def load_results(root: Path, model_kind: str) -> pd.DataFrame:
    records: list[dict] = []
    for path in sorted(root.rglob("summary.json")):
        with path.open() as handle:
            payload = json.load(handle)
        setting = _setting_from_json(path, payload)
        if model_kind == "SNN":
            t_match = re.fullmatch(r"T(\d+)", path.parent.parent.name)
            if not t_match:
                continue
            time_steps = int(t_match.group(1))
        else:
            time_steps = 0
        for row in payload.get("rows", []):
            record = {
                "model_kind": model_kind,
                "T": time_steps,
                **setting,
                "path": str(path),
                **row,
            }
            records.append(record)
    if not records:
        raise RuntimeError(f"No summary.json files found under {root}")
    return pd.DataFrame.from_records(records)


def add_paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_kind", "T", "level", "batch", "seed"]
    wide = frame.pivot_table(index=keys, columns="id", values="mean_acc", aggfunc="first")
    missing = [method for method in BASELINES if method not in wide.columns]
    if missing:
        raise RuntimeError(f"Missing baselines: {missing}")
    refs = pd.DataFrame(index=wide.index)
    refs["source_acc"] = wide["B0_source"]
    refs["best_adapt_baseline_acc"] = wide[list(ADAPT_BASELINES)].max(axis=1)
    refs["best_conventional_acc"] = wide[list(BASELINES)].max(axis=1)
    refs["best_adapt_baseline_id"] = wide[list(ADAPT_BASELINES)].idxmax(axis=1)
    refs["best_conventional_id"] = wide[list(BASELINES)].idxmax(axis=1)
    out = frame.merge(refs.reset_index(), on=keys, how="left", validate="many_to_one")
    out["delta_source"] = out["mean_acc"] - out["source_acc"]
    out["delta_best_adapt_baseline"] = out["mean_acc"] - out["best_adapt_baseline_acc"]
    out["delta_best_conventional"] = out["mean_acc"] - out["best_conventional_acc"]
    return out


def candidate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame[~frame["id"].isin(BASELINES)].copy()
    grouped = candidates.groupby(["model_kind", "T", "id", "method"], as_index=False)
    summary = grouped.agg(
        settings=("mean_acc", "size"),
        mean_acc=("mean_acc", "mean"),
        mean_delta_source=("delta_source", "mean"),
        mean_delta_best=("delta_best_conventional", "mean"),
        median_delta_best=("delta_best_conventional", "median"),
        min_delta_best=("delta_best_conventional", "min"),
        max_delta_best=("delta_best_conventional", "max"),
        win_rate_best=("delta_best_conventional", lambda x: float((x > 0).mean())),
        noninferior_rate_0p2=("delta_best_conventional", lambda x: float((x >= -0.2).mean())),
        mean_adapted_frac=("adapted_frac", "mean"),
        mean_time_s=("time_s", "mean"),
    )
    return summary.sort_values(
        ["model_kind", "T", "mean_delta_best"], ascending=[True, True, False]
    )


def setting_summary(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame[~frame["id"].isin(BASELINES)].copy()
    grouped = candidates.groupby(
        ["model_kind", "T", "level", "batch", "id", "method"], as_index=False
    )
    summary = grouped.agg(
        seeds=("seed", "nunique"),
        mean_acc=("mean_acc", "mean"),
        std_acc=("mean_acc", "std"),
        mean_delta_source=("delta_source", "mean"),
        std_delta_source=("delta_source", "std"),
        mean_delta_best=("delta_best_conventional", "mean"),
        std_delta_best=("delta_best_conventional", "std"),
        wins=("delta_best_conventional", lambda x: int((x > 0).sum())),
    )
    summary["all_seed_win"] = summary["wins"] == summary["seeds"]
    return summary.sort_values("mean_delta_best", ascending=False)


def per_corruption_summary(frame: pd.DataFrame) -> pd.DataFrame:
    corr_cols = [column for column in frame.columns if column.startswith("acc_")]
    id_cols = ["model_kind", "T", "level", "batch", "seed", "id", "method"]
    long = frame[id_cols + corr_cols].melt(
        id_vars=id_cols, var_name="corruption", value_name="acc"
    )
    long["corruption"] = long["corruption"].str.removeprefix("acc_")
    source = (
        long[long["id"] == "B0_source"]
        .rename(columns={"acc": "source_corr_acc"})
        .drop(columns=["id", "method"])
    )
    adaptive = long[long["id"].isin(ADAPT_BASELINES)]
    keys = ["model_kind", "T", "level", "batch", "seed", "corruption"]
    best = adaptive.groupby(keys, as_index=False)["acc"].max().rename(
        columns={"acc": "best_adapt_corr_acc"}
    )
    candidates = long[~long["id"].isin(BASELINES)].merge(source, on=keys, how="left")
    candidates = candidates.merge(best, on=keys, how="left")
    candidates["delta_source"] = candidates["acc"] - candidates["source_corr_acc"]
    candidates["delta_best_adapt"] = candidates["acc"] - candidates["best_adapt_corr_acc"]
    return (
        candidates.groupby(["model_kind", "T", "id", "method", "corruption"], as_index=False)
        .agg(
            settings=("acc", "size"),
            mean_acc=("acc", "mean"),
            mean_delta_source=("delta_source", "mean"),
            mean_delta_best_adapt=("delta_best_adapt", "mean"),
            win_rate_best_adapt=("delta_best_adapt", lambda x: float((x > 0).mean())),
        )
        .sort_values("mean_delta_best_adapt", ascending=False)
    )


def snn_ann_uplift(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame[~frame["id"].isin(BASELINES)].copy()
    snn = candidates[candidates["model_kind"] == "SNN"]
    ann = candidates[candidates["model_kind"] == "ANN"]
    keys = ["level", "batch", "seed", "id"]
    matched = snn.merge(
        ann[keys + ["delta_source"]].rename(columns={"delta_source": "ann_delta_source"}),
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    matched["snn_minus_ann_uplift"] = matched["delta_source"] - matched["ann_delta_source"]
    return (
        matched.groupby(["T", "level", "batch", "id", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            snn_uplift=("delta_source", "mean"),
            ann_uplift=("ann_delta_source", "mean"),
            snn_minus_ann_uplift=("snn_minus_ann_uplift", "mean"),
        )
        .sort_values("snn_minus_ann_uplift", ascending=False)
    )


def plot_best_setting_heatmap(setting: pd.DataFrame, out_dir: Path) -> None:
    snn = setting[setting["model_kind"] == "SNN"]
    best = snn.groupby(["T", "batch"], as_index=False)["mean_delta_best"].max()
    table = best.pivot(index="T", columns="batch", values="mean_delta_best").sort_index()
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    image = ax.imshow(table.values, cmap="RdYlGn", aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("SNN time steps T")
    ax.set_title("Best Phase-3 candidate uplift over best conventional baseline (pp)")
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table.iloc[row, col]
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="accuracy uplift (percentage points)")
    fig.tight_layout()
    fig.savefig(out_dir / "best_candidate_uplift_heatmap.png", dpi=180)
    plt.close(fig)


def plot_candidate_leaderboard(summary: pd.DataFrame, out_dir: Path) -> None:
    snn = summary[summary["model_kind"] == "SNN"]
    pooled = snn.groupby(["id", "method"], as_index=False).agg(
        mean_delta_best=("mean_delta_best", "mean"),
        win_rate_best=("win_rate_best", "mean"),
    ).sort_values("mean_delta_best")
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    colors = plt.cm.viridis(np.clip(pooled["win_rate_best"].to_numpy(), 0, 1))
    ax.barh(pooled["id"], pooled["mean_delta_best"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean uplift over best conventional baseline (pp)")
    ax.set_ylabel("Candidate")
    ax.set_title("Phase-3 SNN candidate leaderboard (all settings pooled)")
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_leaderboard.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snn-root", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--ann-root", type=Path, default=Path("outputs/phase3_ann"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase3_analysis"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.concat(
        [load_results(args.snn_root, "SNN"), load_results(args.ann_root, "ANN")],
        ignore_index=True,
    )
    paired = add_paired_deltas(raw)
    candidates = candidate_summary(paired)
    settings = setting_summary(paired)
    corruptions = per_corruption_summary(paired)
    architecture = snn_ann_uplift(paired)

    paired.to_csv(args.out_dir / "all_paired_results.csv", index=False)
    candidates.to_csv(args.out_dir / "candidate_summary.csv", index=False)
    settings.to_csv(args.out_dir / "setting_summary.csv", index=False)
    corruptions.to_csv(args.out_dir / "per_corruption_summary.csv", index=False)
    architecture.to_csv(args.out_dir / "snn_ann_uplift.csv", index=False)
    plot_best_setting_heatmap(settings, args.out_dir)
    plot_candidate_leaderboard(candidates, args.out_dir)

    print(f"Loaded {len(raw)} method-setting rows")
    print(f"Wrote analysis to {args.out_dir}")
    print("\nTop SNN setting/candidate combinations by paired uplift:")
    columns = ["T", "level", "batch", "id", "mean_acc", "mean_delta_source",
               "mean_delta_best", "std_delta_best", "wins", "seeds"]
    print(settings[settings["model_kind"] == "SNN"][columns].head(20).to_string(index=False))
    print("\nSNN candidate leaderboard by T:")
    columns = ["T", "id", "mean_delta_source", "mean_delta_best", "win_rate_best",
               "min_delta_best", "max_delta_best"]
    print(candidates[candidates["model_kind"] == "SNN"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
