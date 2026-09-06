"""Paper-facing Phase-4 comparison, ablation, architecture, and efficiency tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt

SNN_ROOT = Path("outputs/phase4")
ANN_ROOT = Path("outputs/phase4_ann")
OUT = SNN_ROOT / "analysis"
BLUR = ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur")
FAMILIES = {
    "Noise-3": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "Blur-4": BLUR,
    "Weather-4": ("snow", "frost", "fog", "brightness"),
    "Digital-4": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
LABEL = {
    "source": "Source", "zo_noop": "ZO-noop", "zo_entropy": "ZO-entropy",
    "zo_margin": "ZO-margin", "m307": "VC-SZO-SD", "m301": "VC-SZO-MD",
    "tent": "TENT", "memo": "MEMO", "bn_stats": "BN",
    "bp_margin": "BP-margin", "bp_entropy": "BP-entropy",
}
KEYS = ["T", "level", "batch", "seed", "corruption", "max_samples"]


def load(root: Path, tag: str) -> pd.DataFrame:
    rows = []
    for path in (root / tag).rglob("*.json") if (root / tag).exists() else []:
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "ok" and "acc" in row:
            rows.append(row)
    return pd.DataFrame(rows)


def pair_source(frame: pd.DataFrame, source_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    source_frame = frame if source_frame is None else source_frame
    source = source_frame[source_frame.method.eq("source")][KEYS + ["acc"]].rename(
        columns={"acc": "source_acc"}
    )
    paired = frame.merge(source, on=KEYS, how="inner", validate="many_to_one")
    paired["delta_source"] = paired.acc - paired.source_acc
    return paired


def efficiency() -> None:
    frame = load(SNN_ROOT, "p0_corrected120")
    frame = frame[(frame.level == 5) & (frame.batch == 2) & frame.corruption.isin(BLUR)]
    methods = ["source", "m307", "m301", "bp_margin", "bp_entropy", "tent", "memo"]
    frame = frame[frame.method.isin(methods)]
    numeric = ["time_s", "peak_allocated_mb", "peak_reserved_mb", "prediction_forwards",
               "objective_forwards", "backward_calls", "updates"]
    table = frame.groupby("method")[numeric].median().reset_index()
    table["total_forward_calls"] = table.prediction_forwards + table.objective_forwards
    source_time = float(table.loc[table.method.eq("source"), "time_s"].iloc[0])
    table["time_vs_source"] = table.time_s / source_time
    # Legacy TENT/MEMO engines did not expose exact objective/backward counters.
    table.loc[table.method.isin(["tent", "memo"]),
              ["prediction_forwards", "objective_forwards", "backward_calls", "total_forward_calls"]] = float("nan")
    table["adapter_params"] = table.method.map(
        {"source": 0, "m307": 512, "m301": 512, "bp_margin": 512, "bp_entropy": 512})
    table["requires_backward_graph"] = table.method.isin(
        ["bp_margin", "bp_entropy", "tent", "memo"])
    table["perturbation_forwards"] = table.method.map(
        {"m307": 120, "m301": 240, "source": 0, "bp_margin": 0, "bp_entropy": 0})
    table["collapse_check_forwards"] = table.method.map(
        {"m307": 60, "m301": 60, "source": 0, "bp_margin": 0, "bp_entropy": 0})
    table["method"] = table.method.map(LABEL)
    table.to_csv(OUT / "efficiency_table.csv", index=False, float_format="%.3f")


def efficiency_profile() -> None:
    """Aggregate the second (warm) corruption from one fresh process per method."""
    frame = load(SNN_ROOT, "efficiency_profile")
    frame = frame[frame.corruption.eq("glass_blur")].copy()
    if frame.empty:
        return
    columns = ["method", "time_s", "peak_allocated_mb", "peak_reserved_mb",
               "prediction_forwards", "objective_forwards", "backward_calls", "updates"]
    table = frame[columns].copy()
    table["total_forward_calls"] = table.prediction_forwards + table.objective_forwards
    source_time = float(table.loc[table.method.eq("source"), "time_s"].iloc[0])
    table["time_vs_source"] = table.time_s / source_time
    table["requires_backward_graph"] = table.method.isin(["bp_margin", "tent", "memo"])
    table["adapter_params"] = table.method.map(
        {"source": 0, "m307": 512, "m301": 512, "bp_margin": 512})
    table.loc[table.method.isin(["tent", "memo"]),
              ["prediction_forwards", "objective_forwards", "backward_calls",
               "total_forward_calls"]] = float("nan")
    table["method"] = table.method.map(LABEL)
    table.to_csv(OUT / "efficiency_profile_warm.csv", index=False, float_format="%.3f")


def efficiency_scaling() -> None:
    frames = [load(SNN_ROOT, "efficiency_profile"),
              load(SNN_ROOT, "efficiency_scaling")]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.corruption.eq("glass_blur")].drop_duplicates(["T", "method"])
    methods = ["source", "m307", "m301", "bp_margin", "tent", "memo"]
    frame = frame[frame.method.isin(methods)].copy()
    columns = ["T", "method", "time_s", "peak_allocated_mb", "peak_reserved_mb",
               "prediction_forwards", "objective_forwards", "backward_calls", "updates"]
    table = frame[columns].sort_values(["T", "method"])
    source_time = table[table.method.eq("source")].set_index("T").time_s
    table["time_vs_source"] = [row.time_s / source_time.get(row.T, np.nan)
                               for row in table.itertuples()]
    table["requires_backward_graph"] = table.method.isin(["bp_margin", "tent", "memo"])
    table.loc[table.method.isin(["tent", "memo"]),
              ["prediction_forwards", "objective_forwards", "backward_calls"]] = np.nan
    table["method"] = table.method.map(LABEL)
    table.to_csv(OUT / "efficiency_scaling_warm.csv", index=False, float_format="%.3f")
    complete = table.groupby("method")["T"].nunique()
    if not complete.empty and complete.max() >= 2:
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        for method, values in table.groupby("method"):
            values = values.sort_values("T")
            ax.plot(values["T"], values.peak_allocated_mb, marker="o", label=method)
        ax.set_xlabel("SNN time steps T")
        ax.set_ylabel("Warm peak allocated memory (MB)")
        ax.set_yscale("log")
        ax.grid(alpha=0.2); ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "efficiency_scaling.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def ann_tables() -> None:
    direct = load(ANN_ROOT, "ann_direct120")
    if direct.empty:
        return
    paired = pair_source(direct)
    rows = []
    for subset, group in [("All-15", paired), ("Blur-4", paired[paired.corruption.isin(BLUR)])]:
        for method, values in group.groupby("method"):
            rows.append({"subset": subset, "method": LABEL.get(method, method),
                         "acc": values.acc.mean(), "source_acc": values.source_acc.mean(),
                         "delta_source": values.delta_source.mean(), "pairs": len(values)})
    pd.DataFrame(rows).to_csv(OUT / "ann_direct_summary.csv", index=False, float_format="%.4f")

    source = load(ANN_ROOT, "ann_calib_default")
    calibration = []
    for tag, setting in [("ann_calib_default", "default"),
                         ("ann_calib_lr3e4", "lr=3e-4"),
                         ("ann_calib_eps3e2", "eps=3e-2")]:
        frame = load(ANN_ROOT, tag)
        if frame.empty:
            continue
        p = pair_source(frame, source)
        for method, values in p.groupby("method"):
            calibration.append({"setting": setting, "method": LABEL.get(method, method),
                                "acc": values.acc.mean(), "delta_source": values.delta_source.mean(),
                                "pairs": len(values)})
    pd.DataFrame(calibration).to_csv(OUT / "ann_calibration.csv", index=False, float_format="%.4f")


def confirmatory_windows() -> None:
    """Analyze five disjoint 120-image windows with shared fresh seeds."""
    tagged = [(0, "confirm_w0"), (120, "confirm_w120"),
              (240, "confirm_w240"), (360, "confirm_w360"),
              (480, "confirm_w480")]
    frames = []
    for offset, tag in tagged:
        frame = load(SNN_ROOT, tag)
        if frame.empty:
            continue
        frame["sample_offset"] = offset
        frames.append(frame)
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.groupby("method").size().rename("completed_cells").reset_index()
    counts["expected_cells"] = 60
    counts.to_csv(OUT / "confirmatory_progress.csv", index=False)
    required = {"source", "zo_noop", "m307", "m301", "bp_margin"}
    if not required.issubset(set(counts.method)) or not counts[counts.method.isin(required)].completed_cells.eq(60).all():
        return
    complete_methods = set(counts.loc[counts.completed_cells.eq(60), "method"])
    frame = frame[frame.method.isin(complete_methods)]
    join_keys = ["T", "level", "batch", "seed", "corruption",
                 "max_samples", "sample_offset"]
    source = frame[frame.method.eq("source")][join_keys + ["acc"]].rename(
        columns={"acc": "source_acc"})
    paired = frame.merge(source, on=join_keys, how="inner", validate="many_to_one")
    paired["delta_source"] = paired.acc - paired.source_acc
    paired.to_csv(OUT / "confirmatory_paired_cells.csv", index=False, float_format="%.4f")
    rows = []
    for method, values in paired.groupby("method"):
        image_groups = values.groupby(["sample_offset", "corruption"]).delta_source.mean()
        window_means = values.groupby("sample_offset").delta_source.mean()
        image_sem = scipy_stats.sem(image_groups)
        window_sem = scipy_stats.sem(window_means)
        image_ci = scipy_stats.t.interval(0.95, len(image_groups) - 1,
                                          loc=image_groups.mean(), scale=image_sem)
        window_ci = scipy_stats.t.interval(0.95, len(window_means) - 1,
                                           loc=window_means.mean(), scale=window_sem)
        image_p = np.nan if np.allclose(image_groups, 0) else scipy_stats.ttest_1samp(
            image_groups, 0, alternative="greater").pvalue
        window_p = np.nan if np.allclose(window_means, 0) else scipy_stats.ttest_1samp(
            window_means, 0, alternative="greater").pvalue
        rows.append({"method": LABEL.get(method, method), "acc": values.acc.mean(),
                     "source_acc": values.source_acc.mean(),
                     "delta_source": values.delta_source.mean(),
                     "image_groups": len(image_groups), "image_ci_low": image_ci[0],
                     "image_ci_high": image_ci[1], "image_p_one_sided": image_p,
                     "windows": len(window_means), "window_ci_low": window_ci[0],
                     "window_ci_high": window_ci[1], "window_p_one_sided": window_p,
                     "positive_windows": int((window_means > 0).sum())})
    pd.DataFrame(rows).to_csv(OUT / "confirmatory_summary.csv", index=False,
                              float_format="%.6f")
    confirmatory_contrasts(frame, complete_methods)


def confirmatory_contrasts(frame: pd.DataFrame, complete_methods: set[str]) -> None:
    """Paired VC-SZO contrasts at image-group and conservative window levels."""
    join_keys = ["T", "level", "batch", "seed", "corruption",
                 "max_samples", "sample_offset"]
    rows = []
    for proposed in ["m307", "m301"]:
        left = frame[frame.method.eq(proposed)][join_keys + ["acc"]].rename(
            columns={"acc": "proposed_acc"})
        for baseline in ["source", "bp_margin", "tent", "memo", "bn_stats"]:
            if baseline not in complete_methods:
                continue
            right = frame[frame.method.eq(baseline)][join_keys + ["acc"]].rename(
                columns={"acc": "baseline_acc"})
            direct = left.merge(right, on=join_keys, how="inner", validate="one_to_one")
            direct["delta"] = direct.proposed_acc - direct.baseline_acc
            grouped = [("window_corruption", direct.groupby(
                ["sample_offset", "corruption"]).delta.mean()),
                       ("window", direct.groupby("sample_offset").delta.mean())]
            for unit, values in grouped:
                if np.allclose(values, 0):
                    ci, pvalue, p_two_sided = (0.0, 0.0), np.nan, np.nan
                else:
                    ci = scipy_stats.t.interval(
                        0.95, len(values) - 1, loc=values.mean(),
                        scale=scipy_stats.sem(values))
                    pvalue = scipy_stats.ttest_1samp(
                        values, 0, alternative="greater").pvalue
                    p_two_sided = scipy_stats.ttest_1samp(values, 0).pvalue
                rows.append({"proposed": LABEL[proposed], "baseline": LABEL[baseline],
                             "unit": unit, "delta_pp": values.mean(),
                             "groups": len(values), "ci_low": ci[0],
                             "ci_high": ci[1], "p_one_sided": pvalue,
                             "p_two_sided": p_two_sided,
                             "positive_groups": int((values > 0).sum())})
    pd.DataFrame(rows).to_csv(OUT / "confirmatory_contrasts.csv", index=False,
                              float_format="%.6f")

def clean_safety() -> None:
    tagged = [(0, "clean_w0"), (120, "clean_w120"), (240, "clean_w240"),
              (360, "clean_w360"), (480, "clean_w480")]
    frames = []
    for offset, tag in tagged:
        frame = load(SNN_ROOT, tag)
        if frame.empty:
            continue
        frame["sample_offset"] = offset; frames.append(frame)
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    counts = frame.groupby("method").size()
    complete = set(counts[counts.eq(15)].index)
    if "source" not in complete:
        return
    frame = frame[frame.method.isin(complete)]
    keys = ["T", "level", "batch", "seed", "corruption", "max_samples", "sample_offset"]
    source = frame[frame.method.eq("source")][keys + ["acc"]].rename(columns={"acc": "source_acc"})
    paired = frame.merge(source, on=keys, how="inner", validate="many_to_one")
    paired["delta_source"] = paired.acc - paired.source_acc
    paired.to_csv(OUT / "clean_safety_paired.csv", index=False, float_format="%.4f")
    rows = []
    for method, values in paired.groupby("method"):
        windows = values.groupby("sample_offset").delta_source.mean()
        if np.allclose(windows, 0):
            ci, pvalue = (0.0, 0.0), np.nan
        else:
            ci = scipy_stats.t.interval(0.95, len(windows)-1, loc=windows.mean(),
                                        scale=scipy_stats.sem(windows))
            pvalue = scipy_stats.ttest_1samp(windows, 0, alternative="less").pvalue
        rows.append({"method": LABEL.get(method, method), "clean_acc": values.acc.mean(),
                     "source_acc": values.source_acc.mean(), "delta_source": values.delta_source.mean(),
                     "windows": len(windows), "ci_low": ci[0], "ci_high": ci[1],
                     "harm_p_one_sided": pvalue, "nonnegative_windows": int((windows >= 0).sum())})
    pd.DataFrame(rows).to_csv(OUT / "clean_safety_summary.csv", index=False, float_format="%.6f")


def severity_families() -> None:
    frame = load(SNN_ROOT, "p0_corrected120")
    frame = frame[(frame["T"] == 12) & (frame.batch == 2) &
                  frame.method.isin(["source", "m307", "m301", "bp_margin"])]
    rows = []
    for (level, method), group in frame.groupby(["level", "method"]):
        if group.seed.nunique() != 3:
            continue
        for family, corruptions in FAMILIES.items():
            values = group[group.corruption.isin(corruptions)]
            expected = 3 * len(corruptions)
            if len(values) == expected:
                rows.append({"level": level, "method": LABEL[method], "family": family,
                             "acc": values.acc.mean(), "cells": len(values)})
        if len(group) == 45:
            rows.append({"level": level, "method": LABEL[method], "family": "All-15",
                         "acc": group.acc.mean(), "cells": len(group)})
    table = pd.DataFrame(rows)
    if not table.empty:
        source = table[table.method.eq("Source")][["level", "family", "acc"]].rename(
            columns={"acc": "source_acc"})
        table = table.merge(source, on=["level", "family"], how="left")
        table["delta_source"] = table.acc - table.source_acc
    table.to_csv(OUT / "severity_family_table.csv", index=False, float_format="%.4f")


def ablation() -> None:
    frame = load(SNN_ROOT, "ablation_vczo")
    if frame.empty or "source" not in set(frame.method):
        return
    paired = pair_source(frame)
    rows = []
    for method, values in paired.groupby("method"):
        if len(values) != 12:
            continue
        seed_means = values.groupby("seed").delta_source.mean()
        rows.append({"method": LABEL.get(method, method), "acc": values.acc.mean(),
                     "delta_source": values.delta_source.mean(),
                     "seed_delta_std": seed_means.std(),
                     "all_seed_positive": bool((seed_means > 0).all()),
                     "objective_forwards": values.objective_forwards.median(),
                     "backward_calls": values.backward_calls.median()})
    pd.DataFrame(rows).to_csv(OUT / "vczo_ablation.csv", index=False, float_format="%.4f")


def query_budget() -> None:
    frame = load(SNN_ROOT, "query_curve360")
    if frame.empty or "trajectory" not in frame:
        return
    rows = []
    for record in frame.to_dict("records"):
        for point in record.get("trajectory", []):
            rows.append({"method": record["method"], "seed": record["seed"],
                         "corruption": record["corruption"], "samples": point["samples"],
                         "acc": point["acc"]})
    if not rows:
        return
    long = pd.DataFrame(rows)
    source = long[long.method.eq("source")][["seed", "corruption", "samples", "acc"]].rename(
        columns={"acc": "source_acc"})
    p = long.merge(source, on=["seed", "corruption", "samples"], how="inner")
    p["delta_source"] = p.acc - p.source_acc
    p_all = p.copy()
    # Symmetric perturbation-objective forwards only: 2/update for S, 4/update for M.
    factor = {"m307": 1.0, "m301": 2.0}  # batch=2 => budget = factor * samples
    p = p[p.method.isin(factor)].copy()
    if p.empty:
        return
    p["perturbation_budget"] = p.apply(lambda row: int(factor[row.method] * row.samples), axis=1)
    stats = p.groupby(["perturbation_budget", "method"]).delta_source.agg(
        ["mean", "std", "count"]).reset_index()
    stats = stats[stats["count"].eq(12)]
    common = set(stats[stats.method.eq("m307")].perturbation_budget) & set(
        stats[stats.method.eq("m301")].perturbation_budget)
    stats = stats[stats.perturbation_budget.isin(common)]
    stats["method"] = stats.method.map(LABEL)
    stats.to_csv(OUT / "query_matched_vczo.csv", index=False, float_format="%.4f")

    # Match total model forward calls across VC-ZO and BP; BP backward calls remain explicit.
    forward_factor = {"m307": 2.0, "m301": 3.0, "bp_margin": 1.0}
    compute = p_all[p_all.method.isin(forward_factor)].copy()
    compute["forward_budget"] = compute.apply(
        lambda row: int(forward_factor[row.method] * row.samples), axis=1)
    compute["backward_calls"] = compute.apply(
        lambda row: row.samples / 2 if row.method == "bp_margin" else 0, axis=1)
    compute_stats = compute.groupby(["forward_budget", "method"]).agg(
        samples=("samples", "median"), mean_delta=("delta_source", "mean"),
        std_delta=("delta_source", "std"), cells=("delta_source", "count"),
        backward_calls=("backward_calls", "median")).reset_index()
    compute_stats = compute_stats[compute_stats.cells.eq(12)]
    common_forward = None
    for method in forward_factor:
        budgets = set(compute_stats[compute_stats.method.eq(method)].forward_budget)
        common_forward = budgets if common_forward is None else common_forward & budgets
    compute_stats = compute_stats[compute_stats.forward_budget.isin(common_forward or set())]
    compute_stats["method"] = compute_stats.method.map(LABEL)
    compute_stats.to_csv(OUT / "forward_matched_zo_bp.csv", index=False, float_format="%.4f")



def paper_plots() -> None:
    styles = {"VC-SZO-MD": ("#d62728", "o"), "VC-SZO-SD": ("#1f77b4", "s"),
              "BP-margin": ("#2ca02c", "^")}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    specs = [(OUT / "query_matched_vczo.csv", "perturbation_budget",
              "Perturbation-objective forwards"),
             (OUT / "forward_matched_zo_bp.csv", "forward_budget",
              "Total model forwards")]
    for ax, (path, xcol, xlabel) in zip(axes, specs):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        ycol = "mean" if "mean" in frame else "mean_delta"
        for method in styles:
            values = frame[frame.method.eq(method)].sort_values(xcol)
            if values.empty:
                continue
            color, marker = styles[method]
            ax.plot(values[xcol], values[ycol], marker=marker, color=color,
                    linewidth=1.8, markersize=5, label=method)
        ax.axhline(0, color="0.45", linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Paired uplift over Source (pp)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "matched_budget.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    path = OUT / "vczo_ablation.csv"
    if path.exists():
        frame = pd.read_csv(path)
        order = ["ZO-entropy", "ZO-margin", "VC-SZO-SD", "VC-SZO-MD"]
        frame = frame[frame.method.isin(order)].set_index("method").reindex(order).reset_index()
        colors = ["#9e9e9e", "#7f7f7f", "#1f77b4", "#d62728"]
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        bars = ax.bar(frame.method, frame.delta_source, color=colors)
        ax.axhline(0, color="0.35", linewidth=1)
        ax.set_ylabel("Paired uplift over Source (pp)")
        ax.set_title("Severity-5 Blur-4 component ablation")
        ax.bar_label(bars, fmt="%+.2f", padding=2, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT / "vczo_ablation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def variance_diagnostic() -> None:
    paths = [SNN_ROOT / "variance_diagnostic" / "gpu0.csv",
             SNN_ROOT / "variance_diagnostic" / "gpu1.csv"]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    case = ["offset", "corruption"]
    reference = frame[frame.config.eq("MD-K1")][case + ["trace_variance"]].rename(
        columns={"trace_variance": "k1_trace_variance"})
    frame = frame.merge(reference, on=case, how="left")
    frame["variance_ratio_to_k1"] = frame.trace_variance / frame.k1_trace_variance
    metrics = ["mean_cosine", "mean_estimate_cosine", "mean_norm_ratio",
               "trace_variance", "variance_ratio_to_k1", "snr", "projected_std"]
    summary = frame.groupby(["config", "directions", "sparsity"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(OUT / "variance_diagnostic.csv", index=False, float_format="%.5f")
    if frame[case].drop_duplicates().shape[0] < 8:
        return
    order = ["MD-K1", "MD-K2", "MD-K4", "SD-q50", "SD-q25"]
    values = frame.groupby("config").agg(variance_ratio=("variance_ratio_to_k1", "mean"),
                                           cosine=("mean_estimate_cosine", "mean")).reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    axes[0].bar(values.index, values.variance_ratio, color="#4c78a8")
    axes[0].set_ylabel("Estimator variance / MD-K1")
    axes[1].bar(values.index, values.cosine, color="#f58518")
    axes[1].set_ylabel("Cosine(mean ZO, BP reference)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "variance_diagnostic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def t_absolute() -> None:
    frames = [load(SNN_ROOT, "p0_corrected120"), load(SNN_ROOT, "t_curve")]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(KEYS + ["method"])
    frame = frame[(frame.level == 5) & (frame.batch == 2) & frame.corruption.isin(BLUR)]
    table = frame.groupby(["T", "method"]).acc.agg(["mean", "std", "count"]).reset_index()
    table = table[table["count"].eq(12)]
    table["method"] = table.method.map(LABEL).fillna(table.method)
    table.to_csv(OUT / "t_absolute_blur.csv", index=False, float_format="%.4f")


def snn_ann() -> None:
    rows = []
    for architecture, root, tag in [("SNN", SNN_ROOT, "p0_corrected120"),
                                     ("ANN", ANN_ROOT, "ann_direct120")]:
        frame = load(root, tag)
        if frame.empty:
            continue
        frame = frame[(frame.level == 5) & (frame.batch == 2)]
        paired = pair_source(frame)
        for subset, group in [("All-15", paired), ("Blur-4", paired[paired.corruption.isin(BLUR)])]:
            for method, values in group.groupby("method"):
                expected = 45 if subset == "All-15" else 12
                if len(values) != expected:
                    continue
                rows.append({"architecture": architecture, "subset": subset,
                             "method": LABEL.get(method, method), "acc": values.acc.mean(),
                             "source_acc": values.source_acc.mean(),
                             "delta_source": values.delta_source.mean(), "cells": len(values)})
    pd.DataFrame(rows).to_csv(OUT / "snn_ann_uplift.csv", index=False, float_format="%.4f")
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    efficiency(); efficiency_profile(); efficiency_scaling(); ann_tables(); confirmatory_windows(); clean_safety(); severity_families(); ablation(); query_budget(); paper_plots(); variance_diagnostic(); t_absolute(); snn_ann()
    print(f"refreshed paper experiment tables in {OUT}")


if __name__ == "__main__":
    main()
