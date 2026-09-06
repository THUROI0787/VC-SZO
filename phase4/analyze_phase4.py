"""Refresh paper-facing Phase-4 tables and simple diagnostic plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("outputs/phase4")
OUT = ROOT / "analysis"
BLUR = ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur")
CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)
LABEL = {
    "source": "Source", "zo_noop": "ZO-noop", "zo_entropy": "ZO-entropy", "zo_margin": "ZO-margin", "m307": "VC-ZO-S",
    "m301": "VC-ZO-M", "tent": "TENT", "memo": "MEMO",
    "bn_stats": "BN", "bp_margin": "BP-margin", "bp_entropy": "BP-entropy",
}
ORDER = list(LABEL)


def load_tag(tag: str) -> pd.DataFrame:
    rows = []
    root = ROOT / tag
    for path in root.rglob("*.json") if root.exists() else []:
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "ok" and "acc" in row:
            row["tag"] = tag
            rows.append(row)
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["T", "level", "batch", "seed", "corruption", "max_samples"]
    source = frame[frame.method.eq("source")][keys + ["acc"]].rename(columns={"acc": "source_acc"})
    out = frame.merge(source, on=keys, how="inner", validate="many_to_one")
    out["delta_source"] = out.acc - out.source_acc
    out["is_blur"] = out.corruption.isin(BLUR)
    return out


def main_table(frame: pd.DataFrame) -> None:
    selected = frame[(frame["T"] == 12) & (frame.level == 5) & (frame.batch == 2)]
    cells = selected.groupby(["method", "corruption"]).agg(
        acc=("acc", "mean"), seeds=("seed", "nunique")
    ).reset_index()
    complete = cells[cells.seeds.eq(3)]
    means = complete.pivot(index="method", columns="corruption", values="acc")
    means = means.reindex(columns=CORRUPTIONS)
    counts = complete.groupby("method").corruption.nunique()
    means["All-15"] = complete.groupby("method").acc.mean().where(counts.eq(15))
    blur = complete[complete.corruption.isin(BLUR)]
    blur_counts = blur.groupby("method").corruption.nunique()
    means["Blur-4"] = blur.groupby("method").acc.mean().where(blur_counts.eq(4))
    means = means.reindex([m for m in ORDER if m in means.index])
    means.rename(index=LABEL).to_csv(OUT / "main_table_level5.csv", float_format="%.3f")

    p = paired(selected)
    per_corruption = p.groupby(["method", "corruption"]).delta_source.agg(
        mean_delta="mean", std_seed_delta="std", seed_wins=lambda x: int((x > 0).sum()),
        noninferior_seeds=lambda x: int((x >= 0).sum()), seeds="count"
    ).reset_index()
    per_corruption["method"] = per_corruption.method.map(LABEL).fillna(per_corruption.method)
    per_corruption.to_csv(OUT / "per_corruption_paired_level5.csv", index=False, float_format="%.4f")

    summaries = []
    for subset, group in [("All-15", p), ("Blur-4", p[p.is_blur])]:
        for method, values in group.groupby("method"):
            d = values.delta_source
            summaries.append({"subset": subset, "method": LABEL.get(method, method),
                              "pairs": len(d), "mean_delta": d.mean(), "std_delta": d.std(),
                              "win_rate": (d > 0).mean(), "noninferior_rate": (d >= 0).mean()})
    pd.DataFrame(summaries).to_csv(OUT / "paired_uplift_level5.csv", index=False, float_format="%.4f")

    seed_summaries = []
    for subset, group, expected in [("All-15", p, 15), ("Blur-4", p[p.is_blur], 4)]:
        seed_cells = group.groupby(["method", "seed"]).agg(
            mean_delta=("delta_source", "mean"), corruptions=("corruption", "nunique")
        ).reset_index()
        seed_cells = seed_cells[seed_cells.corruptions.eq(expected)]
        for method, values in seed_cells.groupby("method"):
            n = len(values)
            mean = values.mean_delta.mean()
            std = values.mean_delta.std()
            half = 4.303 * std / (n ** 0.5) if n == 3 else float("nan")
            seed_summaries.append({"subset": subset, "method": LABEL.get(method, method),
                                   "seeds": n, "mean_delta": mean, "std_seed_delta": std,
                                   "ci95_low": mean - half, "ci95_high": mean + half,
                                   "all_seed_positive": bool((values.mean_delta > 0).all())})
    pd.DataFrame(seed_summaries).to_csv(
        OUT / "seed_paired_uplift_level5.csv", index=False, float_format="%.4f"
    )


def severity_plot(frame: pd.DataFrame) -> None:
    selected = frame[(frame["T"] == 12) & (frame.batch == 2) & frame.corruption.isin(BLUR)]
    stats = selected.groupby(["level", "method"]).acc.agg(["mean", "std", "count"]).reset_index()
    stats.to_csv(OUT / "severity_blur.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for method in ["source", "m307", "m301", "bp_margin", "tent"]:
        g = stats[stats.method.eq(method)].sort_values("level")
        if not g.empty:
            ax.errorbar(g.level, g["mean"], yerr=g["std"], marker="o", capsize=3,
                        linewidth=1.8, label=LABEL[method])
    ax.set(xticks=[1, 3, 5], xlabel="CIFAR-10-C severity",
           ylabel="Blur-family accuracy (%)", title="T=12, batch=2, 120 samples/corruption")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "severity_blur.png", dpi=180); plt.close(fig)


def curve(tags: list[str], x: str, filename: str) -> None:
    frames = [load_tag(t) for t in tags]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[(frame["level"] == 5) & frame["corruption"].isin(BLUR)]
    p = paired(frame)
    stats = p.groupby([x, "method"]).delta_source.agg(["mean", "std", "count"]).reset_index()
    stats = stats[stats["count"].eq(12)]
    stats.to_csv(OUT / f"{filename}.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for method in ["m307", "m301", "tent", "memo", "bp_margin"]:
        g = stats[stats.method.eq(method)].sort_values(x)
        if not g.empty:
            ax.errorbar(g[x], g["mean"], yerr=g["std"], marker="o", capsize=3,
                        linewidth=1.7, label=LABEL[method])
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xlabel("Batch size" if x == "batch" else "SNN time steps T")
    ax.set_ylabel("Paired uplift over Source (pp)")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"{filename}.png", dpi=180); plt.close(fig)


def convergence_plot() -> None:
    frame = load_tag("convergence500")
    if frame.empty or "trajectory" not in frame:
        return
    rows = []
    for record in frame.to_dict("records"):
        for point in record.get("trajectory", []):
            rows.append({"seed": record["seed"], "method": record["method"],
                         "corruption": record["corruption"], "samples": point["samples"],
                         "acc": point["acc"]})
    if not rows:
        return
    long = pd.DataFrame(rows)
    keys = ["seed", "corruption", "samples"]
    source = long[long.method.eq("source")][keys + ["acc"]].rename(columns={"acc": "source_acc"})
    paired_long = long.merge(source, on=keys, how="inner")
    paired_long["delta_source"] = paired_long.acc - paired_long.source_acc
    stats = paired_long.groupby(["samples", "method"]).delta_source.agg(
        ["mean", "std", "count"]
    ).reset_index()
    stats = stats[stats["count"].eq(12)]
    stats.to_csv(OUT / "convergence_curve.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for method in ["m307", "m301", "tent", "bp_margin"]:
        g = stats[stats.method.eq(method)].sort_values("samples")
        if not g.empty:
            ax.errorbar(g.samples, g["mean"], yerr=g["std"], marker="o", capsize=3,
                        linewidth=1.7, label=LABEL[method])
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Samples seen per corruption", ylabel="Cumulative paired uplift over Source (pp)")
    ax.set_xscale("log", base=2); ax.grid(axis="y", alpha=.25)
    if ax.lines:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "convergence_curve.png", dpi=180); plt.close(fig)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p0 = load_tag("p0_corrected120")
    if p0.empty:
        raise SystemExit("No Phase-4 P0 results yet")
    main_table(p0)
    severity_plot(p0)
    curve(["p0_corrected120", "batch_fixed120_b1", "batch_fixed120_b4", "batch_fixed120_b8"], "batch", "batch_curve_fixed_samples")
    curve(["p0_corrected120", "batch_b1", "batch_b4", "batch_b8"], "batch", "batch_curve_fixed_updates")
    curve(["p0_corrected120", "t_curve"], "T", "t_curve")
    convergence_plot()
    print(f"loaded {len(p0)} P0 rows; wrote {OUT}")


if __name__ == "__main__":
    main()
