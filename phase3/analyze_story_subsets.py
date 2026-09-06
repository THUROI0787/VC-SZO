"""Build the small-batch corruption-family tables/figures used for story selection."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILIES = {
    "Noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "Blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "Weather": ("snow", "frost", "fog", "brightness"),
    "Digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
METHODS = ("B0_source", "B1_tent_bp", "m307", "m301")
LABELS = {
    "B0_source": "Source",
    "B1_tent_bp": "TENT-BP",
    "m307": "Subspace margin-ZO (m307)",
    "m301": "2-sample margin-ZO (m301)",
}


def main() -> None:
    out_dir = Path("outputs/phase3_analysis")
    frame = pd.read_csv(out_dir / "all_paired_results.csv")
    snn = frame[(frame["model_kind"] == "SNN") & (frame["T"] == 12)].copy()
    for family, names in FAMILIES.items():
        snn[f"family_{family.lower()}"] = snn[[f"acc_{name}" for name in names]].mean(axis=1)

    # Paper-candidate main table: T=12, b=2, three severities and three seeds.
    selected = snn[(snn["batch"] == 2) & snn["id"].isin(METHODS)]
    rows = []
    for (level, method), group in selected.groupby(["level", "id"]):
        row = {"T": 12, "level": level, "batch": 2, "id": method, "method": LABELS[method]}
        for family in FAMILIES:
            values = group[f"family_{family.lower()}"]
            row[f"{family.lower()}_mean"] = values.mean()
            row[f"{family.lower()}_std"] = values.std()
        row["all15_mean"] = group["mean_acc"].mean()
        row["all15_std"] = group["mean_acc"].std()
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["level", "id"])
    table.to_csv(out_dir / "story_t12_b2_families.csv", index=False)

    # Figure 1: the robust blur subset across severity.
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    styles = {
        "B0_source": ("#444444", "o"),
        "B1_tent_bp": ("#c44e52", "s"),
        "m307": ("#2a9d8f", "D"),
        "m301": ("#457b9d", "^"),
    }
    for method in METHODS:
        group = table[table["id"] == method].sort_values("level")
        color, marker = styles[method]
        ax.errorbar(
            group["level"], group["blur_mean"], yerr=group["blur_std"],
            label=LABELS[method], color=color, marker=marker, linewidth=2,
            markersize=6, capsize=3,
        )
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("CIFAR-10-C severity")
    ax.set_ylabel("Blur-family accuracy (%)")
    ax.set_title("T=12, batch=2: margin-ZO remains effective under blur shifts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "story_t12_b2_blur.png", dpi=200)
    plt.close(fig)

    # Figure 2: paired m307 uplift over source over the small-batch range.
    m307 = snn[snn["id"] == "m307"]
    source = snn[snn["id"] == "B0_source"]
    keys = ["T", "level", "batch", "seed"]
    paired = m307.merge(
        source[keys + [f"family_{family.lower()}" for family in FAMILIES]],
        on=keys, suffixes=("", "_source"), validate="one_to_one",
    )
    batches = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), sharey=True)
    for ax, level in zip(axes, [1, 3, 5]):
        level_data = paired[(paired["level"] == level) & paired["batch"].isin(batches)]
        for family, color in zip(FAMILIES, ("#e76f51", "#2a9d8f", "#e9c46a", "#457b9d")):
            delta = level_data[f"family_{family.lower()}"] - level_data[f"family_{family.lower()}_source"]
            stats = level_data.assign(delta=delta).groupby("batch")["delta"].agg(["mean", "std"])
            ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o",
                        label=family, color=color, capsize=2)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches, batches)
        ax.set_title(f"Severity {level}")
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("m307 uplift over source (pp)")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("T=12: the positive regime is concentrated at very small batches", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "story_m307_small_batch_uplift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
