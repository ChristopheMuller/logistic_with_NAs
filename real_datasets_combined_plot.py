"""Combined real-data figure for inclusion at full text width.

Run from the repository root:
    Rscript --vanilla real_datasets_pattern_support.R  # when split inputs change
    .venv/bin/python real_datasets_combined_plot.py
Ranks and AUC differences are regenerated from the current score CSVs.
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/logistic-rank-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import real_datasets_rank_plot as rank_plot

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "real_datasets_results/rank_summary_draft"
BLUE, ORANGE = "#2c6586", "#bc4b26"


def main():
    rank_plot.main()
    ranks = pd.read_csv(OUT / "dataset_ranks.csv")
    averages = ranks.groupby(["Method", "Metric"]).Rank.mean().unstack()
    order = averages.mean(axis=1).sort_values(kind="stable").index.tolist()
    support = pd.read_csv(ROOT / "real_datasets_results/pattern_support_draft/split_pattern_support.csv")
    support = support[support.ThresholdType.eq("per_coefficient") & support.Threshold.eq(3)]
    assert not support.duplicated(["Dataset", "Fold"]).any()
    assert support.groupby("Dataset").Fold.nunique().eq(15).all()
    support = support.groupby("Dataset", as_index=False).agg(
        TrainPatterns=("TrainPatterns", "mean"), Coverage=("Coverage", "mean"))
    # Difference of split-mean AUCs equals the mean of paired differences
    # because every included method has the same 15 split scores.
    auc = ranks[ranks.Metric.eq("AUC")].pivot(index="Dataset", columns="Method", values="Score")
    auc["AUCDifference"] = auc["PbP"] - auc["Mean.IMP"]
    scatter = support.merge(auc.AUCDifference, on="Dataset", validate="one_to_one")
    assert scatter.AUCDifference.notna().all()
    scatter["CoveragePercent"] = 100 * scatter.Coverage
    scatter.to_csv(OUT / "combined_scatter_data_k3.csv", index=False)

    plt.rcdefaults()
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(12.6, 6.2))
    # Nested grids allow a compact rank pair and more room for scatter y labels.
    outer = fig.add_gridspec(1, 2, width_ratios=[2.12, 1], wspace=0.18)
    left = outer[0].subgridspec(1, 2, wspace=0.10)
    right = outer[1].subgridspec(2, 1, hspace=0.42)
    rank_axes = [fig.add_subplot(left[0]), fig.add_subplot(left[1])]
    scatter_axes = [fig.add_subplot(right[0]), fig.add_subplot(right[1])]
    datasets = sorted(ranks.Dataset.unique())
    jitter = dict(zip(datasets, np.random.default_rng(20261102).uniform(-0.24, 0.24, len(datasets))))
    for ax, metric, letter in zip(rank_axes, ["AUC", "Brier"], ["a", "b"]):
        for row, method in enumerate(order):
            part = ranks[ranks.Method.eq(method) & ranks.Metric.eq(metric)]
            color = ORANGE if method == "PbP" else BLUE
            if row % 2 == 0:
                ax.axhspan(row - 0.5, row + 0.5, color="#f3f5f6", zorder=0)
            ax.scatter(part.Rank, row + part.Dataset.map(jitter), s=13,
                       color=color, alpha=0.29, linewidths=0, zorder=2)
            mean = averages.loc[method, metric]
            ax.scatter(mean, row, s=45, color=color, edgecolors="white",
                       linewidths=0.9, zorder=4)
            ax.text(14.6, row, f"{mean:.2f}", va="center", color=color, fontsize=9)
        ax.set_title(f"({letter}) " + ("ROC-AUC" if metric == "AUC" else "Brier score"), pad=4)
        ax.text(14.6, -0.82, "Mean", fontsize=9, color="0.35")
        ax.set_xlim(0.5, 16.1)
        ax.set_ylim(len(order) - 0.5, -1.1)
        ax.set_xticks([1, 4, 7, 10, 14])
        ax.set_xlabel("Rank (1 = best)", labelpad=7)
        ax.set_yticks(range(len(order)), order if ax == rank_axes[0] else [])
        ax.tick_params(axis="both", length=0, labelsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dce1e5", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for label in rank_axes[0].get_yticklabels():
        if label.get_text() == "PbP":
            label.set_color(ORANGE)

    for ax, x, letter in zip(scatter_axes, ["TrainPatterns", "CoveragePercent"], ["c", "d"]):
        ax.scatter(scatter[x], scatter.AUCDifference, s=24, color=ORANGE,
                   alpha=0.7, edgecolors="white", linewidths=0.4, zorder=3)
        ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_ylabel("AUC(PbP) − AUC(Mean)", fontsize=10)
        ax.set_yticks([-0.2, -0.1, 0, 0.1])
        ax.set_ylim(min(-0.205, scatter.AUCDifference.min() - 0.015),
                    max(0.105, scatter.AUCDifference.max() + 0.015))
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.025, 0.96, f"({letter})", transform=ax.transAxes,
                ha="left", va="top", fontsize=11)
        if x == "TrainPatterns":
            ax.set_xscale("log")
            ax.set_xlabel("Training patterns", labelpad=4)
        else:
            ax.set_xlim(-3, 103)
            ax.set_xticks([0, 50, 100])
            ax.set_xlabel("Support coverage (%) · k = 3", labelpad=4)

    handles = [Line2D([], [], marker="o", color=BLUE, alpha=0.35,
                      linestyle="none", markersize=4, label="One dataset"),
               Line2D([], [], marker="o", color=BLUE, markeredgecolor="white",
                      linestyle="none", markersize=7, label="Average rank")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.95, bottom=0.14)
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"rank_and_support.{ext}", dpi=200,
                    bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    missing = sorted(set(datasets) - set(scatter.Dataset))
    (OUT / "rank_and_support_caption.tex").write_text(
        "\\caption{Performance comparison and pattern support on real-world datasets. "
        "(a,b) Within-dataset ranks for ROC-AUC and Brier score across 38 datasets, "
        "computed after averaging scores over 15 random 80/20 train--test splits; "
        "ties receive average ranks. Small dots represent individual datasets; "
        "large dots and numerical annotations give average ranks. Lower ranks are better. "
        "PbP is highlighted in orange. (c,d) AUC difference between PbP and mean imputation "
        "against the number of training patterns and support coverage, respectively, "
        "for the 37 datasets with available split information. Positive differences favour PbP. "
        "Support coverage is the fraction of test observations whose pattern has at least "
        "$3p_m$ training observations, where $p_m$ counts model coefficients, including "
        "the intercept and categorical expansion. Both coordinates are averaged over splits. "
        "NHANES is omitted from (c,d) because its saved split information is unavailable.}\n")
    print(f"Combined figure: {len(datasets)} rank datasets, {len(scatter)} scatter datasets; missing: {missing}")


if __name__ == "__main__":
    main()
