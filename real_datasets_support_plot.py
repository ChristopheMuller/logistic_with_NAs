"""Exploratory PbP performance versus pattern support, using existing scores.

First run: Rscript --vanilla real_datasets_pattern_support.R
Then run: .venv/bin/python real_datasets_support_plot.py
Coverage is a count-based diagnostic, not the rate of successful model fits.
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/logistic-rank-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "real_datasets_results/pattern_support_draft"


def main():
    frames = []
    for source, path in enumerate([
        ROOT / "real_datasets_results/real_datasets_metrics_detailed.csv",
        ROOT / "tabzilla/tabzilla_real_datasets_results/real_datasets_metrics_detailed.csv",
    ]):
        frame = pd.read_csv(path)
        if source == 1:
            frame = frame[frame.Dataset.ne("colic")]
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    shared_methods = set(frames[0].Method) & set(frames[1].Method)
    rank_scores = scores[scores.Method.isin(shared_methods) & scores.Metric.isin(["AUC", "Brier"])]
    rank_means = rank_scores.groupby(["Dataset", "Method", "Metric"], as_index=False).Value.mean()
    rank_means["Rank"] = rank_means.groupby(["Dataset", "Metric"]).Value.rank(method="average")
    auc = rank_means.Metric.eq("AUC")
    rank_means.loc[auc, "Rank"] = rank_means.loc[auc].groupby("Dataset").Value.rank(
        ascending=False, method="average")
    auc_scores = scores[scores.Metric.eq("AUC") & scores.Method.isin(["PbP", "Mean.IMP"])]
    assert not auc_scores.duplicated(["Dataset", "Method", "Fold"]).any()
    auc_paired = auc_scores.pivot(index=["Dataset", "Fold"], columns="Method", values="Value")
    assert auc_paired.notna().all().all()
    auc_difference = (auc_paired["PbP"] - auc_paired["Mean.IMP"]).groupby("Dataset").mean()
    scores = scores[scores.Metric.eq("Brier") & scores.Method.isin(["PbP", "Mean.IMP"])]
    assert not scores.duplicated(["Dataset", "Method", "Fold"]).any()
    paired = scores.pivot(index=["Dataset", "Fold"], columns="Method", values="Value")
    assert paired.notna().all().all()
    paired["BrierDifference"] = paired["PbP"] - paired["Mean.IMP"]
    support = pd.read_csv(OUT / "split_pattern_support.csv")
    support = support[support.ThresholdType.eq("per_coefficient")]
    merged = support.merge(paired.BrierDifference.reset_index(), on=["Dataset", "Fold"],
                           how="left", validate="many_to_one")
    assert merged.BrierDifference.notna().all()
    summary = merged.groupby(["Dataset", "Threshold"], as_index=False).agg(
        TrainPatterns=("TrainPatterns", "mean"), Coverage=("Coverage", "mean"),
        TrainN=("TrainN", "mean"), BrierDifference=("BrierDifference", "mean"),
        Splits=("Fold", "nunique"))
    assert summary.Splits.eq(15).all()
    summary["CoveragePercent"] = 100 * summary.Coverage
    summary.to_csv(OUT / "dataset_support_summary.csv", index=False)
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    rank_data = summary[summary.Threshold.eq(10)].merge(
        rank_means[rank_means.Method.eq("PbP")][["Dataset", "Metric", "Rank"]],
        on="Dataset", validate="one_to_many")
    rank_data.to_csv(OUT / "dataset_rank_vs_support.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, metric in zip(axes, ["AUC", "Brier"]):
        data = rank_data[rank_data.Metric.eq(metric)]
        ax.scatter(data.CoveragePercent, data.Rank, s=38, alpha=0.6,
                   color="tab:blue", edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_title("ROC-AUC" if metric == "AUC" else "Brier score", pad=4)
        ax.set_xlim(-3, 103)
        ax.set_ylim(0.5, len(shared_methods) + 0.5)
        ax.set_yticks([1, 2, 4, 6, 8, 10, 12, 14])
        ax.set_xlabel("Test coverage (%) by patterns with\n≥ 10 training observations per coefficient")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("PbP rank among 14 methods\n(1 = best)")
    fig.tight_layout()
    save(fig, "rank_vs_support")

    def draw(ax, data, x, xlabel, y="BrierDifference"):
        ax.scatter(data[x], data[y], s=34, alpha=0.75,
                   color="tab:blue", edgecolors="white", linewidths=0.5, zorder=3)
        ax.axhline(0, color="0.35", linewidth=0.9, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.2)
        for name in ["colic", "MiceProtein", "pedestrian", "sleep", "Ozone",
                     "hepatitis", "eucalyptus"]:
            if y == "AUCDifference" and name == "hepatitis":
                continue  # Shares the same position as colic in coverage view.
            row = data[data.Dataset.eq(name)]
            if len(row):
                r = row.iloc[0]
                # Keep labels inside the axes at both coverage boundaries.
                dx = -6 if r[x] >= 0.9 * data[x].max() else 5
                offset = (dx, -14 if name in ["pedestrian", "hepatitis"] else 6)
                if y == "AUCDifference" and name == "Ozone":
                    offset = (dx, -14)
                ax.annotate(name, (r[x], r[y]), xytext=offset,
                            textcoords="offset points", fontsize=8,
                            ha="right" if offset[0] < 0 else "left")

    selected = summary[summary.Threshold.eq(10)]
    auc_data = selected.merge(auc_difference.rename("AUCDifference"), on="Dataset", validate="one_to_one")
    auc_data.to_csv(OUT / "dataset_auc_difference_vs_support.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    draw(axes[0], auc_data, "TrainPatterns", "Number of training patterns", y="AUCDifference")
    axes[0].set_xscale("log")
    draw(axes[1], auc_data, "CoveragePercent",
         "Test coverage (%) by patterns with\n≥ 10 training observations per coefficient", y="AUCDifference")
    axes[1].set_xlim(-3, 103)
    axes[0].set_ylabel("ROC-AUC difference\nPbP − mean imputation (higher is better)")
    fig.tight_layout()
    save(fig, "auc_difference_vs_support")

    auc_sensitivity = summary.merge(auc_difference.rename("AUCDifference"),
                                    on="Dataset", validate="many_to_one")
    auc_sensitivity.to_csv(OUT / "dataset_auc_support_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
    for ax, k in zip(axes, [1, 2, 3, 5, 10]):
        data = auc_sensitivity[auc_sensitivity.Threshold.eq(k)]
        ax.scatter(data.CoveragePercent, data.AUCDifference, s=30, alpha=0.7,
                   color="tab:blue", edgecolors="white", linewidths=0.5, zorder=3)
        ax.axhline(0, color="0.35", linewidth=0.9, linestyle="--")
        ax.set_title(f"k = {k}", pad=4)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Test coverage (%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("ROC-AUC difference\nPbP − mean imputation (higher is better)")
    fig.tight_layout()
    save(fig, "auc_support_threshold_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    draw(axes[0], selected, "TrainPatterns", "Number of training patterns")
    axes[0].set_xscale("log")
    draw(axes[1], selected, "CoveragePercent",
         "Test coverage (%) by patterns with\n≥ 10 training observations per coefficient")
    axes[1].set_xlim(-3, 103)
    axes[0].set_ylabel("Brier score difference\nPbP − mean imputation (lower is better)")
    fig.tight_layout()
    save(fig, "performance_vs_support")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, k in zip(axes, [5, 10, 20]):
        draw(ax, summary[summary.Threshold.eq(k)], "CoveragePercent", "Test coverage (%)")
        ax.set_title(f"k = {k}", pad=4)
        ax.set_xlim(-3, 103)
    axes[0].set_ylabel("Brier score difference\nPbP − mean imputation (lower is better)")
    fig.tight_layout()
    save(fig, "support_threshold_sensitivity")
    print(f"Included {selected.Dataset.nunique()} datasets. Outputs: {OUT}")
    print(selected.sort_values("BrierDifference")[["Dataset", "BrierDifference", "CoveragePercent"]].to_string(index=False))


def save(fig, name):
    for extension in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{extension}", dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    main()
