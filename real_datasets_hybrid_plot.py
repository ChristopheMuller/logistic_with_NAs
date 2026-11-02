"""Aggregate hybrid-threshold experiment; every dataset has equal weight.

Run: .venv/bin/python real_datasets_hybrid_plot.py
Optional: --input real_datasets_results/hybrid_smoke
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/logistic-rank-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parent / "real_datasets_results/hybrid_threshold")
    args = parser.parse_args()
    out = args.input
    config = pd.read_csv(out / "run_config.csv", header=None, index_col=0)[1]
    folds = int(config["ExpectedFolds"])
    data = pd.read_csv(out / "metrics.csv")
    if "Variant" not in data:
        data["Variant"] = "Hybrid"
    variants = sorted(data.Variant.unique())
    data["k"] = pd.to_numeric(data.k)
    assert not data.duplicated(["Dataset", "Fold", "Variant", "k"]).any()
    grid = sorted(data.k.unique())
    assert grid[0] == 0 and np.isinf(grid[-1])
    eligible = []
    excluded = []
    statuses = pd.read_csv(out / "fit_status.csv")
    for name in sorted(statuses.Dataset.unique()):
        part = data[data.Dataset.eq(name)]
        valid = (len(part) == folds * len(grid) * len(variants) and
                 part.groupby(["Fold", "Variant"]).size().eq(len(grid)).all() and
                 part.groupby(["k", "Variant"]).Fold.nunique().eq(folds).all() and
                 set(part.Variant) == set(variants) and
                 set(part.k) == set(grid) and
                 np.isfinite(part[["AUC_mean", "AUC_hybrid", "AUC_difference",
                                   "Specialisation_coverage"]]).all().all())
        (eligible if valid else excluded).append(name)
    (out / "plot_exclusions.txt").write_text(
        "Datasets excluded from all thresholds because of incomplete/undefined AUC results:\n"
        + ("\n".join(excluded) if excluded else "None") + "\n")
    if not eligible:
        raise ValueError("No dataset has complete finite AUC results at every threshold")
    data = data[data.Dataset.isin(eligible)]
    endpoint = data[np.isinf(data.k)]
    assert endpoint.AUC_difference.eq(0).all()
    assert endpoint.Specialisation_coverage.eq(0).all()
    assert data.Specialisation_coverage.between(0, 1).all()
    for _, group in data.groupby(["Dataset", "Fold", "Variant"]):
        assert group.sort_values("k").Specialisation_coverage.diff().dropna().le(1e-12).all()
    means = data.groupby(["Dataset", "Variant", "k"], as_index=False).agg(
        AUC_difference=("AUC_difference", "mean"),
        Specialisation_coverage=("Specialisation_coverage", "mean"))
    means.to_csv(out / "dataset_threshold_means.csv", index=False)
    summary = means.groupby(["Variant", "k"]).agg(
        AUC_median=("AUC_difference", "median"),
        AUC_q25=("AUC_difference", lambda x: x.quantile(.25)),
        AUC_q75=("AUC_difference", lambda x: x.quantile(.75)),
        Coverage_median=("Specialisation_coverage", "median"),
        Coverage_q25=("Specialisation_coverage", lambda x: x.quantile(.25)),
        Coverage_q75=("Specialisation_coverage", lambda x: x.quantile(.75)),
        Datasets=("Dataset", "nunique"))
    summary.to_csv(out / "threshold_summary.csv")
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"Hybrid": "#bc4b26", "Ridge hybrid": "#2c6586"}
    x = np.arange(len(grid))
    for individual in [False, True]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, prefix, metric, scale in zip(
            axes, ["AUC", "Coverage"], ["AUC_difference", "Specialisation_coverage"], [1, 100]
        ):
            for variant in variants:
                color = colors[variant]
                part = summary.loc[variant].reindex(grid)
                if individual:
                    for _, group in means[means.Variant.eq(variant)].groupby("Dataset"):
                        ax.plot(x, group.set_index("k").reindex(grid)[metric]*scale,
                                color=color, alpha=.12, linewidth=.7, zorder=1)
                else:
                    ax.fill_between(x, part[prefix+"_q25"]*scale,
                                    part[prefix+"_q75"]*scale, color=color, alpha=.15,
                                    linewidth=0)
                ax.plot(x, part[prefix+"_median"]*scale, color=color,
                        marker="s" if variant == "Ridge hybrid" else "o",
                        linestyle="--" if variant == "Ridge hybrid" else "-",
                        markersize=4, linewidth=1.7, label=variant+" (median)")
            ax.set_xticks(x, ["∞" if np.isinf(k) else f"{k:g}" for k in grid])
            ax.set_xlabel("Minimum training observations per coefficient, k")
            ax.grid(axis="y", alpha=.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].axhline(0, color="0.4", linestyle="--", linewidth=.8)
        axes[0].set_ylabel("AUC(hybrid) − AUC(mean imputation)")
        axes[0].set_title("(a) Predictive performance", pad=4)
        axes[1].set_ylabel("Specialisation coverage (%)")
        axes[1].set_ylim(-2, 102)
        axes[1].set_title("(b) Specialisation coverage", pad=4)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(.5, .005), fontsize=9)
        fig.tight_layout(rect=(0, .075, 1, 1))
        name = "hybrid_individual_curves" if individual else "hybrid_threshold_summary"
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight", pad_inches=.06)
        plt.close(fig)
    print(f"Plotted {len(eligible)} datasets with {folds} splits each; excluded: {excluded}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
