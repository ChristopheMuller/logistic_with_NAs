"""Draft rank distributions from the existing real-data score CSVs.

Run from the repository root: .venv/bin/python real_datasets_rank_plot.py
Scores are averaged over splits before ranking; ties receive average ranks.
No input files are modified. These scores still require a provenance audit.
"""

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/logistic-rank-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "real_datasets_results" / "rank_summary_draft"
SOURCES = {
    "benchmark": ROOT / "real_datasets_results/real_datasets_metrics_detailed.csv",
    "tabzilla": ROOT / "tabzilla/tabzilla_real_datasets_results/real_datasets_metrics_detailed.csv",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    audit = {}
    keys = ["Dataset", "Method", "Fold", "Metric"]
    for source, path in SOURCES.items():
        frame = pd.read_csv(path)
        assert not frame.duplicated(keys).any(), f"Duplicate keys in {path}"
        assert frame.groupby(["Dataset", "Method", "Metric"]).size().eq(15).all()
        finite = frame.Value.dropna()
        assert np.isfinite(finite).all() and finite.between(0, 1).all()
        audit[source] = {
            "rows": len(frame),
            "datasets": int(frame.Dataset.nunique()),
            "methods": int(frame.Method.nunique()),
            "missing_values_by_metric": frame.groupby("Metric").Value.apply(
                lambda values: int(values.isna().sum())
            ).to_dict(),
        }
        frame["Source"] = source
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    # main_old.tex specifies retaining the benchmark preprocessing for colic.
    data = data[~((data.Source == "tabzilla") & (data.Dataset == "colic"))]
    shared = set(frames[0].Method) & set(frames[1].Method)
    data = data[data.Method.isin(shared)].copy()
    assert not data.duplicated(keys).any()
    data = data[data.Metric.isin(["AUC", "Brier"])]
    assert data.Value.notna().all()
    scores = data.groupby(["Dataset", "Method", "Metric"], as_index=False).agg(
        Score=("Value", "mean"), Splits=("Value", "size")
    )
    assert scores.Splits.eq(15).all()
    assert scores.groupby(["Dataset", "Metric"]).size().eq(len(shared)).all()
    scores["Rank"] = scores.groupby(["Dataset", "Metric"]).Score.rank(
        ascending=True, method="average"
    )
    auc = scores.Metric.eq("AUC")
    scores.loc[auc, "Rank"] = scores.loc[auc].groupby("Dataset").Score.rank(
        ascending=False, method="average"
    )
    averages = scores.groupby(["Method", "Metric"]).Rank.mean().unstack()
    order = averages.mean(axis=1).sort_values(kind="stable").index.tolist()
    scores.to_csv(OUT / "dataset_ranks.csv", index=False)
    averages.loc[order].to_csv(OUT / "average_ranks.csv")

    # Identical jitter per dataset in both panels; jitter has no numeric meaning.
    rng = np.random.default_rng(20261102)
    datasets = sorted(scores.Dataset.unique())
    jitter = dict(zip(datasets, rng.uniform(-0.24, 0.24, len(datasets))))
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)
    for ax, metric in zip(axes, ["AUC", "Brier"]):
        for row, method in enumerate(order):
            part = scores[(scores.Method == method) & (scores.Metric == metric)]
            highlight = method == "PbP"
            color = "#bc4b26" if highlight else "#2c6586"
            if row % 2 == 0:
                ax.axhspan(row - 0.5, row + 0.5, color="#f3f5f6", zorder=0)
            ax.scatter(part.Rank, row + part.Dataset.map(jitter), s=22,
                       color=color, alpha=0.29, linewidths=0, zorder=2)
            mean_rank = averages.loc[method, metric]
            ax.scatter(mean_rank, row, s=78, color=color, edgecolors="white",
                       linewidths=1.3, zorder=4)
            ax.text(14.65, row, f"{mean_rank:.2f}", ha="left", va="center",
                    color=color, fontsize=9)
        ax.set_title("ROC-AUC" if metric == "AUC" else "Brier score", pad=3)
        ax.text(14.65, -0.82, "Mean", fontsize=9, color="#555555")
        ax.set_xlim(0.5, 15.75)
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14])
        ax.set_xlabel("Within-dataset rank (1 = best)", labelpad=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dce1e5", linewidth=0.6)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_ylim(len(order) - 0.5, -1.1)
    for label in axes[0].get_yticklabels():
        if label.get_text() == "PbP":
            label.set_color("#bc4b26")
    handles = [Line2D([], [], marker="o", color="#2c6586", alpha=0.35,
                      linestyle="none", markersize=5, label="One dataset"),
               Line2D([], [], marker="o", color="#2c6586", markeredgecolor="white",
                      linestyle="none", markersize=9, label="Average rank")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False)
    # Size the label margin from its rendered text instead of reserving a
    # fixed fraction of the canvas. Centre the legend on the complete figure.
    fig.tight_layout(rect=(0, 0.065, 1, 1), pad=0.6, w_pad=1.5)
    for extension in ["png", "pdf"]:
        fig.savefig(OUT / f"rank_distribution.{extension}", dpi=180,
                    facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    audit["draft_selection"] = {
        "datasets": len(datasets), "methods": len(shared),
        "excluded": ["Tabzilla colic (retain benchmark version)", "PbP.MinObs (not shared)"],
        "method_order": "Average of the AUC and Brier mean ranks",
        "limitations": [
            "CSV completeness does not establish correctness of stored predictions.",
            "Execution failures and intentional baseline fallbacks are not labelled in CSVs.",
            "Metric script uses test-label mean for NA predictions; affected rows unknown.",
            "Metric script does not fix ROC direction explicitly.",
            "SAEM and constant-imputation methods can use fewer covariates on categorical datasets.",
            "Other potential duplicate tasks (e.g. related heart/breast datasets) need provenance review.",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"Plot saved to {OUT}")


if __name__ == "__main__":
    main()
