#####
#
# This script builds extends the matrix of scores for the simulation study,
# using a score per pattern instead of overall score.
#
#####

# %%

import os
import csv
import numpy as np
import pandas as pd
from utils import *

# %%

exp = "SimNL"
set_up = pd.read_csv(os.path.join("data", exp, "set_up.csv"))
simulation = pd.read_csv(os.path.join("data", exp, "simulation.csv"))

score_matrix = pd.read_csv(os.path.join("data", exp, "score_matrix.csv"))
# %%

metrics = {
    "mae_bayes": lambda y, pred_probs, bayes_probs: np.mean(np.abs(bayes_probs - pred_probs))
}

methods = [
"MICE.1.IMP","MICE.1.Y.IMP","MICE.1.M.IMP","MICE.1.Y.M.IMP",
"MICE.1.IMP.M","MICE.1.Y.IMP.M","MICE.1.M.IMP.M","MICE.1.Y.M.IMP.M",
"MICE.10.IMP","MICE.10.Y.IMP","MICE.10.M.IMP","MICE.10.Y.M.IMP",
"MICE.10.IMP.M","MICE.10.Y.IMP.M","MICE.10.M.IMP.M","MICE.10.Y.M.IMP.M",
"SAEM",
"Mean.IMP","Mean.IMP.M","05.IMP","05.IMP.M",
"PbP.Fixed",#"CC",
"MICE.RF.10.IMP","MICE.RF.10.Y.IMP","MICE.RF.10.M.IMP","MICE.RF.10.Y.M.IMP",
"MICE.RF.10.IMP.M","MICE.RF.10.Y.IMP.M","MICE.RF.10.M.IMP.M","MICE.RF.10.Y.M.IMP.M",
"MICE.100.IMP","MICE.100.Y.IMP","MICE.100.M.IMP","MICE.100.Y.M.IMP",
"MICE.100.IMP.M","MICE.100.Y.IMP.M","MICE.100.M.IMP.M","MICE.100.Y.M.IMP.M",
"SAEM.NoReg"
]

# patterns = [
#     0,
#     1,
#     2,
#     3,
#     4
# ]

patterns = [
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
]

# %%

def score_matrix_per_pattern(exp, set_up, simulation, metrics, methods, patterns, score_matrix):
    """Replace per-pattern scores while preserving all other score rows."""
    pattern_labels = [str(pattern) for pattern in patterns]
    simulation = simulation.loc[
        simulation["method"].isin(methods)
        & simulation["set_up"].isin(set_up["set_up"])
    ].copy()
    simulation["n_train"] = simulation["n_train"].round().astype(int)
    score_keys = ["set_up", "method", "n_train"]
    selected_results = pd.MultiIndex.from_frame(simulation[score_keys])
    replace = (
        score_matrix["exp"].eq(exp)
        & score_matrix["metric"].isin(metrics)
        & score_matrix["filter"].isin(pattern_labels)
        & pd.MultiIndex.from_frame(score_matrix[score_keys]).isin(selected_results)
    )
    retained = score_matrix.loc[~replace]
    rows = []

    for i, setup in enumerate(set_up["set_up"], start=1):
        print(f"Set up {i}/{len(set_up)} - {setup}", flush=True)
        with np.load(os.path.join("data", exp, "test_data", f"{setup}.npz")) as test:
            mask, true_y = test["M"], test["y"]
        with np.load(os.path.join("data", exp, "bayes_data", f"{setup}.npz")) as bayes:
            bayes_probs = bayes["y_probs_bayes"]
        indices = [get_index_pattern(pattern, mask) for pattern in patterns]
        if any(len(idx) == 0 for idx in indices):
            raise ValueError(f"A selected pattern has no test observations in {setup}")

        for result in simulation.loc[simulation["set_up"] == setup].itertuples():
            ntrain = int(round(result.n_train))
            path = os.path.join("data", exp, "pred_data", f"{setup}_{result.method}_{ntrain}.npz")
            with np.load(path) as prediction:
                pred_probs = prediction["y_probs_pred"].ravel()
            for pattern, idx in zip(pattern_labels, indices):
                for metric, calculate in metrics.items():
                    score = calculate(true_y[idx], pred_probs[idx], bayes_probs[idx])
                    baseline = calculate(true_y[idx], bayes_probs[idx], bayes_probs[idx])
                    for adjusted, value in ((False, score), (True, score - baseline)):
                        rows.append({"exp": exp, "set_up": setup, "method": result.method,
                                     "n_train": ntrain, "bayes_adj": adjusted,
                                     "metric": metric, "score": value, "filter": pattern})
    return pd.concat([retained, pd.DataFrame(rows)], ignore_index=True)


new_matrix = score_matrix_per_pattern(exp, set_up, simulation, metrics, methods, patterns, score_matrix)
new_matrix.to_csv(os.path.join("data", exp, "score_matrix.csv"), index=False, quoting=csv.QUOTE_ALL)
