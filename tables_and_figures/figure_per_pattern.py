#####
#
# Plot results per pattern (1 metrics)
#
#####

# %%

# if current working directory is "/tables_and_figures", change it to the parent directory
import os
if os.getcwd().endswith("tables_and_figures"):
    os.chdir(os.path.join(os.getcwd(), ".."))

# %% load packages

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from setups_design import metrics_config, methods_config

# %% set up

exp = "SimNL"
names_titles = [
    "Gaussian missing (Z1)",
    "Gaussian missing (Z2)",
    "Exponential missing (Z3)",
    "Cubic missing (Z4)",
    "Non-monotonic missing (Z5)"
]

score_matrix = pd.read_csv(os.path.join("data", exp, "score_matrix.csv"))
score_matrix = score_matrix[score_matrix["exp"] == exp]


metric_sel = "mae_bayes"
patterns_sel = [
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
]
patterns_sel = [str(pattern) for pattern in patterns_sel]

pattern_names = patterns_sel.copy()

methods_sel = [
"Mean.IMP.M",
"PbP.Fixed",
"SAEM.NoReg",
"MICE.100.IMP",
"MICE.100.Y.IMP",
"MICE.RF.10.IMP",
"MICE.RF.10.Y.IMP",
# "MICE.100.Y.IMP.M",
# "MICE.100.Y.M.IMP.M",
# "MICE.RF.10.Y.IMP.M",
# "MICE.RF.10.Y.M.IMP.M",
]


score_matrix_sel = score_matrix[score_matrix["metric"] == metric_sel]
score_matrix_sel = score_matrix_sel[score_matrix_sel["method"].isin(methods_sel)]
score_matrix_sel = score_matrix_sel[score_matrix_sel["exp"] == exp]

score_matrix_sel = score_matrix_sel[score_matrix_sel["filter"].isin(patterns_sel)]

score_matrix_sel["score"] = score_matrix_sel["score"].astype(float)
score_matrix_sel = score_matrix_sel.groupby(["filter", "method", "n_train"]).agg({"score": ["mean", "std", "count"]}).reset_index()
score_matrix_sel.columns = ["filter", "method", "n_train", "mean", "sd", "count"]

score_matrix_sel["se"] = score_matrix_sel["sd"] / np.sqrt(score_matrix_sel["count"])
fig, axes = plt.subplots(1, len(patterns_sel), figsize=(4 * len(patterns_sel), 5))

for i, pattern in enumerate(patterns_sel):

    print(i, pattern)

    # filter the score
    score_matrix_pattern = score_matrix_sel[score_matrix_sel["filter"] == pattern]

    # plot the mean and se
    for method in methods_sel:
        method_config = methods_config[method]

        score_matrix_method = score_matrix_pattern[score_matrix_pattern["method"] == method]
        axes[i].plot(score_matrix_method["n_train"], score_matrix_method["mean"], label=method_config["label"], 
                     color=method_config["color"], linestyle=method_config["linestyle"],
                     marker=method_config["marker"], markersize=5)
        axes[i].fill_between(score_matrix_method["n_train"], score_matrix_method["mean"] - score_matrix_method["se"],
                              score_matrix_method["mean"] + score_matrix_method["se"], alpha=0.2, 
                              color=method_config["color"], linestyle=method_config["linestyle"])
    
    axes[i].set_xscale("log")
    axes[i].set_xlabel("Number of training samples")
    axes[i].set_ylabel(metrics_config[metric_sel]["label"])
    # axes[i].set_title(f"M = {pattern_names[i]}")   
    axes[i].set_title(names_titles[i])

    if i == 1:
        axes[i].legend()
    # axes[i].grid()

    axes[i].set_ylim(-0.01, 0.35)
    # line at
    axes[i].axhline(0, color="black", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join("tables_and_figures", exp, f"{exp}_perPattern_{metric_sel}.pdf"))
plt.show()
