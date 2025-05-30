#####
#
# Plot main results (4 metrics), Simulation C
#
#####

# %% load packages

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from setups_design import metrics_config, methods_config

# %% set up

exp = "SimulationC"
score_matrix = pd.read_csv(os.path.join("data", exp, "score_matrix.csv"))
score_matrix = score_matrix[score_matrix["exp"] == exp]


metrics_config

# %% 

from utils import calculate_ymin_for_R_proportion
score_matrix = score_matrix[score_matrix["filter"] == "all"]


# methods_sel = ["MICE.IMP", "MICE.M.IMP", "MICE.Y.IMP", "MICE.Y.M.IMP", "Mean.IMP", "Mean.IMP.M", "05.IMP", "05.IMP.M"]  ## Single Imputation
# methods_sel = ["Mean.IMP.M", "PbP", "CC", "MICE.IMP", "MICE.Y.IMP",  "MICE.10.Y.IMP", "MICE.100.Y.IMP", "SAEM"]  ## Selected Procedures
methods_sel = ["MICE.10.Y.IMP.M", "MICE.Caliber.10.Y.IMP.M", "MICE.Cart.10.Y.IMP.M", "MICE.RF.10.Y.IMP.M"]  ## MICE Methods

scores_sel = ["misclassification", "calibration", "mse_error", "mae_bayes"]
filter_bayes = [True, True, False, False]


ntrains = [100, 500, 1000, 5000, 10000, 50000]

# ylimsmax = [0.06, 0.035, 1.5, 0.20] ## Single Imputation
ylimsmax = [0.35, 0.1, 5, 0.45]  ## Selected Procedures
ylimsmax = [0.12, 0.02, 3, 0.2]  ## Selected Procedures

ylimsmin = calculate_ymin_for_R_proportion(0.03, ylimsmax)
ylims = [(ylimsmin[i], ylimsmax[i]) for i in range(len(ylimsmax))] 

fig, axes = plt.subplots(1, len(scores_sel), figsize=(4 * len(scores_sel), 4))

for i, score in enumerate(scores_sel):

    print(i, score)

    # filter the score
    score_matrix_sel = score_matrix[score_matrix["metric"] == score]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["method"].isin(methods_sel)]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["exp"] == exp]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["n_train"].isin(ntrains)]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["bayes_adj"] == filter_bayes[i]]

    # group by method and n_train
    score_matrix_sel = score_matrix_sel.groupby(["method", "n_train"]).agg({"score": ["mean", "std", "count"]}).reset_index()
    score_matrix_sel.columns = ["method", "n_train", "mean", "sd", "count"]
    score_matrix_sel["se"] = score_matrix_sel["sd"] / np.sqrt(score_matrix_sel["count"])

    # plot the mean and se
    for method in methods_sel:
        method_config = methods_config[method]

        score_matrix_method = score_matrix_sel[score_matrix_sel["method"] == method]
        axes[i].plot(score_matrix_method["n_train"], score_matrix_method["mean"], label=method_config["label"], 
                     color=method_config["color"], linestyle=method_config["linestyle"],
                     marker=method_config["marker"], markersize=5)
        axes[i].fill_between(score_matrix_method["n_train"], score_matrix_method["mean"] - score_matrix_method["se"],
                              score_matrix_method["mean"] + score_matrix_method["se"], alpha=0.2, 
                              color=method_config["color"], linestyle=method_config["linestyle"])
    
    axes[i].set_xscale("log")
    axes[i].set_xlabel("Number of training samples")
    axes[i].set_ylabel(metrics_config[score]["label"])
    axes[i].set_title(metrics_config[score]["label"])   
    if i == 0:
        axes[i].legend()
    # axes[i].grid()
    axes[i].set_ylim(ylims[i])

    # line at 
    axes[i].axhline(0, color="black", linestyle="--", linewidth=0.5)

plt.tight_layout()
# plt.savefig(os.path.join("plots_scripts", "plots", "SimC_SingleImputation.pdf"))
plt.savefig(os.path.join("plots_scripts", "plots", "SimC_SelectedProcedures.pdf"))
plt.show()
    

# %%


metric_sel = "mae_bayes"
patterns_sel = [
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
]
titles = [
    "Gaussian missing (Z1)",
    "Gaussian missing (Z2)",
    "Exponential missing (Z3)",
    "Cubic missing (Z4)",
    "Non-Monotonic missing (Z5)"
]

# patterns_sel = [
#     0,
#     1,
#     2,
#     3,
#     4
# ]


patterns_sel = [str(pattern) for pattern in patterns_sel]

# sel_methods = ["MICE.IMP", "MICE.M.IMP", "MICE.Y.IMP", "MICE.Y.M.IMP", "Mean.IMP", "Mean.IMP.M", "05.IMP", "05.IMP.M"]  ## Single Imputation
sel_methods = ["Mean.IMP.M", "PbP", "CC", "MICE.IMP", "MICE.Y.IMP",  "MICE.10.Y.IMP", "MICE.100.Y.IMP", "SAEM"]  ## Selected Procedures

score_matrix_sel = score_matrix[score_matrix["metric"] == metric_sel]
score_matrix_sel = score_matrix_sel[score_matrix_sel["method"].isin(sel_methods)]
score_matrix_sel = score_matrix_sel[score_matrix_sel["exp"] == exp]

score_matrix_sel = score_matrix_sel[score_matrix_sel["filter"].isin(patterns_sel)]


score_matrix_sel = score_matrix_sel.groupby(["filter", "method", "n_train"]).agg({"score": ["mean", "std", "count"]}).reset_index()
score_matrix_sel.columns = ["filter", "method", "n_train", "mean", "sd", "count"]
score_matrix_sel["se"] = score_matrix_sel["sd"] / np.sqrt(score_matrix_sel["count"])

fig, axes = plt.subplots(1, len(patterns_sel), figsize=(3.5 * len(patterns_sel), 5))

for i, pattern in enumerate(patterns_sel):

    print(i, pattern)

    # filter the score
    score_matrix_pattern = score_matrix_sel[score_matrix_sel["filter"] == pattern]

    # plot the mean and se
    for method in sel_methods:
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
    axes[i].set_title(titles[i]) 
    if i == 0:
        axes[i].legend()
    # axes[i].grid()

    axes[i].set_ylim(-0.01, 0.4)
    # line at
    axes[i].axhline(0, color="black", linestyle="--", linewidth=0.5)

plt.tight_layout()
# plt.savefig(os.path.join("plots_scripts", "plots", "SimC_SingleImputation_per_pattern.pdf"))
plt.savefig(os.path.join("plots_scripts", "plots", "SimC_SelectedProcedures_per_pattern.pdf"))
plt.show()

# %%

set_up_df = pd.read_csv(os.path.join("data", exp, "set_up.csv"))
set_up = set_up_df.iloc[0]["set_up"]

# plot Xi vs X0 for i = 1, .., 4

original_data = np.load(os.path.join("data", exp, "original_data",  f"{set_up}.npz"))
X = original_data["X_full"]

titles = [
    "Gaussian (Z2)",
    "Exponential (Z3)",
    "Cubic (Z4)",
    "Non-Monotonic (Z5)"
]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for i in range(4):

    axes[i].scatter(X[:, 0], X[:, i + 1], alpha=0.2, s=1)
    axes[i].set_xlabel("Z1")
    axes[i].set_ylabel(f"Z{i + 2}")
    axes[i].grid()
    axes[i].set_title(titles[i])

plt.tight_layout()
plt.savefig(os.path.join("plots_scripts", "plots", "SimC_Xi_vs_X0.jpg"), dpi=300)
plt.show()

# %% 

# %%

from utils import calculate_ymin_for_R_proportion
score_matrix = score_matrix[score_matrix["filter"] == "all"]


# methods_sel = ["MICE.IMP", "MICE.M.IMP", "MICE.Y.IMP", "MICE.Y.M.IMP", "Mean.IMP", "Mean.IMP.M", "05.IMP", "05.IMP.M"]  ## Single Imputation
methods_sel = ["Mean.IMP.M", "PbP", "CC", "MICE.IMP", "MICE.Y.IMP",  "MICE.10.Y.IMP", "MICE.100.Y.IMP", "SAEM"]  ## Selected Procedures
# methods_sel = ["MICE.IMP", "MICE.M.IMP" ,"MICE.Y.IMP", "MICE.Y.M.IMP", "MICE.10.IMP", "MICE.10.IMP.M", "MICE.10.Y.IMP", "MICE.10.Y.IMP.M",
#                "MICE.100.IMP", "MICE.100.IMP.M", "MICE.100.Y.IMP", "MICE.100.Y.IMP.M"]  ## MICE Methods

scores_sel = ["brier", "angular_error"]
filter_bayes = [True, False]

ntrains = [100, 500, 1000, 5000, 10000, 50000]

# ylimsmax = [0.06, 0.035, 1.5, 0.20] ## Single Imputation
ylimsmax = [0.12, 0.75]  ## Selected Procedures
# ylimsmax = [0.06, 0.035, 1.1, 0.20]  ## MICE Procedures


ylimsmin = calculate_ymin_for_R_proportion(0.03, ylimsmax)
ylims = [(ylimsmin[i], ylimsmax[i]) for i in range(len(ylimsmax))] 

fig, axes = plt.subplots(1, len(scores_sel), figsize=(4 * len(scores_sel), 5))

for i, score in enumerate(scores_sel):

    print(i, score)

    # filter the score
    score_matrix_sel = score_matrix[score_matrix["metric"] == score]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["method"].isin(methods_sel)]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["exp"] == exp]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["n_train"].isin(ntrains)]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["bayes_adj"] == filter_bayes[i]]

    # group by method and n_train
    score_matrix_sel = score_matrix_sel.groupby(["method", "n_train"]).agg({"score": ["mean", "std", "count"]}).reset_index()
    score_matrix_sel.columns = ["method", "n_train", "mean", "sd", "count"]
    score_matrix_sel["se"] = score_matrix_sel["sd"] / np.sqrt(score_matrix_sel["count"])

    # plot the mean and se
    for method in methods_sel:
        method_config = methods_config[method]

        score_matrix_method = score_matrix_sel[score_matrix_sel["method"] == method]
        axes[i].plot(score_matrix_method["n_train"], score_matrix_method["mean"], label=method_config["label"], 
                     color=method_config["color"], linestyle=method_config["linestyle"],
                     marker=method_config["marker"], markersize=5)
        axes[i].fill_between(score_matrix_method["n_train"], score_matrix_method["mean"] - score_matrix_method["se"],
                              score_matrix_method["mean"] + score_matrix_method["se"], alpha=0.2, 
                              color=method_config["color"], linestyle=method_config["linestyle"])
    
    axes[i].set_xscale("log")
    axes[i].set_xlabel("Number of training samples")
    axes[i].set_ylabel(metrics_config[score]["label"])
    axes[i].set_title(metrics_config[score]["label"])   
    # if i == 0:
    if i == 1:
        axes[i].legend()
    # axes[i].grid()
    axes[i].set_ylim(ylims[i])

    # line at 
    axes[i].axhline(0, color="black", linestyle="--", linewidth=0.5)

plt.tight_layout()
# plt.savefig(os.path.join("plots_scripts", "plots", "SimA_SingleImputation.pdf"))
plt.savefig(os.path.join("plots_scripts", "plots", "SimC_SelectedProcedures_Brier_Angular.pdf"))
# plt.savefig(os.path.join("plots_scripts", "plots", "SimA_MICE_Procedures.pdf"))
plt.show()
    
