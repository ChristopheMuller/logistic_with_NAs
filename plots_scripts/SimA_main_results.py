#####
#
# Plot main results (4 metrics), Simulation A
#
#####

# %% load packages

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from setups import metrics_config, methods_config

# %% set up

exp = "SimulationA"
score_matrix = pd.read_csv(os.path.join("data", exp, "score_matrix.csv"))


metrics_config

# %% 

from utils import calculate_ymin_for_R_proportion

# methods_sel = ["MICE.IMP", "MICE.M.IMP", "MICE.Y.IMP", "MICE.Y.M.IMP", "Mean.IMP", "Mean.IMP.M", "05.IMP", "05.IMP.M"]  ## Single Imputation
methods_sel = ["Mean.IMP", "PbP", "CC", "MICE.IMP", "MICE.Y.IMP",  "MICE.10.Y.IMP", "MICE.100.Y.IMP", "SAEM"]  ## Selected Procedures

scores_sel = ["misclassification", "calibration", "mse_error", "mae_bayes"]
filter_bayes = [True, True, False, False]

ntrains = [100, 500, 1000, 5000, 10000, 50000]

# ylimsmax = [0.06, 0.035, 1.5, 0.20] ## Single Imputation
ylimsmax = [0.12, 0.1, 1.25, 0.3]  ## Selected Procedures
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
    if i == 0:
        axes[i].legend()
    # axes[i].grid()
    axes[i].set_ylim(ylims[i])

    # line at 
    axes[i].axhline(0, color="black", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join("plots_scripts", "plots", "SimA_SelectedProcedures.pdf"))
plt.show()
    

