#####
#
# This script builds a score matrix for the simulation study,
# instead of re-computing the scores for each plot.
#
# Determine:
# - If the score matrix already exists,
# - What scores to add,
# - What methods to add
#
#####


# %% load packages

import os
import numpy as np
import pandas as pd

# %% set up

exp = "SimulationA"
set_up = pd.read_csv(os.path.join("data", exp, "set_up.csv"))
simulation = pd.read_csv(os.path.join("data", exp, "simulation.csv"))


# %% define metrics we want to calculate


metrics = {

    "brier": lambda y, pred_probs, bayes_probs: np.mean((y - pred_probs)**2),
    "misclassification": lambda y, pred_probs, bayes_probs: 1 - np.mean(y == (pred_probs >= 0.5)),
    "mae_bayes": lambda y, pred_probs, bayes_probs: np.mean(np.abs(bayes_probs - pred_probs))


}

methods = [
    "SAEM",
    "05.IMP",
    "05.IMP.M",
    "Mean.IMP",
    "Mean.IMP.M",
    "PbP",
    "MICE.IMP",
    "MICE.5.IMP",
    "MICE.M.IMP",
    "MICE.Y.IMP",
    "MICE.Y.M.IMP",
    "MICE.10.Y.IMP",
    "MICE.100.Y.IMP"
]

# %% build the score matrix (predictions)

def build_score_matrix(exp, set_up, simulation, metrics, methods, existing_matrix=None):

    n_set_ups = len(set_up)
    n_metrics = len(metrics)

    if existing_matrix is not None:
        score_matrix = existing_matrix.copy()
    else:  
        score_matrix = pd.DataFrame(
            columns = ["exp", "set_up", "method", "n_train", "bayes_adj", "metric", "score", "filter"]
        )

    simulation = simulation.copy()
    simulation = simulation[simulation["method"].isin(methods)]

    for i in range(n_set_ups):

        setup = set_up.iloc[i]["set_up"]
        print(f"Set up {i+1}/{n_set_ups} - {setup}")

        simulation_setup = simulation[simulation["set_up"] == setup]
        
        for j in range(len(simulation_setup)):
            method = simulation_setup.iloc[j]["method"]
            ntrain = np.round(simulation_setup.iloc[j]["n_train"], 0).astype(int)

            # print(f"Simulation {j+1}/{len(simulation_setup)} - {method} - {ntrain}")

            ####

            true_y = np.load(os.path.join("data", exp, "test_data", f"{setup}.npz"))["y"]
            pred_probs = np.load(os.path.join("data", exp, "pred_data", f"{setup}_{method}_{ntrain}.npz"))["y_probs_pred"].ravel()
            bayes_probs = np.load(os.path.join("data", exp, "bayes_data", f"{setup}.npz"))["y_probs_bayes"]

            ####

            for k in range(n_metrics):

                metric = list(metrics.keys())[k]
                score = metrics[metric](true_y, pred_probs, bayes_probs)
                score_bayes = metrics[metric](true_y, bayes_probs, bayes_probs)
                score_bayes_adj = score - score_bayes

                score_matrix = pd.concat([
                    score_matrix,
                    pd.DataFrame({
                        "exp": [exp],
                        "set_up": [setup],
                        "method": [method],
                        "n_train": [ntrain],
                        "bayes_adj": False,
                        "metric": [metric],
                        "score": [score],
                        "filter": ["all"]
                    })
                ], ignore_index=True)

                score_matrix = pd.concat([
                    score_matrix,
                    pd.DataFrame({
                        "exp": [exp],
                        "set_up": [setup],
                        "method": [method],
                        "n_train": [ntrain],
                        "bayes_adj": True,
                        "metric": [metric],
                        "score": [score_bayes_adj],
                        "filter": ["all"]
                    })
                ], ignore_index=True)

    score_matrix = score_matrix.reset_index(drop=True)
    return score_matrix


matrix = build_score_matrix(exp, set_up, simulation, metrics, methods)


# %%

# matrix.to_csv(os.path.join("data", exp, "score_matrix.csv"), index=False)



# %%

score_matrix = matrix.copy()

methods = [
    "SAEM",
    "05.IMP",
    "05.IMP.M",
    "Mean.IMP",
    "Mean.IMP.M",
    "PbP",
    "MICE.IMP",
    "MICE.5.IMP",
    "MICE.M.IMP",
    "MICE.Y.IMP",
    "MICE.Y.M.IMP",
    "MICE.10.Y.IMP",
    "MICE.100.Y.IMP",
    "CC"
]

simulation_setup = pd.read_csv(os.path.join("data", exp, "simulation_set_up.csv"))
simulation_setup = simulation_setup[["set_up", "method", "n_train", "running_time", "angular_error", "mse_error"]]
simulation_setup = simulation_setup[simulation_setup["method"].isin(methods)]


for i in range(len(simulation_setup)):

    setup = simulation_setup.iloc[i]["set_up"]
    method = simulation_setup.iloc[i]["method"]
    ntrain = np.round(simulation_setup.iloc[i]["n_train"], 0).astype(int)

    print(f"Simulation {i+1}/{len(simulation_setup)} - {setup} - {method} - {ntrain}")

    angular_error = simulation_setup.iloc[i]["angular_error"]
    mse_error = simulation_setup.iloc[i]["mse_error"]
    running_time = simulation_setup.iloc[i]["running_time"]

    new_scores = pd.DataFrame({
        "exp": [exp, exp, exp],
        "set_up": [setup, setup, setup],
        "method": [method, method, method],
        "n_train": [ntrain, ntrain, ntrain],
        "bayes_adj": [False, False, False],
        "metric": ["angular_error", "mse_error", "running_time"],
        "score": [angular_error, mse_error, running_time],
        "filter": ["all", "all", "all"]
    })

    score_matrix = pd.concat([score_matrix, new_scores], ignore_index=True)

score_matrix.to_csv(os.path.join("data", exp, "score_matrix.csv"), index=False)

# %%
