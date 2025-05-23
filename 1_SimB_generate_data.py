
# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from utils import *

# %%

experiment_name = "SimulationB"
experiment_data_folder = os.path.join("data", experiment_name)

if os.path.exists(experiment_data_folder) == False:
    os.makedirs(experiment_data_folder)

if os.path.exists(os.path.join(experiment_data_folder, "original_data")) == False:
    os.makedirs(os.path.join(experiment_data_folder, "original_data"))

if os.path.exists(os.path.join(experiment_data_folder, "test_data")) == False:
    os.makedirs(os.path.join(experiment_data_folder, "test_data"))

if os.path.exists(os.path.join(experiment_data_folder, "pred_data")) == False:
    os.makedirs(os.path.join(experiment_data_folder, "pred_data"))

if os.path.exists(os.path.join(experiment_data_folder, "bayes_data")) == False:
    os.makedirs(os.path.join(experiment_data_folder, "bayes_data"))

# %%

n_replicates = 10

_prop_NA = 0.25
_d = 5
_corr = 0.65

n_train = 100_000
n_test = 15_000
n = n_train + n_test

N_MC = 1000


# %%

def toep_matrix(d, corr):
    """
    Generate a Toeplitz matrix with correlation corr.
    """
    return np.array([[corr**abs(i-j) for j in range(d)] for i in range(d)])

def generate_X(n, d, corr, mu=None):
    """
    Generate a design matrix X with n rows and d columns, with a correlation of corr.
    """

    if mu is None:
        mu = np.zeros(d)

    cov = toep_matrix(d, corr)
    
    X = np.random.multivariate_normal(mu, cov, size=n)
    
    return X

def generate_M(n, d, prc):
    """
    Generate a missing data matrix M with n rows and d columns, with a proportion of missing data prop_NA.
    It guarantees no row with all missing data.
    """
    M = np.random.binomial(n=1, p=prc, size=(n, d))

    all_ones = np.all(M == 1, axis=1)

    while np.any(all_ones):
        M[all_ones] = np.random.binomial(n=1, p=prc, size=(all_ones.sum(), d))
        all_ones = np.all(M == 1, axis=1)  # Recheck after redrawing

    return M


def get_y_prob_bayes_same_pattern(X_m, full_mu, full_cov, true_beta, n_mc=1000, intercept=0):

    m = np.isnan(X_m[0])
    
    observed_idx = ~m
    missing_idx = m

    mu_obs = full_mu[observed_idx]
    mu_mis = full_mu[missing_idx]

    cov_obs = full_cov[np.ix_(observed_idx, observed_idx)]
    cov_obs_inv = np.linalg.inv(cov_obs)

    cov_mis = full_cov[np.ix_(missing_idx, missing_idx)]
    cross_cov = full_cov[np.ix_(observed_idx, missing_idx)]

    cond_cov = cov_mis - cross_cov.T @ cov_obs_inv @ cross_cov

    prob_y_all = []
    X_mc_all = []

    for x_obs in X_m:
        x_obs_obs = x_obs[observed_idx]

        cond_mu = mu_mis + cross_cov.T @ cov_obs_inv @ (x_obs_obs - mu_obs)

        if len(cond_mu) == 0:
            X_mc = np.zeros((n_mc, 0))
        else:
            X_mc = np.random.multivariate_normal(cond_mu, cond_cov, size=n_mc)

        X_full_mc = np.tile(x_obs, (n_mc, 1))
        X_full_mc[:, missing_idx] = X_mc

        logits_mc = X_full_mc @ true_beta + intercept
        prob_y_mc = sigma(logits_mc)

        prob_y_all.append(prob_y_mc)
        X_mc_all.append(X_mc)

    return np.array(prob_y_all)


set_up_df = pd.DataFrame({
    "sim": [],
    "replicate": [],
    "n": [],
    "d": [],
    "corr": [],
    "prop_NA": [],
    "center_X": [],
    "set_up": []
})



np.random.seed(1)
random.seed(1)


all_betas = {}
for i in range(2**_d):

    pattern = np.array([int(x) for x in np.binary_repr(i, width=_d)])
    all_betas[tuple(pattern)] = np.random.normal(0, 1, _d)

#save the betas
all_betas = pd.DataFrame(all_betas).T
all_betas.columns = [f"beta_{i}" for i in range(_d)]
all_betas["pattern"] = all_betas.index

all_betas.to_csv(os.path.join(experiment_data_folder, "all_betas.csv"), index=False)

# %%
all_betas = pd.read_csv(os.path.join(experiment_data_folder, "all_betas.csv"))


# %%

for i in range(n_replicates):

    print(f"Set up {i+1}/{n_replicates}")

    # generate X
    X = generate_X(n, _d, _corr)

    # generate M
    M = generate_M(n, _d, _prop_NA)
    X_obs = X.copy()
    X_obs[M == 1] = np.nan

    # generate y
    y_logits = np.zeros(n)
    y_prob_bayes = np.zeros(n_test)

    # for each pattern, generate y from betas
    sum_ = 0
    sum_bayes = 0
    for j in range(len(all_betas)):

        pattern = all_betas.iloc[j]["pattern"]
        pattern = np.array([int(x) for x in pattern[1:-1].split(",")])

        beta0 = all_betas.iloc[j][:-1].values
        beta0 = np.array([float(x) for x in beta0])

        idx_pattern = get_index_pattern(list(pattern), M)

        y_logits[idx_pattern] = np.dot(X[idx_pattern], beta0)
        sum_ += len(idx_pattern)

        # for test set [n_train:]
        idx_pattern = get_index_pattern(list(pattern), M[n_train:])
        if len(idx_pattern) > 0:
            y_prob_bayes[idx_pattern] = get_y_prob_bayes_same_pattern(X_obs[n_train:][idx_pattern], np.zeros(_d), toep_matrix(_d, _corr), beta0, N_MC).mean(axis=1)
            sum_bayes += len(idx_pattern)
    
    assert sum_ == n, "The number of samples does not match the number of rows in X"
    assert sum_bayes == n_test, "The number of samples does not match the number of rows in X"

    y_probs = 1 / (1 + np.exp(-y_logits))
    y = np.random.binomial(1, y_probs)

    # create the params
    sim = experiment_name
    rep = i
    n = n_test + n_train
    d = _d
    corr = np.round(_corr*100,0).astype(int)
    prop_NA = np.round(_prop_NA*100,0).astype(int)
    mu0 = np.zeros(_d)
    set_up = f"{sim}_rep{rep}_n{n}_d{d}_corr{corr}_NA{prop_NA}"

    new_row = pd.DataFrame({
        "sim": [sim],
        "replicate": [rep],
        "n": [n],
        "d": [d],
        "corr": [corr],
        "prop_NA": [prop_NA],
        "center_X": [mu0],
        "set_up": [set_up]
    })
    set_up_df = pd.concat([set_up_df, new_row], ignore_index=True)

    data_to_save = {
        "X_obs": X_obs,
        "M": M,
        "y": y,
        "y_probs": y_probs,
        "X_full": X
    }
    np.savez(os.path.join(experiment_data_folder, "original_data", f"{set_up}.npz"), **data_to_save)

    # save test data
    data_to_save = {
        "X_obs": X_obs[n_train:],
        "M": M[n_train:],
        "y": y[n_train:],
        "y_probs": y_probs[n_train:],
        "X_full": X[n_train:]
    }
    np.savez(os.path.join(experiment_data_folder, "test_data", f"{set_up}.npz"), **data_to_save)

    # save bayes data
    data_to_save = {
        "y_probs_bayes": y_prob_bayes
    }
    np.savez(os.path.join(experiment_data_folder, "bayes_data", f"{set_up}.npz"), **data_to_save)


# save the set up
set_up_df.to_csv(os.path.join(experiment_data_folder, "set_up.csv"), index=False)



# %%
