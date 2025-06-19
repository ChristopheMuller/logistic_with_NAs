####
# SimF3: GPMM, MNAR, 5 dimension, few patterns
####
# %%

import numpy as np
import pandas as pd
import random
import os
from utils import *

# %%

experiment_name = "SimF3"
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
_d =5
_max_var = 1.
_var_mu = 0.5

n_train = 100_000
n_test = 15_000
n = n_train + n_test

N_MC = 10_000

n_patterns = 3

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

def generate_M_specific_patterns(n, patterns):

    d = patterns.shape[1]
    n_patterns = patterns.shape[0]

    base_repeats = n // n_patterns
    remainder = n % n_patterns

    pattern_indices = np.array([i for i in range(n_patterns) for _ in range(base_repeats)] +
                               list(range(remainder)))

    np.random.shuffle(pattern_indices)

    M = patterns[pattern_indices]
    M = M.astype(int)
    assert M.shape == (n, d), "The shape of M does not match the expected shape."

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

# %%

np.random.seed(1)
random.seed(1)

beta0 = np.random.normal(0, 1, _d)

print("beta0", beta0)

all_mus = {}
all_corrs = {}
all_vars = {}

patterns_selected = (np.random.rand(n_patterns, _d) < 0.25).astype(int)
# if pattern fully observed (only 0), sample again:
for i in range(n_patterns):
    while np.all(patterns_selected[i] == 0):
        patterns_selected[i] = (np.random.rand(_d) < 0.25).astype(int)
patterns_df = pd.DataFrame(patterns_selected)
patterns_df.columns = [f"pattern_{i}" for i in range(_d)]
patterns_df.to_csv(os.path.join(experiment_data_folder, "patterns.csv"), index=False)

for pattern in patterns_selected:

    mu_pattern = np.random.normal(0, np.sqrt(_var_mu), _d)
    corr_pattern = np.random.uniform(-1, 1)
    var_pattern = np.random.uniform(0, _max_var)


    print("Pattern:", pattern[:3], "...", pattern[-3:])
    print("\tMu:", np.round(mu_pattern[:3],2), "...", np.round(mu_pattern[-3:],2))
    print("\tCorr:", np.round(corr_pattern, 2))
    print("\tVar:", np.round(var_pattern, 2))
    
    
    all_mus[tuple(pattern)] = mu_pattern
    all_corrs[tuple(pattern)] = corr_pattern
    all_vars[tuple(pattern)] = var_pattern
    


# save the Mus and Corrs
all_mus_df = pd.DataFrame(all_mus).T
all_mus_df.columns = [f"mu_{i}" for i in range(_d)]
all_mus_df["pattern"] = all_mus_df.index
all_mus_df.to_csv(os.path.join(experiment_data_folder, "all_mus.csv"), index=False)

all_corrs_df = pd.Series(all_corrs)
index = all_corrs_df.index
all_corrs_df = pd.DataFrame(all_corrs_df).reset_index(drop=True)
all_corrs_df["pattern"] = index
all_corrs_df.columns = ["corr", "pattern"]
all_corrs_df.to_csv(os.path.join(experiment_data_folder, "all_corrs.csv"), index=False)

all_vars_df = pd.Series(all_vars)
index = all_vars_df.index
all_vars_df = pd.DataFrame(all_vars_df).reset_index(drop=True)
all_vars_df["pattern"] = index
all_vars_df.columns = ["var", "pattern"]
all_vars_df.to_csv(os.path.join(experiment_data_folder, "all_vars.csv"), index=False)


# %%


for i in range(n_replicates):

    print(f"Set up {i+1}/{n_replicates}")

    # Generate M
    M = generate_M_specific_patterns(n, patterns_selected)

    # generate X: mu and corr based on the pattern of M
    X = np.zeros_like(M, dtype=float)
    unique_patterns = np.unique(M, axis=0)
    assert unique_patterns.shape[0] <= n_patterns, "The number of unique patterns exceeds the number of patterns."
    total_pats = 0
    for pat in unique_patterns:

        rows_with_pat = np.all(M == pat, axis=1)
        mu_pat = all_mus[tuple(pat)]
        corr_pat = all_corrs[tuple(pat)]
        var_pat = all_vars[tuple(pat)]
        cov_pat = toep_matrix(_d, corr_pat) * var_pat
        X_temp = np.random.multivariate_normal(mu_pat, cov_pat, size=np.sum(rows_with_pat))

        X[rows_with_pat] = X_temp

        total_pats += np.sum(rows_with_pat)

    assert total_pats == n, "The number of rows with the same pattern does not match the total number of rows."  

    # generate y
    y_logits = np.dot(X, beta0)
    y_probs = 1 / (1 + np.exp(-y_logits))
    y = np.random.binomial(1, y_probs)

    # Mask X
    X_obs = X.copy()
    X_obs[M == 1] = np.nan

    # create the params
    sim = experiment_name
    rep = i
    n = n_test + n_train
    d = _d
    corr = "MIXTURE"
    prop_NA = np.round(_prop_NA*100,0).astype(int)
    beta0 = beta0
    mu0 = np.zeros(_d)
    set_up = f"{sim}_rep{rep}_n{n}_d{d}_corr{corr}_NA{prop_NA}"

    # save the data
    new_row = pd.DataFrame({
        "sim": [sim],
        "replicate": [rep],
        "n": [n],
        "d": [d],
        "corr": [corr],
        "prop_NA": [prop_NA],
        "true_beta": [beta0],
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
    y_probs_bayes = np.zeros(n_test)
    total_pats = 0
    for pat in unique_patterns:
        rows_with_pat = np.all(M[n_train:] == pat, axis=1)
        mu_pat = all_mus[tuple(pat)]
        corr_pat = all_corrs[tuple(pat)]
        var_pat = all_vars[tuple(pat)]
        cov_pat = toep_matrix(_d, corr_pat) * var_pat

        total_pats += np.sum(rows_with_pat)

        y_probs_bayes_pat = get_y_prob_bayes_same_pattern(
            X_obs[n_train:][rows_with_pat],
            mu_pat,
            cov_pat,
            beta0,
            n_mc=N_MC
        )

        y_probs_bayes[rows_with_pat] = np.mean(y_probs_bayes_pat, axis=1)

    assert total_pats == n_test, "The number of rows with the same pattern does not match the number of test rows."

    data_to_save = {
        "y_probs_bayes": y_probs_bayes
    }
    np.savez(os.path.join(experiment_data_folder, "bayes_data", f"{set_up}.npz"), **data_to_save)


# save the set up
set_up_df.to_csv(os.path.join(experiment_data_folder, "set_up.csv"), index=False)


# %%

# import matplotlib.pyplot as plt

# bayes_data = np.load("data/SimE/bayes_data/SimE_rep0_n115000_d25_corrMIXTURE_NA25.npz")
# test_data = np.load("data/SimE/test_data/SimE_rep0_n115000_d25_corrMIXTURE_NA25.npz")

# M = test_data["M"]
# y_probs = test_data["y_probs"]
# y_probs_bayes = bayes_data["y_probs_bayes"]

# idx = get_index_pattern(None, M)

# #array([ 5,  6,  4,  6,  8,  5, 11, 11])

# plt.scatter(y_probs[idx], y_probs_bayes[idx], alpha=0.1)
# plt.hist(y_probs[idx])