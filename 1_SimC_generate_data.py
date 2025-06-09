
# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from utils import *

# %%

experiment_name = "SimC"
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
_corr = 0.95

n_train = 100_000
n_test = 15_000
n = n_train + n_test

N_MC = 10_000

# %%

np.random.seed(1)
random.seed(1)

beta0 = np.random.normal(0, 1.0, _d)

print("beta0", beta0)

# %%

def toep_matrix(d, corr):
    """
    Generate a Toeplitz matrix with correlation corr.
    """
    return np.array([[corr**abs(i-j) for j in range(d)] for i in range(d)])

def generate_Z(n, d, corr, mu=None):
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

def transform_X_to_Z(X):
    """
    Transforms Z back to X based on the inverse operations.
    Handles potential issues with domains of operations (log, sqrt, fractional powers)
    and ensures correct sign recovery, accounting for shifts introduced in transform_X_to_Z.
    """
    Z = np.zeros_like(X, dtype=float) # Ensure output is float and matches Z's shape

    Z[:,0] = X[:,0]
    Z[:,1] = X[:,1]
    
    # Inverse for Z[:,2] = np.exp(X[:,2]) - 1.67
    Z[:,2] = np.log(X[:,2] + 1.67)

    Z[:,3] = np.sign(X[:,3]) * np.power(np.abs(X[:,3]), 1/3)

    # Inverse for Z[:,4] = np.where(...) + 2
    X_prime_col4 = X[:,4] - 2

    mask_X4_positive = X_prime_col4 > 0
    mask_X4_zero = X_prime_col4 == 0
    mask_X4_negative = X_prime_col4 < 0

    # If X_prime_col4 > 0, then X[:,4] was >= 0. Inverse: X[:,4] = sqrt(Z_prime_col4)
    Z[mask_X4_positive, 4] = np.sqrt(X_prime_col4[mask_X4_positive])
    
    # If X_prime_col4 == 0, then Z[:,4] was 0. Inverse: X[:,4] = 0.0
    Z[mask_X4_zero, 4] = 0.0
    
    # If X_prime_col4 < 0, then X[:,4] was < 0. Inverse: X[:,4] = log(-Z_prime_col4 / 10)
    Z[mask_X4_negative, 4] = np.log(-X_prime_col4[mask_X4_negative] / 10)

    return Z

def transform_Z_to_X(Z):
    """
    Transforms X to Z based on the specified piecewise and power transformations.
    (This function is provided by the user and is the 'target' for inversion).
    """
    X = np.zeros_like(Z, dtype=float)

    X[:, 0] = Z[:,0]
    X[:, 1] = Z[:,1]
    
    X[:, 2] = np.exp(Z[:,2]) - 1.67 # Added constant
    
    X[:, 3] = np.power(Z[:,3], 3)
    
    X[:, 4] = np.where(Z[:,4] >= 0, Z[:,4]**2, -10*np.exp(Z[:,4])) + 2 # Added constant

    return X


def get_y_prob_bayes(X_m, full_mu, full_cov, true_beta, n_mc=1000, intercept=0):

    M = np.isnan(X_m)
    unique_patterns = np.unique(M, axis=0)
    
    prob_y_all = np.zeros((X_m.shape[0], n_mc))
    
    for pattern in unique_patterns:

        pattern_indices = np.all(M == pattern, axis=1)
        X_m_subset = X_m[pattern_indices]
        
        prob_y_subset = get_y_prob_bayes_same_pattern(X_m_subset, full_mu, full_cov, true_beta, n_mc, intercept)
        
        prob_y_all[pattern_indices] = prob_y_subset
    
    return prob_y_all

def get_y_prob_bayes_same_pattern(X_m, full_mu, full_cov, true_beta, n_mc=1000, intercept=0):

    m = np.isnan(X_m[0])

    Z_m = transform_X_to_Z(X_m)
    
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

    for z_m in Z_m:
        z_m_obs = z_m[observed_idx]
        cond_mu = mu_mis + cross_cov.T @ cov_obs_inv @ (z_m_obs - mu_obs)

        if len(cond_mu) == 0:
            Z_mc = np.zeros((n_mc, 0))
        else:
            Z_mc = np.random.multivariate_normal(cond_mu, cond_cov, size=n_mc)

        Z_full_mc = np.tile(z_m, (n_mc, 1))
        Z_full_mc[:, missing_idx] = Z_mc

        X_full_mc = transform_Z_to_X(Z_full_mc)

        logits_mc = X_full_mc @ true_beta + intercept
        prob_y_mc = sigma(logits_mc)

        prob_y_all.append(prob_y_mc)

    return np.array(prob_y_all)

# %% 

set_up_df = pd.DataFrame({
    "sim": [],
    "replicate": [],
    "n": [],
    "d": [],
    "corr": [],
    "prop_NA": [],
    "true_beta": [],
    "center_X": [],
    "set_up": []
})


for i in range(n_replicates):

    print(f"Set up {i+1}/{n_replicates}")

    # generate X, Z
    Z = generate_Z(n, _d, _corr)
    X = transform_Z_to_X(Z)

    # generate y
    y_logits = np.dot(X, beta0)
    y_probs = 1 / (1 + np.exp(-y_logits))
    y = np.random.binomial(1, y_probs)

    # generate M
    M = generate_M(n, _d, _prop_NA)
    Z_obs = Z.copy()
    Z_obs[M == 1] = np.nan
    X_obs = X.copy()
    X_obs[M == 1] = np.nan

    # create the params
    sim = experiment_name
    rep = i
    n = n_test + n_train
    d = _d
    corr = np.round(_corr*100,0).astype(int)
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
    y_probs_bayes = get_y_prob_bayes(X_obs[n_train:], np.zeros(_d), toep_matrix(_d, _corr), beta0, N_MC)
    y_probs_bayes = np.mean(y_probs_bayes, axis=1)

    data_to_save = {
        "y_probs_bayes": y_probs_bayes
    }
    np.savez(os.path.join(experiment_data_folder, "bayes_data", f"{set_up}.npz"), **data_to_save)


# save the set up
set_up_df.to_csv(os.path.join(experiment_data_folder, "set_up.csv"), index=False)


# %%