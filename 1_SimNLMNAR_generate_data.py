### 
# SimNL: Non-linear, MCAR, 5 dim
###
# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from utils import *

# %%

experiment_name = "SimNLMNAR"
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

def generate_M(X, intercept, slope): #intercept = -0.5, slope = 1.0
    n, d = X.shape
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / (X_std + 1e-8)

    logits = intercept + slope * X_norm
    probs = sigma(logits)

    M = np.random.binomial(n=1, p=probs)

    all_ones = np.all(M == 1, axis=1)
    
    while np.any(all_ones):
        idx = np.where(all_ones)[0]
        M[idx] = np.random.binomial(n=1, p=probs[idx])
        all_ones = np.all(M == 1, axis=1)

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



# %% Approximate the Bayesian probabilities with a large, XGBoost model
n_bayes_mc = 100000
Z_bayes_mc = generate_Z(n_bayes_mc, _d, _corr)
X_bayes_mc = transform_Z_to_X(Z_bayes_mc)
M_bayes_mc = generate_M(X_bayes_mc, intercept=-0.5, slope=1.0)
y_logits_bayes_mc = np.dot(X_bayes_mc, beta0)
y_probs_bayes_mc = 1 / (1 + np.exp(-y_logits_bayes_mc))
y_bayes_mc = np.random.binomial(1, y_probs_bayes_mc)
X_bayes_mc[M_bayes_mc == 1] = np.nan

# train the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
bayes_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=1)
bayes_model.fit(X_bayes_mc, y_bayes_mc)

test_set_bayes_mc = 20000
Z_test_bayes_mc = generate_Z(test_set_bayes_mc, _d, _corr)
X_test_bayes_mc = transform_Z_to_X(Z_test_bayes_mc)
y_logits_test_bayes_mc = np.dot(X_test_bayes_mc, beta0)
y_probs_test_bayes_mc = 1 / (1 + np.exp(-y_logits_test_bayes_mc))
y_test_bayes_mc = np.random.binomial(1, y_probs_test_bayes_mc)
M_test_bayes_mc = generate_M(X_test_bayes_mc, intercept=-0.5, slope=1.0)
X_test_bayes_mc[M_test_bayes_mc == 1] = np.nan

y_probs_bayes_mc = bayes_model.predict_proba(X_test_bayes_mc)[:,1]
print("Bayes MC AUC:", roc_auc_score(y_test_bayes_mc, y_probs_bayes_mc))

def get_bayes_probs(X,M,beta):

    # for rows with no missing data: use the logistic model
    idx_no_missing = np.where(np.all(M == 0, axis=1))[0]
    y_logits = np.dot(X[idx_no_missing], beta)
    y_probs = 1 / (1 + np.exp(-y_logits))

    # for rows with missing data: use the bayes model
    idx_missing = np.where(np.any(M == 1, axis=1))[0]
    y_probs_bayes = bayes_model.predict_proba(X[idx_missing])[:,1]
    y_probs_full = np.zeros(X.shape[0])
    y_probs_full[idx_no_missing] = y_probs
    y_probs_full[idx_missing] = y_probs_bayes

    return y_probs_full

#%%

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
    M = generate_M(X, intercept=-0.5, slope=1.0)
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
    y_probs_bayes = get_bayes_probs(X_obs, M, beta0)

    data_to_save = {
        "y_probs_bayes": y_probs_bayes
    }
    np.savez(os.path.join(experiment_data_folder, "bayes_data", f"{set_up}.npz"), **data_to_save)


# save the set up
set_up_df.to_csv(os.path.join(experiment_data_folder, "set_up.csv"), index=False)


# %%

import matplotlib.pyplot as plt

data = np.load(os.path.join(experiment_data_folder, "test_data", f"SimNLMNAR_rep0_n115000_d5_corr95_NA25.npz"))
y_probs = data["y_probs"]
M = data["M"]

data_bayes = np.load(os.path.join(experiment_data_folder, "bayes_data", f"SimNLMNAR_rep0_n115000_d5_corr95_NA25.npz"))
y_probs_bayes = data_bayes["y_probs_bayes"]

idx = get_index_pattern(4, M)

plt.figure(figsize=(8,6))
plt.scatter(y_probs[idx], y_probs_bayes[idx], alpha=0.3)
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.show()


#%%
