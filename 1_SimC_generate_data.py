
# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from utils import *

# %%

experiment_name = "SimulationC"
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

N_MC = 1000

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

def generate_X(n, d, corr, mu=None):
    """
    Generate a design matrix X with n rows and d columns, with a correlation of corr.
    """

    if mu is None:
        mu = np.zeros(d)

    cov = toep_matrix(d, corr)
    
    X = np.random.multivariate_normal(mu, cov, size=n)
    
    return X

def generate_Z(X):

    Z = np.zeros_like(X)

    Z[:, 0] = X[:,0]
    Z[:, 1] = X[:,1]
    
    Z[:, 2] = np.exp(X[:,2])
    Z[:, 3] = np.power(X[:,3], 3)
    Z[:, 4] = np.where(X[:,4] >= 0, X[:,4]**2, -10*np.exp(X[:,4]))

    return Z

def transform_Z_to_X(Z):

    X = np.zeros((Z.shape[0], Z.shape[1]))

    X[:,0] = Z[:,0]
    X[:,1] = Z[:,1]
    
    X[:,2] = np.log(Z[:,2])
    X[:,3] = np.power(Z[:,3], 3)
    X[:,4] = np.where(Z[:,4] > 0, np.sqrt(Z[:,4]), -np.log(-Z[:,4]/10))

    return X

def transform_X_to_Z(X):

    Z = np.zeros((X.shape[0], X.shape[1]))

    Z[:, 0] = X[:,0]
    Z[:, 1] = X[:,1]
    
    Z[:, 2] = np.exp(X[:,2])
    Z[:, 3] = np.power(X[:,3], 1/3)
    Z[:, 4] = np.where(X[:,4] >= 0, X[:,4]**2, -10*np.exp(X[:,4]))

    return Z

# %% 

