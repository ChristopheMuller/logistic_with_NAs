
# %%

import numpy as np
import pandas as pd
import os

# %%

exp = "SimA"

df_set_up = pd.read_csv(os.path.join("data",exp,"set_up.csv"))
df_simulations = pd.read_csv(os.path.join("data",exp,"simulation.csv"))


# %%

# some cols of set_up not in simulations
if not np.all([df_set_up.columns[i] in df_simulations.columns for i in range(len(df_set_up.columns))]):
    df_simulations_enlarged = pd.merge(left=df_simulations, right=df_set_up, on="set_up")

# %%

import ast
def deal_with_estimated_beta(beta_str, method, d=5):

    if type(beta_str) != str:
        return None

    beta_list = ast.literal_eval(beta_str)

    if len(beta_list[0]) == d:
        return beta_list[0]
    
    if len(beta_list[0]) == 2*d:
        return beta_list[0][:d]
    
    else:
        raise ValueError(f"The length of beta_list is not 5 or 10 -- Method {method}")
    
def deal_with_estimated_intercept(beta_str):
    if type(beta_str) != str:
        return None
    
    beta_list = ast.literal_eval(beta_str)
    
    # The intercept is the second element of the outer list and is a single-element list
    return beta_list[1][0] # Access the value inside the single-element list


df_simulations_enlarged["pred_beta"] = df_simulations_enlarged.apply(lambda x: deal_with_estimated_beta(x["estimated_beta"], x["method"], d=np.round(x["d"]).astype(int)), axis=1)
df_simulations_enlarged["pred_intercept"] = df_simulations_enlarged.apply(lambda x: deal_with_estimated_intercept(x["estimated_beta"]), axis=1)
df_simulations_enlarged.to_csv(os.path.join("data",exp,"simulation_set_up.csv"), index=False)

# %%

# fix your results if necessary

def filter_simulations(method, n_train, exp):

    df_simulations_temp = pd.read_csv(os.path.join("data",exp,"simulation.csv"))
    df_simulations_copy = df_simulations_temp.copy()
    all_n_train = df_simulations_temp["n_train"].unique()
    if n_train is None:
        n_train = all_n_train

    df_simulations_temp = df_simulations_temp[df_simulations_temp["n_train"].isin(n_train)]
    df_simulations_temp = df_simulations_temp[df_simulations_temp["method"].isin(method)]

    for i, row in df_simulations_temp.iterrows():
        file_name = row["file_name"]

        # delete the file
        if os.path.exists(os.path.join("data",exp,"pred_data", f"{file_name}.npz")):
            os.remove(os.path.join("data",exp,"pred_data", f"{file_name}.npz"))

        # delete row in the dataframe
        df_simulations_copy = df_simulations_copy.drop(i)

        print(f"{i}..Deleted {file_name}")

    df_simulations_copy.to_csv(os.path.join("data",exp,"simulation.csv"), index=False)

import os
import pandas as pd

# make a copy of all files with characterisitcs
method = ["MICE.100.Y.IMP"]
n_train = None

exp = "ExpA"
folder_goal = os.path.join("data",exp,"temp2")

if not os.path.exists(folder_goal):
    os.makedirs(folder_goal)

def copy_files(method, n_train, exp, folder_goal):

    df_simulations_temp = pd.read_csv(os.path.join("data",exp,"simulation.csv"))
    all_n_train = df_simulations_temp["n_train"].unique()
    if n_train is None:
        n_train = all_n_train

    df_simulations_temp = df_simulations_temp[df_simulations_temp["n_train"].isin(n_train)]
    df_simulations_temp = df_simulations_temp[df_simulations_temp["method"].isin(method)]

    for i, row in df_simulations_temp.iterrows():
        file_name = row["file_name"]

        # copy the file
        if os.path.exists(os.path.join("data",exp,"temp", f"{file_name}.npz")):
            os.system(f"cp {os.path.join('data',exp,'temp', f'{file_name}.npz')} {os.path.join(folder_goal, f'{file_name}.npz')}")

        print(f"{i}..Copied {file_name}")

