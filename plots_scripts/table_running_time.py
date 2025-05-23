#####
#
# Summary of running time
#
#####

# %% load packages

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

exp = "SimulationA"

simulation_setup_df = pd.read_csv(os.path.join("data", exp, "simulation_set_up.csv"))[["set_up", "method", "n_train", "running_time"]]

simulation_setup_df = simulation_setup_df.groupby(["method", "n_train"]).agg({"running_time": ["mean", "std", "count"]}).reset_index()
simulation_setup_df.columns = ["method", "n_train", "mean", "sd", "count"]

methods_sel = [
    "05.IMP",
    "05.IMP.M",
    "CC",
    "Mean.IMP",
    "Mean.IMP.M",
    "MICE.IMP",
    "MICE.M.IMP",
    "MICE.Y.IMP",
    "MICE.Y.M.IMP",
    "MICE.10.Y.IMP",
    "MICE.100.Y.IMP",
    "PbP",
    "SAEM"
]

n_trains = [500, 5000, 50_000]

# Filter the data
simulation_setup_df = simulation_setup_df[simulation_setup_df["method"].isin(methods_sel)]
simulation_setup_df = simulation_setup_df[simulation_setup_df["n_train"].isin(n_trains)]

# make a table (cols = n-train, rows = methods)
simulation_setup_df_pivot = simulation_setup_df.pivot(index="method", columns="n_train", values="mean")
simulation_setup_df_pivot = simulation_setup_df_pivot.reset_index()

simulation_setup_df_pivot

