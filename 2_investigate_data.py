
# %%

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% SIM A - distribution of p

sim = "SimulationA"

set_ups = pd.read_csv(os.path.join("data", sim, "set_up.csv"))

all_p = []

for i in range(len(set_ups)):

    set_up = set_ups.loc[i, "set_up"]
    p_i = load_data(set_up=set_up, data_type="original", exp=sim)["y_probs"]
    
    all_p.append(p_i)

all_p = np.array(all_p)

plt.hist(all_p.flatten(), bins=200)
plt.xlabel("p")
plt.ylabel("Frequency")
plt.title("Distribution of p")
plt.show()

for i in range(len(set_ups)):

    set_up = set_ups.loc[i, "set_up"]
    p_i = load_data(set_up=set_up, data_type="original", exp=sim)["y_probs"]
    
    plt.hist(p_i.flatten(), bins=50, alpha=0.5, label=set_up)
    plt.show()

# %%


# simulations_df = pd.read_csv(os.path.join("data", sim, "simulation.csv"))
# score_matrix = pd.read_csv(os.path.join("data", sim, "score_matrix.csv"))

# all_methods = score_matrix["method"].unique()
# print(all_methods)

# methods_to_remove = [
#     "MICE.Y.M.IMP",
#     "MICE.M.IMP"
# ]

# # Remove methods from the score matrix
# print(score_matrix.shape)

# score_matrix = score_matrix[~score_matrix["method"].isin(methods_to_remove)]
# print(score_matrix["method"].unique())

# print(score_matrix.shape)
# print(score_matrix.head())

# # save score matrix

# score_matrix.to_csv(os.path.join("data", sim, "score_matrix.csv"), index=False)

# # loop over the simulation df:
# # if method is in methods to remove => remove the file corresponding (name of the file in "file_name" column)

# simulation_with_methods = simulations_df[simulations_df["method"].isin(methods_to_remove)]
# print(simulation_with_methods.shape)

# sum_ = 0
# for i in range(len(simulations_df)):

#     method = simulations_df.loc[i, "method"]
    
#     if method in methods_to_remove:
#         file_name = simulations_df.loc[i, "file_name"]
#         # print(file_name)

#         full_file_path = os.path.join("data", sim, "pred_data", f"{file_name}.npz")
#         # print(full_file_path)

#         if os.path.exists(full_file_path):
#             sum_ += 1
#             os.remove(full_file_path)
#             print(f"Removed {full_file_path}")

# print(sum_)

# simulations_df = simulations_df[~simulations_df["method"].isin(methods_to_remove)]
# simulations_df.to_csv(os.path.join("data", sim, "simulation.csv"), index=False)