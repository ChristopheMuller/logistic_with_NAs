
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
