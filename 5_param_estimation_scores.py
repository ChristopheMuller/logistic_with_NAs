
# %%

from utils import *
import numpy as np
import pandas as pd

# %%

exp = "SimulationA"

simulation_set_up = pd.read_csv(os.path.join("data",exp,"simulation_set_up.csv"))

# %%

import re

def parse_vector_from_string(s):
    """Parses a vector stored as a string in a CSV cell."""

    if pd.isnull(s):
        return None
    
    s = s.strip()
    
    # Remove brackets if present
    s = s.lstrip("[").rstrip("]")
    
    # Try parsing with commas first
    if "," in s:
        values = [float(x) for x in s.split(",")]
    else:
        # Use regex to extract numbers (handles spaces between numbers)
        values = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
    
    return np.array(values)

# %%

def compute_beta_estimation_error(true_beta, pred_beta, error="mse"):

    true_beta = parse_vector_from_string(true_beta)
    pred_beta = parse_vector_from_string(pred_beta)

    if pred_beta is None:
        return None

    if error == "mse":
        return np.mean((np.array(true_beta) - np.array(pred_beta)) ** 2)
    
    if error == "angular":
        return np.arccos(np.dot(true_beta, pred_beta) / (np.linalg.norm(true_beta) * np.linalg.norm(pred_beta))) ** 2
    
# %%

simulation_set_up["angular_error"] = simulation_set_up.apply(lambda x: compute_beta_estimation_error(x["true_beta"], x["pred_beta"], "angular"), axis=1)
simulation_set_up["mse_error"] = simulation_set_up.apply(lambda x: compute_beta_estimation_error(x["true_beta"], x["pred_beta"], "mse"), axis=1)

# %%

simulation_set_up.to_csv(os.path.join("data",exp,"simulation_set_up.csv"), index=False)

# %%
