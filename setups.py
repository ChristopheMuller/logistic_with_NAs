import numpy as np
import matplotlib.pyplot as plt

color_palette = plt.cm.tab10.colors

uncertainties = {
    "sd": np.std,
    "se": lambda x: np.std(x) / np.sqrt(len(x)),
}

uncertainties_config = {
    "sd": {"label": "St. Dev."},
    "se": {"label": "St. Err."},
}

metrics_set_up = {
    "brier_score": lambda true, pred, true_y, bayes: np.mean(((true >= 0.5).astype(int) - pred)**2),
    "missclassification_error": lambda true, pred, true_y, bayes: 1 - np.mean(true_y == (pred >= 0.5)),
    "mae_score": lambda true, pred, true_y, bayes: np.mean(np.abs(true - pred)),
    "mae_score_bayes": lambda true, pred, true_y, bayes: np.mean(np.abs(bayes - pred)),
}

variable_config = {
    "corr": {"label": "Correlation"},
    "prcNA": {"label": "Missingness"},
    "prop1": {"label": "Proportion of 1"},
    "d": {"label": "Dimensions"},
    "centricity": {"label": "Centricity"},
}

metrics_config = {
    "angular_error": {"label": "Angular Error"},
    "mse_error": {"label": "MSE"},
    "missclassification_error": {"label": "Missclassification"},
    "brier_score": {"label": "Brier Score"},
    "mae_score": {"label": "MAE"},
    "running_time": {"label": "Running Time (s.)"},
    "mae_score_bayes": {"label": "Mean Absolute Distance to Bayes Probs"},
}

methods_config = {
    "05.IMP": {"label": "05.IMP", "color": color_palette[0], "linestyle": "-", "marker":"o"},
    "05.IMP.M": {"label": "05.IMP.M", "color": color_palette[0], "linestyle": "--", "marker":"x"},
    "CC": {"label": "CC", "color": color_palette[1], "linestyle": "-", "marker":"o"},
    # "ICE.IMP": {"label": "ICE.IMP", "color": color_palette[2], "linestyle": "-", "marker":"o"},
    # "ICE.IMP.M": {"label": "ICE.IMP.M", "color": color_palette[2], "linestyle": "--", "marker":"x"},
    # "ICEY.IMP": {"label": "ICEY.IMP", "color": color_palette[3], "linestyle": "-", "marker":"o"},
    # "ICEY.IMP.M": {"label": "ICEY.IMP.M", "color": color_palette[3], "linestyle": "--", "marker":"x"},
    "Mean.IMP": {"label": "Mean.IMP", "color": color_palette[5], "linestyle": "-", "marker":"o"},
    "Mean.IMP.M": {"label": "Mean.IMP.M", "color": color_palette[5], "linestyle": "--", "marker":"x"},
    "PbP": {"label": "PbP", "color": color_palette[6], "linestyle": "-", "marker":"o"},
    "SAEM": {"label": "SAEM", "color": color_palette[7], "linestyle": "-", "marker":"o"},
    "MICE.IMP": {"label": "MICE.IMP", "color": color_palette[2], "linestyle": "-", "marker":"o"},
    "MICE.5.IMP": {"label": "MICE.5.IMP", "color": color_palette[2], "linestyle": "--", "marker":"x"},
    "MICE.10.Y.IMP": {"label": "MICE.10.Y.IMP", "color": color_palette[2], "linestyle": ":", "marker":"^"},
    "MICE.100.Y.IMP": {"label": "MICE.100.Y.IMP", "color": color_palette[2], "linestyle": "-.", "marker":"v"},
    "MICE.M.IMP": {"label": "MICE.IMP.M", "color": color_palette[4], "linestyle": "--", "marker":"x"},
    "MICE.Y.IMP": {"label": "MICE.Y.IMP", "color": color_palette[9], "linestyle": "-", "marker":"o"},
    "MICE.Y.M.IMP": {"label": "MICE.Y.IMP.M", "color": color_palette[9], "linestyle": "--", "marker":"x"},
    "PY.SAEM": {"label": "PY.SAEM", "color": color_palette[8], "linestyle": "-", "marker":"o"},
}

methods_no_beta_estimate = [
    "PbP",
]

methods_no_pred_estimate = [
    "CC",
]