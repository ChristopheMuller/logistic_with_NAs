"""Plot the complete covariates from the first non-linear MCAR replicate.

Run from any directory with ``python tables_and_figures/figure_nonlinear_features.py``.
The simulation seed and draws match 1_SimNL_generate_data.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    n, d, rho = 115_000, 5, 0.95
    rng = np.random.RandomState(1)
    rng.normal(0, 1, d)  # The simulation draws its coefficients first.
    covariance = rho ** np.abs(np.arange(d)[:, None] - np.arange(d))
    z = rng.multivariate_normal(np.zeros(d), covariance, size=n)

    x = z.copy()
    x[:, 2] = np.exp(z[:, 2]) - 1.67
    x[:, 3] = z[:, 3] ** 3
    x[:, 4] = np.where(z[:, 4] >= 0, z[:, 4] ** 2, -10 * np.exp(z[:, 4])) + 2

    fig, axes = plt.subplots(1, 4, figsize=(16.7, 4.8))
    titles = ("Gaussian", "Exponential", "Cubic", "Non-Monotonic")
    for j, (ax, title) in enumerate(zip(axes, titles), start=1):
        ax.scatter(x[:, 0], x[:, j], s=2, alpha=0.2, linewidths=0)
        ax.set_title(f"{title} (X{j + 1})")
        ax.set_xlabel("X1")
        ax.set_ylabel(f"X{j + 1}")
        ax.grid(True)

    fig.tight_layout()
    output = Path(__file__).resolve().parent / "SimNL_Xi_vs_X0.jpg"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
