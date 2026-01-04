"""Diagnostic plots for prior predictive checks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_diagnostic_plots(
    alpha: np.ndarray,
    beta_weight: np.ndarray,
    sigma: np.ndarray,
    mpg_rep: np.ndarray,
    obs_mpg: np.ndarray,
    output_dir: Path,
) -> None:
    """Create and save prior predictive diagnostic plots."""
    obs_min, obs_max = obs_mpg.min(), obs_mpg.max()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1: alpha prior
    ax = axes[0, 0]
    ax.hist(alpha, bins=50, density=True, alpha=0.7)
    ax.axvline(3.1, color="red", linestyle="--", label="Prior mean")
    ax.axvline(np.log(np.mean(obs_mpg)), color="green", linestyle=":",
               label=f"log(mean MPG)={np.log(np.mean(obs_mpg)):.2f}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Density")
    ax.set_title("Prior: alpha ~ Normal(3.1, 0.3)")
    ax.legend(fontsize=8)

    # Plot 2: beta_weight prior
    ax = axes[0, 1]
    ax.hist(beta_weight, bins=50, density=True, alpha=0.7, color="orange")
    ax.axvline(-1, color="red", linestyle="--", label="Prior mean = -1")
    ax.set_xlabel("beta_weight")
    ax.set_ylabel("Density")
    ax.set_title("Prior: beta_weight ~ Normal(-1, 0.3)")
    ax.legend(fontsize=8)

    # Plot 3: MPG histogram comparison
    ax = axes[1, 0]
    mpg_flat = mpg_rep.flatten()
    mpg_sample = np.random.choice(mpg_flat, size=min(50000, len(mpg_flat)), replace=False)
    ax.hist(mpg_sample, bins=100, density=True, alpha=0.7, color="steelblue",
            range=(0, 100), label="Prior predictive")
    ax.hist(obs_mpg, bins=30, density=True, alpha=0.5, color="orange", label="Observed")
    ax.axvline(obs_min, color="red", linestyle="--", alpha=0.7)
    ax.axvline(obs_max, color="red", linestyle="--", alpha=0.7,
               label=f"Observed range [{obs_min:.0f}, {obs_max:.0f}]")
    ax.set_xlabel("MPG")
    ax.set_ylabel("Density")
    ax.set_title("Prior Predictive Distribution (MPG scale)")
    ax.set_xlim(0, 80)
    ax.legend(fontsize=8)

    # Plot 4: ECDF comparison
    ax = axes[1, 1]
    n_draws_plot = 100
    draw_indices = np.random.choice(len(mpg_rep), n_draws_plot, replace=False)
    for i in draw_indices:
        sorted_vals = np.sort(mpg_rep[i])
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, ecdf, alpha=0.05, color="steelblue")

    sorted_obs = np.sort(obs_mpg)
    ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
    ax.plot(sorted_obs, ecdf_obs, color="orange", linewidth=2, label="Observed")
    ax.set_xlabel("MPG")
    ax.set_ylabel("ECDF")
    ax.set_title("Prior Predictive ECDF (100 draws) vs Observed")
    ax.set_xlim(0, 80)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "prior_predictive_check.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Sigma plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sigma, bins=50, density=True, alpha=0.7, color="purple")
    ax.axvline(0.2, color="red", linestyle="--", label="Expected ~0.2")
    ax.set_xlabel("sigma")
    ax.set_ylabel("Density")
    ax.set_title("Prior: sigma ~ Exponential(5)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prior_sigma.png", dpi=150, bbox_inches="tight")
    plt.close()
