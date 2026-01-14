#!/usr/bin/env python3
"""
Create visualizations for prior predictive check - Experiment 2
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Paths
prior_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2/prior_predictive")
idata = az.from_netcdf(prior_dir / "prior_predictive.nc")

# Set style
plt.style.use("default")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# Figure 1: Distribution comparison (ECDF and KDE)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ECDF
ax = axes[0]
y_prior = idata.prior_predictive["y"].values.flatten()
y_obs = idata.observed_data["y"].values

# Plot prior predictive ECDF
sorted_prior = np.sort(y_prior)
ecdf_prior = np.arange(1, len(sorted_prior) + 1) / len(sorted_prior)
ax.plot(sorted_prior, ecdf_prior, label="Prior predictive", alpha=0.7, linewidth=2)

# Plot observed ECDF
sorted_obs = np.sort(y_obs)
ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
ax.plot(sorted_obs, ecdf_obs, label="Observed", alpha=0.7, linewidth=2, color="red")

# Add plausible range shading (40-110)
ax.axvspan(40, 110, alpha=0.1, color="green", label="Plausible range")
ax.axvline(0, color="black", linestyle="--", alpha=0.3, label="Zero (implausible)")

ax.set_xlabel("Test Score")
ax.set_ylabel("Cumulative Probability")
ax.set_title("ECDF: Prior Predictive vs Observed")
ax.legend()
ax.grid(True, alpha=0.3)

# KDE
ax = axes[1]

kde_prior = gaussian_kde(y_prior)
kde_obs = gaussian_kde(y_obs)

x_range = np.linspace(-50, 150, 500)
ax.plot(x_range, kde_prior(x_range), label="Prior predictive", alpha=0.7, linewidth=2)
ax.plot(x_range, kde_obs(x_range), label="Observed", alpha=0.7, linewidth=2, color="red")

# Add plausible range shading
ax.axvspan(40, 110, alpha=0.1, color="green", label="Plausible range")
ax.axvline(0, color="black", linestyle="--", alpha=0.3)

ax.set_xlabel("Test Score")
ax.set_ylabel("Density")
ax.set_title("KDE: Prior Predictive vs Observed")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(prior_dir / "prior_predictive_distribution.png")
print(f"Saved: {prior_dir / 'prior_predictive_distribution.png'}")
plt.close()

# Figure 2: Prior hyperparameter distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

params = ["alpha_0", "tau_alpha", "beta", "sigma"]
titles = [
    "alpha_0: Population mean intercept",
    "tau_alpha: SD of school intercepts",
    "beta: Treatment effect",
    "sigma: Residual SD"
]
expected = [77, None, 5, None]

for i, (param, title, exp) in enumerate(zip(params, titles, expected)):
    ax = axes[i // 2, i % 2]
    values = idata.prior[param].values.flatten()

    ax.hist(values, bins=50, density=True, alpha=0.6, edgecolor="black")
    ax.axvline(values.mean(), color="blue", linestyle="--", linewidth=2, label=f"Mean: {values.mean():.1f}")

    if exp is not None:
        ax.axvline(exp, color="red", linestyle="--", linewidth=2, label=f"Expected: {exp}")

    # Mark problematic regions
    if param == "tau_alpha" or param == "sigma":
        # These should be positive and reasonable
        ax.axvline(0, color="black", linestyle=":", alpha=0.5)
        prob_negative = np.mean(values < 0)
        if prob_negative > 0:
            ax.text(0.05, 0.95, f"P(< 0) = {prob_negative:.3f}",
                   transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="wheat"))

    ax.set_xlabel(param)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(prior_dir / "prior_hyperparameters.png")
print(f"Saved: {prior_dir / 'prior_hyperparameters.png'}")
plt.close()

# Figure 3: Extreme values analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Distribution of min/max per chain
ax = axes[0]
y_prior_per_chain = idata.prior_predictive["y"].values  # shape: (chains, draws, N)
mins = y_prior_per_chain.min(axis=2).flatten()
maxs = y_prior_per_chain.max(axis=2).flatten()

ax.hist(mins, bins=50, alpha=0.6, label="Min values", edgecolor="black")
ax.hist(maxs, bins=50, alpha=0.6, label="Max values", edgecolor="black")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (implausible)")
ax.axvline(y_obs.min(), color="green", linestyle="--", linewidth=2, label=f"Obs min: {y_obs.min():.1f}")
ax.axvline(y_obs.max(), color="blue", linestyle="--", linewidth=2, label=f"Obs max: {y_obs.max():.1f}")

ax.set_xlabel("Test Score")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Extremes Across Prior Draws")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Percentage of implausible values
ax = axes[1]
categories = ["< 0\n(impossible)", "< 40\n(very low)", "40-110\n(plausible)", "> 110\n(very high)"]
percentages = [
    100 * np.mean(y_prior < 0),
    100 * np.mean((y_prior >= 0) & (y_prior < 40)),
    100 * np.mean((y_prior >= 40) & (y_prior <= 110)),
    100 * np.mean(y_prior > 110)
]
colors = ["red", "orange", "green", "orange"]

bars = ax.bar(categories, percentages, color=colors, alpha=0.6, edgecolor="black")
ax.set_ylabel("Percentage (%)")
ax.set_title("Prior Predictive Distribution by Plausibility")
ax.grid(True, alpha=0.3, axis="y")

# Add percentage labels on bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
           f'{pct:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(prior_dir / "prior_extremes.png")
print(f"Saved: {prior_dir / 'prior_extremes.png'}")
plt.close()

# Summary statistics
print("\n" + "="*70)
print("PLAUSIBILITY ASSESSMENT")
print("="*70)
print(f"\nPrior predictive range: [{y_prior.min():.1f}, {y_prior.max():.1f}]")
print(f"Observed data range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
print("\nPercentage of prior predictive values:")
print(f"  < 0 (impossible): {100 * np.mean(y_prior < 0):.2f}%")
print(f"  < 40 (very low): {100 * np.mean((y_prior >= 0) & (y_prior < 40)):.2f}%")
print(f"  40-110 (plausible): {100 * np.mean((y_prior >= 40) & (y_prior <= 110)):.2f}%")
print(f"  > 110 (very high): {100 * np.mean(y_prior > 110):.2f}%")
print("\n" + "="*70)
