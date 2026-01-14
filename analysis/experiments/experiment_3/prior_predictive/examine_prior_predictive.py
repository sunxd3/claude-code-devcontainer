#!/usr/bin/env python3
"""Examine and visualize prior predictive distributions."""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# Load prior predictive data
output_dir = Path(
    "/home/user/claude-code-devcontainer/analysis/experiments/experiment_3/prior_predictive"
)
idata = az.from_netcdf(output_dir / "prior_predictive.nc")

# Extract samples (prior samples are in posterior group due to fixed_param sampling)
y_rep = idata.posterior["y_rep"].values.flatten()
y_obs = idata.observed_data["y"].values

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Distribution comparison (ECDF)
ax = axes[0, 0]
sorted_obs = np.sort(y_obs)
sorted_rep = np.sort(y_rep)
ax.plot(
    sorted_obs,
    np.linspace(0, 1, len(sorted_obs)),
    "ko-",
    linewidth=2,
    label="Observed",
)
ax.plot(
    sorted_rep,
    np.linspace(0, 1, len(sorted_rep)),
    "b-",
    alpha=0.3,
    label="Prior Predictive",
)
ax.set_xlabel("Test Score")
ax.set_ylabel("Cumulative Probability")
ax.set_title("ECDF: Observed vs Prior Predictive")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Histogram comparison
ax = axes[0, 1]
ax.hist(y_obs, bins=20, alpha=0.5, label="Observed", density=True, color="black")
ax.hist(y_rep, bins=50, alpha=0.3, label="Prior Predictive", density=True, color="blue")
ax.set_xlabel("Test Score")
ax.set_ylabel("Density")
ax.set_title("Distribution Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Range examination
ax = axes[1, 0]
stats = {
    "Observed": [y_obs.min(), np.percentile(y_obs, 5), np.percentile(y_obs, 95), y_obs.max()],
    "Prior Pred": [y_rep.min(), np.percentile(y_rep, 5), np.percentile(y_rep, 95), y_rep.max()],
}
positions = [0, 1]
colors = ["black", "blue"]

for i, (label, values) in enumerate(stats.items()):
    ax.plot([positions[i], positions[i]], [values[0], values[3]], "o-", color=colors[i], linewidth=2)
    ax.plot([positions[i] - 0.1, positions[i] + 0.1], [values[1], values[1]], "-", color=colors[i], linewidth=3)
    ax.plot([positions[i] - 0.1, positions[i] + 0.1], [values[2], values[2]], "-", color=colors[i], linewidth=3)

ax.axhline(0, linestyle="--", color="red", alpha=0.5, label="Impossible (y<0)")
ax.axhline(120, linestyle="--", color="orange", alpha=0.5, label="Implausible (y>120)")
ax.set_xticks(positions)
ax.set_xticklabels(stats.keys())
ax.set_ylabel("Test Score")
ax.set_title("Range Comparison (min, 5%, 95%, max)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# 4. Proportion of extreme values
ax = axes[1, 1]
prop_negative = (y_rep < 0).mean() * 100
prop_very_low = (y_rep < 40).mean() * 100
prop_very_high = (y_rep > 110).mean() * 100
prop_extreme = (y_rep > 150).mean() * 100

categories = ["< 0\n(Impossible)", "< 40\n(Very Low)", "> 110\n(Very High)", "> 150\n(Extreme)"]
proportions = [prop_negative, prop_very_low, prop_very_high, prop_extreme]
colors_bar = ["red", "orange", "orange", "red"]

bars = ax.bar(categories, proportions, color=colors_bar, alpha=0.6)
ax.set_ylabel("Percentage of Prior Predictive Samples (%)")
ax.set_title("Proportion of Implausible Values")
ax.grid(True, alpha=0.3, axis="y")

for bar, prop in zip(bars, proportions):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{prop:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(output_dir / "prior_predictive_check.png", dpi=300, bbox_inches="tight")
print(f"Saved figure to {output_dir / 'prior_predictive_check.png'}")

# Print detailed statistics
print("\nPrior Predictive Check Summary:")
print("=" * 60)
print("\nObserved Data:")
print(f"  Range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
print(f"  Mean: {y_obs.mean():.1f} (SD: {y_obs.std():.1f})")
print(f"  5th-95th percentile: [{np.percentile(y_obs, 5):.1f}, {np.percentile(y_obs, 95):.1f}]")

print("\nPrior Predictive:")
print(f"  Range: [{y_rep.min():.1f}, {y_rep.max():.1f}]")
print(f"  Mean: {y_rep.mean():.1f} (SD: {y_rep.std():.1f})")
print(f"  5th-95th percentile: [{np.percentile(y_rep, 5):.1f}, {np.percentile(y_rep, 95):.1f}]")

print("\nProblematic Samples:")
print(f"  Negative values: {(y_rep < 0).sum()} ({prop_negative:.2f}%)")
print(f"  Very low (< 40): {(y_rep < 40).sum()} ({prop_very_low:.2f}%)")
print(f"  Very high (> 110): {(y_rep > 110).sum()} ({prop_very_high:.2f}%)")
print(f"  Extreme (> 150): {(y_rep > 150).sum()} ({prop_extreme:.2f}%)")

print("\n" + "=" * 60)
