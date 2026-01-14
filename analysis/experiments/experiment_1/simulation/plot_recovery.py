#!/usr/bin/env python3
"""Create recovery visualizations for Complete Pooling model."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load recovery results
output_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_1/simulation")
results_path = output_dir / "recovery_results.json"

with open(results_path) as f:
    results = json.load(f)

# Extract data for each parameter
scenarios = [r["scenario"] for r in results]
params = ["alpha", "beta", "sigma"]
param_labels = {"alpha": r"$\alpha$ (Intercept)", "beta": r"$\beta$ (Treatment Effect)", "sigma": r"$\sigma$ (Residual SD)"}

# Create figure with 2 rows: scatter plot and interval plot
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Parameter Recovery Check: Complete Pooling Model", fontsize=14, fontweight="bold")

# Row 1: Scatter plots (posterior mean vs true value)
for idx, param in enumerate(params):
    ax = axes[0, idx]

    true_vals = [r[f"true_{param}"] for r in results]
    post_means = [r[f"post_{param}_mean"] for r in results]

    # Scatter plot
    ax.scatter(true_vals, post_means, s=100, alpha=0.7, color="steelblue", edgecolors="black", linewidths=1.5)

    # Identity line
    min_val = min(min(true_vals), min(post_means))
    max_val = max(max(true_vals), max(post_means))
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
            'k--', alpha=0.5, label="Perfect recovery")

    ax.set_xlabel(f"True {param}", fontsize=11)
    ax.set_ylabel(f"Posterior mean {param}", fontsize=11)
    ax.set_title(param_labels[param], fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

# Row 2: Interval plots (credible intervals with true values)
for idx, param in enumerate(params):
    ax = axes[1, idx]

    true_vals = [r[f"true_{param}"] for r in results]
    post_means = [r[f"post_{param}_mean"] for r in results]
    post_sds = [r[f"post_{param}_sd"] for r in results]
    post_q05 = [r[f"post_{param}_q05"] for r in results]
    post_q95 = [r[f"post_{param}_q95"] for r in results]

    x_pos = np.arange(len(scenarios))

    # Plot posterior means with error bars
    ax.errorbar(x_pos, post_means,
                yerr=[np.array(post_means) - np.array(post_q05),
                      np.array(post_q95) - np.array(post_means)],
                fmt='o', markersize=8, capsize=5, capthick=2,
                color="steelblue", ecolor="steelblue",
                label="Posterior mean ± 90% CI", alpha=0.8)

    # Plot true values as horizontal lines
    for i, (x, true_val) in enumerate(zip(x_pos, true_vals)):
        ax.plot([x - 0.3, x + 0.3], [true_val, true_val],
                'r-', linewidth=3, alpha=0.8,
                label="True value" if i == 0 else "")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=15, ha="right")
    ax.set_ylabel(f"{param}", fontsize=11)
    ax.set_title(f"{param_labels[param]}: Recovery Check", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

# Save figure
fig_path = output_dir / "recovery_plot.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved recovery plot to {fig_path}")
plt.close()

# Create convergence diagnostics summary figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Convergence Diagnostics Summary", fontsize=14, fontweight="bold")

# R-hat values
ax = axes[0]
for idx, param in enumerate(params):
    rhat_vals = [r[f"{param}_rhat"] for r in results]
    x_pos = np.arange(len(scenarios)) + idx * 0.25
    ax.bar(x_pos, rhat_vals, width=0.2, label=param, alpha=0.7)

ax.axhline(y=1.01, color='r', linestyle='--', linewidth=2, label="Threshold (1.01)")
ax.set_xticks(np.arange(len(scenarios)) + 0.25)
ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=15, ha="right")
ax.set_ylabel("R-hat", fontsize=11)
ax.set_title("R-hat Convergence Statistic", fontsize=12, fontweight="bold")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim([0.995, 1.015])

# ESS bulk
ax = axes[1]
for idx, param in enumerate(params):
    ess_vals = [r[f"{param}_ess_bulk"] for r in results]
    x_pos = np.arange(len(scenarios)) + idx * 0.25
    ax.bar(x_pos, ess_vals, width=0.2, label=param, alpha=0.7)

ax.axhline(y=400, color='r', linestyle='--', linewidth=2, label="Threshold (400)")
ax.set_xticks(np.arange(len(scenarios)) + 0.25)
ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=15, ha="right")
ax.set_ylabel("ESS Bulk", fontsize=11)
ax.set_title("Effective Sample Size (Bulk)", fontsize=12, fontweight="bold")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# Recovery errors
ax = axes[2]
for idx, param in enumerate(params):
    errors = [r[f"{param}_error"] for r in results]
    x_pos = np.arange(len(scenarios)) + idx * 0.25
    ax.bar(x_pos, errors, width=0.2, label=param, alpha=0.7)

ax.set_xticks(np.arange(len(scenarios)) + 0.25)
ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=15, ha="right")
ax.set_ylabel("Absolute Error", fontsize=11)
ax.set_title("Recovery Error (|Posterior Mean - True Value|)", fontsize=12, fontweight="bold")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

# Save figure
fig_path = output_dir / "convergence_diagnostics.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved convergence diagnostics to {fig_path}")
plt.close()

print("\nVisualization complete!")
