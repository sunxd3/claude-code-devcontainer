"""Generate parameter recovery visualization plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Output directory
output_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2/simulation")

# Load recovery results
results_file = output_dir / "recovery_results.json"
with open(results_file) as f:
    results = json.load(f)

# Parameters to visualize
params = ["alpha_0", "tau_alpha", "beta", "sigma"]
param_labels = {
    "alpha_0": r"$\alpha_0$ (pop. intercept)",
    "tau_alpha": r"$\tau_\alpha$ (school SD)",
    "beta": r"$\beta$ (treatment effect)",
    "sigma": r"$\sigma$ (residual SD)"
}

# Extract data for plotting
true_vals = {p: [] for p in params}
post_means = {p: [] for p in params}
post_sds = {p: [] for p in params}
hdi_lower = {p: [] for p in params}
hdi_upper = {p: [] for p in params}
scenario_names = []

for result in results:
    scenario_names.append(result['scenario'])
    for param in params:
        recovery = result['recovery'][param]
        true_vals[param].append(recovery['true'])
        post_means[param].append(recovery['posterior_mean'])
        post_sds[param].append(recovery['posterior_sd'])
        hdi_lower[param].append(recovery['hdi_lower'])
        hdi_upper[param].append(recovery['hdi_upper'])

# Plot 1: Scatter plots (posterior mean vs true value)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, param in enumerate(params):
    ax = axes[idx]

    # Extract values
    true_arr = np.array(true_vals[param])
    post_arr = np.array(post_means[param])

    # Scatter plot
    ax.scatter(true_arr, post_arr, s=100, alpha=0.7, edgecolors='black')

    # Identity line
    min_val = min(true_arr.min(), post_arr.min())
    max_val = max(true_arr.max(), post_arr.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Identity')

    # Labels for each point
    for i, name in enumerate(scenario_names):
        ax.annotate(name.replace('_clustering', ''),
                   (true_arr[i], post_arr[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel(f'True {param_labels[param]}', fontsize=10)
    ax.set_ylabel(f'Posterior Mean {param_labels[param]}', fontsize=10)
    ax.set_title(param_labels[param], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Parameter Recovery: Posterior Mean vs True Value', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "recovery_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Interval plots (true values with posterior credible intervals)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, param in enumerate(params):
    ax = axes[idx]

    # Extract values
    true_arr = np.array(true_vals[param])
    post_arr = np.array(post_means[param])
    lower_arr = np.array(hdi_lower[param])
    upper_arr = np.array(hdi_upper[param])

    x_pos = np.arange(len(scenario_names))

    # Plot posterior intervals
    ax.errorbar(x_pos, post_arr,
               yerr=[post_arr - lower_arr, upper_arr - post_arr],
               fmt='o', capsize=5, capthick=2, markersize=8,
               label='Posterior Mean + 94% HDI')

    # Plot true values
    ax.scatter(x_pos, true_arr, marker='x', s=100, color='red',
              linewidths=3, label='True Value', zorder=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace('_clustering', '') for n in scenario_names], rotation=45)
    ax.set_ylabel(param_labels[param], fontsize=10)
    ax.set_title(param_labels[param], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

plt.suptitle('Parameter Recovery: True Values vs Posterior Intervals', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "recovery_intervals.png", dpi=150, bbox_inches="tight")
plt.close()

print("Recovery visualization complete.")
print("  Saved: recovery_scatter.png")
print("  Saved: recovery_intervals.png")
