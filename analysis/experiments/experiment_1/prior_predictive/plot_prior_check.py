"""Create visualizations for prior predictive check."""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# Load results
exp_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_1")
output_dir = exp_dir / "prior_predictive"
idata = az.from_netcdf(output_dir / "prior_predictive.nc")

# Extract data
y_prior = idata.prior_predictive['y_prior_pred'].values.flatten()
y_obs = idata.observed_data['y'].values

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram comparison
ax = axes[0, 0]
ax.hist(y_prior, bins=50, alpha=0.5, label='Prior predictive', density=True, color='C0')
ax.hist(y_obs, bins=30, alpha=0.5, label='Observed', density=True, color='C1')
ax.axvline(y_obs.mean(), color='C1', linestyle='--', linewidth=2, label=f'Obs mean: {y_obs.mean():.1f}')
ax.axvline(y_prior.mean(), color='C0', linestyle='--', linewidth=2, label=f'Prior pred mean: {y_prior.mean():.1f}')
ax.set_xlabel('Test Score')
ax.set_ylabel('Density')
ax.set_title('Prior Predictive vs Observed Distribution')
ax.legend()
ax.grid(alpha=0.3)

# 2. ECDF comparison
ax = axes[0, 1]
y_prior_sorted = np.sort(y_prior)
y_obs_sorted = np.sort(y_obs)
ax.plot(y_prior_sorted, np.linspace(0, 1, len(y_prior_sorted)),
        label='Prior predictive', color='C0', linewidth=2)
ax.plot(y_obs_sorted, np.linspace(0, 1, len(y_obs_sorted)),
        label='Observed', color='C1', linewidth=2)
ax.set_xlabel('Test Score')
ax.set_ylabel('Cumulative Probability')
ax.set_title('Empirical CDF Comparison')
ax.legend()
ax.grid(alpha=0.3)

# 3. Box plot comparison
ax = axes[1, 0]
box_data = [y_prior, y_obs]
bp = ax.boxplot(box_data, labels=['Prior Predictive', 'Observed'], patch_artist=True)
bp['boxes'][0].set_facecolor('C0')
bp['boxes'][1].set_facecolor('C1')
ax.set_ylabel('Test Score')
ax.set_title('Distribution Summary')
ax.grid(alpha=0.3, axis='y')
ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='Zero (impossible)')
ax.legend()

# 4. Plausibility diagnostics
ax = axes[1, 1]
checks = [
    f"Range: [{y_prior.min():.1f}, {y_prior.max():.1f}]",
    f"Mean: {y_prior.mean():.1f} (obs: {y_obs.mean():.1f})",
    f"SD: {y_prior.std():.1f} (obs: {y_obs.std():.1f})",
    f"Negative: {100*np.sum(y_prior < 0)/len(y_prior):.1f}%",
    f"Below 40: {100*np.sum(y_prior < 40)/len(y_prior):.1f}%",
    f"Above 110: {100*np.sum(y_prior > 110)/len(y_prior):.1f}%",
]

y_pos = np.arange(len(checks))
ax.barh(y_pos, [1]*len(checks), alpha=0.3, color='lightgray')
ax.set_yticks(y_pos)
ax.set_yticklabels(checks, fontsize=10)
ax.set_xlim(0, 1.2)
ax.set_title('Prior Predictive Diagnostics')
ax.axis('off')

for i, check in enumerate(checks):
    ax.text(0.05, i, check, fontsize=11, verticalalignment='center')

plt.tight_layout()
plt.savefig(output_dir / "prior_predictive_check.png", dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_dir / 'prior_predictive_check.png'}")
