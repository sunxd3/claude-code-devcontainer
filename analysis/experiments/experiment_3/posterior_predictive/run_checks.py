#!/usr/bin/env python3
"""Comprehensive posterior predictive checks for Experiment 3."""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Configure plotting
plt.style.use('default')
az.style.use('arviz-doc')

# Paths
base_dir = Path("/home/user/claude-code-devcontainer/analysis")
fit_path = base_dir / "experiments/experiment_3/fit/posterior.nc"
data_path = base_dir / "data/student_scores.csv"
output_dir = base_dir / "experiments/experiment_3/posterior_predictive"

# Load data
print("Loading data...")
idata = az.from_netcdf(fit_path)
df = pd.read_csv(data_path)

# Extract observed and replicated data
y_obs = df['score'].values
school_id = df['school_id'].values
treatment = df['treatment'].values

# Get posterior predictive samples
y_rep = idata.posterior_predictive['y_rep'].values  # shape: (chains, draws, N)
n_chains, n_draws, n_obs = y_rep.shape
print(f"Posterior predictive samples: {n_chains} chains x {n_draws} draws x {n_obs} observations")

# Compute LOO for LOO-PIT
print("\nComputing LOO...")
loo_result = az.loo(idata, pointwise=True)

# ============================================================================
# 1. Overall Distribution Checks
# ============================================================================
print("\n1. Checking overall score distribution...")

# PPC plot with multiple replications
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_ppc(idata, num_pp_samples=100, data_pairs={'y': 'y_rep'}, ax=ax)
ax.set_xlabel('Score')
ax.set_ylabel('Density')
ax.set_title('Posterior Predictive Check: Overall Distribution')
plt.tight_layout()
plt.savefig(output_dir / "overall_ppc.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: overall_ppc.png")

# LOO-PIT calibration (preferred over regular PIT)
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_loo_pit(idata, y='y', y_hat='y_rep', ecdf=True, ax=ax)
ax.set_title('LOO-PIT ECDF Calibration')
plt.tight_layout()
plt.savefig(output_dir / "loo_pit_calibration.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: loo_pit_calibration.png")

# ============================================================================
# 2. Summary Statistics (Test Statistics)
# ============================================================================
print("\n2. Checking summary statistics...")

# Multiple test statistics combined
def compute_tstats(y):
    """Compute test statistics for posterior predictive checks."""
    return {
        'median': np.median(y, axis=-1),
        'mad': np.median(np.abs(y - np.median(y, axis=-1, keepdims=True)), axis=-1),
        'iqr': np.percentile(y, 75, axis=-1) - np.percentile(y, 25, axis=-1),
        'min': np.min(y, axis=-1),
        'max': np.max(y, axis=-1)
    }

obs_stats = {
    'median': np.median(y_obs),
    'mad': np.median(np.abs(y_obs - np.median(y_obs))),
    'iqr': np.percentile(y_obs, 75) - np.percentile(y_obs, 25),
    'min': np.min(y_obs),
    'max': np.max(y_obs)
}

rep_stats = compute_tstats(y_rep)

# Plot test statistics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (stat_name, obs_val) in enumerate(obs_stats.items()):
    if idx < len(axes):
        ax = axes[idx]
        rep_vals = rep_stats[stat_name].flatten()
        ax.hist(rep_vals, bins=50, alpha=0.7, edgecolor='black', label='Replicated')
        ax.axvline(obs_val, color='red', linewidth=2, label='Observed')
        ax.set_xlabel(stat_name.upper())
        ax.set_ylabel('Count')
        ax.set_title(f'Test Statistic: {stat_name.upper()}')
        ax.legend()

        # Compute p-value (two-tailed)
        p_val = 2 * min(np.mean(rep_vals >= obs_val), np.mean(rep_vals <= obs_val))
        ax.text(0.95, 0.95, f'p = {p_val:.3f}',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Remove extra subplot
if len(obs_stats) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig(output_dir / "test_statistics.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: test_statistics.png")

# ============================================================================
# 3. School-Level Patterns
# ============================================================================
print("\n3. Checking school-level patterns...")

# Compute school-level observed and replicated means
schools = np.sort(df['school_id'].unique())
n_schools = len(schools)

school_obs_means = []
school_rep_means = []

for school in schools:
    school_mask = (school_id == school)
    school_obs_means.append(y_obs[school_mask].mean())

    # Replicated means for this school
    school_y_rep = y_rep[:, :, school_mask]  # (chains, draws, n_students_in_school)
    school_rep_means.append(school_y_rep.mean(axis=2))  # (chains, draws)

# Plot school means
fig, ax = plt.subplots(figsize=(12, 6))

positions = np.arange(n_schools)
for i, school in enumerate(schools):
    rep_means_flat = school_rep_means[i].flatten()

    # Violin plot of replicated means
    parts = ax.violinplot([rep_means_flat], positions=[positions[i]],
                           widths=0.7, showmeans=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)

    # Overlay observed mean
    ax.plot(positions[i], school_obs_means[i], 'ro', markersize=10,
            label='Observed' if i == 0 else '', zorder=10)

    # Add 95% interval
    q025 = np.percentile(rep_means_flat, 2.5)
    q975 = np.percentile(rep_means_flat, 97.5)
    ax.plot([positions[i], positions[i]], [q025, q975], 'k-', linewidth=2, alpha=0.5)

ax.set_xlabel('School')
ax.set_ylabel('Mean Score')
ax.set_title('School-Level Mean Scores: Observed vs Posterior Predictive')
ax.set_xticks(positions)
ax.set_xticklabels([f'School {s}' for s in schools], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "school_means.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: school_means.png")

# ============================================================================
# 4. School-Specific Treatment Effects
# ============================================================================
print("\n4. Checking school-specific treatment effects...")

# Extract school-specific treatment effects from posterior
beta = idata.posterior['beta'].values  # (chains, draws, n_schools)

# Compute observed treatment effects by school
school_obs_effects = []
for school in schools:
    school_mask = (school_id == school)
    treat_mask = treatment[school_mask] == 1
    ctrl_mask = treatment[school_mask] == 0

    if treat_mask.sum() > 0 and ctrl_mask.sum() > 0:
        obs_effect = y_obs[school_mask][treat_mask].mean() - y_obs[school_mask][ctrl_mask].mean()
    else:
        obs_effect = np.nan
    school_obs_effects.append(obs_effect)

# Compute replicated treatment effects
school_rep_effects = []
for school in schools:
    school_mask = (school_id == school)
    treat_mask = treatment[school_mask] == 1
    ctrl_mask = treatment[school_mask] == 0

    if treat_mask.sum() > 0 and ctrl_mask.sum() > 0:
        school_treat_idx = np.where(school_mask)[0][treat_mask]
        school_ctrl_idx = np.where(school_mask)[0][ctrl_mask]

        rep_treat = y_rep[:, :, school_treat_idx].mean(axis=2)
        rep_ctrl = y_rep[:, :, school_ctrl_idx].mean(axis=2)
        rep_effect = rep_treat - rep_ctrl
    else:
        rep_effect = np.full((n_chains, n_draws), np.nan)

    school_rep_effects.append(rep_effect)

# Plot treatment effects
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for i, school in enumerate(schools):
    ax = axes[i]

    # Posterior distribution of beta (model parameter)
    beta_school = beta[:, :, i].flatten()
    ax.hist(beta_school, bins=40, alpha=0.6, edgecolor='black',
            label='Beta (parameter)', color='skyblue', density=True)

    # Replicated treatment effect distribution
    if not np.isnan(school_rep_effects[i]).all():
        rep_effect_flat = school_rep_effects[i].flatten()
        ax.hist(rep_effect_flat, bins=40, alpha=0.6, edgecolor='black',
                label='Replicated ATE', color='lightgreen', density=True)

    # Observed treatment effect
    if not np.isnan(school_obs_effects[i]):
        ax.axvline(school_obs_effects[i], color='red', linewidth=2,
                  label='Observed ATE')

    ax.set_xlabel('Treatment Effect')
    ax.set_ylabel('Density')
    ax.set_title(f'School {school}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "school_treatment_effects.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: school_treatment_effects.png")

# ============================================================================
# 5. Extreme Schools Analysis (School 1 and School 3)
# ============================================================================
print("\n5. Analyzing extreme schools (School 1 with high ATE, School 3 with near-zero ATE)...")

extreme_schools = [1, 3]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, school in enumerate(extreme_schools):
    ax = axes[idx]
    school_mask = (school_id == school)

    # Observed data for this school
    school_y_obs = y_obs[school_mask]
    school_treat = treatment[school_mask]

    # Replicated data for this school
    school_y_rep = y_rep[:, :, school_mask]  # (chains, draws, n_students)

    # Plot distributions by treatment group
    treat_obs = school_y_obs[school_treat == 1]
    ctrl_obs = school_y_obs[school_treat == 0]

    # Treatment group
    if len(treat_obs) > 0:
        treat_indices = np.where(school_mask)[0][school_treat == 1]
        treat_rep = y_rep[:, :, treat_indices].reshape(-1, len(treat_indices))

        for i in range(len(treat_obs)):
            parts = ax.violinplot([treat_rep[:, i]], positions=[i], widths=0.4,
                                 showmeans=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor('salmon')
                pc.set_alpha(0.6)
            ax.plot(i, treat_obs[i], 'ro', markersize=8, zorder=10)

    # Control group
    offset = len(treat_obs) + 1
    if len(ctrl_obs) > 0:
        ctrl_indices = np.where(school_mask)[0][school_treat == 0]
        ctrl_rep = y_rep[:, :, ctrl_indices].reshape(-1, len(ctrl_indices))

        for i in range(len(ctrl_obs)):
            parts = ax.violinplot([ctrl_rep[:, i]], positions=[offset + i], widths=0.4,
                                 showmeans=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.6)
            ax.plot(offset + i, ctrl_obs[i], 'bo', markersize=8, zorder=10)

    # Add group labels
    ax.axvline(len(treat_obs) + 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(treat_obs)/2, ax.get_ylim()[1]*0.95, 'Treatment',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.3))
    ax.text(offset + len(ctrl_obs)/2, ax.get_ylim()[1]*0.95, 'Control',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.3))

    ax.set_title(f'School {school}')
    ax.set_xlabel('Student Index')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "extreme_schools_individual.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: extreme_schools_individual.png")

# ============================================================================
# 6. Compute Quantitative Metrics
# ============================================================================
print("\n6. Computing quantitative metrics...")

metrics = {}

# Overall calibration (LOO-PIT uniformity test)
loo_pit_values = az.loo_pit(idata=idata, y='y', y_hat='y_rep')
if loo_pit_values is not None:
    # Kolmogorov-Smirnov test for uniformity
    loo_pit_flat = loo_pit_values.flatten() if isinstance(loo_pit_values, np.ndarray) else loo_pit_values.values.flatten()
    ks_stat, ks_pval = stats.kstest(loo_pit_flat, 'uniform')
    metrics['loo_pit_ks_stat'] = ks_stat
    metrics['loo_pit_ks_pval'] = ks_pval

# School-level coverage
school_coverage = []
for i, school in enumerate(schools):
    obs_mean = school_obs_means[i]
    rep_means_flat = school_rep_means[i].flatten()
    q025 = np.percentile(rep_means_flat, 2.5)
    q975 = np.percentile(rep_means_flat, 97.5)
    in_interval = (obs_mean >= q025) and (obs_mean <= q975)
    school_coverage.append(in_interval)

metrics['school_mean_coverage'] = np.mean(school_coverage)
metrics['schools_covered'] = f"{sum(school_coverage)}/{len(schools)}"

# Treatment effect coverage
te_coverage = []
for i, school in enumerate(schools):
    if not np.isnan(school_obs_effects[i]) and not np.isnan(school_rep_effects[i]).all():
        obs_effect = school_obs_effects[i]
        rep_effects_flat = school_rep_effects[i].flatten()
        q025 = np.percentile(rep_effects_flat, 2.5)
        q975 = np.percentile(rep_effects_flat, 97.5)
        in_interval = (obs_effect >= q025) and (obs_effect <= q975)
        te_coverage.append(in_interval)

metrics['treatment_effect_coverage'] = np.mean(te_coverage) if te_coverage else np.nan
metrics['treatment_effects_covered'] = f"{sum(te_coverage)}/{len(te_coverage)}" if te_coverage else "N/A"

# Save metrics
with open(output_dir / "ppc_metrics.txt", "w") as f:
    f.write("Posterior Predictive Check Metrics\n")
    f.write("=" * 50 + "\n\n")

    for key, val in metrics.items():
        f.write(f"{key}: {val}\n")

    f.write("\n" + "=" * 50 + "\n")
    f.write("Observed Test Statistics:\n")
    for key, val in obs_stats.items():
        f.write(f"  {key}: {val:.2f}\n")

    f.write("\nReplicated Test Statistics (mean):\n")
    for key, vals in rep_stats.items():
        f.write(f"  {key}: {vals.mean():.2f} (SD: {vals.std():.2f})\n")

print("  Saved: ppc_metrics.txt")

print("\nPosterior predictive checks complete!")
print(f"All outputs saved to: {output_dir}")
