#!/usr/bin/env python3
"""
Posterior Predictive Check for Experiment 2: Random Intercepts Only

Checks if the model can reproduce:
1. Overall score distribution
2. School-level mean differences
3. Treatment effect (constant across schools)
4. Systematic residuals suggesting treatment effect heterogeneity
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2")
POSTERIOR_PATH = BASE_DIR / "fit" / "posterior.nc"
DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/student_scores.csv")
STAN_DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/stan_data.json")
OUTPUT_DIR = BASE_DIR / "posterior_predictive"

# Plotting settings
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})


def main():
    print("=" * 80)
    print("POSTERIOR PREDICTIVE CHECK: Random Intercepts Only Model")
    print("=" * 80)

    # Load data
    print("\n[1/8] Loading data...")
    df = pd.read_csv(DATA_PATH)

    print(f"  N = {len(df)} students")
    print(f"  J = {df['school_id'].nunique()} schools")

    # Load posterior
    print("\n[2/8] Loading posterior...")
    idata = az.from_netcdf(POSTERIOR_PATH)
    print(f"  Posterior draws: {idata.posterior.dims['chain']} chains x {idata.posterior.dims['draw']} draws")
    print(f"  y_rep shape: {idata.posterior_predictive['y_rep'].shape}")

    # Check 1: Overall distribution
    print("\n[3/8] Checking overall score distribution...")

    # ECDF
    fig = az.plot_ppc(idata, kind="cumulative", data_pairs={"y": "y_rep"}, figsize=(10, 6))
    plt.savefig(OUTPUT_DIR / "ppc_ecdf.png", bbox_inches='tight')
    plt.close()
    print("  Saved: ppc_ecdf.png")

    # KDE
    fig = az.plot_ppc(idata, kind="kde", data_pairs={"y": "y_rep"}, figsize=(10, 6))
    plt.savefig(OUTPUT_DIR / "ppc_kde.png", bbox_inches='tight')
    plt.close()
    print("  Saved: ppc_kde.png")

    # Check 2: LOO-PIT calibration
    print("\n[4/8] Checking LOO-PIT calibration...")
    try:
        fig = az.plot_loo_pit(idata, y="y", figsize=(10, 6))
        plt.savefig(OUTPUT_DIR / "loo_pit.png", bbox_inches='tight')
        plt.close()
        print("  Saved: loo_pit.png")
    except Exception as e:
        print(f"  Warning: LOO-PIT failed - {e}")
        print("  Skipping LOO-PIT check")

    # Check 3: Test statistics (mean, SD, min, max)
    print("\n[5/8] Checking summary statistics...")

    y_obs = idata.observed_data['y'].values
    y_rep = idata.posterior_predictive['y_rep'].values.reshape(-1, len(y_obs))

    # Compute statistics
    stats = {
        'Mean': (np.mean(y_obs), [np.mean(rep) for rep in y_rep]),
        'SD': (np.std(y_obs), [np.std(rep) for rep in y_rep]),
        'Min': (np.min(y_obs), [np.min(rep) for rep in y_rep]),
        'Max': (np.max(y_obs), [np.max(rep) for rep in y_rep])
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (stat_name, (obs_val, rep_vals)) in enumerate(stats.items()):
        ax = axes[idx]
        ax.hist(rep_vals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(obs_val, color='red', linewidth=2, label='Observed')
        ax.set_xlabel(f'{stat_name}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Posterior Predictive: {stat_name}')
        ax.legend()

        # Compute p-value
        p_val = np.mean(np.array(rep_vals) >= obs_val) if stat_name in ['Max'] else \
                np.mean(np.array(rep_vals) <= obs_val) if stat_name in ['Min'] else \
                2 * min(np.mean(np.array(rep_vals) >= obs_val), np.mean(np.array(rep_vals) <= obs_val))
        ax.text(0.05, 0.95, f'p = {p_val:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppc_summary_stats.png", bbox_inches='tight')
    plt.close()
    print("  Saved: ppc_summary_stats.png")

    # Check 4: School-level means
    print("\n[6/8] Checking school-level means...")

    # Compute observed school means
    school_means_obs = df.groupby('school_id')['score'].mean().values

    # Compute replicated school means
    school_means_rep = []
    for rep in y_rep:
        rep_df = df.copy()
        rep_df['score_rep'] = rep
        means = rep_df.groupby('school_id')['score_rep'].mean().values
        school_means_rep.append(means)
    school_means_rep = np.array(school_means_rep)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot posterior predictive intervals for each school
    schools = np.arange(1, len(school_means_obs) + 1)
    lower = np.percentile(school_means_rep, 5, axis=0)
    upper = np.percentile(school_means_rep, 95, axis=0)
    median = np.median(school_means_rep, axis=0)

    ax.fill_between(schools, lower, upper, alpha=0.3, color='skyblue', label='90% Posterior Predictive')
    ax.plot(schools, median, 'b-', linewidth=2, label='Median')
    ax.plot(schools, school_means_obs, 'ro', markersize=8, label='Observed', zorder=10)

    ax.set_xlabel('School')
    ax.set_ylabel('Mean Score')
    ax.set_title('School-Level Mean Scores: Observed vs Posterior Predictive')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppc_school_means.png", bbox_inches='tight')
    plt.close()
    print("  Saved: ppc_school_means.png")

    # Check 5: Treatment effect by school
    print("\n[7/8] Checking treatment effects by school...")

    # Compute observed treatment effects per school
    treatment_effects_obs = []
    for school_id in sorted(df['school_id'].unique()):
        school_df = df[df['school_id'] == school_id]
        treated = school_df[school_df['treatment'] == 1]['score'].mean()
        control = school_df[school_df['treatment'] == 0]['score'].mean()
        treatment_effects_obs.append(treated - control)
    treatment_effects_obs = np.array(treatment_effects_obs)

    # Compute replicated treatment effects per school
    treatment_effects_rep = []
    for rep in y_rep:
        rep_df = df.copy()
        rep_df['score_rep'] = rep
        effects = []
        for school_id in sorted(df['school_id'].unique()):
            school_df = rep_df[rep_df['school_id'] == school_id]
            treated = school_df[school_df['treatment'] == 1]['score_rep'].mean()
            control = school_df[school_df['treatment'] == 0]['score_rep'].mean()
            effects.append(treated - control)
        treatment_effects_rep.append(effects)
    treatment_effects_rep = np.array(treatment_effects_rep)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot posterior predictive intervals for treatment effect in each school
    lower = np.percentile(treatment_effects_rep, 5, axis=0)
    upper = np.percentile(treatment_effects_rep, 95, axis=0)
    median = np.median(treatment_effects_rep, axis=0)

    ax.fill_between(schools, lower, upper, alpha=0.3, color='lightcoral', label='90% Posterior Predictive')
    ax.plot(schools, median, 'r-', linewidth=2, label='Median')
    ax.plot(schools, treatment_effects_obs, 'ko', markersize=8, label='Observed', zorder=10)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add pooled treatment effect from posterior
    beta_samples = idata.posterior['beta'].values.flatten()
    beta_mean = beta_samples.mean()
    beta_lower = np.percentile(beta_samples, 5)
    beta_upper = np.percentile(beta_samples, 95)
    ax.axhline(beta_mean, color='blue', linestyle='-', linewidth=2, label=f'Pooled β (mean={beta_mean:.2f})', alpha=0.7)
    ax.axhspan(beta_lower, beta_upper, color='blue', alpha=0.1)

    ax.set_xlabel('School')
    ax.set_ylabel('Treatment Effect')
    ax.set_title('Treatment Effects by School: Observed vs Posterior Predictive\n(Model assumes constant effect)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ppc_treatment_effects.png", bbox_inches='tight')
    plt.close()
    print("  Saved: ppc_treatment_effects.png")

    # Check 6: Residuals by school and treatment
    print("\n[8/8] Checking residual patterns...")

    # Compute posterior mean predictions
    mu_samples = idata.posterior['mu'].values.reshape(-1, len(y_obs))
    mu_mean = mu_samples.mean(axis=0)
    residuals = y_obs - mu_mean

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals by school
    ax = axes[0]
    school_ids = df['school_id'].values
    for school in sorted(df['school_id'].unique()):
        mask = school_ids == school
        ax.scatter(np.full(mask.sum(), school), residuals[mask], alpha=0.5, s=30)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('School')
    ax.set_ylabel('Residual (observed - predicted)')
    ax.set_title('Residuals by School')
    ax.grid(True, alpha=0.3)

    # Residuals by treatment
    ax = axes[1]
    treatment = df['treatment'].values
    for t in [0, 1]:
        mask = treatment == t
        label = 'Control' if t == 0 else 'Treatment'
        ax.scatter(np.full(mask.sum(), t), residuals[mask], alpha=0.5, s=30, label=label)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Group')
    ax.set_ylabel('Residual (observed - predicted)')
    ax.set_title('Residuals by Treatment Group')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Treatment'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_patterns.png", bbox_inches='tight')
    plt.close()
    print("  Saved: residual_patterns.png")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nObserved vs Replicated:")
    for stat_name, (obs_val, rep_vals) in stats.items():
        rep_mean = np.mean(rep_vals)
        rep_lower = np.percentile(rep_vals, 5)
        rep_upper = np.percentile(rep_vals, 95)
        print(f"  {stat_name:8s}: Obs={obs_val:6.2f}, Rep={rep_mean:6.2f} [{rep_lower:6.2f}, {rep_upper:6.2f}]")

    print("\nSchool means (observed vs replicated):")
    for i, school in enumerate(schools):
        obs = school_means_obs[i]
        rep_mean = median[i]
        rep_lower = lower[i]
        rep_upper = upper[i]
        coverage = "✓" if rep_lower <= obs <= rep_upper else "✗"
        print(f"  School {school}: Obs={obs:6.2f}, Rep={rep_mean:6.2f} [{rep_lower:6.2f}, {rep_upper:6.2f}] {coverage}")

    print("\nTreatment effects by school (observed vs replicated):")
    for i, school in enumerate(schools):
        obs = treatment_effects_obs[i]
        rep_median = median[i] if i < len(median) else np.median(treatment_effects_rep[:, i])
        rep_lower = np.percentile(treatment_effects_rep[:, i], 5)
        rep_upper = np.percentile(treatment_effects_rep[:, i], 95)
        coverage = "✓" if rep_lower <= obs <= rep_upper else "✗"
        print(f"  School {school}: Obs={obs:6.2f}, Rep={rep_median:6.2f} [{rep_lower:6.2f}, {rep_upper:6.2f}] {coverage}")

    # Compute variance of treatment effects
    var_obs = np.var(treatment_effects_obs)
    var_rep = np.var(treatment_effects_rep, axis=1)
    var_rep_mean = np.mean(var_rep)
    var_rep_lower = np.percentile(var_rep, 5)
    var_rep_upper = np.percentile(var_rep, 95)

    print("\nVariance of treatment effects across schools:")
    print(f"  Observed: {var_obs:.2f}")
    print(f"  Replicated: {var_rep_mean:.2f} [{var_rep_lower:.2f}, {var_rep_upper:.2f}]")

    if var_obs > var_rep_upper:
        print("  ⚠ Observed variance exceeds 95% of replicated variance")
        print("  → Suggests treatment effect heterogeneity not captured by model")
    else:
        print("  ✓ Observed variance within replicated range")

    print("\n" + "=" * 80)
    print("Posterior predictive check complete")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
