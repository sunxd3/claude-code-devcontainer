"""Posterior predictive checks for Complete Pooling model."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from shared_utils import load_posterior

# Load data
data = pd.read_csv(
    "/home/user/claude-code-devcontainer/analysis/data/student_scores.csv"
)
idata = load_posterior(
    "/home/user/claude-code-devcontainer/analysis/experiments/experiment_1/fit"
)

# Output directory
output_dir = (
    "/home/user/claude-code-devcontainer/analysis/experiments/experiment_1/posterior_predictive"
)

print("Posterior predictive checks for Complete Pooling model")
print("=" * 60)
print(f"Number of observations: {len(data)}")
print(f"Number of schools: {data['school_id'].nunique()}")
print(
    f"Posterior draws: {idata.posterior.dims['chain']} chains x {idata.posterior.dims['draw']} draws"
)

# 1. Distribution checks: ECDF and KDE
print("\n1. Creating distribution checks (ECDF and KDE)...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ECDF
ax = axes[0]
y_obs = idata.observed_data["y"].values
y_rep = idata.posterior_predictive["y_rep"].values
y_rep_flat = y_rep.reshape(-1, y_rep.shape[-1])

# Plot a sample of replications
np.random.seed(42)
n_samples = 50
for i in np.random.choice(y_rep_flat.shape[0], n_samples, replace=False):
    sorted_vals = np.sort(y_rep_flat[i])
    ecdf_vals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, ecdf_vals, color="lightblue", alpha=0.3, linewidth=0.5)

# Observed ECDF
sorted_obs = np.sort(y_obs)
ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
ax.plot(sorted_obs, ecdf_obs, color="red", linewidth=2, label="Observed")
ax.set_xlabel("Score")
ax.set_ylabel("ECDF")
ax.set_title("ECDF: Observed vs Replicated Data")
ax.legend()
ax.grid(True, alpha=0.3)

# KDE
ax = axes[1]
for i in np.random.choice(y_rep_flat.shape[0], n_samples, replace=False):
    try:
        kde = stats.gaussian_kde(y_rep_flat[i])
        x_vals = np.linspace(y_rep_flat.min(), y_rep_flat.max(), 200)
        ax.plot(x_vals, kde(x_vals), color="lightblue", alpha=0.3, linewidth=0.5)
    except Exception:
        continue

kde_obs = stats.gaussian_kde(y_obs)
x_vals = np.linspace(y_obs.min(), y_obs.max(), 200)
ax.plot(x_vals, kde_obs(x_vals), color="red", linewidth=2, label="Observed")
ax.set_xlabel("Score")
ax.set_ylabel("Density")
ax.set_title("KDE: Observed vs Replicated Data")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/distribution_checks.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: distribution_checks.png")

# 2. Calibration check: LOO-PIT
print("\n2. Creating calibration check (LOO-PIT)...")
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_loo_pit(idata, y="y", y_hat="y_rep", ecdf=True, ax=ax)
ax.set_title("LOO-PIT: Calibration Check")
plt.tight_layout()
plt.savefig(f"{output_dir}/loo_pit_calibration.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: loo_pit_calibration.png")

# 3. Test statistics: median, MAD, IQR
print("\n3. Creating test statistic checks...")


def compute_test_stats(data):
    """Compute test statistics for observed or replicated data."""
    return {
        "median": np.median(data),
        "mad": np.median(np.abs(data - np.median(data))),
        "iqr": np.percentile(data, 75) - np.percentile(data, 25),
        "min": np.min(data),
        "max": np.max(data),
    }


# Compute for observed
obs_stats = compute_test_stats(y_obs)

# Compute for replicated data
rep_stats = {key: [] for key in obs_stats.keys()}
for i in range(y_rep_flat.shape[0]):
    stats_i = compute_test_stats(y_rep_flat[i])
    for key in rep_stats:
        rep_stats[key].append(stats_i[key])

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (stat_name, stat_label) in enumerate(
    [
        ("median", "Median"),
        ("mad", "MAD"),
        ("iqr", "IQR"),
        ("min", "Minimum"),
        ("max", "Maximum"),
    ]
):
    ax = axes[idx]
    ax.hist(rep_stats[stat_name], bins=30, alpha=0.7, density=True, label="Replicated")
    ax.axvline(
        obs_stats[stat_name],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Observed",
    )

    # Compute p-value (proportion of replications more extreme than observed)
    p_val = np.mean(
        np.abs(rep_stats[stat_name] - np.mean(rep_stats[stat_name]))
        >= np.abs(obs_stats[stat_name] - np.mean(rep_stats[stat_name]))
    )

    ax.set_xlabel(stat_label)
    ax.set_ylabel("Density")
    ax.set_title(f"{stat_label}\n(p-value: {p_val:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove the 6th subplot
axes[5].axis("off")

plt.tight_layout()
plt.savefig(f"{output_dir}/test_statistics.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: test_statistics.png")

# 4. School-level residual analysis
print("\n4. Analyzing school-level residuals...")

# Get posterior predictive mean for each observation
y_rep_mean = y_rep_flat.mean(axis=0)

# Calculate residuals
residuals = y_obs - y_rep_mean
data["residual"] = residuals

# School-level summaries
school_stats = (
    data.groupby("school_id")
    .agg({"residual": ["mean", "std", "count"], "score": "mean", "school_name": "first"})
    .reset_index()
)
school_stats.columns = [
    "school_id",
    "mean_residual",
    "std_residual",
    "n_students",
    "mean_score",
    "school_name",
]

print("\nSchool-level residual summary:")
print(school_stats.to_string(index=False))

# Compute RMSE by school
school_stats["rmse"] = np.sqrt(
    data.groupby("school_id")["residual"].apply(lambda x: (x**2).mean()).values
)

# Plot school-level residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Mean residuals by school
ax = axes[0, 0]
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
bars = ax.bar(school_stats["school_id"], school_stats["mean_residual"])
# Color bars by magnitude
colors = ["red" if abs(x) > 2 else "steelblue" for x in school_stats["mean_residual"]]
for bar, color in zip(bars, colors):
    bar.set_color(color)
ax.set_xlabel("School ID")
ax.set_ylabel("Mean Residual")
ax.set_title(
    "Mean Residuals by School\n(Red bars indicate systematic bias |residual| > 2)"
)
ax.set_xticks(school_stats["school_id"])
ax.grid(True, alpha=0.3, axis="y")

# Panel 2: RMSE by school
ax = axes[0, 1]
ax.bar(school_stats["school_id"], school_stats["rmse"], color="steelblue")
ax.set_xlabel("School ID")
ax.set_ylabel("RMSE")
ax.set_title("RMSE by School")
ax.set_xticks(school_stats["school_id"])
ax.grid(True, alpha=0.3, axis="y")

# Panel 3: Residuals by school (boxplot)
ax = axes[1, 0]
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
data.boxplot(column="residual", by="school_id", ax=ax)
ax.set_xlabel("School ID")
ax.set_ylabel("Residual")
ax.set_title("Distribution of Residuals by School")
plt.suptitle("")  # Remove default title
ax.grid(True, alpha=0.3, axis="y")

# Panel 4: Q-Q plot of residuals
ax = axes[1, 1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q Plot of Residuals")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/school_level_residuals.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: school_level_residuals.png")

# 5. Treatment effect analysis
print("\n5. Analyzing treatment effect predictions...")
treatment_stats = (
    data.groupby("treatment")
    .agg({"score": ["mean", "std", "count"], "residual": ["mean", "std"]})
    .reset_index()
)
treatment_stats.columns = [
    "treatment",
    "mean_score",
    "std_score",
    "n",
    "mean_residual",
    "std_residual",
]
treatment_stats["treatment_label"] = treatment_stats["treatment"].map(
    {0: "Control", 1: "Treatment"}
)

print("\nTreatment group summary:")
print(
    treatment_stats[
        [
            "treatment_label",
            "mean_score",
            "std_score",
            "n",
            "mean_residual",
            "std_residual",
        ]
    ].to_string(index=False)
)

# Observed treatment effect
obs_effect = (
    treatment_stats.loc[treatment_stats["treatment"] == 1, "mean_score"].values[0]
    - treatment_stats.loc[treatment_stats["treatment"] == 0, "mean_score"].values[0]
)

# Posterior treatment effect (beta parameter)
beta_samples = idata.posterior["beta"].values.flatten()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(beta_samples, bins=50, alpha=0.7, density=True, label="Posterior")
ax.axvline(
    beta_samples.mean(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"Posterior mean: {beta_samples.mean():.2f}",
)
ax.axvline(
    obs_effect,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Observed effect: {obs_effect:.2f}",
)
ax.set_xlabel("Treatment Effect")
ax.set_ylabel("Density")
ax.set_title("Treatment Effect: Posterior vs Observed")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/treatment_effect.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: treatment_effect.png")

print("\n" + "=" * 60)
print("Posterior predictive checks completed successfully!")
print(f"All outputs saved to: {output_dir}")
