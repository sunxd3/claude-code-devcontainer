"""Posterior Predictive Checks for A3-Robust Model (Student-t Errors).

Compares observed data against posterior predictive distributions to assess
whether the Student-t error model adequately captures key features of the
Auto-MPG data, including:
- Marginal distributions (density, ECDF)
- Tail behavior and extreme observations
- Residual patterns vs fitted values, year, and origin
- Comparison with Normal errors model (Experiment 2)

Key question: Does the Student-t better capture extreme observations?
"""

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Paths
DATA_PATH = Path("/workspace/analysis/eda/auto_mpg_cleaned.csv")
POSTERIOR_PATH = Path("/workspace/analysis/experiments/experiment_3/fit/posterior.nc")
POSTERIOR_EXP2_PATH = Path("/workspace/analysis/experiments/experiment_2/fit/posterior.nc")
OUTPUT_DIR = Path("/workspace/analysis/experiments/experiment_3/posterior_predictive")

# Configure plots
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


def load_data():
    """Load observed data and compute transformations."""
    df = pd.read_csv(DATA_PATH)
    df_clean = df.dropna(subset=["horsepower"])

    # Compute transformations
    df_clean = df_clean.copy()
    df_clean["log_mpg"] = np.log(df_clean["mpg"])
    df_clean["log_weight_c"] = np.log(df_clean["weight"]) - 7.96
    df_clean["year_c"] = df_clean["model_year"] - 76

    return df_clean


def compute_fitted_values(idata, df):
    """Compute posterior mean fitted values."""
    alpha = idata.posterior["alpha"].values.mean()
    beta_weight = idata.posterior["beta_weight"].values.mean()
    beta_year = idata.posterior["beta_year"].values.mean()

    fitted = alpha + beta_weight * df["log_weight_c"].values + beta_year * df["year_c"].values
    return fitted


def compute_residuals(df, fitted):
    """Compute residuals on log scale."""
    return df["log_mpg"].values - fitted


def plot_observed_vs_predicted_density(y_obs, y_rep, output_path):
    """Plot density comparison: observed vs posterior predictive."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: KDE comparison (sample of y_rep draws)
    ax = axes[0]

    # Sample posterior draws for plotting
    n_draws = min(100, y_rep.shape[0] * y_rep.shape[1])
    y_rep_flat = y_rep.reshape(-1, y_rep.shape[-1])
    sample_idx = np.random.choice(len(y_rep_flat), size=n_draws, replace=False)

    for idx in sample_idx:
        ax.hist(y_rep_flat[idx], bins=30, density=True, alpha=0.02, color="C0")

    ax.hist(y_obs, bins=30, density=True, alpha=0.7, color="C1",
            label="Observed", edgecolor="white")
    ax.axvline(y_obs.mean(), color="C1", linestyle="--", linewidth=2, label="Observed mean")
    ax.axvline(y_rep.mean(), color="C0", linestyle="--", linewidth=2, label="Replications mean")
    ax.set_xlabel("log(mpg)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive: Density")
    ax.legend()

    # Right: ECDF comparison
    ax = axes[1]

    # Observed ECDF
    y_sorted = np.sort(y_obs)
    ecdf_obs = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    ax.step(y_sorted, ecdf_obs, where="post", color="C1", linewidth=2, label="Observed")

    # Sample replications for ECDF envelope
    for idx in sample_idx[:50]:
        rep_sorted = np.sort(y_rep_flat[idx])
        ecdf_rep = np.arange(1, len(rep_sorted) + 1) / len(rep_sorted)
        ax.step(rep_sorted, ecdf_rep, where="post", color="C0", alpha=0.05)

    ax.set_xlabel("log(mpg)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Posterior Predictive: ECDF")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_loo_pit(idata, output_path):
    """Plot LOO-PIT histogram and ECDF for calibration check."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Get observed data - check both possible variable names
    if hasattr(idata, "observed_data"):
        obs_data = idata.observed_data
        if "y" in obs_data:
            y_obs = obs_data["y"].values
        elif "observed_data" in obs_data:
            y_obs = obs_data["observed_data"].values
        else:
            y_obs = list(obs_data.data_vars.values())[0].values
    else:
        raise ValueError("No observed_data group in idata")

    y_rep = idata.posterior_predictive["y_rep"].values

    # Compute empirical PIT (approximate - not true LOO-PIT)
    pit_values = []
    for i in range(len(y_obs)):
        rep_at_i = y_rep[:, :, i].flatten()
        pit = np.mean(rep_at_i < y_obs[i])
        pit_values.append(pit)
    pit_values = np.array(pit_values)

    # Left: Histogram
    axes[0].hist(pit_values, bins=20, density=True, alpha=0.7, edgecolor="white", color="C0")
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Uniform reference")
    axes[0].set_xlabel("PIT value")
    axes[0].set_ylabel("Density")
    axes[0].set_title("PIT Histogram")
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    # Right: ECDF
    pit_sorted = np.sort(pit_values)
    ecdf = np.arange(1, len(pit_sorted) + 1) / len(pit_sorted)
    axes[1].plot(pit_sorted, ecdf, color="C0", linewidth=2, label="PIT ECDF")
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Uniform reference")
    axes[1].set_xlabel("PIT value")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("PIT ECDF")
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_residuals_vs_fitted(residuals, fitted, output_path):
    """Plot residuals vs fitted values to check homoscedasticity."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(fitted, residuals, alpha=0.5, s=30, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # Add smoothed trend line
    from scipy.ndimage import uniform_filter1d
    sorted_idx = np.argsort(fitted)
    window = max(20, len(fitted) // 20)
    smooth_resid = uniform_filter1d(residuals[sorted_idx], size=window, mode="nearest")
    ax.plot(fitted[sorted_idx], smooth_resid, color="orange", linewidth=2, label="Smoothed trend")

    ax.set_xlabel("Fitted values (log scale)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted Values")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_residuals_by_year(residuals, df, output_path):
    """Plot residuals by model year."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    years = df["model_year"].values

    # Scatter plot
    ax = axes[0]
    ax.scatter(years, residuals, alpha=0.5, s=30, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # Add mean per year
    unique_years = np.sort(df["model_year"].unique())
    year_means = [residuals[years == y].mean() for y in unique_years]
    ax.plot(unique_years, year_means, "o-", color="orange", linewidth=2,
            markersize=6, label="Year mean")

    ax.set_xlabel("Model Year (19XX)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Model Year")
    ax.legend()

    # Boxplot by year
    ax = axes[1]
    year_groups = [residuals[years == y] for y in unique_years]
    bp = ax.boxplot(year_groups, labels=[str(y) for y in unique_years], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("C0")
        patch.set_alpha(0.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Model Year (19XX)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Distribution by Year")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_residuals_by_origin(residuals, df, output_path):
    """Plot residuals by car origin (1=USA, 2=Europe, 3=Japan)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    origin_labels = {1: "USA", 2: "Europe", 3: "Japan"}
    origins = df["origin"].values

    # Boxplot
    ax = axes[0]
    origin_groups = [residuals[origins == o] for o in [1, 2, 3]]
    bp = ax.boxplot(origin_groups, labels=[origin_labels[o] for o in [1, 2, 3]],
                    patch_artist=True)
    colors = ["C0", "C1", "C2"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Origin")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Distribution by Origin")

    # Summary statistics
    ax = axes[1]
    means = [residuals[origins == o].mean() for o in [1, 2, 3]]
    stds = [residuals[origins == o].std() for o in [1, 2, 3]]
    counts = [np.sum(origins == o) for o in [1, 2, 3]]

    x = np.arange(3)
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors,
           edgecolor="black", linewidth=1)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{origin_labels[o]}\n(n={counts[i]})" for i, o in enumerate([1, 2, 3])])
    ax.set_ylabel("Mean Residual (+/- SD)")
    ax.set_title("Mean Residuals by Origin")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def compute_tail_statistics(y_obs, y_rep):
    """Compute statistics for tail behavior assessment."""
    stats_dict = {}

    # Observed statistics
    stats_dict["obs_min"] = y_obs.min()
    stats_dict["obs_max"] = y_obs.max()
    stats_dict["obs_q01"] = np.percentile(y_obs, 1)
    stats_dict["obs_q99"] = np.percentile(y_obs, 99)
    stats_dict["obs_range"] = y_obs.max() - y_obs.min()
    stats_dict["obs_iqr"] = np.percentile(y_obs, 75) - np.percentile(y_obs, 25)
    stats_dict["obs_kurtosis"] = float(stats.kurtosis(y_obs))
    stats_dict["obs_skewness"] = float(stats.skew(y_obs))

    # Replicated statistics (posterior predictive)
    y_rep_flat = y_rep.reshape(-1, y_rep.shape[-1])
    n_reps = y_rep_flat.shape[0]

    rep_mins = np.array([y_rep_flat[i].min() for i in range(n_reps)])
    rep_maxs = np.array([y_rep_flat[i].max() for i in range(n_reps)])
    rep_ranges = rep_maxs - rep_mins
    rep_q01 = np.array([np.percentile(y_rep_flat[i], 1) for i in range(n_reps)])
    rep_q99 = np.array([np.percentile(y_rep_flat[i], 99) for i in range(n_reps)])
    rep_kurtosis = np.array([stats.kurtosis(y_rep_flat[i]) for i in range(n_reps)])
    rep_skewness = np.array([stats.skew(y_rep_flat[i]) for i in range(n_reps)])

    # P-values (proportion of replications more extreme than observed)
    stats_dict["pval_min"] = float(np.mean(rep_mins <= stats_dict["obs_min"]))
    stats_dict["pval_max"] = float(np.mean(rep_maxs >= stats_dict["obs_max"]))
    stats_dict["pval_range"] = float(np.mean(rep_ranges >= stats_dict["obs_range"]))
    stats_dict["pval_kurtosis"] = float(np.mean(np.abs(rep_kurtosis) >= np.abs(stats_dict["obs_kurtosis"])))

    # Store replicated distribution summaries
    stats_dict["rep_min_mean"] = float(rep_mins.mean())
    stats_dict["rep_min_std"] = float(rep_mins.std())
    stats_dict["rep_max_mean"] = float(rep_maxs.mean())
    stats_dict["rep_max_std"] = float(rep_maxs.std())
    stats_dict["rep_kurtosis_mean"] = float(rep_kurtosis.mean())

    return stats_dict


def plot_tail_comparison(y_obs_exp3, y_rep_exp3, y_obs_exp2, y_rep_exp2, output_path):
    """Compare tail behavior between Student-t (exp3) and Normal (exp2) models."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Compute replicated minima and maxima
    def get_extremes(y_rep):
        y_flat = y_rep.reshape(-1, y_rep.shape[-1])
        mins = [y_flat[i].min() for i in range(len(y_flat))]
        maxs = [y_flat[i].max() for i in range(len(y_flat))]
        return np.array(mins), np.array(maxs)

    mins_exp3, maxs_exp3 = get_extremes(y_rep_exp3)
    mins_exp2, maxs_exp2 = get_extremes(y_rep_exp2)

    # Top left: Minima comparison
    ax = axes[0, 0]
    ax.hist(mins_exp2, bins=30, alpha=0.5, density=True, label="Normal (A2)", color="C0")
    ax.hist(mins_exp3, bins=30, alpha=0.5, density=True, label="Student-t (A3)", color="C1")
    ax.axvline(y_obs_exp3.min(), color="red", linestyle="--", linewidth=2,
               label=f"Observed min: {y_obs_exp3.min():.2f}")
    ax.set_xlabel("Minimum log(mpg)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive: Minimum Values")
    ax.legend(fontsize=8)

    # Top right: Maxima comparison
    ax = axes[0, 1]
    ax.hist(maxs_exp2, bins=30, alpha=0.5, density=True, label="Normal (A2)", color="C0")
    ax.hist(maxs_exp3, bins=30, alpha=0.5, density=True, label="Student-t (A3)", color="C1")
    ax.axvline(y_obs_exp3.max(), color="red", linestyle="--", linewidth=2,
               label=f"Observed max: {y_obs_exp3.max():.2f}")
    ax.set_xlabel("Maximum log(mpg)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive: Maximum Values")
    ax.legend(fontsize=8)

    # Bottom left: Tail probability comparison (low tail)
    ax = axes[1, 0]
    obs_low_threshold = np.percentile(y_obs_exp3, 5)

    def get_tail_prob_low(y_rep, threshold):
        y_flat = y_rep.reshape(-1, y_rep.shape[-1])
        return [np.mean(y_flat[i] < threshold) for i in range(len(y_flat))]

    tail_probs_exp2 = get_tail_prob_low(y_rep_exp2, obs_low_threshold)
    tail_probs_exp3 = get_tail_prob_low(y_rep_exp3, obs_low_threshold)

    ax.hist(tail_probs_exp2, bins=30, alpha=0.5, density=True, label="Normal (A2)", color="C0")
    ax.hist(tail_probs_exp3, bins=30, alpha=0.5, density=True, label="Student-t (A3)", color="C1")
    ax.axvline(0.05, color="red", linestyle="--", linewidth=2, label="Expected 5%")
    ax.set_xlabel("Proportion below 5th percentile")
    ax.set_ylabel("Density")
    ax.set_title("Lower Tail Coverage")
    ax.legend(fontsize=8)

    # Bottom right: Tail probability comparison (high tail)
    ax = axes[1, 1]
    obs_high_threshold = np.percentile(y_obs_exp3, 95)

    def get_tail_prob_high(y_rep, threshold):
        y_flat = y_rep.reshape(-1, y_rep.shape[-1])
        return [np.mean(y_flat[i] > threshold) for i in range(len(y_flat))]

    tail_probs_exp2 = get_tail_prob_high(y_rep_exp2, obs_high_threshold)
    tail_probs_exp3 = get_tail_prob_high(y_rep_exp3, obs_high_threshold)

    ax.hist(tail_probs_exp2, bins=30, alpha=0.5, density=True, label="Normal (A2)", color="C0")
    ax.hist(tail_probs_exp3, bins=30, alpha=0.5, density=True, label="Student-t (A3)", color="C1")
    ax.axvline(0.05, color="red", linestyle="--", linewidth=2, label="Expected 5%")
    ax.set_xlabel("Proportion above 95th percentile")
    ax.set_ylabel("Density")
    ax.set_title("Upper Tail Coverage")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_extreme_observations(df, residuals, fitted, output_path):
    """Identify and plot extreme observations to assess outlier handling."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Identify extreme residuals
    resid_threshold = 2.5 * np.std(residuals)
    extreme_idx = np.abs(residuals) > resid_threshold

    # Left: Scatter plot highlighting extremes
    ax = axes[0]
    ax.scatter(fitted[~extreme_idx], residuals[~extreme_idx], alpha=0.5, s=30,
               color="C0", label="Normal obs", edgecolors="none")
    ax.scatter(fitted[extreme_idx], residuals[extreme_idx], alpha=0.8, s=50,
               color="C3", marker="^", label=f"Extreme (|r| > 2.5 SD)", edgecolors="black")
    ax.axhline(0, color="gray", linestyle="--")
    ax.axhline(resid_threshold, color="red", linestyle=":", alpha=0.5)
    ax.axhline(-resid_threshold, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Extreme Observations Identified")
    ax.legend()

    # Right: Distribution of observed vs fitted for extreme points
    ax = axes[1]
    extreme_df = df.iloc[extreme_idx].copy()
    extreme_df["residual"] = residuals[extreme_idx]
    extreme_df["fitted"] = fitted[extreme_idx]
    extreme_df["observed"] = df["log_mpg"].values[extreme_idx]

    # Sort by residual magnitude
    extreme_df = extreme_df.sort_values("residual", key=np.abs, ascending=False)

    n_show = min(10, len(extreme_df))
    if n_show > 0:
        x = np.arange(n_show)
        obs_vals = extreme_df["observed"].values[:n_show]
        fit_vals = extreme_df["fitted"].values[:n_show]
        resid_vals = extreme_df["residual"].values[:n_show]

        width = 0.35
        ax.barh(x - width/2, obs_vals, height=width, label="Observed", color="C0", alpha=0.7)
        ax.barh(x + width/2, fit_vals, height=width, label="Fitted", color="C1", alpha=0.7)

        # Add car names
        car_names = extreme_df["car_name"].values[:n_show]
        ax.set_yticks(x)
        ax.set_yticklabels([f"{n[:20]}..." if len(n) > 20 else n for n in car_names], fontsize=8)
        ax.set_xlabel("log(mpg)")
        ax.set_title(f"Top {n_show} Extreme Observations")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No extreme observations found", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    return extreme_df if n_show > 0 else None


def plot_test_statistics(y_obs, y_rep, output_path):
    """Plot posterior predictive check for test statistics not directly fit by model."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    y_rep_flat = y_rep.reshape(-1, y_rep.shape[-1])
    n_reps = len(y_rep_flat)

    # 1. Median (central tendency - not directly in likelihood)
    ax = axes[0, 0]
    rep_medians = [np.median(y_rep_flat[i]) for i in range(n_reps)]
    ax.hist(rep_medians, bins=40, density=True, alpha=0.7, color="C0", edgecolor="white")
    obs_median = np.median(y_obs)
    ax.axvline(obs_median, color="red", linestyle="--", linewidth=2,
               label=f"Observed: {obs_median:.3f}")
    pval = np.mean(np.array(rep_medians) <= obs_median)
    ax.set_xlabel("Median log(mpg)")
    ax.set_ylabel("Density")
    ax.set_title(f"Median (p = {pval:.2f})")
    ax.legend()

    # 2. MAD (spread via robust statistic)
    ax = axes[0, 1]
    rep_mads = [stats.median_abs_deviation(y_rep_flat[i]) for i in range(n_reps)]
    ax.hist(rep_mads, bins=40, density=True, alpha=0.7, color="C0", edgecolor="white")
    obs_mad = stats.median_abs_deviation(y_obs)
    ax.axvline(obs_mad, color="red", linestyle="--", linewidth=2,
               label=f"Observed: {obs_mad:.3f}")
    pval = np.mean(np.array(rep_mads) >= obs_mad)
    ax.set_xlabel("MAD")
    ax.set_ylabel("Density")
    ax.set_title(f"Median Absolute Deviation (p = {pval:.2f})")
    ax.legend()

    # 3. Skewness (shape - not directly fit)
    ax = axes[1, 0]
    rep_skews = [stats.skew(y_rep_flat[i]) for i in range(n_reps)]
    ax.hist(rep_skews, bins=40, density=True, alpha=0.7, color="C0", edgecolor="white")
    obs_skew = stats.skew(y_obs)
    ax.axvline(obs_skew, color="red", linestyle="--", linewidth=2,
               label=f"Observed: {obs_skew:.3f}")
    pval = np.mean(np.abs(np.array(rep_skews)) >= np.abs(obs_skew))
    ax.set_xlabel("Skewness")
    ax.set_ylabel("Density")
    ax.set_title(f"Skewness (two-sided p = {pval:.2f})")
    ax.legend()

    # 4. Kurtosis (tail behavior - key for Student-t)
    ax = axes[1, 1]
    rep_kurts = [stats.kurtosis(y_rep_flat[i]) for i in range(n_reps)]
    ax.hist(rep_kurts, bins=40, density=True, alpha=0.7, color="C0", edgecolor="white")
    obs_kurt = stats.kurtosis(y_obs)
    ax.axvline(obs_kurt, color="red", linestyle="--", linewidth=2,
               label=f"Observed: {obs_kurt:.3f}")
    pval = np.mean(np.abs(np.array(rep_kurts)) >= np.abs(obs_kurt))
    ax.set_xlabel("Excess Kurtosis")
    ax.set_ylabel("Density")
    ax.set_title(f"Kurtosis (two-sided p = {pval:.2f})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    # Return p-values for summary
    return {
        "median_pval": float(np.mean(np.array(rep_medians) <= np.median(y_obs))),
        "mad_pval": float(np.mean(np.array(rep_mads) >= stats.median_abs_deviation(y_obs))),
        "skewness_pval": float(np.mean(np.abs(np.array(rep_skews)) >= np.abs(stats.skew(y_obs)))),
        "kurtosis_pval": float(np.mean(np.abs(np.array(rep_kurts)) >= np.abs(stats.kurtosis(y_obs)))),
    }


def main():
    print("=" * 70)
    print("POSTERIOR PREDICTIVE CHECKS: A3-Robust Model (Student-t Errors)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data()
    y_obs = df["log_mpg"].values
    print(f"  N = {len(df)} observations")

    # Load posteriors
    print("\nLoading posterior samples...")
    idata_exp3 = az.from_netcdf(POSTERIOR_PATH)
    print(f"  Experiment 3 (Student-t): loaded")

    idata_exp2 = az.from_netcdf(POSTERIOR_EXP2_PATH)
    print(f"  Experiment 2 (Normal): loaded")

    # Extract posterior predictive samples
    y_rep_exp3 = idata_exp3.posterior_predictive["y_rep"].values
    y_rep_exp2 = idata_exp2.posterior_predictive["y_rep"].values
    print(f"  y_rep shape (exp3): {y_rep_exp3.shape}")
    print(f"  y_rep shape (exp2): {y_rep_exp2.shape}")

    # Compute fitted values and residuals
    print("\nComputing fitted values and residuals...")
    fitted = compute_fitted_values(idata_exp3, df)
    residuals = compute_residuals(df, fitted)

    # -------------------------------------------------------------------------
    # 1. Observed vs Predicted Distributions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("1. MARGINAL DISTRIBUTION CHECKS")
    print("-" * 50)

    plot_observed_vs_predicted_density(
        y_obs, y_rep_exp3,
        OUTPUT_DIR / "ppc_density_ecdf.png"
    )

    # -------------------------------------------------------------------------
    # 2. LOO-PIT Calibration
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("2. LOO-PIT CALIBRATION")
    print("-" * 50)

    plot_loo_pit(idata_exp3, OUTPUT_DIR / "loo_pit.png")

    # -------------------------------------------------------------------------
    # 3. Test Statistics (not directly fit)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("3. TEST STATISTICS")
    print("-" * 50)

    test_stats_pvals = plot_test_statistics(
        y_obs, y_rep_exp3,
        OUTPUT_DIR / "ppc_test_statistics.png"
    )
    print(f"  Median p-value: {test_stats_pvals['median_pval']:.3f}")
    print(f"  MAD p-value: {test_stats_pvals['mad_pval']:.3f}")
    print(f"  Skewness p-value: {test_stats_pvals['skewness_pval']:.3f}")
    print(f"  Kurtosis p-value: {test_stats_pvals['kurtosis_pval']:.3f}")

    # -------------------------------------------------------------------------
    # 4. Residual Diagnostics
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("4. RESIDUAL DIAGNOSTICS")
    print("-" * 50)

    plot_residuals_vs_fitted(residuals, fitted, OUTPUT_DIR / "residuals_vs_fitted.png")
    plot_residuals_by_year(residuals, df, OUTPUT_DIR / "residuals_by_year.png")
    plot_residuals_by_origin(residuals, df, OUTPUT_DIR / "residuals_by_origin.png")

    # Summary statistics
    print(f"\n  Residual mean: {residuals.mean():.4f}")
    print(f"  Residual std: {residuals.std():.4f}")
    print(f"  Residual range: [{residuals.min():.3f}, {residuals.max():.3f}]")

    # -------------------------------------------------------------------------
    # 5. Tail Behavior Comparison
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("5. TAIL BEHAVIOR: Student-t vs Normal")
    print("-" * 50)

    plot_tail_comparison(
        y_obs, y_rep_exp3,
        y_obs, y_rep_exp2,
        OUTPUT_DIR / "tail_comparison.png"
    )

    # Compute tail statistics
    tail_stats_exp3 = compute_tail_statistics(y_obs, y_rep_exp3)
    tail_stats_exp2 = compute_tail_statistics(y_obs, y_rep_exp2)

    print("\n  Observed extremes:")
    print(f"    Min log(mpg): {tail_stats_exp3['obs_min']:.3f} (MPG: {np.exp(tail_stats_exp3['obs_min']):.1f})")
    print(f"    Max log(mpg): {tail_stats_exp3['obs_max']:.3f} (MPG: {np.exp(tail_stats_exp3['obs_max']):.1f})")

    print("\n  Model capability to generate extremes (p-values):")
    print(f"    Min - Student-t: {tail_stats_exp3['pval_min']:.3f}, Normal: {tail_stats_exp2['pval_min']:.3f}")
    print(f"    Max - Student-t: {tail_stats_exp3['pval_max']:.3f}, Normal: {tail_stats_exp2['pval_max']:.3f}")

    print("\n  Kurtosis check:")
    print(f"    Observed kurtosis: {tail_stats_exp3['obs_kurtosis']:.3f}")
    print(f"    Replicated kurtosis (Student-t): {tail_stats_exp3['rep_kurtosis_mean']:.3f}")
    print(f"    Kurtosis p-value: {tail_stats_exp3['pval_kurtosis']:.3f}")

    # -------------------------------------------------------------------------
    # 6. Extreme Observations Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("6. EXTREME OBSERVATIONS ANALYSIS")
    print("-" * 50)

    extreme_df = plot_extreme_observations(
        df, residuals, fitted,
        OUTPUT_DIR / "extreme_observations.png"
    )

    if extreme_df is not None and len(extreme_df) > 0:
        print(f"\n  Found {len(extreme_df)} extreme observations (|residual| > 2.5 SD)")
        print("\n  Top 5 by residual magnitude:")
        for i, (_, row) in enumerate(extreme_df.head(5).iterrows()):
            direction = "under-predicted" if row["residual"] > 0 else "over-predicted"
            print(f"    {i+1}. {row['car_name'][:30]}: MPG={row['mpg']:.1f}, {direction}")

    # -------------------------------------------------------------------------
    # Save Summary Statistics
    # -------------------------------------------------------------------------
    summary = {
        "model": "A3-Robust (Student-t errors)",
        "n_observations": len(df),
        "residual_statistics": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
        },
        "test_statistic_pvalues": test_stats_pvals,
        "tail_statistics": {
            "student_t": tail_stats_exp3,
            "normal": tail_stats_exp2,
        },
        "extreme_observations": {
            "count": len(extreme_df) if extreme_df is not None else 0,
            "threshold": "2.5 SD",
        },
    }

    with open(OUTPUT_DIR / "ppc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'ppc_summary.json'}")

    print("\n" + "=" * 70)
    print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    summary = main()
