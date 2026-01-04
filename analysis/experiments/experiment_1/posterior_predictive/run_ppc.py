"""Posterior Predictive Checks for A1-Baseline Model.

This script performs comprehensive posterior predictive checks to assess
model fit quality, including:
- Distribution comparisons (observed vs replicated)
- Residual analysis vs fitted values and omitted predictors
- Test statistics and p-values
- LOO-PIT calibration
"""

import json
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
FIT_DIR = Path("/workspace/analysis/experiments/experiment_1/fit")
DATA_FILE = Path("/workspace/analysis/eda/auto_mpg_cleaned.csv")
OUTPUT_DIR = Path("/workspace/analysis/experiments/experiment_1/posterior_predictive")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_and_posterior():
    """Load observed data and posterior samples."""
    # Load raw data
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["horsepower"])
    df = df[df["horsepower"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    # Compute transforms (matching fit script)
    log_mpg = np.log(df["mpg"].values)
    log_weight = np.log(df["weight"].values)
    log_weight_mean = log_weight.mean()
    log_weight_c = log_weight - log_weight_mean

    # Load posterior
    idata = az.from_netcdf(FIT_DIR / "posterior.nc")

    # Extract posterior predictive samples (y_rep)
    y_rep = idata.posterior_predictive["y_rep"].values  # shape: (chains, draws, N)
    n_chains, n_draws, n_obs = y_rep.shape
    y_rep_flat = y_rep.reshape(n_chains * n_draws, n_obs)  # (total_draws, N)

    # Extract posterior parameters for computing fitted values
    alpha = idata.posterior["alpha"].values.flatten()
    beta_weight = idata.posterior["beta_weight"].values.flatten()
    sigma = idata.posterior["sigma"].values.flatten()

    # Compute posterior mean predictions (fitted values)
    mu_hat = alpha.mean() + beta_weight.mean() * log_weight_c

    data_dict = {
        "df": df,
        "log_mpg": log_mpg,
        "log_weight": log_weight,
        "log_weight_c": log_weight_c,
        "log_weight_mean": log_weight_mean,
        "year": df["model_year"].values,
        "origin": df["origin"].values,
        "mu_hat": mu_hat,
        "alpha_mean": alpha.mean(),
        "beta_weight_mean": beta_weight.mean(),
        "sigma_mean": sigma.mean(),
    }

    return idata, y_rep_flat, data_dict


def compute_residuals(y_obs, mu_hat):
    """Compute residuals (observed - predicted)."""
    return y_obs - mu_hat


def compute_test_statistics(y, label=""):
    """Compute summary statistics for a single realization or observed data."""
    return {
        f"mean{label}": np.mean(y),
        f"sd{label}": np.std(y),
        f"median{label}": np.median(y),
        f"min{label}": np.min(y),
        f"max{label}": np.max(y),
        f"iqr{label}": np.percentile(y, 75) - np.percentile(y, 25),
        f"skewness{label}": stats.skew(y),
        f"kurtosis{label}": stats.kurtosis(y),
        f"q10{label}": np.percentile(y, 10),
        f"q90{label}": np.percentile(y, 90),
    }


def compute_ppc_pvalues(y_obs, y_rep):
    """Compute posterior predictive p-values for various test statistics.

    p-value = P(T(y_rep) >= T(y_obs)) where T is a test statistic.
    Values near 0 or 1 indicate model misfit.
    """
    n_draws = y_rep.shape[0]

    # Observed statistics
    T_obs = compute_test_statistics(y_obs)

    # Replicated statistics
    T_rep = {k: [] for k in T_obs.keys()}
    for i in range(n_draws):
        stats_i = compute_test_statistics(y_rep[i, :])
        for k, v in stats_i.items():
            T_rep[k].append(v)

    # Compute p-values
    p_values = {}
    for k in T_obs.keys():
        T_rep_arr = np.array(T_rep[k])
        # Two-sided p-value: proportion more extreme than observed
        p_values[k] = np.mean(T_rep_arr >= T_obs[k])

    return p_values, T_obs, {k: np.array(v) for k, v in T_rep.items()}


def plot_density_overlay(y_obs, y_rep, output_path):
    """Plot density comparison: observed vs posterior predictive."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: log(mpg) scale
    ax = axes[0]
    # Plot subset of replications
    n_plot = min(100, y_rep.shape[0])
    idx = np.random.choice(y_rep.shape[0], n_plot, replace=False)
    for i in idx:
        ax.hist(y_rep[i, :], bins=30, alpha=0.03, color="blue", density=True)
    ax.hist(y_obs, bins=30, alpha=0.7, color="black", density=True,
            label="Observed", histtype="step", linewidth=2)
    ax.set_xlabel("log(mpg)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive: log(MPG)")
    ax.legend()

    # Right: original MPG scale
    ax = axes[1]
    for i in idx:
        ax.hist(np.exp(y_rep[i, :]), bins=30, alpha=0.03, color="blue", density=True)
    ax.hist(np.exp(y_obs), bins=30, alpha=0.7, color="black", density=True,
            label="Observed", histtype="step", linewidth=2)
    ax.set_xlabel("MPG")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive: MPG (original scale)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ppc_ecdf(y_obs, y_rep, output_path):
    """Plot ECDF comparison with uncertainty bands."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Observed ECDF
    y_sorted = np.sort(y_obs)
    n = len(y_obs)
    ecdf_obs = np.arange(1, n + 1) / n

    # Replicated ECDFs - compute pointwise quantiles
    n_draws = y_rep.shape[0]
    ecdf_reps = np.zeros((n_draws, n))
    for i in range(n_draws):
        y_rep_sorted = np.sort(y_rep[i, :])
        # For each observed quantile position, find where it falls in replicated
        for j, q in enumerate(ecdf_obs):
            ecdf_reps[i, j] = np.mean(y_rep[i, :] <= y_sorted[j])

    # Compute bands
    ecdf_median = np.median(ecdf_reps, axis=0)
    ecdf_lo = np.percentile(ecdf_reps, 2.5, axis=0)
    ecdf_hi = np.percentile(ecdf_reps, 97.5, axis=0)

    ax.fill_between(y_sorted, ecdf_lo, ecdf_hi, alpha=0.3, color="blue", label="95% interval")
    ax.plot(y_sorted, ecdf_median, "b-", alpha=0.7, label="Median y_rep")
    ax.step(y_sorted, ecdf_obs, "k-", where="post", linewidth=2, label="Observed")

    ax.set_xlabel("log(mpg)")
    ax.set_ylabel("ECDF")
    ax.set_title("Posterior Predictive ECDF Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_residuals_vs_fitted(residuals, mu_hat, output_path):
    """Plot residuals vs fitted values to check homoscedasticity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: residuals vs fitted
    ax = axes[0]
    ax.scatter(mu_hat, residuals, alpha=0.5, s=20)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    # Add LOESS-like smoothing (binned means)
    bins = np.linspace(mu_hat.min(), mu_hat.max(), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (mu_hat >= bins[i]) & (mu_hat < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(residuals[mask].mean())
        else:
            bin_means.append(np.nan)
    ax.plot(bin_centers, bin_means, "r-o", linewidth=2, markersize=8, label="Binned mean")
    ax.set_xlabel("Fitted values (log scale)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted Values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: scale-location plot (sqrt absolute residuals)
    ax = axes[1]
    sqrt_abs_res = np.sqrt(np.abs(residuals))
    ax.scatter(mu_hat, sqrt_abs_res, alpha=0.5, s=20)
    # Binned means for spread
    bin_means_spread = []
    for i in range(len(bins) - 1):
        mask = (mu_hat >= bins[i]) & (mu_hat < bins[i+1])
        if mask.sum() > 0:
            bin_means_spread.append(sqrt_abs_res[mask].mean())
        else:
            bin_means_spread.append(np.nan)
    ax.plot(bin_centers, bin_means_spread, "r-o", linewidth=2, markersize=8, label="Binned mean")
    ax.set_xlabel("Fitted values (log scale)")
    ax.set_ylabel("sqrt(|Residuals|)")
    ax.set_title("Scale-Location Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_residuals_vs_omitted(residuals, data_dict, output_path):
    """Plot residuals vs predictors NOT in the model (year, origin)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    year = data_dict["year"]
    origin = data_dict["origin"]

    # Left: residuals vs year
    ax = axes[0]
    ax.scatter(year, residuals, alpha=0.5, s=20)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # Compute mean residual by year
    years_unique = np.sort(np.unique(year))
    year_means = [residuals[year == y].mean() for y in years_unique]
    year_sems = [residuals[year == y].std() / np.sqrt((year == y).sum()) for y in years_unique]
    ax.errorbar(years_unique, year_means, yerr=1.96*np.array(year_sems),
                fmt="ro-", linewidth=2, markersize=8, capsize=3, label="Mean +/- 95% CI")

    ax.set_xlabel("Model Year")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Model Year (NOT in model)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: residuals by origin
    ax = axes[1]
    origin_labels = {1: "USA", 2: "Europe", 3: "Japan"}
    origins_unique = [1, 2, 3]

    # Box plot
    data_by_origin = [residuals[origin == o] for o in origins_unique]
    bp = ax.boxplot(data_by_origin, labels=[origin_labels[o] for o in origins_unique])
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # Add means
    for i, o in enumerate(origins_unique):
        mean_val = residuals[origin == o].mean()
        ax.scatter(i + 1, mean_val, color="red", s=100, zorder=5, marker="D")

    ax.set_xlabel("Origin")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals by Origin (NOT in model)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_test_statistic_ppc(T_obs, T_rep, output_path):
    """Plot posterior predictive distributions of test statistics."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    stats_to_plot = ["mean", "sd", "median", "iqr", "skewness", "min", "max", "q90"]

    for i, stat in enumerate(stats_to_plot):
        ax = axes[i]
        T_rep_vals = T_rep[stat]
        T_obs_val = T_obs[stat]

        ax.hist(T_rep_vals, bins=40, alpha=0.7, color="blue", density=True)
        ax.axvline(T_obs_val, color="red", linewidth=2, linestyle="--", label=f"Observed: {T_obs_val:.3f}")

        # Compute p-value
        p_val = np.mean(T_rep_vals >= T_obs_val)
        p_val_2sided = 2 * min(p_val, 1 - p_val)

        ax.set_title(f"{stat}\np = {p_val:.3f}")
        ax.set_xlabel(stat)
        ax.legend(fontsize=8)

    plt.suptitle("Posterior Predictive Check: Test Statistics", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_loo_pit(idata, output_path):
    """Plot LOO-PIT to check calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compute LOO-PIT values
    try:
        # ArviZ LOO-PIT
        az.plot_loo_pit(idata, y="y", ax=axes[0])
        axes[0].set_title("LOO-PIT: ECDF")
    except Exception as e:
        print(f"LOO-PIT plot failed: {e}")
        # Manual PIT using posterior predictive
        y_obs = idata.observed_data["y"].values
        y_rep = idata.posterior_predictive["y_rep"].values
        n_chains, n_draws, n_obs = y_rep.shape
        y_rep_flat = y_rep.reshape(-1, n_obs)

        # Compute PIT values (proportion of y_rep <= y_obs for each observation)
        pit_values = np.mean(y_rep_flat <= y_obs, axis=0)

        # Plot ECDF of PIT values
        pit_sorted = np.sort(pit_values)
        n = len(pit_sorted)
        ecdf = np.arange(1, n + 1) / n

        axes[0].plot(pit_sorted, ecdf, "b-", linewidth=2, label="PIT ECDF")
        axes[0].plot([0, 1], [0, 1], "k--", label="Uniform")
        axes[0].set_xlabel("PIT value")
        axes[0].set_ylabel("ECDF")
        axes[0].set_title("PIT ECDF (should be uniform if calibrated)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Histogram of PIT values
    try:
        y_obs = idata.observed_data["y"].values
        y_rep = idata.posterior_predictive["y_rep"].values
        n_chains, n_draws, n_obs = y_rep.shape
        y_rep_flat = y_rep.reshape(-1, n_obs)
        pit_values = np.mean(y_rep_flat <= y_obs, axis=0)

        axes[1].hist(pit_values, bins=20, alpha=0.7, density=True, edgecolor="black")
        axes[1].axhline(1, color="red", linestyle="--", linewidth=2, label="Uniform")
        axes[1].set_xlabel("PIT value")
        axes[1].set_ylabel("Density")
        axes[1].set_title("PIT Histogram (should be flat if calibrated)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    except Exception as e:
        print(f"PIT histogram failed: {e}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def compute_residual_statistics_by_group(residuals, groups, group_name):
    """Compute residual statistics by group for reporting."""
    results = {}
    for g in np.unique(groups):
        mask = groups == g
        r = residuals[mask]
        results[f"{group_name}={g}"] = {
            "n": int(mask.sum()),
            "mean_residual": float(r.mean()),
            "sd_residual": float(r.std()),
            "t_stat": float(r.mean() / (r.std() / np.sqrt(len(r)))),
        }
    return results


def main():
    """Run all posterior predictive checks."""
    print("=" * 60)
    print("Loading data and posterior samples...")
    idata, y_rep, data_dict = load_data_and_posterior()
    y_obs = data_dict["log_mpg"]
    mu_hat = data_dict["mu_hat"]

    print(f"Observations: {len(y_obs)}")
    print(f"Posterior draws: {y_rep.shape[0]}")
    print(f"Posterior mean alpha: {data_dict['alpha_mean']:.4f}")
    print(f"Posterior mean beta_weight: {data_dict['beta_weight_mean']:.4f}")
    print(f"Posterior mean sigma: {data_dict['sigma_mean']:.4f}")

    # Compute residuals
    residuals = compute_residuals(y_obs, mu_hat)
    print(f"Residual mean: {residuals.mean():.4f}")
    print(f"Residual SD: {residuals.std():.4f}")

    # 1. Density overlay plots
    print("\n" + "=" * 60)
    print("Generating density overlay plots...")
    plot_density_overlay(y_obs, y_rep, OUTPUT_DIR / "ppc_density_overlay.png")

    # 2. ECDF comparison
    print("\n" + "=" * 60)
    print("Generating ECDF comparison...")
    plot_ppc_ecdf(y_obs, y_rep, OUTPUT_DIR / "ppc_ecdf.png")

    # 3. Residuals vs fitted
    print("\n" + "=" * 60)
    print("Generating residual plots...")
    plot_residuals_vs_fitted(residuals, mu_hat, OUTPUT_DIR / "residuals_vs_fitted.png")

    # 4. Residuals vs omitted predictors (KEY CHECK)
    print("\n" + "=" * 60)
    print("Generating residuals vs omitted predictors (year, origin)...")
    plot_residuals_vs_omitted(residuals, data_dict, OUTPUT_DIR / "residuals_vs_omitted.png")

    # 5. Test statistics and p-values
    print("\n" + "=" * 60)
    print("Computing posterior predictive p-values...")
    p_values, T_obs, T_rep = compute_ppc_pvalues(y_obs, y_rep)

    print("\nTest Statistic P-values:")
    for stat, p in p_values.items():
        flag = " ***" if p < 0.05 or p > 0.95 else ""
        print(f"  {stat}: p = {p:.3f}{flag}")

    plot_test_statistic_ppc(T_obs, T_rep, OUTPUT_DIR / "ppc_test_statistics.png")

    # 6. LOO-PIT calibration
    print("\n" + "=" * 60)
    print("Generating LOO-PIT calibration plot...")
    plot_loo_pit(idata, OUTPUT_DIR / "loo_pit.png")

    # 7. Compute residual statistics by year and origin
    print("\n" + "=" * 60)
    print("Residual analysis by omitted predictors:")

    year_stats = compute_residual_statistics_by_group(residuals, data_dict["year"], "year")
    origin_stats = compute_residual_statistics_by_group(residuals, data_dict["origin"], "origin")

    print("\nBy Year:")
    for k, v in sorted(year_stats.items()):
        print(f"  {k}: n={v['n']}, mean={v['mean_residual']:.3f}, t={v['t_stat']:.2f}")

    print("\nBy Origin:")
    for k, v in sorted(origin_stats.items()):
        origin_labels = {1: "USA", 2: "Europe", 3: "Japan"}
        origin_num = int(k.split("=")[1])
        print(f"  {origin_labels[origin_num]}: n={v['n']}, mean={v['mean_residual']:.3f}, t={v['t_stat']:.2f}")

    # Save results
    results = {
        "model": "A1-Baseline (log(mpg) ~ log(weight))",
        "n_observations": len(y_obs),
        "n_posterior_draws": int(y_rep.shape[0]),
        "posterior_means": {
            "alpha": float(data_dict["alpha_mean"]),
            "beta_weight": float(data_dict["beta_weight_mean"]),
            "sigma": float(data_dict["sigma_mean"]),
        },
        "residual_summary": {
            "mean": float(residuals.mean()),
            "sd": float(residuals.std()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
        },
        "ppc_p_values": {k: float(v) for k, v in p_values.items()},
        "observed_statistics": {k: float(v) for k, v in T_obs.items()},
        "residuals_by_year": year_stats,
        "residuals_by_origin": origin_stats,
    }

    with open(OUTPUT_DIR / "ppc_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'ppc_results.json'}")

    print("\n" + "=" * 60)
    print("Posterior predictive checks complete!")
    print(f"Output directory: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    main()
