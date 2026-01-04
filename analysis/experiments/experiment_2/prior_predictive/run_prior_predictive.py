"""Prior predictive check for A2-Year model.

Samples from priors only (no likelihood) to generate prior predictive
distributions for model plausibility assessment.
"""

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import CmdStanModel

# Paths
EXPERIMENT_DIR = Path("/workspace/analysis/experiments/experiment_2")
MODEL_FILE = EXPERIMENT_DIR / "model.stan"
PRIOR_MODEL_FILE = EXPERIMENT_DIR / "prior_predictive" / "model_prior_only.stan"
OUTPUT_DIR = EXPERIMENT_DIR / "prior_predictive"
DATA_FILE = Path("/workspace/analysis/data/auto-mpg.data")

# Constants from data context
N = 392
MPG_MIN = 9.0
MPG_MAX = 46.6
YEAR_C_MIN = -6  # centered at 76
YEAR_C_MAX = 6

# Log transformations of MPG bounds
LOG_MPG_MIN = np.log(MPG_MIN)  # ~2.20
LOG_MPG_MAX = np.log(MPG_MAX)  # ~3.84


def load_and_prepare_data() -> dict:
    """Load auto-mpg data and prepare for Stan."""
    # Read raw data
    data = []
    with open(DATA_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "?" or parts[3] == "?":
                continue  # skip missing values
            mpg = float(parts[0])
            weight = float(parts[4].rstrip("."))
            year = int(parts[6])
            data.append((mpg, weight, year))

    mpg = np.array([d[0] for d in data])
    weight = np.array([d[1] for d in data])
    year = np.array([d[2] for d in data])

    # Transform and center
    log_mpg = np.log(mpg)
    log_weight = np.log(weight)
    log_weight_c = log_weight - 7.96  # centering constant from model comment
    year_c = year - 76  # centered at 1976

    return {
        "N": len(mpg),
        "log_mpg": log_mpg.tolist(),
        "log_weight_c": log_weight_c.tolist(),
        "year_c": year_c.tolist(),
        # Keep originals for reference
        "_mpg": mpg,
        "_weight": weight,
        "_year": year,
        "_log_mpg": log_mpg,
    }


def run_prior_predictive(data: dict, n_draws: int = 1000) -> az.InferenceData:
    """Run prior predictive simulation using prior-only model."""
    # Compile prior-only model (no likelihood term)
    print("Compiling prior-only model...")
    model = CmdStanModel(stan_file=str(PRIOR_MODEL_FILE))

    # Prepare Stan data (remove internal fields)
    stan_data = {k: v for k, v in data.items() if not k.startswith("_")}

    # Sample from priors using the prior-only model
    # This model has no likelihood, so MCMC samples directly from priors
    print("Sampling from priors...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=500,
        iter_sampling=n_draws,
        show_progress=True,
    )

    # Convert to ArviZ InferenceData
    # For prior predictive, we use prior and prior_predictive groups
    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive=["y_rep"],
        observed_data={"y": data["_log_mpg"]},
    )

    # Rename posterior groups to prior groups for proper semantics
    # ArviZ from_cmdstanpy puts samples in posterior by default
    # Use xarray Dataset operations for renaming
    import xarray as xr

    groups = {}
    if "posterior" in idata:
        groups["prior"] = idata.posterior
    if "posterior_predictive" in idata:
        groups["prior_predictive"] = idata.posterior_predictive
    if "observed_data" in idata:
        groups["observed_data"] = idata.observed_data

    idata = az.InferenceData(**groups)

    return idata


def compute_statistics(idata: az.InferenceData, data: dict) -> dict:
    """Compute summary statistics for prior predictive check."""
    # Extract prior predictive samples
    y_rep = idata.prior_predictive["y_rep"].values.flatten()

    # Observed data bounds (in log scale)
    log_mpg_obs = data["_log_mpg"]

    # Coverage statistics
    in_range = (y_rep >= LOG_MPG_MIN) & (y_rep <= LOG_MPG_MAX)
    coverage = np.mean(in_range)

    # Extreme value frequencies
    below_min = np.mean(y_rep < LOG_MPG_MIN)
    above_max = np.mean(y_rep > LOG_MPG_MAX)

    # Convert to MPG scale for interpretability
    mpg_rep = np.exp(y_rep)
    mpg_rep_clipped = mpg_rep[(mpg_rep > 0) & (mpg_rep < 1000)]  # remove numerical issues

    # Summary stats on MPG scale
    stats = {
        "log_scale": {
            "mean": float(np.mean(y_rep)),
            "std": float(np.std(y_rep)),
            "min": float(np.min(y_rep)),
            "max": float(np.max(y_rep)),
            "p2_5": float(np.percentile(y_rep, 2.5)),
            "p97_5": float(np.percentile(y_rep, 97.5)),
        },
        "mpg_scale": {
            "mean": float(np.mean(mpg_rep_clipped)),
            "median": float(np.median(mpg_rep_clipped)),
            "p2_5": float(np.percentile(mpg_rep_clipped, 2.5)),
            "p97_5": float(np.percentile(mpg_rep_clipped, 97.5)),
            "min": float(np.min(mpg_rep_clipped)),
            "max": float(np.max(mpg_rep_clipped)),
        },
        "coverage": {
            "within_observed_range": float(coverage),
            "below_min": float(below_min),
            "above_max": float(above_max),
        },
        "observed_bounds": {
            "log_mpg_min": float(LOG_MPG_MIN),
            "log_mpg_max": float(LOG_MPG_MAX),
            "mpg_min": float(MPG_MIN),
            "mpg_max": float(MPG_MAX),
        },
    }

    # Prior parameter samples
    prior_samples = idata.prior
    stats["prior_parameters"] = {
        "alpha": {
            "mean": float(prior_samples["alpha"].values.mean()),
            "std": float(prior_samples["alpha"].values.std()),
        },
        "beta_weight": {
            "mean": float(prior_samples["beta_weight"].values.mean()),
            "std": float(prior_samples["beta_weight"].values.std()),
        },
        "beta_year": {
            "mean": float(prior_samples["beta_year"].values.mean()),
            "std": float(prior_samples["beta_year"].values.std()),
        },
        "sigma": {
            "mean": float(prior_samples["sigma"].values.mean()),
            "std": float(prior_samples["sigma"].values.std()),
        },
    }

    return stats


def create_diagnostic_plots(idata: az.InferenceData, data: dict, output_dir: Path):
    """Create diagnostic visualizations for prior predictive check."""
    # Extract data
    y_rep = idata.prior_predictive["y_rep"].values
    y_obs = data["_log_mpg"]

    # Prior parameter samples
    prior = idata.prior

    # Figure 1: Prior predictive distribution vs observed data range
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1a: Prior predictive distribution (log scale)
    ax = axes[0, 0]
    # Flatten y_rep across chains and draws
    y_rep_flat = y_rep.flatten()
    ax.hist(y_rep_flat, bins=100, density=True, alpha=0.7, color="steelblue", label="Prior predictive")
    ax.axvline(LOG_MPG_MIN, color="red", linestyle="--", linewidth=2, label=f"Observed min ({MPG_MIN} MPG)")
    ax.axvline(LOG_MPG_MAX, color="red", linestyle="--", linewidth=2, label=f"Observed max ({MPG_MAX} MPG)")
    ax.axvline(np.mean(y_obs), color="green", linestyle="-", linewidth=2, label="Observed mean")
    ax.set_xlabel("log(MPG)")
    ax.set_ylabel("Density")
    ax.set_title("Prior Predictive Distribution (log scale)")
    ax.legend(fontsize=8)

    # 1b: Prior predictive in MPG scale
    ax = axes[0, 1]
    mpg_rep = np.exp(y_rep_flat)
    # Clip to reasonable range for visualization
    mpg_rep_viz = mpg_rep[(mpg_rep > 0) & (mpg_rep < 200)]
    ax.hist(mpg_rep_viz, bins=100, density=True, alpha=0.7, color="steelblue", label="Prior predictive")
    ax.axvline(MPG_MIN, color="red", linestyle="--", linewidth=2, label=f"Min ({MPG_MIN})")
    ax.axvline(MPG_MAX, color="red", linestyle="--", linewidth=2, label=f"Max ({MPG_MAX})")
    ax.axvline(np.mean(data["_mpg"]), color="green", linestyle="-", linewidth=2, label=f"Mean ({np.mean(data['_mpg']):.1f})")
    ax.set_xlabel("MPG")
    ax.set_ylabel("Density")
    ax.set_title("Prior Predictive Distribution (MPG scale)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8)

    # 1c: Prior parameter distributions
    ax = axes[1, 0]
    params = ["alpha", "beta_weight", "beta_year", "sigma"]
    colors = ["C0", "C1", "C2", "C3"]
    for i, (param, color) in enumerate(zip(params, colors)):
        samples = prior[param].values.flatten()
        ax.hist(samples, bins=50, alpha=0.5, color=color, label=param, density=True)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Density")
    ax.set_title("Prior Parameter Distributions")
    ax.legend()

    # 1d: Coverage by percentile
    ax = axes[1, 1]
    percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
    pct_values = np.percentile(y_rep_flat, percentiles)
    ax.bar(range(len(percentiles)), pct_values, tick_label=[f"{p}%" for p in percentiles], color="steelblue", alpha=0.7)
    ax.axhline(LOG_MPG_MIN, color="red", linestyle="--", linewidth=2, label=f"Observed min")
    ax.axhline(LOG_MPG_MAX, color="red", linestyle="--", linewidth=2, label=f"Observed max")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("log(MPG)")
    ax.set_title("Prior Predictive Percentiles vs Observed Range")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "prior_predictive_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: ECDF comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw subset of prior predictive ECDFs (thin lines)
    n_draws_to_plot = 50
    n_chains = y_rep.shape[0]
    n_samples = y_rep.shape[1]
    indices = np.random.choice(n_chains * n_samples, min(n_draws_to_plot, n_chains * n_samples), replace=False)

    for idx in indices:
        chain_idx = idx // n_samples
        sample_idx = idx % n_samples
        y_sample = y_rep[chain_idx, sample_idx, :]
        sorted_y = np.sort(y_sample)
        ecdf = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
        ax.step(sorted_y, ecdf, color="steelblue", alpha=0.1, linewidth=0.5)

    # Observed ECDF
    sorted_obs = np.sort(y_obs)
    ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
    ax.step(sorted_obs, ecdf_obs, color="black", linewidth=2, label="Observed data")

    ax.axvline(LOG_MPG_MIN, color="red", linestyle="--", alpha=0.5)
    ax.axvline(LOG_MPG_MAX, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("log(MPG)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Prior Predictive ECDFs vs Observed Data")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "prior_predictive_ecdf.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved diagnostic plots to {output_dir}")


def main():
    """Run prior predictive checks."""
    print("=" * 60)
    print("Prior Predictive Check: A2-Year Model")
    print("=" * 60)

    # Load data
    print("\nLoading and preparing data...")
    data = load_and_prepare_data()
    print(f"  N = {data['N']} observations")
    print(f"  log(MPG) range: [{LOG_MPG_MIN:.3f}, {LOG_MPG_MAX:.3f}]")
    print(f"  MPG range: [{MPG_MIN}, {MPG_MAX}]")

    # Run prior predictive simulation
    print("\nRunning prior predictive simulation...")
    idata = run_prior_predictive(data, n_draws=1000)

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(idata, data)

    # Print summary
    print("\n" + "=" * 60)
    print("PRIOR PREDICTIVE SUMMARY")
    print("=" * 60)
    print(f"\nCoverage of observed MPG range ({MPG_MIN}-{MPG_MAX}):")
    print(f"  Within range: {stats['coverage']['within_observed_range']:.1%}")
    print(f"  Below min:    {stats['coverage']['below_min']:.1%}")
    print(f"  Above max:    {stats['coverage']['above_max']:.1%}")

    print(f"\nPrior predictive on log(MPG) scale:")
    print(f"  Mean: {stats['log_scale']['mean']:.3f}")
    print(f"  Std:  {stats['log_scale']['std']:.3f}")
    print(f"  95% interval: [{stats['log_scale']['p2_5']:.3f}, {stats['log_scale']['p97_5']:.3f}]")

    print(f"\nPrior predictive on MPG scale:")
    print(f"  Median: {stats['mpg_scale']['median']:.1f}")
    print(f"  95% interval: [{stats['mpg_scale']['p2_5']:.1f}, {stats['mpg_scale']['p97_5']:.1f}]")

    print(f"\nPrior parameter means (sampled):")
    for param, pstats in stats["prior_parameters"].items():
        print(f"  {param}: {pstats['mean']:.4f} (std: {pstats['std']:.4f})")

    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    create_diagnostic_plots(idata, data, OUTPUT_DIR)

    # Save results
    print("\nSaving results...")
    idata.to_netcdf(str(OUTPUT_DIR / "prior_predictive.nc"))
    with open(OUTPUT_DIR / "prior_predictive_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Assessment
    print("\n" + "=" * 60)
    print("ASSESSMENT")
    print("=" * 60)

    coverage = stats["coverage"]["within_observed_range"]
    issues = []

    if coverage < 0.3:
        issues.append(f"Low coverage ({coverage:.1%}) - priors may be too narrow or miscentered")
    if coverage > 0.99:
        issues.append(f"Very high coverage ({coverage:.1%}) - priors may be too vague")
    if stats["coverage"]["below_min"] > 0.3:
        issues.append(f"Too many predictions below minimum ({stats['coverage']['below_min']:.1%})")
    if stats["coverage"]["above_max"] > 0.3:
        issues.append(f"Too many predictions above maximum ({stats['coverage']['above_max']:.1%})")

    # Check for numerical issues
    y_rep = idata.prior_predictive["y_rep"].values.flatten()
    if np.any(np.isnan(y_rep)) or np.any(np.isinf(y_rep)):
        issues.append("Numerical issues detected (NaN or Inf values)")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        recommendation = "FAIL" if len(issues) > 1 or coverage < 0.2 else "PASS with minor concerns"
    else:
        recommendation = "PASS"
        print("No issues found. Priors generate plausible data.")

    print(f"\nRecommendation: {recommendation}")
    print("=" * 60)

    # Save recommendation
    with open(OUTPUT_DIR / "recommendation.txt", "w") as f:
        f.write(f"Recommendation: {recommendation}\n")
        if issues:
            f.write("\nIssues:\n")
            for issue in issues:
                f.write(f"  - {issue}\n")

    return recommendation


if __name__ == "__main__":
    main()
