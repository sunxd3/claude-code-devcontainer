"""Prior predictive check for A3-Robust model (Student-t errors).

Samples from priors only (no likelihood) to generate prior predictive
distributions for model plausibility assessment.
"""

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# Paths
EXPERIMENT_DIR = Path("/workspace/analysis/experiments/experiment_3")
PRIOR_MODEL_FILE = EXPERIMENT_DIR / "prior_predictive" / "model_prior_only.stan"
OUTPUT_DIR = EXPERIMENT_DIR / "prior_predictive"
DATA_FILE = Path("/workspace/analysis/eda/auto_mpg_cleaned.csv")

# Data bounds from EDA
MPG_MIN = 9.0
MPG_MAX = 46.6
LOG_MPG_MIN = np.log(MPG_MIN)
LOG_MPG_MAX = np.log(MPG_MAX)


def load_and_prepare_data() -> dict:
    """Load auto-mpg data and prepare for Stan."""
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["horsepower"])

    mpg = df["mpg"].values
    weight = df["weight"].values
    year = df["model_year"].values

    log_mpg = np.log(mpg)
    log_weight_c = np.log(weight) - 7.96
    year_c = year - 76

    return {
        "N": len(mpg),
        "log_mpg": log_mpg.tolist(),
        "log_weight_c": log_weight_c.tolist(),
        "year_c": year_c.tolist(),
        "_mpg": mpg,
        "_log_mpg": log_mpg,
    }


def run_prior_predictive(data: dict, n_draws: int = 1000) -> az.InferenceData:
    """Run prior predictive simulation using prior-only model."""
    print("Compiling prior-only model...")
    model = CmdStanModel(stan_file=str(PRIOR_MODEL_FILE))

    stan_data = {k: v for k, v in data.items() if not k.startswith("_")}

    print("Sampling from priors...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=500,
        iter_sampling=n_draws,
        show_progress=True,
    )

    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive=["y_rep"],
        observed_data={"y": data["_log_mpg"]},
    )

    # Rename posterior groups to prior groups
    groups = {}
    if "posterior" in idata:
        groups["prior"] = idata.posterior
    if "posterior_predictive" in idata:
        groups["prior_predictive"] = idata.posterior_predictive
    if "observed_data" in idata:
        groups["observed_data"] = idata.observed_data

    return az.InferenceData(**groups)


def compute_statistics(idata: az.InferenceData, data: dict) -> dict:
    """Compute summary statistics for prior predictive check."""
    y_rep = idata.prior_predictive["y_rep"].values.flatten()
    log_mpg_obs = data["_log_mpg"]

    in_range = (y_rep >= LOG_MPG_MIN) & (y_rep <= LOG_MPG_MAX)
    coverage = np.mean(in_range)
    below_min = np.mean(y_rep < LOG_MPG_MIN)
    above_max = np.mean(y_rep > LOG_MPG_MAX)

    mpg_rep = np.exp(y_rep)
    mpg_rep_clipped = mpg_rep[(mpg_rep > 0) & (mpg_rep < 1000)]

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
        },
        "coverage": {
            "within_observed_range": float(coverage),
            "below_min": float(below_min),
            "above_max": float(above_max),
        },
    }

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
        "nu": {
            "mean": float(prior_samples["nu"].values.mean()),
            "std": float(prior_samples["nu"].values.std()),
        },
    }

    return stats


def create_diagnostic_plots(idata: az.InferenceData, data: dict, output_dir: Path):
    """Create diagnostic visualizations for prior predictive check."""
    y_rep = idata.prior_predictive["y_rep"].values
    y_obs = data["_log_mpg"]
    prior = idata.prior

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1a: Prior predictive distribution (log scale)
    ax = axes[0, 0]
    y_rep_flat = y_rep.flatten()
    ax.hist(y_rep_flat, bins=100, density=True, alpha=0.7, color="steelblue")
    ax.axvline(LOG_MPG_MIN, color="red", linestyle="--", linewidth=2, label=f"Obs min ({MPG_MIN} MPG)")
    ax.axvline(LOG_MPG_MAX, color="red", linestyle="--", linewidth=2, label=f"Obs max ({MPG_MAX} MPG)")
    ax.axvline(np.mean(y_obs), color="green", linestyle="-", linewidth=2, label="Obs mean")
    ax.set_xlabel("log(MPG)")
    ax.set_ylabel("Density")
    ax.set_title("Prior Predictive Distribution (log scale)")
    ax.legend(fontsize=8)

    # 1b: Prior predictive in MPG scale
    ax = axes[0, 1]
    mpg_rep = np.exp(y_rep_flat)
    mpg_rep_viz = mpg_rep[(mpg_rep > 0) & (mpg_rep < 200)]
    ax.hist(mpg_rep_viz, bins=100, density=True, alpha=0.7, color="steelblue")
    ax.axvline(MPG_MIN, color="red", linestyle="--", linewidth=2)
    ax.axvline(MPG_MAX, color="red", linestyle="--", linewidth=2)
    ax.axvline(np.mean(data["_mpg"]), color="green", linestyle="-", linewidth=2)
    ax.set_xlabel("MPG")
    ax.set_ylabel("Density")
    ax.set_title("Prior Predictive Distribution (MPG scale)")
    ax.set_xlim(0, 100)

    # 1c: Prior parameter distributions
    ax = axes[1, 0]
    params = ["alpha", "beta_weight", "beta_year", "sigma"]
    colors = ["C0", "C1", "C2", "C3"]
    for param, color in zip(params, colors):
        samples = prior[param].values.flatten()
        ax.hist(samples, bins=50, alpha=0.5, color=color, label=param, density=True)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Density")
    ax.set_title("Prior Parameter Distributions")
    ax.legend()

    # 1d: Nu prior distribution (key parameter)
    ax = axes[1, 1]
    nu_samples = prior["nu"].values.flatten()
    ax.hist(nu_samples, bins=50, density=True, alpha=0.7, color="purple")
    ax.axvline(30, color="red", linestyle="--", linewidth=2, label="nu=30 (approx normal)")
    ax.axvline(15, color="orange", linestyle="--", linewidth=2, label="nu=15 (heavy tails)")
    ax.axvline(4, color="darkred", linestyle="--", linewidth=2, label="nu=4 (very heavy)")
    ax.set_xlabel("nu (degrees of freedom)")
    ax.set_ylabel("Density")
    ax.set_title("Prior on nu (Student-t df)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "prior_predictive_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ECDF plot
    fig, ax = plt.subplots(figsize=(10, 6))
    n_draws_to_plot = 50
    n_chains, n_samples = y_rep.shape[0], y_rep.shape[1]
    indices = np.random.choice(n_chains * n_samples, min(n_draws_to_plot, n_chains * n_samples), replace=False)

    for idx in indices:
        chain_idx = idx // n_samples
        sample_idx = idx % n_samples
        y_sample = y_rep[chain_idx, sample_idx, :]
        sorted_y = np.sort(y_sample)
        ecdf = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
        ax.step(sorted_y, ecdf, color="steelblue", alpha=0.1, linewidth=0.5)

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
    print("Prior Predictive Check: A3-Robust Model (Student-t)")
    print("=" * 60)

    print("\nLoading and preparing data...")
    data = load_and_prepare_data()
    print(f"  N = {data['N']} observations")

    print("\nRunning prior predictive simulation...")
    idata = run_prior_predictive(data, n_draws=1000)

    print("\nComputing statistics...")
    stats = compute_statistics(idata, data)

    print("\n" + "=" * 60)
    print("PRIOR PREDICTIVE SUMMARY")
    print("=" * 60)
    print(f"\nCoverage of observed MPG range ({MPG_MIN}-{MPG_MAX}):")
    print(f"  Within range: {stats['coverage']['within_observed_range']:.1%}")
    print(f"  Below min:    {stats['coverage']['below_min']:.1%}")
    print(f"  Above max:    {stats['coverage']['above_max']:.1%}")

    print(f"\nPrior on nu (degrees of freedom):")
    print(f"  Mean: {stats['prior_parameters']['nu']['mean']:.1f}")
    print(f"  Std:  {stats['prior_parameters']['nu']['std']:.1f}")

    print("\nCreating diagnostic plots...")
    create_diagnostic_plots(idata, data, OUTPUT_DIR)

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
        issues.append(f"Low coverage ({coverage:.1%}) - priors may be too narrow")
    if coverage > 0.99:
        issues.append(f"Very high coverage ({coverage:.1%}) - priors may be too vague")
    if stats["coverage"]["below_min"] > 0.3:
        issues.append(f"Too many below minimum ({stats['coverage']['below_min']:.1%})")
    if stats["coverage"]["above_max"] > 0.3:
        issues.append(f"Too many above maximum ({stats['coverage']['above_max']:.1%})")

    y_rep = idata.prior_predictive["y_rep"].values.flatten()
    if np.any(np.isnan(y_rep)) or np.any(np.isinf(y_rep)):
        issues.append("Numerical issues detected (NaN or Inf values)")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        recommendation = "FAIL" if len(issues) > 1 or coverage < 0.2 else "PASS with concerns"
    else:
        recommendation = "PASS"
        print("No issues found. Priors generate plausible data.")

    print(f"\nRecommendation: {recommendation}")
    print("=" * 60)

    with open(OUTPUT_DIR / "recommendation.txt", "w") as f:
        f.write(f"Recommendation: {recommendation}\n")
        if issues:
            f.write("\nIssues:\n")
            for issue in issues:
                f.write(f"  - {issue}\n")

    return recommendation


if __name__ == "__main__":
    main()
