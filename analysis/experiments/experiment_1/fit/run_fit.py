"""Fit A1-Baseline model to Auto-MPG data."""

import numpy as np
import pandas as pd

from shared_utils import (
    compile_model,
    fit_model,
    to_arviz,
    save_results,
    check_convergence,
    compute_loo,
    ensure_dir,
)

# Paths
MODEL_FILE = "/workspace/analysis/experiments/experiment_1/model.stan"
DATA_FILE = "/workspace/analysis/eda/auto_mpg_cleaned.csv"
OUTPUT_DIR = "/workspace/analysis/experiments/experiment_1/fit"


def prepare_data():
    """Load and prepare data for Stan."""
    df = pd.read_csv(DATA_FILE)

    # Remove rows with missing horsepower (stored as empty string or NaN)
    df = df.dropna(subset=["horsepower"])
    # Also drop rows where horsepower is empty string (from CSV parsing)
    df = df[df["horsepower"].astype(str).str.strip() != ""]

    n_original = 398  # Original dataset size
    n_after = len(df)
    print(f"Data: {n_original} rows -> {n_after} rows (removed {n_original - n_after} with missing horsepower)")

    # Compute log transforms
    log_mpg = np.log(df["mpg"].values)
    log_weight = np.log(df["weight"].values)
    log_weight_mean = log_weight.mean()
    log_weight_c = log_weight - log_weight_mean

    print(f"log(weight) mean: {log_weight_mean:.4f}")
    print(f"log(mpg) range: [{log_mpg.min():.3f}, {log_mpg.max():.3f}]")
    print(f"log_weight_c range: [{log_weight_c.min():.3f}, {log_weight_c.max():.3f}]")

    stan_data = {
        "N": n_after,
        "log_mpg": log_mpg.tolist(),
        "log_weight_c": log_weight_c.tolist(),
    }

    return stan_data, log_mpg


def main():
    """Run model fitting."""
    output_dir = ensure_dir(OUTPUT_DIR)

    # Prepare data
    print("=" * 60)
    print("Preparing data...")
    stan_data, y_obs = prepare_data()

    # Compile model
    print("=" * 60)
    print("Compiling model...")
    model = compile_model(MODEL_FILE)
    print(f"Model compiled: {model.stan_file}")

    # Fit model
    print("=" * 60)
    print("Fitting model (4 chains, 1000 warmup, 1000 sampling)...")
    fit = fit_model(
        model,
        stan_data,
        chains=4,
        warmup=1000,
        sampling=1000,
        adapt_delta=0.9,
        output_dir=output_dir,
    )

    # Convert to ArviZ
    print("=" * 60)
    print("Converting to ArviZ InferenceData...")
    idata = to_arviz(fit, y_obs=y_obs)

    # Check convergence
    print("=" * 60)
    print("Checking convergence...")
    conv = check_convergence(idata, var_names=["alpha", "beta_weight", "sigma"])
    print(conv)

    # Compute LOO
    print("=" * 60)
    print("Computing LOO-CV...")
    loo_result = compute_loo(idata)
    print(loo_result)

    # Save results
    print("=" * 60)
    print("Saving results...")
    save_results(idata, output_dir, convergence=conv, loo=loo_result)
    print(f"Saved posterior.nc, convergence.json, loo.json to {output_dir}")

    # Print parameter summary
    print("=" * 60)
    print("Parameter Summary:")
    import arviz as az
    summary = az.summary(idata, var_names=["alpha", "beta_weight", "sigma"])
    print(summary.to_string())

    return idata, conv, loo_result


if __name__ == "__main__":
    main()
