"""Fit A2-Year model to Auto-MPG data.

Model: log(mpg) ~ log(weight) + year
Tests whether technology trend adds predictive value beyond weight.
"""

import numpy as np
import pandas as pd

from shared_utils import (
    check_convergence,
    compile_model,
    compute_loo,
    fit_model,
    save_results,
    to_arviz,
)

# Paths
DATA_PATH = "/workspace/analysis/eda/auto_mpg_cleaned.csv"
MODEL_PATH = "/workspace/analysis/experiments/experiment_2/model.stan"
OUTPUT_DIR = "/workspace/analysis/experiments/experiment_2/fit"

# Load and prepare data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Original dataset: {len(df)} rows")

# Remove rows with missing horsepower
df_clean = df.dropna(subset=["horsepower"])
print(f"After removing missing horsepower: {len(df_clean)} rows")
print(f"Rows removed: {len(df) - len(df_clean)}")

# Compute transformations
log_mpg = np.log(df_clean["mpg"].values)
log_weight_c = np.log(df_clean["weight"].values) - 7.96
year_c = df_clean["model_year"].values - 76

N = len(df_clean)
print(f"\nData summary:")
print(f"  N = {N}")
print(f"  log_mpg: mean={log_mpg.mean():.3f}, sd={log_mpg.std():.3f}")
print(f"  log_weight_c: mean={log_weight_c.mean():.3f}, sd={log_weight_c.std():.3f}")
print(f"  year_c: mean={year_c.mean():.2f}, range=[{year_c.min()}, {year_c.max()}]")

# Prepare Stan data
stan_data = {
    "N": N,
    "log_mpg": log_mpg.tolist(),
    "log_weight_c": log_weight_c.tolist(),
    "year_c": year_c.tolist(),
}

# Compile model
print("\nCompiling model...")
model = compile_model(MODEL_PATH)
print("Model compiled successfully.")

# Fit model
print("\nFitting model (4 chains, 1000 warmup, 1000 sampling)...")
fit = fit_model(
    model,
    stan_data,
    chains=4,
    warmup=1000,
    sampling=1000,
    adapt_delta=0.9,
    output_dir=OUTPUT_DIR,
)

# Check CmdStanPy diagnostics
print("\n" + "=" * 60)
print("CmdStanPy diagnostics:")
print("=" * 60)
print(fit.diagnose())

# Convert to ArviZ
print("\nConverting to ArviZ InferenceData...")
idata = to_arviz(fit, y_obs=log_mpg, log_likelihood="log_lik")

# Check convergence
print("\n" + "=" * 60)
print("Convergence diagnostics:")
print("=" * 60)
convergence = check_convergence(
    idata, var_names=["alpha", "beta_weight", "beta_year", "sigma"]
)
print(convergence)

# Compute LOO
print("\n" + "=" * 60)
print("LOO-CV:")
print("=" * 60)
loo_result = compute_loo(idata)
print(loo_result)

# Save results
print("\nSaving results...")
save_results(idata, OUTPUT_DIR, convergence=convergence, loo=loo_result)
print(f"  posterior.nc saved to {OUTPUT_DIR}")
print(f"  convergence.json saved to {OUTPUT_DIR}")
print(f"  loo.json saved to {OUTPUT_DIR}")

# Print parameter summary
print("\n" + "=" * 60)
print("Parameter estimates:")
print("=" * 60)
import arviz as az

summary = az.summary(
    idata,
    var_names=["alpha", "beta_weight", "beta_year", "sigma"],
    hdi_prob=0.95,
)
print(summary.to_string())

print("\nFitting complete.")
