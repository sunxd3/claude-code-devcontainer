"""Fit A3-Robust model (Student-t errors) to Auto-MPG data.

Model: log(mpg) ~ Student-t(nu, mu, sigma)
       mu = alpha + beta_weight * log_weight_c + beta_year * year_c

Tests robustness to outliers via Student-t likelihood with estimated df (nu).
Key output: posterior estimate of nu
  - nu > 30: Normal errors adequate (essentially Gaussian)
  - nu < 15: Heavy tails needed (data has outliers)
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
MODEL_PATH = "/workspace/analysis/experiments/experiment_3/model.stan"
OUTPUT_DIR = "/workspace/analysis/experiments/experiment_3/fit"

# Load and prepare data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Original dataset: {len(df)} rows")

# Remove rows with missing horsepower (6 rows expected)
df_clean = df.dropna(subset=["horsepower"])
print(f"After removing missing horsepower: {len(df_clean)} rows")
print(f"Rows removed: {len(df) - len(df_clean)}")

# Compute transformations per specification
log_mpg = np.log(df_clean["mpg"].values)
log_weight_c = np.log(df_clean["weight"].values) - 7.96  # centered at exp(7.96) ~ 2860 lbs
year_c = df_clean["model_year"].values - 76              # centered at 1976

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

# Check convergence (include nu in parameters)
print("\n" + "=" * 60)
print("Convergence diagnostics:")
print("=" * 60)
convergence = check_convergence(
    idata, var_names=["alpha", "beta_weight", "beta_year", "sigma", "nu"]
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
    var_names=["alpha", "beta_weight", "beta_year", "sigma", "nu"],
    hdi_prob=0.95,
)
print(summary.to_string())

# Interpret nu
print("\n" + "=" * 60)
print("Interpretation of nu (degrees of freedom):")
print("=" * 60)
nu_samples = idata.posterior["nu"].values.flatten()
nu_mean = nu_samples.mean()
nu_median = np.median(nu_samples)
nu_q025 = np.percentile(nu_samples, 2.5)
nu_q975 = np.percentile(nu_samples, 97.5)

print(f"  nu mean: {nu_mean:.1f}")
print(f"  nu median: {nu_median:.1f}")
print(f"  nu 95% CI: [{nu_q025:.1f}, {nu_q975:.1f}]")

if nu_median > 30:
    interpretation = "Normal errors adequate - Student-t approximately Gaussian"
elif nu_median > 15:
    interpretation = "Mild heavy tails - modest improvement over Normal"
else:
    interpretation = "Heavy tails needed - outliers present, Student-t beneficial"

print(f"\n  Interpretation: {interpretation}")

# Compare with experiment 2 (Normal errors)
print("\n" + "=" * 60)
print("Comparison with Experiment 2 (Normal errors):")
print("=" * 60)
exp2_elpd = 279.7
exp2_se = 17.27
elpd_diff = loo_result.elpd_loo - exp2_elpd
# Approximate SE of difference (conservative)
se_diff = np.sqrt(exp2_se**2 + loo_result.se**2)
z_score = elpd_diff / se_diff if se_diff > 0 else 0

print(f"  Experiment 2 ELPD: {exp2_elpd:.1f} +/- {exp2_se:.1f}")
print(f"  Experiment 3 ELPD: {loo_result.elpd_loo:.1f} +/- {loo_result.se:.1f}")
print(f"  Difference (A3 - A2): {elpd_diff:.1f}")
print(f"  Approximate z-score: {z_score:.2f}")

if abs(z_score) < 2:
    model_comparison = "No significant difference between models"
elif elpd_diff > 0:
    model_comparison = "A3-Robust shows improved fit"
else:
    model_comparison = "A2-Year shows better fit (unexpected)"

print(f"\n  {model_comparison}")

print("\nFitting complete.")
