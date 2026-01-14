#!/usr/bin/env python3
"""Run prior predictive check for Experiment 3."""

from pathlib import Path

import arviz as az
import numpy as np
from shared_utils import compile_model, load_stan_data

# Paths
exp_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_3")
output_dir = exp_dir / "prior_predictive"
data_path = Path("/home/user/claude-code-devcontainer/analysis/data/stan_data.json")

# Load data
data = load_stan_data(data_path)
print(f"Loaded data: N={data['N']}, J={data['J']}")

# Create prior predictive Stan model (no likelihood)
prior_model_code = """
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> school;
  array[N] int<lower=0, upper=1> treatment;
}

generated quantities {
  // Sample from priors
  real alpha_0 = normal_rng(77, 15);
  real<lower=0> tau_alpha = abs(normal_rng(0, 10));
  vector[J] z_alpha;

  real beta_0 = normal_rng(5, 5);
  real<lower=0> tau_beta = abs(normal_rng(0, 5));
  vector[J] z_beta;

  real<lower=0> sigma = abs(normal_rng(0, 15));

  // Transformed parameters
  vector[J] alpha;
  vector[J] beta;

  // Generate replicated data
  array[N] real y_rep;

  // Sample standardized random effects
  for (j in 1:J) {
    z_alpha[j] = normal_rng(0, 1);
    z_beta[j] = normal_rng(0, 1);
  }

  // Non-centered parameterization
  alpha = alpha_0 + tau_alpha * z_alpha;
  beta = beta_0 + tau_beta * z_beta;

  // Generate replicated observations
  for (n in 1:N) {
    real mu = alpha[school[n]] + beta[school[n]] * treatment[n];
    y_rep[n] = normal_rng(mu, sigma);
  }
}
"""

# Write and compile prior predictive model
prior_model_path = output_dir / "prior_predictive_model.stan"
with open(prior_model_path, "w") as f:
    f.write(prior_model_code)

print("Compiling prior predictive model...")
try:
    model = compile_model(prior_model_path)
    print("Model compiled successfully")
except Exception as e:
    print(f"Compilation failed: {e}")
    raise

# Sample from prior predictive (using fixed_param since no parameters to sample)
print("Generating prior predictive samples...")
try:
    # Remove 'y' from data for prior predictive
    prior_data = {k: v for k, v in data.items() if k != 'y'}

    fit = model.sample(
        data=prior_data,
        fixed_param=True,
        iter_sampling=1000,
        chains=4,
        seed=12345
    )
    print("Prior predictive sampling completed")
except Exception as e:
    print(f"Sampling failed: {e}")
    raise

# Convert to ArviZ InferenceData
print("Converting to ArviZ InferenceData...")
idata = az.from_cmdstanpy(
    fit,
    prior_predictive=["y_rep"],
    log_likelihood=None,
    observed_data={"y": data["y"]}
)

# Save InferenceData
output_file = output_dir / "prior_predictive.nc"
idata.to_netcdf(output_file)
print(f"Saved prior predictive data to {output_file}")

# Print summary statistics
print("\nPrior predictive summary:")
print(f"  alpha_0: {fit.stan_variable('alpha_0').mean():.2f} ± {fit.stan_variable('alpha_0').std():.2f}")
print(f"  tau_alpha: {fit.stan_variable('tau_alpha').mean():.2f} ± {fit.stan_variable('tau_alpha').std():.2f}")
print(f"  beta_0: {fit.stan_variable('beta_0').mean():.2f} ± {fit.stan_variable('beta_0').std():.2f}")
print(f"  tau_beta: {fit.stan_variable('tau_beta').mean():.2f} ± {fit.stan_variable('tau_beta').std():.2f}")
print(f"  sigma: {fit.stan_variable('sigma').mean():.2f} ± {fit.stan_variable('sigma').std():.2f}")

y_rep = fit.stan_variable('y_rep')
print("\nPrior predictive y_rep:")
print(f"  Range: [{y_rep.min():.1f}, {y_rep.max():.1f}]")
print(f"  Mean: {y_rep.mean():.2f} ± {y_rep.std():.2f}")
print(f"  Observed y range: [{np.min(data['y']):.1f}, {np.max(data['y']):.1f}]")
print(f"  Observed y mean: {np.mean(data['y']):.2f}")

print("\nPrior predictive check completed successfully!")
