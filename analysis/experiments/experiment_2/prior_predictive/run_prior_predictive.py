#!/usr/bin/env python3
"""
Run prior predictive check for Experiment 2: Random Intercepts Only
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr
from shared_utils import compile_model

# Paths
exp_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2")
prior_dir = exp_dir / "prior_predictive"
data_path = Path("/home/user/claude-code-devcontainer/analysis/data/stan_data.json")

# Load data (only N, J, school, treatment - not y)
with open(data_path) as f:
    full_data = json.load(f)

# Prior predictive only needs structure, not observed outcomes
prior_data = {
    "N": full_data["N"],
    "J": full_data["J"],
    "school": full_data["school"],
    "treatment": full_data["treatment"]
}

# Compile prior predictive model
print("Compiling prior predictive model...")
model = compile_model(prior_dir / "prior_predictive_model.stan")

# Sample from prior
print("\nSampling from prior predictive distribution...")
try:
    prior_samples = model.sample(
        data=prior_data,
        chains=4,
        parallel_chains=4,
        iter_sampling=1000,
        iter_warmup=0,  # No warmup needed for generated quantities only
        fixed_param=True,  # No parameters to sample
        adapt_engaged=False,  # No adaptation needed for fixed_param
        show_console=False
    )
    print("Prior predictive sampling completed successfully.")
except Exception as e:
    print(f"Error during sampling: {e}")
    raise

# Convert to ArviZ InferenceData
print("\nConverting to ArviZ InferenceData...")

# For fixed_param sampling, all variables are in the sample
# We need to separate them into prior and prior_predictive groups manually

# Extract draws
draws = prior_samples.draws_xr()

# Create prior group (parameters and intermediate quantities)
prior_vars = ["alpha_0", "tau_alpha", "beta", "sigma", "z_alpha", "alpha", "mu"]
prior_data = {var: draws[var] for var in prior_vars if var in draws}
prior_group = xr.Dataset(prior_data)

# Create prior_predictive group (replicated observations)
prior_pred_data = {"y": draws["y_prior_pred"]}
prior_predictive_group = xr.Dataset(prior_pred_data)

# Create observed_data group
observed_data_group = xr.Dataset({"y": xr.DataArray(full_data["y"], dims=["y_dim_0"])})

# Create InferenceData object
idata = az.InferenceData(
    prior=prior_group,
    prior_predictive=prior_predictive_group,
    observed_data=observed_data_group
)

# Save to NetCDF
output_path = prior_dir / "prior_predictive.nc"
idata.to_netcdf(output_path)
print(f"\nSaved InferenceData to {output_path}")

# Print summary statistics
print("\n" + "="*70)
print("PRIOR PREDICTIVE SUMMARY")
print("="*70)

print("\n--- Hyperparameters ---")
prior_summary = az.summary(idata, group="prior", var_names=["alpha_0", "tau_alpha", "beta", "sigma"])
print(prior_summary)

print("\n--- School-level Intercepts (alpha) ---")
alpha_summary = az.summary(idata, group="prior", var_names=["alpha"])
print(alpha_summary)

print("\n--- Prior Predictive Observations (y) ---")
y_prior = idata.prior_predictive["y"].values.flatten()
print(f"Min: {y_prior.min():.1f}")
print(f"5th percentile: {np.percentile(y_prior, 5):.1f}")
print(f"Median: {np.median(y_prior):.1f}")
print(f"Mean: {y_prior.mean():.1f}")
print(f"95th percentile: {np.percentile(y_prior, 95):.1f}")
print(f"Max: {y_prior.max():.1f}")

print("\n--- Observed Data (for comparison) ---")
y_obs = full_data["y"]
print(f"Min: {min(y_obs):.1f}")
print(f"Median: {sorted(y_obs)[len(y_obs)//2]:.1f}")
print(f"Mean: {sum(y_obs)/len(y_obs):.1f}")
print(f"Max: {max(y_obs):.1f}")

print("\n" + "="*70)
print("Done. Run visualization script next.")
print("="*70)
