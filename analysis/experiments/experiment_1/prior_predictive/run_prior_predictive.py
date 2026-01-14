"""Run prior predictive check for Experiment 1: Complete Pooling model."""

import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr
from shared_utils import compile_model

# Set paths
exp_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_1")
data_path = Path("/home/user/claude-code-devcontainer/analysis/data/stan_data.json")
stan_file = exp_dir / "complete_pooling_prior.stan"
output_dir = exp_dir / "prior_predictive"

# Load data
with open(data_path) as f:
    data = json.load(f)

# For prior predictive, we don't need the observed y
prior_data = {
    'N': data['N'],
    'treatment': data['treatment']
}

# Compile and sample from prior
print("Compiling Stan model...")
model = compile_model(stan_file)

print("Sampling from prior predictive distribution...")
# Sample from priors (no fixed_param - we want to draw from prior distributions)
prior_fit = model.sample(
    data=prior_data,
    iter_warmup=500,
    iter_sampling=1000,
    chains=4,
    show_console=False,
    adapt_engaged=False  # No adaptation needed for prior-only sampling
)

# Convert to ArviZ InferenceData
print("Converting to ArviZ InferenceData...")
# When using fixed_param, ArviZ puts everything in posterior group
# We'll manually construct the InferenceData with correct groups

# Extract draws from the fit
draws = prior_fit.draws_xr()

# Create prior group with parameters
prior_dict = {
    'alpha': draws['alpha'],
    'beta': draws['beta'],
    'sigma': draws['sigma']
}
prior_ds = xr.Dataset(prior_dict)

# Create prior_predictive group with y_prior_pred
prior_pred_dict = {
    'y_prior_pred': draws['y_prior_pred']
}
prior_pred_ds = xr.Dataset(prior_pred_dict)

# Create observed_data group
obs_dict = {'y': (['y_dim_0'], np.array(data['y']))}
obs_ds = xr.Dataset(obs_dict)

# Combine into InferenceData
idata = az.InferenceData(
    prior=prior_ds,
    prior_predictive=prior_pred_ds,
    observed_data=obs_ds
)

# Save
output_file = output_dir / "prior_predictive.nc"
idata.to_netcdf(output_file)
print(f"Saved prior predictive samples to {output_file}")

# Print summary statistics
print("\n=== Prior Distributions ===")
print(az.summary(idata, group="prior", var_names=["alpha", "beta", "sigma"]))

print("\n=== Prior Predictive Distribution ===")
# Check available groups
print(f"Available groups: {list(idata.groups())}")

# Get prior predictive from correct group
if 'prior_predictive' in idata.groups():
    y_prior = idata.prior_predictive['y_prior_pred'].values.flatten()
elif 'posterior_predictive' in idata.groups():
    y_prior = idata.posterior_predictive['y_prior_pred'].values.flatten()
else:
    raise ValueError(f"Cannot find prior predictive data. Available groups: {list(idata.groups())}")

print(f"Range: [{y_prior.min():.1f}, {y_prior.max():.1f}]")
print(f"Mean: {y_prior.mean():.1f}")
print(f"SD: {y_prior.std():.1f}")
print(f"2.5th percentile: {np.percentile(y_prior, 2.5):.1f}")
print(f"97.5th percentile: {np.percentile(y_prior, 97.5):.1f}")

print("\n=== Observed Data ===")
y_obs = np.array(data['y'])
print(f"Range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
print(f"Mean: {y_obs.mean():.1f}")
print(f"SD: {y_obs.std():.1f}")

# Check for issues
print("\n=== Plausibility Checks ===")
n_negative = np.sum(y_prior < 0)
n_extreme_low = np.sum(y_prior < 20)
n_extreme_high = np.sum(y_prior > 150)

print(f"Negative values: {n_negative} / {len(y_prior)} ({100*n_negative/len(y_prior):.1f}%)")
print(f"Values < 20: {n_extreme_low} / {len(y_prior)} ({100*n_extreme_low/len(y_prior):.1f}%)")
print(f"Values > 150: {n_extreme_high} / {len(y_prior)} ({100*n_extreme_high/len(y_prior):.1f}%)")
