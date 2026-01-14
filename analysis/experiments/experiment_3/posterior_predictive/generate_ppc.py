#!/usr/bin/env python3
"""Generate posterior predictive samples for Experiment 3."""

import arviz as az
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
base_dir = Path("/home/user/claude-code-devcontainer/analysis")
fit_path = base_dir / "experiments/experiment_3/fit/posterior.nc"
data_path = base_dir / "data/student_scores.csv"
output_dir = base_dir / "experiments/experiment_3/posterior_predictive"
output_dir.mkdir(parents=True, exist_ok=True)

# Load fitted model
print("Loading fitted model...")
idata = az.from_netcdf(fit_path)

# Load observed data
print("Loading observed data...")
df = pd.read_csv(data_path)
print(f"Observed data: {len(df)} students from {df['school_id'].nunique()} schools")

# Check model structure
print("\nInferenceData groups:")
print(idata.groups())

# Check posterior group
print("\nPosterior variables:")
print(list(idata.posterior.data_vars))

# Check if posterior_predictive group exists
if "posterior_predictive" in idata.groups():
    print("\nPosterior predictive group exists:")
    print(list(idata.posterior_predictive.data_vars))
else:
    print("\nNo posterior_predictive group found. Need to generate samples.")

# Save basic info
with open(output_dir / "model_structure.txt", "w") as f:
    f.write("InferenceData Groups:\n")
    f.write(str(idata.groups()) + "\n\n")
    f.write("Posterior Variables:\n")
    for var in idata.posterior.data_vars:
        shape = idata.posterior[var].shape
        f.write(f"  {var}: {shape}\n")

    if "posterior_predictive" in idata.groups():
        f.write("\nPosterior Predictive Variables:\n")
        for var in idata.posterior_predictive.data_vars:
            shape = idata.posterior_predictive[var].shape
            f.write(f"  {var}: {shape}\n")

print(f"\nModel structure saved to {output_dir / 'model_structure.txt'}")
