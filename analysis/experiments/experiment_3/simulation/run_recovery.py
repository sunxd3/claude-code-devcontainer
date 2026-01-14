"""
Parameter recovery test for Experiment 3: Random Intercepts + Random Slopes model.

Runs 3 recovery scenarios to verify the model can recover known parameters from synthetic data.
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
from shared_utils import check_convergence, compile_model, fit_model

# Set random seed for reproducibility
np.random.seed(20260114)

# Output directory
output_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_3/simulation")
output_dir.mkdir(parents=True, exist_ok=True)

# Data structure
J = 8  # number of schools
n_per_school = 20  # students per school
N = J * n_per_school  # total students

# School assignments
school = np.repeat(np.arange(1, J + 1), n_per_school)

# Treatment assignment (50% treated, balanced within schools)
treatment = np.tile(np.concatenate([np.zeros(n_per_school // 2), np.ones(n_per_school // 2)]), J).astype(int)

# Define recovery scenarios
scenarios = [
    {
        "name": "scenario_1_low_heterogeneity",
        "description": "Low heterogeneity",
        "alpha_0": 75.0,
        "tau_alpha": 5.0,
        "beta_0": 5.0,
        "tau_beta": 2.0,
        "sigma": 12.0,
    },
    {
        "name": "scenario_2_true_dgp",
        "description": "True DGP",
        "alpha_0": 70.0,
        "tau_alpha": 8.0,
        "beta_0": 5.0,
        "tau_beta": 3.0,
        "sigma": 10.0,
    },
    {
        "name": "scenario_3_high_heterogeneity",
        "description": "High heterogeneity",
        "alpha_0": 70.0,
        "tau_alpha": 10.0,
        "beta_0": 7.0,
        "tau_beta": 6.0,
        "sigma": 8.0,
    },
]

# Compile model once
print("Compiling Stan model...")
model_path = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_3/model.stan")
model = compile_model(model_path)
print("Model compiled successfully.\n")

# Run recovery for each scenario
results = []

for scenario in scenarios:
    print("=" * 80)
    print(f"SCENARIO: {scenario['description']}")
    print("=" * 80)

    # Extract true parameters
    alpha_0_true = scenario["alpha_0"]
    tau_alpha_true = scenario["tau_alpha"]
    beta_0_true = scenario["beta_0"]
    tau_beta_true = scenario["tau_beta"]
    sigma_true = scenario["sigma"]

    print("\nTrue parameters:")
    print(f"  alpha_0 = {alpha_0_true}")
    print(f"  tau_alpha = {tau_alpha_true}")
    print(f"  beta_0 = {beta_0_true}")
    print(f"  tau_beta = {tau_beta_true}")
    print(f"  sigma = {sigma_true}")

    # Generate school-level random effects (using z-score parameterization)
    z_alpha_true = np.random.standard_normal(J)
    z_beta_true = np.random.standard_normal(J)

    alpha_true = alpha_0_true + tau_alpha_true * z_alpha_true
    beta_true = beta_0_true + tau_beta_true * z_beta_true

    # Generate synthetic data
    mu_true = np.zeros(N)
    for n in range(N):
        j = school[n] - 1  # Convert to 0-indexed
        mu_true[n] = alpha_true[j] + beta_true[j] * treatment[n]

    y_synthetic = mu_true + np.random.normal(0, sigma_true, N)

    print("\nGenerated synthetic data:")
    print(f"  N = {N}, J = {J}")
    print(f"  y range: [{y_synthetic.min():.1f}, {y_synthetic.max():.1f}]")
    print(f"  y mean: {y_synthetic.mean():.1f}, y SD: {y_synthetic.std():.1f}")

    # Prepare data for Stan
    stan_data = {
        "N": N,
        "J": J,
        "school": school.tolist(),
        "treatment": treatment.tolist(),
        "y": y_synthetic.tolist(),
    }

    # Fit model
    print("\nFitting model...")
    try:
        fit = fit_model(
            model=model,
            data=stan_data,
            sampling=2000,
            warmup=1000,
            chains=4,
            show_progress=True,
        )

        # Convert to InferenceData
        idata = az.from_cmdstanpy(
            fit,
            log_likelihood="log_lik",
            posterior_predictive={"y_rep": "y_rep"},
            observed_data={"y": y_synthetic},
            coords={"school": np.arange(1, J + 1)},
            dims={
                "z_alpha": ["school"],
                "z_beta": ["school"],
                "alpha": ["school"],
                "beta": ["school"],
            },
        )

        # Check convergence
        print("\nChecking convergence diagnostics...")
        convergence = check_convergence(
            idata, var_names=["alpha_0", "tau_alpha", "beta_0", "tau_beta", "sigma"]
        )

        # Save InferenceData
        idata_path = output_dir / f"{scenario['name']}_posterior.nc"
        idata.to_netcdf(idata_path)
        print(f"Saved posterior to {idata_path}")

        # Extract posterior summaries for key parameters
        summary = az.summary(
            idata,
            var_names=["alpha_0", "tau_alpha", "beta_0", "tau_beta", "sigma"],
        )

        print("\nPosterior summary (population parameters):")
        print(summary)

        # Store results
        result = {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "converged": convergence.converged,
            "num_divergences": convergence.n_divergent,
            "max_rhat": convergence.max_rhat,
            "min_ess_bulk": convergence.min_ess_bulk,
            "true_params": {
                "alpha_0": alpha_0_true,
                "tau_alpha": tau_alpha_true,
                "beta_0": beta_0_true,
                "tau_beta": tau_beta_true,
                "sigma": sigma_true,
            },
            "posterior_means": {
                "alpha_0": summary.loc["alpha_0", "mean"],
                "tau_alpha": summary.loc["tau_alpha", "mean"],
                "beta_0": summary.loc["beta_0", "mean"],
                "tau_beta": summary.loc["tau_beta", "mean"],
                "sigma": summary.loc["sigma", "mean"],
            },
            "posterior_sds": {
                "alpha_0": summary.loc["alpha_0", "sd"],
                "tau_alpha": summary.loc["tau_alpha", "sd"],
                "beta_0": summary.loc["beta_0", "sd"],
                "tau_beta": summary.loc["tau_beta", "sd"],
                "sigma": summary.loc["sigma", "sd"],
            },
            "fit_success": True,
        }

        results.append(result)

    except Exception as e:
        print(f"\nERROR: Model fitting failed for {scenario['name']}")
        print(f"Error message: {e}")

        result = {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "fit_success": False,
            "error": str(e),
        }
        results.append(result)

    print("\n")

# Save all results
results_path = output_dir / "recovery_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 80}")
print(f"Recovery test complete. Results saved to {results_path}")
print(f"{'=' * 80}")
