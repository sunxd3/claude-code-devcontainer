"""Parameter recovery check for Random Intercepts Only model.

Tests whether the model can recover known parameters from synthetic data.
Runs 3 scenarios with varying clustering levels.
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
from shared_utils import check_convergence, compile_model, fit_model

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2/simulation")
output_dir.mkdir(exist_ok=True, parents=True)

# Model path
model_path = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2/model.stan")

# Data structure constants
J = 8  # number of schools
N = 160  # total students
n_per_school = N // J  # students per school

# Define scenarios
scenarios = [
    {
        "name": "low_clustering",
        "alpha_0": 75.0,
        "tau_alpha": 3.0,
        "beta": 5.0,
        "sigma": 12.0,
        "description": "Low clustering: small between-school variation"
    },
    {
        "name": "medium_clustering",
        "alpha_0": 70.0,
        "tau_alpha": 8.0,
        "beta": 5.0,
        "sigma": 10.0,
        "description": "Medium clustering: moderate between-school variation (close to true DGP)"
    },
    {
        "name": "high_clustering",
        "alpha_0": 80.0,
        "tau_alpha": 12.0,
        "beta": 3.0,
        "sigma": 8.0,
        "description": "High clustering: large between-school variation"
    }
]

print("Compiling Stan model...")
try:
    model = compile_model(model_path)
    print("Model compiled successfully.\n")
except Exception as e:
    print(f"ERROR: Model compilation failed: {e}")
    raise

# Store all results
all_results = []

# Run recovery for each scenario
for scenario in scenarios:
    print("=" * 80)
    print(f"SCENARIO: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print("True parameters:")
    print(f"  alpha_0 = {scenario['alpha_0']}")
    print(f"  tau_alpha = {scenario['tau_alpha']}")
    print(f"  beta = {scenario['beta']}")
    print(f"  sigma = {scenario['sigma']}")
    print("=" * 80)

    # Generate synthetic data
    print("\nGenerating synthetic data...")

    # True school-level random intercepts
    z_alpha_true = np.random.standard_normal(J)
    alpha_true = scenario['alpha_0'] + scenario['tau_alpha'] * z_alpha_true

    # Student-level data
    school_ids = np.repeat(np.arange(1, J + 1), n_per_school)  # 1-indexed
    treatment = np.tile([0, 1], N // 2)  # Alternate 0/1 for 50% treated
    np.random.shuffle(treatment)  # Randomize treatment assignment

    # Generate outcomes
    mu_true = alpha_true[school_ids - 1] + scenario['beta'] * treatment
    y = mu_true + np.random.normal(0, scenario['sigma'], N)

    print(f"  Generated N={N} observations across J={J} schools")
    print(f"  Treatment: {treatment.mean()*100:.1f}% treated")
    print(f"  Outcome range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}")

    # Prepare Stan data
    stan_data = {
        "N": N,
        "J": J,
        "school": school_ids.tolist(),
        "treatment": treatment.tolist(),
        "y": y.tolist()
    }

    # Save synthetic data
    data_file = output_dir / f"{scenario['name']}_data.json"
    with open(data_file, 'w') as f:
        json.dump(stan_data, f, indent=2)
    print(f"  Saved synthetic data to {data_file.name}")

    # Fit model
    print("\nFitting model to synthetic data...")
    try:
        fit = fit_model(
            model,
            stan_data,
            warmup=1000,
            sampling=1000,
            chains=4,
            show_progress=False
        )
        print("  Fit completed successfully")
    except Exception as e:
        print(f"  ERROR: Fit failed: {e}")
        all_results.append({
            "scenario": scenario['name'],
            "status": "FAILED",
            "error": str(e)
        })
        continue

    # Convert to InferenceData
    idata = az.from_cmdstanpy(
        fit,
        log_likelihood="log_lik",
        posterior_predictive={"y_rep": "y_rep"},
        observed_data={"y": y}
    )

    # Save posterior
    posterior_file = output_dir / f"{scenario['name']}_posterior.nc"
    idata.to_netcdf(posterior_file)
    print(f"  Saved posterior to {posterior_file.name}")

    # Check convergence
    print("\nChecking convergence diagnostics...")
    convergence_result = check_convergence(idata, var_names=["alpha_0", "tau_alpha", "beta", "sigma"])

    print(convergence_result)
    convergence_ok = convergence_result.converged

    # Extract posterior summaries for key parameters
    summary_df = az.summary(idata, var_names=["alpha_0", "tau_alpha", "beta", "sigma"])

    print("\nPosterior summary:")
    print(summary_df[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]])

    # Calculate recovery metrics
    print("\nRecovery assessment:")

    recovery_metrics = {}
    for param in ["alpha_0", "tau_alpha", "beta", "sigma"]:
        true_val = scenario[param]
        post_mean = summary_df.loc[param, "mean"]
        post_sd = summary_df.loc[param, "sd"]
        hdi_lower = summary_df.loc[param, "hdi_3%"]
        hdi_upper = summary_df.loc[param, "hdi_97%"]

        # Check if true value is in 94% HDI
        in_hdi = hdi_lower <= true_val <= hdi_upper

        # Calculate relative error
        rel_error = abs(post_mean - true_val) / abs(true_val) * 100

        # Calculate z-score (how many SDs away from posterior mean)
        z_score = abs(post_mean - true_val) / post_sd

        recovery_metrics[param] = {
            "true": float(true_val),
            "posterior_mean": float(post_mean),
            "posterior_sd": float(post_sd),
            "hdi_lower": float(hdi_lower),
            "hdi_upper": float(hdi_upper),
            "in_hdi": bool(in_hdi),
            "relative_error_pct": float(rel_error),
            "z_score": float(z_score)
        }

        status = "GOOD" if in_hdi and z_score < 2 else "CHECK"
        print(f"  {param}: true={true_val:.2f}, posterior={post_mean:.2f} +/- {post_sd:.2f}")
        print(f"    94% HDI=[{hdi_lower:.2f}, {hdi_upper:.2f}], in_HDI={in_hdi}, z={z_score:.2f} [{status}]")

    # Store results
    result = {
        "scenario": scenario['name'],
        "description": scenario['description'],
        "status": "PASSED" if convergence_ok else "CONVERGENCE_WARNING",
        "convergence": {
            "converged": convergence_ok,
            "max_rhat": convergence_result.max_rhat,
            "min_ess_bulk": convergence_result.min_ess_bulk,
            "min_ess_tail": convergence_result.min_ess_tail,
            "n_divergent": convergence_result.n_divergent
        },
        "recovery": recovery_metrics
    }
    all_results.append(result)

    print("\n")

# Save all results
results_file = output_dir / "recovery_results.json"
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print("=" * 80)
print("RECOVERY CHECK COMPLETE")
print(f"Results saved to {results_file}")
print("=" * 80)
