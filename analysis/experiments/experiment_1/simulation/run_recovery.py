#!/usr/bin/env python3
"""Parameter recovery check for Complete Pooling model.

Tests whether the model can recover known parameters from synthetic data.
"""

import json
from pathlib import Path

import numpy as np
from shared_utils import compile_model, fit_model

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_1/simulation")
output_dir.mkdir(parents=True, exist_ok=True)

# Model path
model_path = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_1/model.stan")

# Test scenarios: (name, alpha, beta, sigma)
scenarios = [
    ("weak_effect", 75.0, 2.0, 12.0),
    ("medium_effect", 70.0, 5.0, 10.0),
    ("strong_effect", 80.0, 10.0, 8.0),
]

# Data structure
N = 160
treatment = np.array([0] * 80 + [1] * 80)  # 50% treated

print("=" * 60)
print("Parameter Recovery Check: Complete Pooling Model")
print("=" * 60)
print(f"\nData: N={N}, 50% treated")
print(f"Testing {len(scenarios)} scenarios\n")

# Compile model once
print("Compiling model...")
try:
    model = compile_model(model_path)
    print("Model compiled successfully\n")
except Exception as e:
    print(f"FAILED: Model compilation error: {e}")
    exit(1)

# Store results for all scenarios
all_results = []

# Run recovery for each scenario
for scenario_name, true_alpha, true_beta, true_sigma in scenarios:
    print(f"\n{'=' * 60}")
    print(f"Scenario: {scenario_name}")
    print(f"True parameters: alpha={true_alpha}, beta={true_beta}, sigma={true_sigma}")
    print('=' * 60)

    # Generate synthetic data
    mu = true_alpha + true_beta * treatment
    y = np.random.normal(mu, true_sigma)

    print(f"Generated data: mean(y)={y.mean():.2f}, sd(y)={y.std():.2f}")
    print(f"  Control group: mean={y[:80].mean():.2f}, sd={y[:80].std():.2f}")
    print(f"  Treatment group: mean={y[80:].mean():.2f}, sd={y[80:].std():.2f}")

    # Prepare Stan data
    stan_data = {
        "N": N,
        "treatment": treatment.tolist(),
        "y": y.tolist(),
    }

    # Fit model
    print("\nFitting model...")
    try:
        fit = fit_model(
            model,
            stan_data,
            warmup=1000,
            sampling=1000,
            chains=4,
            show_progress=False,
        )

        # Extract posterior summary
        summary = fit.summary()

        # Extract key parameters
        alpha_post = summary.loc["alpha"]
        beta_post = summary.loc["beta"]
        sigma_post = summary.loc["sigma"]

        print("\nPosterior summaries:")
        print(f"  alpha: mean={alpha_post['Mean']:.2f}, sd={alpha_post['StdDev']:.2f}, "
              f"R-hat={alpha_post['R_hat']:.4f}, ESS_bulk={alpha_post['ESS_bulk']:.0f}")
        print(f"  beta:  mean={beta_post['Mean']:.2f}, sd={beta_post['StdDev']:.2f}, "
              f"R-hat={beta_post['R_hat']:.4f}, ESS_bulk={beta_post['ESS_bulk']:.0f}")
        print(f"  sigma: mean={sigma_post['Mean']:.2f}, sd={sigma_post['StdDev']:.2f}, "
              f"R-hat={sigma_post['R_hat']:.4f}, ESS_bulk={sigma_post['ESS_bulk']:.0f}")

        # Check recovery (is true value within posterior 95% CI?)
        alpha_recovered = alpha_post["5%"] <= true_alpha <= alpha_post["95%"]
        beta_recovered = beta_post["5%"] <= true_beta <= beta_post["95%"]
        sigma_recovered = sigma_post["5%"] <= true_sigma <= sigma_post["95%"]

        # Calculate recovery error (posterior mean vs true value)
        alpha_error = abs(alpha_post['Mean'] - true_alpha)
        beta_error = abs(beta_post['Mean'] - true_beta)
        sigma_error = abs(sigma_post['Mean'] - true_sigma)

        print("\nRecovery check (true value in 90% CI?):")
        print(f"  alpha: {alpha_recovered} (error={alpha_error:.2f})")
        print(f"  beta:  {beta_recovered} (error={beta_error:.2f})")
        print(f"  sigma: {sigma_recovered} (error={sigma_error:.2f})")

        # Check convergence
        rhat_ok = (alpha_post['R_hat'] < 1.01 and
                   beta_post['R_hat'] < 1.01 and
                   sigma_post['R_hat'] < 1.01)
        ess_ok = (alpha_post['ESS_bulk'] >= 400 and
                  beta_post['ESS_bulk'] >= 400 and
                  sigma_post['ESS_bulk'] >= 400)

        converged = rhat_ok and ess_ok

        print(f"\nConvergence: {'PASS' if converged else 'FAIL'}")
        print(f"  R-hat < 1.01: {rhat_ok}")
        print(f"  ESS_bulk >= 400: {ess_ok}")

        # Check for divergences
        try:
            divergences = fit.method_variables()["divergent__"].sum()
            print(f"  Divergences: {divergences}")
        except (KeyError, AttributeError):
            divergences = 0

        # Save fit
        fit_path = output_dir / f"{scenario_name}_fit"
        fit.save_csvfiles(dir=str(fit_path))
        print(f"\nSaved fit to {fit_path}/")

        # Store results
        result = {
            "scenario": scenario_name,
            "true_alpha": true_alpha,
            "true_beta": true_beta,
            "true_sigma": true_sigma,
            "post_alpha_mean": float(alpha_post['Mean']),
            "post_alpha_sd": float(alpha_post['StdDev']),
            "post_alpha_q05": float(alpha_post['5%']),
            "post_alpha_q95": float(alpha_post['95%']),
            "post_beta_mean": float(beta_post['Mean']),
            "post_beta_sd": float(beta_post['StdDev']),
            "post_beta_q05": float(beta_post['5%']),
            "post_beta_q95": float(beta_post['95%']),
            "post_sigma_mean": float(sigma_post['Mean']),
            "post_sigma_sd": float(sigma_post['StdDev']),
            "post_sigma_q05": float(sigma_post['5%']),
            "post_sigma_q95": float(sigma_post['95%']),
            "alpha_recovered": bool(alpha_recovered),
            "beta_recovered": bool(beta_recovered),
            "sigma_recovered": bool(sigma_recovered),
            "alpha_error": float(alpha_error),
            "beta_error": float(beta_error),
            "sigma_error": float(sigma_error),
            "alpha_rhat": float(alpha_post['R_hat']),
            "beta_rhat": float(beta_post['R_hat']),
            "sigma_rhat": float(sigma_post['R_hat']),
            "alpha_ess_bulk": float(alpha_post['ESS_bulk']),
            "beta_ess_bulk": float(beta_post['ESS_bulk']),
            "sigma_ess_bulk": float(sigma_post['ESS_bulk']),
            "divergences": int(divergences),
            "converged": bool(converged),
            "fit_success": True,
        }

        all_results.append(result)
        print(f"\nScenario {scenario_name}: SUCCESS")

    except Exception as e:
        print(f"\nScenario {scenario_name}: FAILED - {e}")
        result = {
            "scenario": scenario_name,
            "true_alpha": true_alpha,
            "true_beta": true_beta,
            "true_sigma": true_sigma,
            "fit_success": False,
            "error": str(e),
        }
        all_results.append(result)

# Save all results
results_path = output_dir / "recovery_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'=' * 60}")
print("Recovery check complete")
print(f"Results saved to {results_path}")
print('=' * 60)

# Summary
successful = sum(1 for r in all_results if r.get("fit_success", False))
print(f"\nSuccessful fits: {successful}/{len(scenarios)}")

if successful > 0:
    recovered = sum(1 for r in all_results
                   if r.get("fit_success", False)
                   and r.get("alpha_recovered", False)
                   and r.get("beta_recovered", False)
                   and r.get("sigma_recovered", False))
    converged = sum(1 for r in all_results
                   if r.get("fit_success", False)
                   and r.get("converged", False))

    print(f"Full recovery (all params in 90% CI): {recovered}/{successful}")
    print(f"Converged (R-hat < 1.01, ESS >= 400): {converged}/{successful}")
