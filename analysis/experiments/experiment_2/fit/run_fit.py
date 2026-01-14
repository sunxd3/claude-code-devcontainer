#!/usr/bin/env python3
"""
Fit Experiment 2: Random Intercepts Only model to real data

Performs:
1. Model compilation
2. Initial probe (short chains)
3. Convergence diagnostics
4. Full sampling (if probe succeeds)
5. ArviZ conversion and save
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
from shared_utils import check_convergence, compile_model, fit_model

# Paths
BASE_DIR = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2")
MODEL_PATH = BASE_DIR / "model.stan"
DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/stan_data.json")
OUTPUT_DIR = BASE_DIR / "fit"

def main():
    print("=" * 80)
    print("EXPERIMENT 2: Random Intercepts Only - Model Fitting")
    print("=" * 80)

    # Load data
    print("\n[1/7] Loading data...")
    with open(DATA_PATH) as f:
        stan_data = json.load(f)
    print(f"  Data: N={stan_data['N']}, J={stan_data['J']}")

    # Compile model
    print("\n[2/7] Compiling model...")
    try:
        model = compile_model(str(MODEL_PATH))
        print("  Compilation successful")
    except Exception as e:
        print(f"  ERROR: Compilation failed - {e}")
        return False

    # Run initial probe
    print("\n[3/7] Running initial probe (4 chains, 200 iterations)...")
    try:
        probe = fit_model(
            model=model,
            data=stan_data,
            warmup=100,
            sampling=100,
            chains=4,
            adapt_delta=0.8,
            show_progress=True
        )
        print("  Probe completed")
    except Exception as e:
        print(f"  ERROR: Probe failed - {e}")
        return False

    # Check probe diagnostics
    print("\n[4/7] Checking probe diagnostics...")
    probe_idata = az.from_cmdstanpy(probe)
    probe_result = check_convergence(
        probe_idata,
        rhat_threshold=1.01,
        ess_bulk_threshold=100,  # Lower threshold for probe
        ess_tail_threshold=100
    )

    print(f"  Max R̂: {probe_result.max_rhat:.4f}")
    print(f"  Min ESS bulk: {probe_result.min_ess_bulk:.0f}")
    print(f"  Divergences: {probe_result.n_divergent}")

    if not probe_result.converged:
        print("\n  WARNING: Probe shows convergence issues")
        print("  Proceeding with full sampling to see if longer chains help...")
    else:
        print("  Probe diagnostics look good")

    # Adjust adapt_delta based on probe divergences
    if probe_result.n_divergent > 0:
        print("  Increasing adapt_delta to 0.95 for full run")
        adapt_delta = 0.95
    else:
        adapt_delta = 0.8

    # Run full sampling
    print("\n[5/7] Running full sampling (4 chains, 1000 iterations)...")
    try:
        fit = fit_model(
            model=model,
            data=stan_data,
            warmup=1000,
            sampling=1000,
            chains=4,
            adapt_delta=adapt_delta,
            show_progress=True
        )
        print("  Sampling completed")
    except Exception as e:
        print(f"  ERROR: Sampling failed - {e}")
        return False

    # Check full diagnostics
    print("\n[6/7] Checking convergence diagnostics...")

    # Get summary for key parameters
    summary = fit.summary()
    print("\n  Key parameters:")
    key_params = ['alpha_0', 'tau_alpha', 'beta', 'sigma']
    for param in key_params:
        if param in summary.index:
            row = summary.loc[param]
            print(f"    {param:12s}: {row['Mean']:7.3f} ± {row['StdDev']:7.3f}  "
                  f"[R̂={row['R_hat']:.4f}, ESS_bulk={row['ESS_bulk']:.0f}]")

    # Convert to ArviZ and check convergence
    print("\n[7/7] Converting to ArviZ InferenceData...")
    try:
        idata = az.from_cmdstanpy(
            fit,
            posterior_predictive=["y_rep"],
            log_likelihood="log_lik",
            observed_data={"y": stan_data["y"]},
            coords={
                "school": np.arange(1, stan_data['J'] + 1),
                "obs": np.arange(stan_data['N'])
            },
            dims={
                "alpha": ["school"],
                "z_alpha": ["school"],
                "y": ["obs"],
                "y_rep": ["obs"],
                "log_lik": ["obs"]
            }
        )
        print("  Conversion successful")

        # Check convergence with ArviZ
        result = check_convergence(idata)
        print(f"\n  Max R̂: {result.max_rhat:.4f}")
        print(f"  Min ESS bulk: {result.min_ess_bulk:.0f}")
        print(f"  Min ESS tail: {result.min_ess_tail:.0f}")
        print(f"  Divergences: {result.n_divergent}")

        if result.converged:
            print("\n  ✓ All convergence diagnostics passed")
            converged = True
        else:
            print("\n  ✗ Convergence issues detected - review diagnostics carefully")
            converged = False

        # Save as NetCDF
        output_path = OUTPUT_DIR / "posterior.nc"
        idata.to_netcdf(output_path)
        print(f"\n  Saved to: {output_path}")

        # Also save fit object
        fit.save_csvfiles(str(OUTPUT_DIR / "stan_output"))
        print(f"  Stan CSV files saved to: {OUTPUT_DIR / 'stan_output'}")

    except Exception as e:
        print(f"  ERROR: ArviZ conversion failed - {e}")
        return False

    # Print final diagnostics
    print("\n" + "=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)

    print(f"Divergent transitions: {result.n_divergent}")
    print(f"Max R̂: {result.max_rhat:.4f}")
    print(f"Min ESS_bulk: {result.min_ess_bulk:.0f}")
    print(f"Min ESS_tail: {result.min_ess_tail:.0f}")

    print("\nKey parameter estimates:")
    print(f"  Population mean (alpha_0): {summary.loc['alpha_0', 'Mean']:.2f} ± {summary.loc['alpha_0', 'StdDev']:.2f}")
    print(f"  School SD (tau_alpha):     {summary.loc['tau_alpha', 'Mean']:.2f} ± {summary.loc['tau_alpha', 'StdDev']:.2f}")
    print(f"  Treatment effect (beta):   {summary.loc['beta', 'Mean']:.2f} ± {summary.loc['beta', 'StdDev']:.2f}")
    print(f"  Residual SD (sigma):       {summary.loc['sigma', 'Mean']:.2f} ± {summary.loc['sigma', 'StdDev']:.2f}")

    print("\n" + "=" * 80)

    if converged:
        print("STATUS: ✓ Model fit successfully with good convergence")
    else:
        print("STATUS: ✗ Model fit completed but with convergence issues")

    print("=" * 80)

    return converged

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
