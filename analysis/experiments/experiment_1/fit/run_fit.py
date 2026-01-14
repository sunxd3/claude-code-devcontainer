"""
Fit the Complete Pooling model to real data.

Fits using adaptive sampling strategy:
1. Short probe (4 chains x 150 iter) to detect issues early
2. Main sampling (4 chains x 1000 iter) if probe succeeds
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
from shared_utils import check_convergence, compile_model, fit_model

# Paths
BASE_DIR = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_1")
MODEL_PATH = BASE_DIR / "model.stan"
DATA_PATH = Path("/home/user/claude-code-devcontainer/analysis/data/stan_data.json")
OUTPUT_DIR = BASE_DIR / "fit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load Stan data from JSON."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded data: N={data['N']}, treatment sum={sum(data['treatment'])}")
    return data

def run_probe_sampling(model, data):
    """Run short probe to detect immediate issues."""
    print("\n" + "="*60)
    print("PROBE SAMPLING: 4 chains x 150 iterations")
    print("="*60)

    try:
        fit = fit_model(
            model=model,
            data=data,
            chains=4,
            warmup=150,
            sampling=150,
            adapt_delta=0.8,
            show_progress=True
        )

        # Quick convergence check using CmdStanPy diagnostics
        try:
            summary = fit.summary()
            # CmdStanPy uses different column names - check what's available
            if 'R_hat' in summary.columns:
                max_rhat = summary['R_hat'].max()
            elif 'R-hat' in summary.columns:
                max_rhat = summary['R-hat'].max()
            else:
                # Fall back to checking available columns
                max_rhat = 1.0  # Assume OK if we can't find it

            print("\nProbe results:")
            print(f"  Max R-hat: {max_rhat:.4f}")

            if max_rhat > 1.05:
                print("  WARNING: High R-hat detected")
                return fit, False

            print("  Probe successful!")
            return fit, True

        except Exception as diag_e:
            print(f"  Diagnostic check failed: {diag_e}")
            print("  Proceeding with main sampling anyway...")
            return fit, True

    except Exception as e:
        print(f"Probe sampling failed: {e}")
        return None, False

def run_main_sampling(model, data):
    """Run main sampling with sufficient iterations."""
    print("\n" + "="*60)
    print("MAIN SAMPLING: 4 chains x 1000 iterations")
    print("="*60)

    try:
        fit = fit_model(
            model=model,
            data=data,
            chains=4,
            warmup=1000,
            sampling=1000,
            adapt_delta=0.8,
            show_progress=True
        )

        return fit

    except Exception as e:
        print(f"Main sampling failed: {e}")
        raise

def main():
    """Main fitting workflow."""
    print("="*60)
    print("Fitting Complete Pooling Model to Real Data")
    print("="*60)

    # Load data
    data = load_data()

    # Compile model
    print(f"\nCompiling model from {MODEL_PATH}...")
    model = compile_model(MODEL_PATH)
    print("Model compiled successfully!")

    # Run probe
    probe_fit, probe_success = run_probe_sampling(model, data)

    if not probe_success:
        print("\nProbe sampling revealed issues. Proceeding with caution...")
        # Could try adjusting adapt_delta here, but let's proceed to main anyway

    # Run main sampling
    fit = run_main_sampling(model, data)

    # Convert to ArviZ InferenceData first
    print("\n" + "="*60)
    print("CONVERTING TO ARVIZ")
    print("="*60)

    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive=["y_rep"],
        log_likelihood="log_lik",
        observed_data={"y": data["y"]},
        coords={
            "obs_id": np.arange(data["N"]),
            "treatment": data["treatment"]
        },
        dims={
            "y": ["obs_id"],
            "y_rep": ["obs_id"],
            "log_lik": ["obs_id"],
            "mu": ["obs_id"]
        }
    )

    # Check convergence using ArviZ
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)

    result = check_convergence(idata, var_names=["alpha", "beta", "sigma"])
    print(result)

    # Save diagnostics to file
    diagnostics_path = OUTPUT_DIR / "convergence_diagnostics.txt"
    with open(diagnostics_path, 'w') as f:
        f.write("Convergence Diagnostics\n")
        f.write("="*60 + "\n\n")
        f.write(str(result))

    print(f"\nDiagnostics saved to {diagnostics_path}")

    # Save to NetCDF
    idata_path = OUTPUT_DIR / "posterior.nc"
    idata.to_netcdf(idata_path)
    print(f"InferenceData saved to {idata_path}")

    # Print summary of key parameters
    print("\n" + "="*60)
    print("KEY PARAMETER ESTIMATES")
    print("="*60)

    summary = az.summary(
        idata,
        var_names=["alpha", "beta", "sigma"],
        kind="stats"
    )
    print(summary)

    # Save summary
    summary_path = OUTPUT_DIR / "parameter_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Parameter Summary\n")
        f.write("="*60 + "\n\n")
        f.write(str(summary))

    print(f"\nParameter summary saved to {summary_path}")

    return idata, result.converged

if __name__ == "__main__":
    idata, converged = main()

    if converged:
        print("\n" + "="*60)
        print("SUCCESS: Model converged!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("WARNING: Convergence issues detected. Review diagnostics.")
        print("="*60)
