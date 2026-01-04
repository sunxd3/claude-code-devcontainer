"""Parameter recovery test for A1-Baseline model.

Tests whether the model can recover known parameters from synthetic data.
This is a sanity check before fitting to real data.
"""

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from shared_utils import check_convergence

# Configuration
EXPERIMENT_DIR = Path("/workspace/analysis/experiments/experiment_1")
OUTPUT_DIR = EXPERIMENT_DIR / "simulation"
MODEL_FILE = EXPERIMENT_DIR / "model.stan"

# True parameter values to recover
TRUE_PARAMS = {
    "alpha": 3.1,
    "beta_weight": -0.9,
    "sigma": 0.25,
}

# Data generation settings
N_OBS = 392
LOG_WEIGHT_C_SD = 0.28  # realistic SD from actual data

# Recovery test settings
N_RECOVERY_TESTS = 5
SEED_BASE = 42


def simulate_data(
    true_params: dict,
    n: int,
    log_weight_c_sd: float,
    seed: int,
) -> dict:
    """Generate synthetic data from the model with known parameters."""
    rng = np.random.default_rng(seed)

    # Generate centered log-weight predictor
    log_weight_c = rng.normal(0, log_weight_c_sd, n)

    # Compute expected log-mpg
    mu = true_params["alpha"] + true_params["beta_weight"] * log_weight_c

    # Generate observed log-mpg with noise
    log_mpg = rng.normal(mu, true_params["sigma"])

    return {
        "N": n,
        "log_mpg": log_mpg.tolist(),
        "log_weight_c": log_weight_c.tolist(),
    }


def check_recovery(
    posterior_summary: pd.DataFrame,
    true_params: dict,
) -> dict:
    """Check if true parameters fall within credible intervals.

    Uses HDI columns from ArviZ summary (hdi_5% and hdi_95% for 90% HDI).
    """
    results = {}

    for param, true_val in true_params.items():
        row = posterior_summary.loc[param]
        mean_val = row["mean"]
        sd_val = row["sd"]
        # ArviZ uses hdi_X% column names for HDI bounds
        lower = row["hdi_5%"]
        upper = row["hdi_95%"]

        in_ci = lower <= true_val <= upper
        bias = mean_val - true_val
        z_score = bias / sd_val

        results[param] = {
            "true": true_val,
            "mean": mean_val,
            "sd": sd_val,
            "lower": lower,
            "upper": upper,
            "in_ci": in_ci,
            "bias": bias,
            "z_score": z_score,
        }

    return results


def run_single_recovery(
    model: CmdStanModel,
    true_params: dict,
    data_settings: dict,
    seed: int,
    test_id: int,
) -> dict:
    """Run a single parameter recovery test."""
    print(f"\n--- Recovery Test {test_id} (seed={seed}) ---")

    # Simulate data
    data = simulate_data(
        true_params,
        data_settings["n"],
        data_settings["log_weight_c_sd"],
        seed,
    )

    # Fit model
    fit = model.sample(
        data=data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        adapt_delta=0.9,
        seed=seed,
        show_progress=False,
    )

    # Convert to ArviZ
    idata = az.from_cmdstanpy(
        fit,
        log_likelihood="log_lik",
        posterior_predictive="y_rep",
        observed_data={"log_mpg": data["log_mpg"]},
    )

    # Check convergence
    convergence = check_convergence(
        idata,
        var_names=["alpha", "beta_weight", "sigma"],
    )
    print(convergence)

    # Get summary with 90% CI
    summary = az.summary(
        idata,
        var_names=["alpha", "beta_weight", "sigma"],
        hdi_prob=0.90,
    )

    # Check recovery
    recovery = check_recovery(posterior_summary=summary, true_params=true_params)

    for param, res in recovery.items():
        status = "OK" if res["in_ci"] else "MISS"
        print(f"  {param}: true={res['true']:.3f}, mean={res['mean']:.3f}, "
              f"90% CI=[{res['lower']:.3f}, {res['upper']:.3f}] [{status}]")

    return {
        "test_id": test_id,
        "seed": seed,
        "convergence": {
            "converged": convergence.converged,
            "max_rhat": convergence.max_rhat,
            "min_ess_bulk": convergence.min_ess_bulk,
            "min_ess_tail": convergence.min_ess_tail,
            "n_divergent": convergence.n_divergent,
        },
        "recovery": recovery,
        "idata": idata,
    }


def plot_recovery_scatter(results: list, output_path: Path):
    """Create scatter plot of posterior mean vs true parameter."""
    params = ["alpha", "beta_weight", "sigma"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, param in zip(axes, params):
        true_vals = []
        post_means = []
        post_sds = []

        for r in results:
            rec = r["recovery"][param]
            true_vals.append(rec["true"])
            post_means.append(rec["mean"])
            post_sds.append(rec["sd"])

        true_vals = np.array(true_vals)
        post_means = np.array(post_means)
        post_sds = np.array(post_sds)

        # Plot error bars
        ax.errorbar(
            true_vals, post_means, yerr=1.645 * post_sds,
            fmt="o", capsize=4, markersize=8, alpha=0.8
        )

        # Identity line
        lims = [
            min(true_vals.min(), post_means.min()) - 0.1,
            max(true_vals.max(), post_means.max()) + 0.1,
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Identity")

        ax.set_xlabel("True value")
        ax.set_ylabel("Posterior mean")
        ax.set_title(param)
        ax.set_aspect("equal", adjustable="box")

    plt.suptitle("Parameter Recovery: Posterior Mean vs True Value\n(error bars = 90% CI)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_recovery_intervals(results: list, output_path: Path):
    """Create interval plot showing true values with posterior CIs."""
    params = ["alpha", "beta_weight", "sigma"]
    n_tests = len(results)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for ax, param in zip(axes, params):
        for i, r in enumerate(results):
            rec = r["recovery"][param]
            color = "green" if rec["in_ci"] else "red"

            # Plot CI as horizontal line
            ax.plot(
                [rec["lower"], rec["upper"]], [i, i],
                color=color, linewidth=2, alpha=0.7
            )

            # Plot posterior mean
            ax.scatter([rec["mean"]], [i], color=color, s=50, zorder=3)

            # Plot true value as vertical line
            ax.axvline(rec["true"], color="black", linestyle="--", alpha=0.5)

        ax.set_ylabel(param)
        ax.set_yticks(range(n_tests))
        ax.set_yticklabels([f"Test {i+1}" for i in range(n_tests)])

    axes[-1].set_xlabel("Parameter value")
    plt.suptitle("Parameter Recovery: 90% Credible Intervals\n(green=recovered, red=missed, dashed=true)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_trace_example(idata, output_path: Path):
    """Create trace plot for one example fit."""
    az.plot_trace(
        idata,
        var_names=["alpha", "beta_weight", "sigma"],
        figsize=(12, 8),
    )
    plt.suptitle("Trace Plot (Test 1)", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run parameter recovery tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PARAMETER RECOVERY TEST: A1-Baseline Model")
    print("=" * 60)
    print(f"\nTrue parameters: {TRUE_PARAMS}")
    print(f"Number of recovery tests: {N_RECOVERY_TESTS}")
    print(f"Observations per test: {N_OBS}")

    # Compile model
    print(f"\nCompiling model: {MODEL_FILE}")
    model = CmdStanModel(stan_file=str(MODEL_FILE))

    # Run recovery tests
    data_settings = {"n": N_OBS, "log_weight_c_sd": LOG_WEIGHT_C_SD}
    results = []

    for i in range(N_RECOVERY_TESTS):
        seed = SEED_BASE + i * 100
        result = run_single_recovery(
            model, TRUE_PARAMS, data_settings, seed, test_id=i + 1
        )
        results.append(result)

    # Summary statistics
    print("\n" + "=" * 60)
    print("RECOVERY SUMMARY")
    print("=" * 60)

    all_converged = all(r["convergence"]["converged"] for r in results)
    all_recovered = {}

    for param in TRUE_PARAMS:
        recoveries = [r["recovery"][param]["in_ci"] for r in results]
        all_recovered[param] = sum(recoveries)
        print(f"{param}: {sum(recoveries)}/{len(recoveries)} tests recovered true value")

    convergence_issues = sum(1 for r in results if not r["convergence"]["converged"])
    print(f"\nConvergence: {N_RECOVERY_TESTS - convergence_issues}/{N_RECOVERY_TESTS} tests converged")

    # Generate plots
    print("\nGenerating diagnostic plots...")
    plot_recovery_scatter(results, OUTPUT_DIR / "recovery_scatter.png")
    plot_recovery_intervals(results, OUTPUT_DIR / "recovery_intervals.png")
    plot_trace_example(results[0]["idata"], OUTPUT_DIR / "trace_example.png")

    # Determine PASS/FAIL
    total_recovered = sum(all_recovered.values())
    total_tests = len(TRUE_PARAMS) * N_RECOVERY_TESTS

    # Pass criteria:
    # - All tests converged (R-hat < 1.01, ESS > 400, no divergences)
    # - At least 80% of individual parameter recoveries successful (allows for 90% CI sampling variability)
    recovery_rate = total_recovered / total_tests

    passed = all_converged and recovery_rate >= 0.80

    # Helper to convert numpy types for JSON
    def to_native(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    # Save results
    summary_data = {
        "model": "A1-Baseline",
        "true_params": TRUE_PARAMS,
        "n_tests": N_RECOVERY_TESTS,
        "n_obs": N_OBS,
        "all_converged": all_converged,
        "convergence_issues": convergence_issues,
        "recovery_by_param": all_recovered,
        "total_recovered": total_recovered,
        "total_tests": total_tests,
        "recovery_rate": recovery_rate,
        "passed": passed,
        "test_results": [
            {
                "test_id": r["test_id"],
                "seed": r["seed"],
                "convergence": {k: to_native(v) for k, v in r["convergence"].items()},
                "recovery": {
                    param: {k: to_native(v) for k, v in rec.items() if k != "idata"}
                    for param, rec in r["recovery"].items()
                },
            }
            for r in results
        ],
    }

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(OUTPUT_DIR / "recovery_results.json", "w") as f:
        json.dump(summary_data, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved: {OUTPUT_DIR / 'recovery_results.json'}")

    # Final verdict
    print("\n" + "=" * 60)
    if passed:
        print("RESULT: PASS")
        print("Model successfully recovers true parameters from synthetic data.")
    else:
        print("RESULT: FAIL")
        if not all_converged:
            print("  - Convergence issues detected")
        if recovery_rate < 0.80:
            print(f"  - Recovery rate {recovery_rate:.1%} below 80% threshold")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    main()
