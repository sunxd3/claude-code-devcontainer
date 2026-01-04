"""Parameter recovery check for A3-Robust model (Student-t errors).

Tests whether the model can recover known parameters from synthetic data.
Key test: Can we recover nu (degrees of freedom)? nu=15 is closer to normal,
making it harder to identify than heavy-tailed values like nu=5.
"""

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from scipy import stats as scipy_stats

from shared_utils import check_convergence

# Configuration
EXP_DIR = Path("/workspace/analysis/experiments/experiment_3")
OUTPUT_DIR = EXP_DIR / "simulation"
MODEL_FILE = EXP_DIR / "model.stan"

# True parameters to recover (as specified in task)
TRUE_PARAMS = {
    "alpha": 3.1,
    "beta_weight": -0.9,
    "beta_year": 0.03,
    "sigma": 0.12,
    "nu": 15.0,  # Closer to normal - harder to identify than heavy tails
}

# Simulation settings
N_SIMS = 5
N_OBS = 200
SEED = 42


def simulate_data(true_params: dict, n_obs: int, rng: np.random.Generator) -> dict:
    """Generate synthetic data from the Student-t model."""
    log_weight_c = rng.uniform(-0.7, 0.7, n_obs)
    year_c = rng.uniform(-6, 6, n_obs)

    mu = (
        true_params["alpha"]
        + true_params["beta_weight"] * log_weight_c
        + true_params["beta_year"] * year_c
    )

    # Student-t errors
    log_mpg = mu + true_params["sigma"] * rng.standard_t(true_params["nu"], n_obs)

    return {
        "N": n_obs,
        "log_mpg": log_mpg.tolist(),
        "log_weight_c": log_weight_c.tolist(),
        "year_c": year_c.tolist(),
    }


def run_single_recovery(model: CmdStanModel, data: dict, sim_id: int) -> dict:
    """Run a single parameter recovery test."""
    print(f"\n--- Simulation {sim_id} ---")

    fit = model.sample(
        data=data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        adapt_delta=0.95,  # Higher for Student-t
        seed=SEED + sim_id,
        show_progress=False,
    )

    idata = az.from_cmdstanpy(fit, log_likelihood="log_lik", posterior_predictive=["y_rep"])

    conv = check_convergence(
        idata, var_names=["alpha", "beta_weight", "beta_year", "sigma", "nu"]
    )
    print(conv)

    summary = az.summary(
        idata, var_names=["alpha", "beta_weight", "beta_year", "sigma", "nu"],
        hdi_prob=0.90  # 90% CI as requested
    )

    results = {"sim_id": sim_id, "converged": conv.converged}

    for param in TRUE_PARAMS:
        true_val = TRUE_PARAMS[param]
        post_mean = summary.loc[param, "mean"]
        post_sd = summary.loc[param, "sd"]
        hdi_low = summary.loc[param, "hdi_5%"]
        hdi_high = summary.loc[param, "hdi_95%"]

        covered = hdi_low <= true_val <= hdi_high

        results[f"{param}_true"] = true_val
        results[f"{param}_mean"] = post_mean
        results[f"{param}_sd"] = post_sd
        results[f"{param}_hdi_low"] = hdi_low
        results[f"{param}_hdi_high"] = hdi_high
        results[f"{param}_covered"] = covered
        results[f"{param}_error"] = post_mean - true_val
        results[f"{param}_z"] = (post_mean - true_val) / post_sd

    results["max_rhat"] = conv.max_rhat
    results["min_ess_bulk"] = conv.min_ess_bulk
    results["n_divergent"] = conv.n_divergent

    return results


def create_recovery_plots(df: pd.DataFrame, output_dir: Path):
    """Create parameter recovery visualizations."""
    params = list(TRUE_PARAMS.keys())
    n_params = len(params)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        true_val = TRUE_PARAMS[param]

        for j, row in df.iterrows():
            color = "forestgreen" if row[f"{param}_covered"] else "crimson"
            ax.errorbar(
                row["sim_id"],
                row[f"{param}_mean"],
                yerr=[
                    [row[f"{param}_mean"] - row[f"{param}_hdi_low"]],
                    [row[f"{param}_hdi_high"] - row[f"{param}_mean"]],
                ],
                fmt="o",
                color=color,
                capsize=4,
                markersize=6,
            )

        ax.axhline(true_val, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Simulation")
        ax.set_ylabel(param)
        ax.set_title(f"{param} (true = {true_val})")
        ax.set_xticks(range(N_SIMS))

    # Hide extra subplot
    if n_params < len(axes):
        axes[-1].axis("off")

    plt.suptitle("90% CI Coverage (green=covered, red=missed)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "recovery_intervals.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'recovery_intervals.png'}")


def create_scatter_plot(df: pd.DataFrame, output_dir: Path):
    """Create scatter plot of posterior mean vs true value."""
    fig, ax = plt.subplots(figsize=(7, 7))

    true_vals = []
    post_means = []
    labels = []
    colors = {"alpha": "C0", "beta_weight": "C1", "beta_year": "C2", "sigma": "C3", "nu": "C4"}

    for param in TRUE_PARAMS:
        for _, row in df.iterrows():
            true_vals.append(row[f"{param}_true"])
            post_means.append(row[f"{param}_mean"])
            labels.append(param)

    for param in TRUE_PARAMS:
        mask = [l == param for l in labels]
        true_p = [t for t, m in zip(true_vals, mask) if m]
        post_p = [p for p, m in zip(post_means, mask) if m]
        ax.scatter(true_p, post_p, label=param, alpha=0.7, s=60, c=colors[param])

    all_vals = true_vals + post_means
    lims = [min(all_vals) - 0.1, max(all_vals) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=1, label="Identity")

    ax.set_xlabel("True Parameter Value")
    ax.set_ylabel("Posterior Mean")
    ax.set_title("Parameter Recovery: Posterior Mean vs True")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "recovery_scatter.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'recovery_scatter.png'}")


def create_z_score_plot(df: pd.DataFrame, output_dir: Path):
    """Create z-score histogram for calibration check."""
    fig, ax = plt.subplots(figsize=(8, 5))

    z_scores = []
    for param in TRUE_PARAMS:
        z_scores.extend(df[f"{param}_z"].tolist())

    ax.hist(z_scores, bins=15, density=True, alpha=0.7, edgecolor="black")

    x = np.linspace(-4, 4, 100)
    ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "r-", linewidth=2, label="N(0,1)")

    ax.set_xlabel("Z-score (posterior mean - true) / posterior SD")
    ax.set_ylabel("Density")
    ax.set_title("Recovery Calibration: Z-scores should follow N(0,1)")
    ax.legend()
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "recovery_zscore.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'recovery_zscore.png'}")


def create_nu_focus_plot(df: pd.DataFrame, output_dir: Path):
    """Create detailed plot focused on nu recovery."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: nu intervals with wider context
    ax = axes[0]
    true_nu = TRUE_PARAMS["nu"]

    for j, row in df.iterrows():
        color = "forestgreen" if row["nu_covered"] else "crimson"
        ax.errorbar(
            row["sim_id"],
            row["nu_mean"],
            yerr=[
                [row["nu_mean"] - row["nu_hdi_low"]],
                [row["nu_hdi_high"] - row["nu_mean"]],
            ],
            fmt="o",
            color=color,
            capsize=4,
            markersize=8,
        )

    ax.axhline(true_nu, color="black", linestyle="--", linewidth=2, label=f"True nu = {true_nu}")
    ax.set_xlabel("Simulation")
    ax.set_ylabel("nu (degrees of freedom)")
    ax.set_title("nu Recovery: 90% CI")
    ax.legend()
    ax.set_xticks(range(N_SIMS))

    # Right: nu posterior width comparison
    ax = axes[1]
    widths = df["nu_hdi_high"] - df["nu_hdi_low"]
    ax.bar(range(N_SIMS), widths, color="steelblue", alpha=0.7)
    ax.set_xlabel("Simulation")
    ax.set_ylabel("90% CI Width")
    ax.set_title("nu Posterior Uncertainty")
    ax.set_xticks(range(N_SIMS))

    plt.tight_layout()
    fig.savefig(output_dir / "recovery_nu_focus.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'recovery_nu_focus.png'}")


def main():
    """Run parameter recovery tests."""
    print("=" * 60)
    print("Parameter Recovery Check: A3-Robust Model (Student-t)")
    print("=" * 60)
    print(f"\nTrue parameters: {TRUE_PARAMS}")
    print(f"Running {N_SIMS} simulations with N={N_OBS} observations each")
    print(f"\nNote: nu=15 is closer to normal, making it harder to identify")

    print(f"\nCompiling model: {MODEL_FILE}")
    model = CmdStanModel(stan_file=str(MODEL_FILE))

    rng = np.random.default_rng(SEED)
    results = []

    for sim_id in range(N_SIMS):
        data = simulate_data(TRUE_PARAMS, N_OBS, rng)
        result = run_single_recovery(model, data, sim_id)
        results.append(result)

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("RECOVERY SUMMARY")
    print("=" * 60)

    all_converged = df["converged"].all()
    print(f"\nAll simulations converged: {all_converged}")
    print(f"Max R-hat across all: {df['max_rhat'].max():.4f}")
    print(f"Min ESS bulk across all: {df['min_ess_bulk'].min():.0f}")
    print(f"Total divergences: {df['n_divergent'].sum()}")

    print("\n90% CI Coverage by Parameter:")
    coverage_results = {}
    for param in TRUE_PARAMS:
        coverage = df[f"{param}_covered"].mean() * 100
        coverage_results[param] = coverage
        print(f"  {param}: {coverage:.0f}% ({df[f'{param}_covered'].sum()}/{N_SIMS})")

    print("\nMean Error (posterior mean - true):")
    for param in TRUE_PARAMS:
        mean_error = df[f"{param}_error"].mean()
        print(f"  {param}: {mean_error:+.4f}")

    print("\nnu Recovery Details (key identifiability check):")
    print(f"  Mean posterior nu: {df['nu_mean'].mean():.1f} (true: {TRUE_PARAMS['nu']})")
    print(f"  Mean posterior SD: {df['nu_sd'].mean():.1f}")
    print(f"  Coverage: {coverage_results['nu']:.0f}%")

    print("\nCreating diagnostic plots...")
    create_recovery_plots(df, OUTPUT_DIR)
    create_scatter_plot(df, OUTPUT_DIR)
    create_z_score_plot(df, OUTPUT_DIR)
    create_nu_focus_plot(df, OUTPUT_DIR)

    df.to_csv(OUTPUT_DIR / "recovery_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'recovery_results.csv'}")

    # PASS/FAIL criteria (90% CI should cover ~90% of the time)
    overall_coverage = np.mean([coverage_results[p] for p in TRUE_PARAMS])

    # For 5 simulations and 90% CI, we expect some misses but not systematic failure
    # Coverage >= 60% is reasonable given small sample (5 sims)
    pass_criteria = {
        "all_converged": all_converged,
        "no_divergences": df["n_divergent"].sum() == 0,
        "coverage_adequate": overall_coverage >= 60,
        "max_rhat_ok": df["max_rhat"].max() < 1.02,
        "nu_identifiable": coverage_results["nu"] >= 60,  # Special attention to nu
    }

    passed = all(pass_criteria.values())

    print("\n" + "=" * 60)
    print("ASSESSMENT")
    print("=" * 60)
    print(f"\nPass criteria:")
    for criterion, met in pass_criteria.items():
        status = "PASS" if met else "FAIL"
        print(f"  {criterion}: {status}")

    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"OVERALL RESULT: {status}")
    print(f"{'=' * 60}")

    summary = {
        "model": "A3-Robust (Student-t)",
        "true_params": TRUE_PARAMS,
        "n_simulations": N_SIMS,
        "n_observations": N_OBS,
        "ci_level": 0.90,
        "all_converged": bool(all_converged),
        "total_divergences": int(df["n_divergent"].sum()),
        "max_rhat": float(df["max_rhat"].max()),
        "min_ess_bulk": float(df["min_ess_bulk"].min()),
        "coverage_by_param": coverage_results,
        "overall_coverage": float(overall_coverage),
        "nu_recovery": {
            "mean_posterior": float(df["nu_mean"].mean()),
            "mean_sd": float(df["nu_sd"].mean()),
            "coverage": float(coverage_results["nu"]),
            "note": "nu=15 is close to normal, harder to identify than heavy tails"
        },
        "pass_criteria": {k: bool(v) for k, v in pass_criteria.items()},
        "passed": bool(passed),
    }

    with open(OUTPUT_DIR / "recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'recovery_summary.json'}")

    return passed


if __name__ == "__main__":
    passed = main()
    exit(0 if passed else 1)
