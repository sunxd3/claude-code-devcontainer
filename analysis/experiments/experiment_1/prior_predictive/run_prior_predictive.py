"""Prior predictive check for A1-Baseline model."""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from shared_utils import compile_model

from plots import create_diagnostic_plots

EXPERIMENT_DIR = Path("/workspace/analysis/experiments/experiment_1")
PRIOR_MODEL_FILE = EXPERIMENT_DIR / "prior_predictive" / "prior_model.stan"
DATA_FILE = Path("/workspace/analysis/eda/auto_mpg_cleaned.csv")
OUTPUT_DIR = EXPERIMENT_DIR / "prior_predictive"


def load_and_prepare_data() -> dict:
    """Load Auto-MPG data and prepare for Stan."""
    df = pd.read_csv(DATA_FILE)
    log_mpg = np.log(df["mpg"].values)
    log_weight = np.log(df["weight"].values)
    log_weight_c = log_weight - np.mean(log_weight)

    return {
        "N": len(df),
        "log_mpg": log_mpg.tolist(),
        "log_weight_c": log_weight_c.tolist(),
        "mpg": df["mpg"].values,
    }


def sample_prior_predictive(model, data: dict, n_draws: int = 4000):
    """Sample from prior predictive using fixed_param."""
    return model.sample(
        data=data,
        chains=4,
        iter_warmup=0,
        iter_sampling=n_draws // 4,
        fixed_param=True,
        adapt_engaged=False,
        show_progress=True,
    )


def assess_prior_predictive(mpg_rep: np.ndarray, obs_mpg: np.ndarray) -> tuple[str, list[str]]:
    """Assess prior predictive distribution against criteria."""
    obs_min, obs_max = obs_mpg.min(), obs_mpg.max()
    total_draws = mpg_rep.size
    issues = []

    n_over_100 = np.sum(mpg_rep > 100)
    if n_over_100 / total_draws > 0.01:
        issues.append(f"WARN: {100*n_over_100/total_draws:.2f}% exceed 100 MPG")

    pct_in_range = np.mean((mpg_rep >= obs_min) & (mpg_rep <= obs_max))
    if pct_in_range < 0.30:
        issues.append(f"WARN: Only {pct_in_range:.1%} in observed range")

    prior_90_width = np.percentile(mpg_rep, 95) - np.percentile(mpg_rep, 5)
    if prior_90_width < (obs_max - obs_min):
        issues.append(f"WARN: Prior too narrow ({prior_90_width:.1f} < {obs_max-obs_min:.1f})")

    if np.percentile(mpg_rep, 99.5) > 200:
        issues.append(f"WARN: 99.5th percentile > 200 MPG")

    verdict = "PASS" if not any("FAIL" in i for i in issues) else "FAIL"
    return verdict, issues


def main():
    print("Prior Predictive Check: A1-Baseline Model")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare_data()
    print(f"N={data['N']}, MPG range: {min(data['mpg']):.1f}-{max(data['mpg']):.1f}")

    model = compile_model(PRIOR_MODEL_FILE)
    fit = sample_prior_predictive(model, data)

    # Extract samples
    alpha = fit.stan_variable("alpha")
    beta_weight = fit.stan_variable("beta_weight")
    sigma = fit.stan_variable("sigma")
    y_rep = fit.stan_variable("y_rep")
    mpg_rep = np.exp(y_rep)

    print(f"alpha: {np.mean(alpha):.3f} +/- {np.std(alpha):.3f}")
    print(f"beta_weight: {np.mean(beta_weight):.3f} +/- {np.std(beta_weight):.3f}")
    print(f"sigma: {np.mean(sigma):.3f} +/- {np.std(sigma):.3f}")
    print(f"MPG 95% interval: [{np.percentile(mpg_rep, 2.5):.1f}, {np.percentile(mpg_rep, 97.5):.1f}]")

    # Save InferenceData (prior model has no log_lik)
    coords = {"obs_id": np.arange(data["N"])}
    dims = {"y_rep": ["obs_id"]}
    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive=["y_rep"],
        observed_data={"y": np.array(data["log_mpg"])},
        coords=coords,
        dims=dims,
    )
    idata = az.InferenceData(
        prior=idata.posterior,
        prior_predictive=idata.posterior_predictive,
        observed_data=idata.observed_data,
    )
    idata.to_netcdf(str(OUTPUT_DIR / "prior_predictive.nc"))

    # Save summary
    summary = {
        "parameters": {
            "alpha": {"mean": float(np.mean(alpha)), "sd": float(np.std(alpha))},
            "beta_weight": {"mean": float(np.mean(beta_weight)), "sd": float(np.std(beta_weight))},
            "sigma": {"mean": float(np.mean(sigma)), "sd": float(np.std(sigma))},
        },
        "mpg_percentiles": {
            f"p{p}": float(np.percentile(mpg_rep, p))
            for p in [0.5, 2.5, 10, 25, 50, 75, 90, 97.5, 99.5]
        },
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create plots
    create_diagnostic_plots(
        alpha, beta_weight, sigma, mpg_rep, data["mpg"], OUTPUT_DIR
    )

    # Assessment
    verdict, issues = assess_prior_predictive(mpg_rep, data["mpg"])
    print(f"\nVERDICT: {verdict}")
    for issue in issues:
        print(f"  {issue}")

    with open(OUTPUT_DIR / "verdict.txt", "w") as f:
        f.write(f"{verdict}\n")

    return verdict


if __name__ == "__main__":
    main()
