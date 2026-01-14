"""Create visual diagnostics for model convergence."""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_1")
OUTPUT_DIR = BASE_DIR / "fit"
IDATA_PATH = OUTPUT_DIR / "posterior.nc"

def create_diagnostics():
    """Create comprehensive diagnostic plots."""
    print("Loading posterior samples...")
    idata = az.from_netcdf(IDATA_PATH)

    # Trace plots
    print("Creating trace plots...")
    az.plot_trace(
        idata,
        var_names=["alpha", "beta", "sigma"],
        compact=False,
        figsize=(12, 8)
    )
    plt.tight_layout()
    trace_path = OUTPUT_DIR / "trace_plots.png"
    plt.savefig(trace_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved trace plots to {trace_path}")

    # Rank plots
    print("Creating rank plots...")
    az.plot_rank(
        idata,
        var_names=["alpha", "beta", "sigma"],
        figsize=(12, 4)
    )
    plt.tight_layout()
    rank_path = OUTPUT_DIR / "rank_plots.png"
    plt.savefig(rank_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved rank plots to {rank_path}")

    # Pair plot
    print("Creating pair plot...")
    az.plot_pair(
        idata,
        var_names=["alpha", "beta", "sigma"],
        kind="kde",
        marginals=True,
        figsize=(10, 10)
    )
    pair_path = OUTPUT_DIR / "pair_plot.png"
    plt.savefig(pair_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pair plot to {pair_path}")

    # Posterior distributions
    print("Creating posterior distributions...")
    az.plot_posterior(
        idata,
        var_names=["alpha", "beta", "sigma"],
        hdi_prob=0.94,
        figsize=(12, 4)
    )
    plt.tight_layout()
    posterior_path = OUTPUT_DIR / "posterior_distributions.png"
    plt.savefig(posterior_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved posterior distributions to {posterior_path}")

    # ESS evolution
    print("Creating ESS evolution plot...")
    az.plot_ess(
        idata,
        var_names=["alpha", "beta", "sigma"],
        kind="evolution",
        figsize=(12, 4)
    )
    plt.tight_layout()
    ess_path = OUTPUT_DIR / "ess_evolution.png"
    plt.savefig(ess_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ESS evolution to {ess_path}")

    print("\nAll diagnostic plots created successfully!")

if __name__ == "__main__":
    create_diagnostics()
