"""Diagnostic plots for A1-Baseline model fit."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from shared_utils import load_posterior

OUTPUT_DIR = "/workspace/analysis/experiments/experiment_1/fit"


def create_trace_plots(idata, output_dir):
    """Create trace plots for model parameters."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    az.plot_trace(
        idata,
        var_names=["alpha", "beta_weight", "sigma"],
        axes=axes,
    )

    plt.suptitle("A1-Baseline: Trace Plots", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trace_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved trace_plots.png")


def create_rank_plots(idata, output_dir):
    """Create rank plots for chain mixing diagnostics."""
    az.plot_rank(
        idata,
        var_names=["alpha", "beta_weight", "sigma"],
    )

    plt.suptitle("A1-Baseline: Rank Plots (Chain Mixing)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rank_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved rank_plots.png")


def create_pair_plot(idata, output_dir):
    """Create pair plot showing parameter correlations and divergences."""
    ax = az.plot_pair(
        idata,
        var_names=["alpha", "beta_weight", "sigma"],
        divergences=True,
        kind="kde",
        marginals=True,
        figsize=(10, 10),
    )

    plt.suptitle("A1-Baseline: Parameter Correlations", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pair_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pair_plot.png")


def create_posterior_plots(idata, output_dir):
    """Create posterior distribution plots with HDI."""
    az.plot_posterior(
        idata,
        var_names=["alpha", "beta_weight", "sigma"],
        hdi_prob=0.94,
    )

    plt.suptitle("A1-Baseline: Posterior Distributions (94% HDI)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/posterior_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved posterior_plots.png")


def create_energy_plot(idata, output_dir):
    """Create energy plot for HMC diagnostics."""
    fig, ax = plt.subplots(figsize=(8, 4))

    az.plot_energy(idata, ax=ax)

    plt.title("A1-Baseline: Energy Distribution (BFMI Diagnostic)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved energy_plot.png")


def create_ess_evolution_plot(idata, output_dir):
    """Create ESS evolution plot."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    az.plot_ess(
        idata,
        var_names=["alpha", "beta_weight", "sigma"],
        kind="evolution",
        ax=axes,
    )

    plt.suptitle("A1-Baseline: ESS Evolution", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ess_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ess_evolution.png")


def main():
    """Generate all diagnostic plots."""
    print("Loading posterior...")
    idata = load_posterior(OUTPUT_DIR)

    print("\nGenerating diagnostic plots...")
    create_trace_plots(idata, OUTPUT_DIR)
    create_rank_plots(idata, OUTPUT_DIR)
    create_pair_plot(idata, OUTPUT_DIR)
    create_posterior_plots(idata, OUTPUT_DIR)
    create_energy_plot(idata, OUTPUT_DIR)
    create_ess_evolution_plot(idata, OUTPUT_DIR)

    print("\nAll plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
