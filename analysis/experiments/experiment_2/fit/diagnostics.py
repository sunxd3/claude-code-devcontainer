#!/usr/bin/env python3
"""
Generate visual diagnostics for Experiment 2 fit

Creates:
- Trace plots for key parameters
- Rank plots for convergence assessment
- Pair plots for correlations
- Energy plot for HMC diagnostics
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2")
POSTERIOR_PATH = BASE_DIR / "fit" / "posterior.nc"
OUTPUT_DIR = BASE_DIR / "fit"

def main():
    print("=" * 80)
    print("EXPERIMENT 2: Visual Diagnostics")
    print("=" * 80)

    # Load posterior
    print("\n[1/5] Loading posterior...")
    idata = az.from_netcdf(POSTERIOR_PATH)
    print(f"  Loaded from: {POSTERIOR_PATH}")
    print(f"  Posterior shape: {idata.posterior.dims}")

    # Trace plots
    print("\n[2/5] Generating trace plots...")
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    az.plot_trace(
        idata,
        var_names=["alpha_0", "tau_alpha", "beta", "sigma"],
        compact=False,
        axes=axes
    )
    fig.suptitle("Experiment 2: Trace Plots (Key Parameters)", fontsize=14, y=0.995)
    plt.tight_layout()
    trace_path = OUTPUT_DIR / "trace_plots.png"
    plt.savefig(trace_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {trace_path}")

    # Rank plots
    print("\n[3/5] Generating rank plots...")
    fig = plt.figure(figsize=(12, 10))
    az.plot_rank(
        idata,
        var_names=["alpha_0", "tau_alpha", "beta", "sigma"],
        kind="bars"
    )
    fig.suptitle("Experiment 2: Rank Plots (Key Parameters)", fontsize=14, y=0.995)
    plt.tight_layout()
    rank_path = OUTPUT_DIR / "rank_plots.png"
    plt.savefig(rank_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {rank_path}")

    # Pair plot for correlations
    print("\n[4/5] Generating pair plot...")
    fig = az.plot_pair(
        idata,
        var_names=["alpha_0", "tau_alpha", "beta", "sigma"],
        kind="hexbin",
        divergences=True,
        figsize=(10, 10)
    )
    plt.suptitle("Experiment 2: Parameter Correlations", fontsize=14, y=0.995)
    plt.tight_layout()
    pair_path = OUTPUT_DIR / "pair_plot.png"
    plt.savefig(pair_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pair_path}")

    # Energy plot
    print("\n[5/5] Generating energy plot...")
    fig = az.plot_energy(idata, figsize=(10, 6))
    plt.suptitle("Experiment 2: Energy Plot (HMC Diagnostics)", fontsize=14)
    plt.tight_layout()
    energy_path = OUTPUT_DIR / "energy_plot.png"
    plt.savefig(energy_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {energy_path}")

    print("\n" + "=" * 80)
    print("Visual diagnostics complete!")
    print("=" * 80)
    print("\nFiles created:")
    print(f"  - {trace_path}")
    print(f"  - {rank_path}")
    print(f"  - {pair_path}")
    print(f"  - {energy_path}")

if __name__ == "__main__":
    main()
