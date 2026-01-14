"""Generate convergence diagnostic plots for recovery tests."""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt

# Output directory
output_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2/simulation")

# Scenarios
scenarios = ["low_clustering", "medium_clustering", "high_clustering"]

for scenario in scenarios:
    print(f"Generating diagnostics for {scenario}...")

    # Load posterior
    posterior_file = output_dir / f"{scenario}_posterior.nc"
    idata = az.from_netcdf(posterior_file)

    # Trace plots
    fig = az.plot_trace(
        idata,
        var_names=["alpha_0", "tau_alpha", "beta", "sigma"],
        compact=True,
        kind="trace"
    )
    plt.suptitle(f"Trace Plots: {scenario}", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"{scenario}_trace.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Rank plots
    fig = az.plot_rank(
        idata,
        var_names=["alpha_0", "tau_alpha", "beta", "sigma"]
    )
    plt.suptitle(f"Rank Plots: {scenario}", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"{scenario}_rank.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved trace and rank plots for {scenario}")

print("\nDiagnostic plots complete.")
