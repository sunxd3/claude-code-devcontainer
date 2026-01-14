"""Generate recovery diagnostic plots."""

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10})

# Load results
output_dir = Path(
    "/home/user/claude-code-devcontainer/analysis/experiments/experiment_3/simulation"
)
results_path = output_dir / "recovery_results.json"

with open(results_path) as f:
    results = json.load(f)

# Extract data for plotting
scenarios = []
params = ["alpha_0", "tau_alpha", "beta_0", "tau_beta", "sigma"]
true_values = {param: [] for param in params}
posterior_means = {param: [] for param in params}
posterior_sds = {param: [] for param in params}

for result in results:
    if result["fit_success"]:
        scenarios.append(result["description"])
        for param in params:
            true_values[param].append(result["true_params"][param])
            posterior_means[param].append(result["posterior_means"][param])
            posterior_sds[param].append(result["posterior_sds"][param])

# Create figure with recovery plots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

param_labels = {
    "alpha_0": r"$\alpha_0$ (Population intercept)",
    "tau_alpha": r"$\tau_\alpha$ (School intercept SD)",
    "beta_0": r"$\beta_0$ (Population treatment effect)",
    "tau_beta": r"$\tau_\beta$ (School treatment effect SD)",
    "sigma": r"$\sigma$ (Residual SD)",
}

for idx, param in enumerate(params):
    ax = axes[idx]

    true_vals = np.array(true_values[param])
    post_means = np.array(posterior_means[param])
    post_sds = np.array(posterior_sds[param])

    # Scatter plot with error bars
    ax.errorbar(
        true_vals,
        post_means,
        yerr=1.96 * post_sds,
        fmt="o",
        markersize=8,
        capsize=5,
        alpha=0.7,
        label="Posterior mean ± 95% CI",
    )

    # Identity line
    all_vals = np.concatenate([true_vals, post_means])
    min_val = all_vals.min()
    max_val = all_vals.max()
    margin = 0.1 * (max_val - min_val)
    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        "k--",
        alpha=0.5,
        linewidth=2,
        label="Identity",
    )

    # Labels and formatting
    ax.set_xlabel(f"True {param_labels[param]}")
    ax.set_ylabel("Posterior mean")
    ax.set_title(f"{param_labels[param]}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

# Remove extra subplot
axes[-1].axis("off")

plt.tight_layout()
plt.savefig(output_dir / "recovery_scatter.png", dpi=300, bbox_inches="tight")
print(f"Saved: {output_dir / 'recovery_scatter.png'}")
plt.close()

# Create interval plot for each scenario
fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 4 * len(scenarios)))
if len(scenarios) == 1:
    axes = [axes]

for scenario_idx, scenario in enumerate(scenarios):
    ax = axes[scenario_idx]

    x_pos = np.arange(len(params))
    for param_idx, param in enumerate(params):
        true_val = true_values[param][scenario_idx]
        post_mean = posterior_means[param][scenario_idx]
        post_sd = posterior_sds[param][scenario_idx]

        # Plot posterior interval
        ax.errorbar(
            x_pos[param_idx],
            post_mean,
            yerr=1.96 * post_sd,
            fmt="o",
            markersize=10,
            capsize=8,
            color="steelblue",
            alpha=0.7,
            linewidth=2,
        )

        # Plot true value
        ax.scatter(
            x_pos[param_idx],
            true_val,
            marker="x",
            s=200,
            color="red",
            linewidth=3,
            label="True value" if param_idx == 0 else "",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([param_labels[p] for p in params], rotation=45, ha="right")
    ax.set_ylabel("Parameter value")
    ax.set_title(f"Recovery: {scenario}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_dir / "recovery_intervals.png", dpi=300, bbox_inches="tight")
print(f"Saved: {output_dir / 'recovery_intervals.png'}")
plt.close()

# Create trace plots for one scenario (True DGP)
print("\nGenerating trace plots for True DGP scenario...")
idata = az.from_netcdf(output_dir / "scenario_2_true_dgp_posterior.nc")

az.plot_trace(
    idata,
    var_names=params,
    divergences="top",
    figsize=(12, 2.5 * len(params)),
)
plt.suptitle("Trace plots: True DGP scenario", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(output_dir / "trace_plots_true_dgp.png", dpi=300, bbox_inches="tight")
print(f"Saved: {output_dir / 'trace_plots_true_dgp.png'}")
plt.close()

# Create rank plots for convergence diagnostics
print("Generating rank plots for True DGP scenario...")
az.plot_rank(idata, var_names=params, figsize=(12, 8))
plt.suptitle("Rank plots: True DGP scenario", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(output_dir / "rank_plots_true_dgp.png", dpi=300, bbox_inches="tight")
print(f"Saved: {output_dir / 'rank_plots_true_dgp.png'}")
plt.close()

print("\nAll plots generated successfully.")
