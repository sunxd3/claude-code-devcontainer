"""Generate diagnostic plots for A3-Robust model fit."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from shared_utils import load_posterior

# Paths
OUTPUT_DIR = "/workspace/analysis/experiments/experiment_3/fit"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"

# Load posterior
print("Loading posterior...")
idata = load_posterior(OUTPUT_DIR)

# Set plotting style
plt.style.use("default")
az.style.use("arviz-darkgrid")

VAR_NAMES = ["alpha", "beta_weight", "beta_year", "sigma", "nu"]

# 1. Trace plots
print("Creating trace plots...")
fig = az.plot_trace(idata, var_names=VAR_NAMES, figsize=(12, 10))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/trace_plots.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Rank plots
print("Creating rank plots...")
fig = az.plot_rank(idata, var_names=VAR_NAMES, figsize=(12, 10))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/rank_plots.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Posterior distributions
print("Creating posterior distributions...")
fig = az.plot_posterior(idata, var_names=VAR_NAMES, figsize=(12, 8), hdi_prob=0.95)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/posterior_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Pair plot (for key parameters)
print("Creating pair plot...")
fig = az.plot_pair(
    idata,
    var_names=["beta_weight", "beta_year", "sigma", "nu"],
    kind="kde",
    marginals=True,
    figsize=(10, 10),
)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/pair_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. ESS evolution
print("Creating ESS evolution plot...")
fig = az.plot_ess(idata, var_names=VAR_NAMES, kind="evolution", figsize=(12, 8))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/ess_evolution.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Energy plot
print("Creating energy plot...")
fig, ax = plt.subplots(figsize=(8, 5))
az.plot_energy(idata, ax=ax)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/energy_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# 7. Autocorrelation
print("Creating autocorrelation plot...")
fig = az.plot_autocorr(idata, var_names=VAR_NAMES, figsize=(12, 10))
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/autocorrelation.png", dpi=150, bbox_inches="tight")
plt.close()

# 8. LOO-PIT plot
print("Creating LOO-PIT plot...")
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_loo_pit(idata, y="y", y_hat="y_rep", ax=ax)
ax.set_title("LOO-PIT: Probability Integral Transform")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/loo_pit.png", dpi=150, bbox_inches="tight")
plt.close()

# 9. Nu posterior with interpretation regions
print("Creating nu interpretation plot...")
fig, ax = plt.subplots(figsize=(10, 6))

nu_samples = idata.posterior["nu"].values.flatten()
nu_median = np.median(nu_samples)
nu_q025 = np.percentile(nu_samples, 2.5)
nu_q975 = np.percentile(nu_samples, 97.5)

# Plot posterior density
az.plot_posterior(idata, var_names=["nu"], ax=ax, hdi_prob=0.95)

# Add interpretation regions
ax.axvline(x=15, color="orange", linestyle="--", linewidth=2, label="nu=15 threshold")
ax.axvline(x=30, color="green", linestyle="--", linewidth=2, label="nu=30 threshold")

# Add annotations
ax.annotate("Heavy tails\n(outliers present)", xy=(5, ax.get_ylim()[1]*0.8),
            fontsize=10, ha="center", color="red")
ax.annotate("Normal-like", xy=(40, ax.get_ylim()[1]*0.8),
            fontsize=10, ha="center", color="green")

ax.set_title(f"Posterior of nu (degrees of freedom)\nMedian = {nu_median:.1f}, 95% CI: [{nu_q025:.1f}, {nu_q975:.1f}]")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/nu_interpretation.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll diagnostic plots saved to {FIGURES_DIR}/")
print("  - trace_plots.png")
print("  - rank_plots.png")
print("  - posterior_distributions.png")
print("  - pair_plot.png")
print("  - ess_evolution.png")
print("  - energy_plot.png")
print("  - autocorrelation.png")
print("  - loo_pit.png")
print("  - nu_interpretation.png")
