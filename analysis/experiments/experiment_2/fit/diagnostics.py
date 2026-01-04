"""Create diagnostic plots for A2-Year model fit."""

import arviz as az
import matplotlib.pyplot as plt

from shared_utils import load_posterior

# Paths
OUTPUT_DIR = "/workspace/analysis/experiments/experiment_2/fit"
FIGURES_DIR = "/workspace/analysis/experiments/experiment_2/fit/figures"

# Load posterior
print("Loading posterior...")
idata = load_posterior(OUTPUT_DIR)

# Create figures directory
import os
os.makedirs(FIGURES_DIR, exist_ok=True)

# Parameter names for diagnostics
param_names = ["alpha", "beta_weight", "beta_year", "sigma"]

# 1. Trace plots
print("Creating trace plots...")
fig = az.plot_trace(idata, var_names=param_names, compact=False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/trace_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/trace_plots.png")

# 2. Rank plots (uniform histograms indicate good mixing)
print("Creating rank plots...")
fig = az.plot_rank(idata, var_names=param_names)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/rank_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/rank_plots.png")

# 3. Posterior distributions
print("Creating posterior distributions...")
fig = az.plot_posterior(
    idata,
    var_names=param_names,
    hdi_prob=0.95,
)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/posterior_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/posterior_distributions.png")

# 4. Pair plot (check correlations between parameters)
print("Creating pair plot...")
fig = az.plot_pair(
    idata,
    var_names=param_names,
    divergences=True,
    kind="kde",
    marginals=True,
)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/pair_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/pair_plot.png")

# 5. ESS evolution plot
print("Creating ESS evolution plot...")
fig = az.plot_ess(idata, var_names=param_names, kind="evolution")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/ess_evolution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/ess_evolution.png")

# 6. Energy plot (HMC diagnostic)
print("Creating energy plot...")
fig = az.plot_energy(idata)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/energy_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/energy_plot.png")

# 7. Autocorrelation plot
print("Creating autocorrelation plot...")
fig = az.plot_autocorr(idata, var_names=param_names, combined=True)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/autocorrelation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/autocorrelation.png")

# 8. LOO-PIT plot (posterior predictive check)
print("Creating LOO-PIT plot...")
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_loo_pit(idata, y="y", y_hat="y_rep", ax=ax)
ax.set_title("LOO-PIT (calibration check)")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/loo_pit.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURES_DIR}/loo_pit.png")

print("\nAll diagnostic plots created successfully.")
