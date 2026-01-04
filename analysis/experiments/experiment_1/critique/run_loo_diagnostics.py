#!/usr/bin/env python3
"""Run additional LOO diagnostics for model critique."""

import json
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# Load posterior
idata = az.from_netcdf("/workspace/analysis/experiments/experiment_1/fit/posterior.nc")

# Compute LOO
loo = az.loo(idata, pointwise=True)

# Print summary
print("=" * 60)
print("LOO-CV Summary")
print("=" * 60)
print(f"ELPD LOO: {loo.elpd_loo:.1f} +/- {loo.se:.1f}")
print(f"p_loo (effective parameters): {loo.p_loo:.2f}")
print()

# Pareto k summary
k_vals = loo.pareto_k.values
print("Pareto k summary:")
print(f"  Good (k < 0.5): {np.sum(k_vals < 0.5)}")
print(f"  OK (0.5 <= k < 0.7): {np.sum((k_vals >= 0.5) & (k_vals < 0.7))}")
print(f"  Bad (0.7 <= k < 1.0): {np.sum((k_vals >= 0.7) & (k_vals < 1.0))}")
print(f"  Very bad (k >= 1.0): {np.sum(k_vals >= 1.0)}")
print()
print(f"Max Pareto k: {np.max(k_vals):.3f}")
print(f"Mean Pareto k: {np.mean(k_vals):.3f}")

# Plot k-hat
fig, ax = plt.subplots(figsize=(10, 4))
az.plot_khat(loo, ax=ax)
ax.set_title("Pareto k Diagnostic Plot")
ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='k=0.5 (OK threshold)')
ax.axhline(0.7, color='red', linestyle='--', alpha=0.7, label='k=0.7 (problematic threshold)')
ax.legend()
plt.tight_layout()
plt.savefig("/workspace/analysis/experiments/experiment_1/critique/loo_khat.png", dpi=150)
plt.close()
print("\nSaved: loo_khat.png")

# Skip LOO-PIT here - already computed in PPC stage (loo_pit.png)

# Save extended LOO results for model comparison
loo_results = {
    "model": "A1-Baseline",
    "elpd_loo": float(loo.elpd_loo),
    "se": float(loo.se),
    "p_loo": float(loo.p_loo),
    "pareto_k_max": float(np.max(k_vals)),
    "pareto_k_mean": float(np.mean(k_vals)),
    "n_k_good": int(np.sum(k_vals < 0.5)),
    "n_k_ok": int(np.sum((k_vals >= 0.5) & (k_vals < 0.7))),
    "n_k_bad": int(np.sum((k_vals >= 0.7) & (k_vals < 1.0))),
    "n_k_very_bad": int(np.sum(k_vals >= 1.0)),
    "n_obs": len(k_vals)
}

with open("/workspace/analysis/experiments/experiment_1/critique/loo_extended.json", "w") as f:
    json.dump(loo_results, f, indent=2)
print("Saved: loo_extended.json")

print("\n" + "=" * 60)
print("LOO Diagnostics Complete")
print("=" * 60)
