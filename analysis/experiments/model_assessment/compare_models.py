#!/usr/bin/env python3
"""
Compare validated models using LOO-CV and posterior predictive checks.

Loads InferenceData from all validated experiments, runs az.compare(),
checks LOO reliability, and generates comparison visualizations.
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
from shared_utils import compute_loo, load_posterior, write_json

# Experiment directories
exp_dirs = {
    "Complete Pooling": Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_1"),
    "Random Intercepts": Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_2"),
    "Random Slopes": Path("/home/user/claude-code-devcontainer/analysis/experiments/experiment_3"),
}

output_dir = Path("/home/user/claude-code-devcontainer/analysis/experiments/model_assessment")

# Load all InferenceData objects
print("Loading InferenceData from validated experiments...")
idatas = {}
for name, exp_dir in exp_dirs.items():
    fit_dir = exp_dir / "fit"
    if (fit_dir / "posterior.nc").exists():
        idatas[name] = load_posterior(fit_dir)
        print(f"  Loaded {name}: {fit_dir / 'posterior.nc'}")
    else:
        print(f"  WARNING: Missing {fit_dir / 'posterior.nc'}")

if len(idatas) < 2:
    raise ValueError("Need at least 2 models to compare")

print(f"\nComparing {len(idatas)} models...")

# Run az.compare()
compare_df = az.compare(idatas, ic="loo", method="stacking")
print("\n" + "="*80)
print("MODEL COMPARISON (LOO-CV)")
print("="*80)
print(compare_df.to_string())
print()

# Save comparison table
compare_df.to_csv(output_dir / "loo_comparison.csv")
print(f"Saved comparison table to {output_dir / 'loo_comparison.csv'}")

# Check for clear winner vs. close race
top_model = compare_df.index[0]
if len(compare_df) > 1:
    elpd_diff = compare_df.loc[compare_df.index[1], "elpd_diff"]
    se_diff = compare_df.loc[compare_df.index[1], "dse"]

    print(f"\nTop model: {top_model}")
    print(f"ELPD difference to 2nd place: {elpd_diff:.2f} ± {se_diff:.2f}")

    if abs(elpd_diff) > 4 * se_diff:
        print("  → CLEAR WINNER (difference > 4×SE)")
    elif abs(elpd_diff) < 2 * se_diff:
        print("  → TOO CLOSE TO CALL (difference < 2×SE)")
    else:
        print("  → MODERATE ADVANTAGE (2×SE < difference < 4×SE)")

# Check LOO reliability (Pareto k diagnostics)
print("\n" + "="*80)
print("LOO RELIABILITY (Pareto k diagnostics)")
print("="*80)

loo_results = {}
for name, idata in idatas.items():
    loo_result = compute_loo(idata)
    loo_raw = az.loo(idata, pointwise=True)
    pareto_k = loo_raw.pareto_k.values

    n_total = loo_result.k_good + loo_result.k_ok + loo_result.k_bad + loo_result.k_very_bad
    pct_high = 100 * (loo_result.k_bad + loo_result.k_very_bad) / n_total

    loo_results[name] = {
        "elpd_loo": loo_result.elpd_loo,
        "se": loo_result.se,
        "p_loo": loo_result.p_loo,
        "k_good": loo_result.k_good,
        "k_ok": loo_result.k_ok,
        "k_bad": loo_result.k_bad,
        "k_very_bad": loo_result.k_very_bad,
        "pct_high_k": float(pct_high),
        "max_k": float(pareto_k.max()),
    }

    print(f"\n{name}:")
    print(f"  ELPD_LOO: {loo_result.elpd_loo:.1f} ± {loo_result.se:.1f}")
    print(f"  p_loo (effective parameters): {loo_result.p_loo:.1f}")
    print(f"  Pareto k > 0.7: {loo_result.k_bad + loo_result.k_very_bad}/{n_total} ({pct_high:.1f}%)")
    print(f"  Pareto k > 0.5: {loo_result.k_ok}/{n_total}")
    print(f"  Max Pareto k: {pareto_k.max():.3f}")

    if pct_high > 10:
        print(f"  ⚠ WARNING: LOO may be unreliable (>{10}% high k values)")
    elif loo_result.k_bad + loo_result.k_very_bad > 0:
        print("  ⚠ CAUTION: Some influential observations (k > 0.7)")
    else:
        print("  ✓ LOO reliable (all k < 0.7)")

# Save LOO diagnostics
write_json(output_dir / "loo_diagnostics.json", loo_results)
print(f"\nSaved LOO diagnostics to {output_dir / 'loo_diagnostics.json'}")

# Generate comparison plots
print("\n" + "="*80)
print("GENERATING COMPARISON PLOTS")
print("="*80)

# Plot 1: az.plot_compare()
fig, ax = plt.subplots(figsize=(10, 4))
az.plot_compare(compare_df, insample_dev=False, ax=ax)
ax.set_title("Model Comparison (LOO-CV)", fontsize=14, fontweight="bold")
plt.tight_layout()
plot_path = output_dir / "loo_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved comparison plot to {plot_path}")
plt.close()

# Plot 2: Pareto k diagnostics for each model
n_models = len(idatas)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
if n_models == 1:
    axes = [axes]

for ax, (name, idata) in zip(axes, idatas.items()):
    az.plot_khat(idata, ax=ax, show_bins=True)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Data point", fontsize=10)
    ax.set_ylabel("Pareto k", fontsize=10)

plt.tight_layout()
plot_path = output_dir / "pareto_k_diagnostics.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved Pareto k diagnostics to {plot_path}")
plt.close()

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
print(f"Results saved to: {output_dir}")
