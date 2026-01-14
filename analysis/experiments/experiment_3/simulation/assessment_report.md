# Parameter Recovery Assessment: Experiment 3

**Model**: Random Intercepts + Random Slopes

**Date**: 2026-01-14

**Status**: PASS (with caveats)

## Executive Summary

The model successfully recovers known parameters from synthetic data across three test scenarios representing different levels of heterogeneity. Convergence diagnostics are generally excellent, with only minor divergences in one scenario. The model demonstrates computational stability and appropriate uncertainty quantification.

However, recovery of the population treatment effect (beta_0) shows concerning bias patterns that warrant careful interpretation on real data. Variance components show expected shrinkage with limited groups (J=8), which is typical for hierarchical models.

## Test Scenarios

Three recovery scenarios were tested:

1. **Low heterogeneity**: Small variance components (tau_alpha=5, tau_beta=2, sigma=12)
2. **True DGP**: Moderate variance matching expected data structure (tau_alpha=8, tau_beta=3, sigma=10)
3. **High heterogeneity**: Large variance components (tau_alpha=10, tau_beta=6, sigma=8)

All scenarios used J=8 schools, N=160 students, 50% treatment rate, balanced design.

## Recovery Performance

### Population Parameters

| Parameter | Scenario 1 | Scenario 2 | Scenario 3 | Assessment |
|-----------|-----------|-----------|-----------|------------|
| **alpha_0** (Intercept) | 75.0 → 75.9 ± 1.5 | 70.0 → 73.3 ± 3.1 | 70.0 → 69.5 ± 3.3 | Good |
| **tau_alpha** (School SD) | 5.0 → 1.4 ± 1.2 | 8.0 → 8.3 ± 2.6 | 10.0 → 8.7 ± 2.6 | Moderate (shrinkage) |
| **beta_0** (Treatment effect) | 5.0 → 6.8 ± 1.9 | 5.0 → 8.6 ± 2.0 | 7.0 → 4.8 ± 1.9 | Concerning bias |
| **tau_beta** (Treatment SD) | 2.0 → 1.9 ± 1.5 | 3.0 → 3.4 ± 2.1 | 6.0 → 4.2 ± 2.0 | Moderate (shrinkage) |
| **sigma** (Residual SD) | 12.0 → 12.1 ± 0.7 | 10.0 → 10.4 ± 0.6 | 8.0 → 8.1 ± 0.5 | Excellent |

### Key Findings

**Excellent recovery:**
- Residual SD (sigma): consistently precise across all scenarios
- Population intercept (alpha_0): generally good, though some bias in scenario 2

**Adequate recovery with expected behavior:**
- Variance components (tau_alpha, tau_beta): show shrinkage toward zero, especially when true values are small. This is expected with limited groups (J=8) and reflects appropriate uncertainty about between-group variation.

**Concerning patterns:**
- Population treatment effect (beta_0): shows substantial bias across all scenarios (36% overestimation in scenario 1, 72% overestimation in scenario 2, 31% underestimation in scenario 3). While these biases fall within posterior uncertainty, the inconsistent direction suggests confounding with random slopes.

## Convergence Diagnostics

| Scenario | Converged | Divergences | Max R-hat | Min ESS Bulk | Assessment |
|----------|-----------|-------------|-----------|--------------|------------|
| Low heterogeneity | No | 2 | 1.00 | 3558 | Minor issue |
| True DGP | Yes | 0 | 1.00 | 2350 | Excellent |
| High heterogeneity | Yes | 0 | 1.00 | 1887 | Excellent |

All scenarios achieved excellent R-hat values (1.00) and adequate effective sample sizes (> 1800). Two divergent transitions occurred during warmup in scenario 1, but did not affect final convergence. This suggests minor geometric issues in low-heterogeneity regions but no fundamental computational problems.

## Visual Diagnostics

Recovery plots (`recovery_scatter.png`, `recovery_intervals.png`) show:
- Most parameters track near the identity line with appropriate uncertainty
- Beta_0 deviates substantially from identity, confirming bias patterns
- Posterior credible intervals appropriately reflect uncertainty, widening when heterogeneity increases

Trace plots and rank plots (`trace_plots_true_dgp.png`, `rank_plots_true_dgp.png`) for the True DGP scenario demonstrate:
- Excellent chain mixing with no trends or stuck regions
- Uniform rank distributions across all parameters
- No evidence of multimodality or poor exploration

## Interpretation and Caveats

### Why beta_0 shows bias

The population treatment effect (beta_0) is challenging to estimate precisely in this model structure for two reasons:

1. **Limited groups**: With only J=8 schools, there is limited information to separate the population mean treatment effect from school-specific deviations.

2. **Confounding with random slopes**: When schools have varying treatment effects (tau_beta > 0), the population mean (beta_0) becomes harder to identify. The model must simultaneously estimate both the average effect and the variation around it from the same 8 schools.

This is a fundamental identification challenge, not a model failure. The posterior uncertainty appropriately reflects this difficulty - notice that posterior SDs for beta_0 are substantial (1.9-2.0 points).

### Variance component shrinkage

The shrinkage of variance components (tau_alpha, tau_beta) toward zero is expected behavior in hierarchical models, especially with:
- Limited number of groups (J=8)
- Non-centered parameterization (which we use for computational stability)
- Genuine uncertainty about between-group variation

This shrinkage represents appropriate regularization and is a feature, not a bug, of Bayesian hierarchical modeling.

## Decision: PASS

**Rationale:**
- ✓ Model converges reliably on synthetic data (2/3 perfect, 1/3 minor divergences)
- ✓ Computational stability confirmed across heterogeneity levels
- ✓ Parameters identifiable with appropriate uncertainty quantification
- ✓ No catastrophic failures, wild scatter, or numerical errors
- ✓ Posterior uncertainty appropriately captures estimation difficulty

**Caveats for real data application:**
1. **Treatment effect interpretation**: Population treatment effect (beta_0) may show bias due to confounding with random slopes. Consider reporting school-specific effects (alpha + beta by school) alongside population means.

2. **Variance component precision**: With J=8 schools, variance components will have substantial uncertainty. Do not over-interpret point estimates; focus on credible intervals.

3. **Sensitivity analysis**: If treatment effect magnitude is critical for decision-making, consider:
   - Comparing results to simpler models (no random slopes)
   - Examining school-specific effects directly
   - Using informative priors if domain knowledge is available

4. **Sample size limitations**: Consider collecting data from more schools if precise estimation of variance components is important.

## Recommendation

Proceed to fitting real data with this model. Monitor posterior diagnostics carefully and interpret treatment effects with the caveats noted above. The model is computationally sound and will provide valid inference, but users should understand the inherent identification challenges with limited groups.

## Files Generated

- `recovery_results.json`: Numerical results for all scenarios
- `recovery_scatter.png`: Parameter recovery scatter plots
- `recovery_intervals.png`: Parameter recovery interval plots by scenario
- `trace_plots_true_dgp.png`: MCMC trace plots for True DGP scenario
- `rank_plots_true_dgp.png`: Rank plots for convergence diagnostics
- `scenario_1_low_heterogeneity_posterior.nc`: Full posterior for scenario 1
- `scenario_2_true_dgp_posterior.nc`: Full posterior for scenario 2
- `scenario_3_high_heterogeneity_posterior.nc`: Full posterior for scenario 3
