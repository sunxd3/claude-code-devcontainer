# Complete Pooling Model - Fit Report

**Model**: Experiment 1 - Complete Pooling
**Data**: 160 observations, binary treatment assignment
**Date**: 2026-01-14

## Summary

The Complete Pooling model converged successfully with excellent diagnostics. All chains mixed well and explored the posterior efficiently. No convergence issues detected.

## Convergence Diagnostics

**Status**: PASSED

The model meets all convergence criteria:

- **R-hat**: 1.0000 (target < 1.01) - Perfect chain agreement
- **ESS bulk**: 2151 minimum (target > 400) - Excellent effective sample size
- **ESS tail**: 2235 minimum (target > 400) - Good tail exploration
- **Divergences**: 0 (target <= 0) - No pathological geometry issues

All parameters show R-hat = 1.00, indicating perfect convergence across all four chains.

## Parameter Estimates

The posterior distributions for model parameters:

| Parameter | Mean   | SD    | HDI 94%        | Interpretation                           |
|-----------|--------|-------|----------------|------------------------------------------|
| alpha     | 73.65  | 1.11  | [71.43, 75.60] | Baseline (control) mean outcome          |
| beta      | 6.84   | 1.51  | [4.05, 9.77]   | Treatment effect (positive)              |
| sigma     | 10.00  | 0.57  | [8.98, 11.10]  | Residual standard deviation              |

The treatment effect (beta) is clearly positive with a 94% HDI entirely above zero, indicating a beneficial treatment effect of approximately 6.8 units on average.

## Sampling Details

**Probe sampling** (4 chains x 150 iterations):
- Completed successfully in < 1 second
- Initial R-hat check showed values < 1.03
- No immediate issues detected

**Main sampling** (4 chains x 1000 iterations):
- Warmup: 1000 iterations per chain
- Sampling: 1000 iterations per chain
- Total draws: 4000 post-warmup samples
- Sampling time: ~1 second
- Adapt delta: 0.8 (default)

## Visual Diagnostics

All visual diagnostics confirm excellent convergence (see figures in this directory):

1. **trace_plots.png**: Chains show stationary "fat fuzzy caterpillar" behavior with no trends or drift. All four chains thoroughly overlap.

2. **rank_plots.png**: Rank histograms are uniform across chains, indicating good mixing and no chain-specific behavior.

3. **pair_plot.png**: Parameter correlations are visible but not extreme. No pathological posterior geometry detected.

4. **posterior_distributions.png**: Smooth unimodal posteriors for all parameters with clear central tendencies.

5. **ess_evolution.png**: ESS grows steadily throughout sampling, confirming efficient exploration.

## Model Structure

This model assumes:
- Single global intercept (alpha) and treatment effect (beta)
- Constant residual variance (sigma)
- No hierarchical structure across schools
- Complete pooling of all observations

The model ignores potential school-level variation, treating all observations as exchangeable conditional on treatment assignment.

## Saved Outputs

All results saved to `/home/user/claude-code-devcontainer/analysis/experiments/experiment_1/fit/`:

- `posterior.nc` - ArviZ InferenceData with full posterior (4000 draws)
- `convergence_diagnostics.txt` - Numerical diagnostics summary
- `parameter_summary.txt` - Parameter estimates table
- `trace_plots.png` - MCMC trace plots
- `rank_plots.png` - Rank histogram diagnostics
- `pair_plot.png` - Parameter correlation plot
- `posterior_distributions.png` - Marginal posterior densities
- `ess_evolution.png` - ESS evolution over iterations

The InferenceData includes:
- Posterior samples for all parameters
- Pointwise log-likelihood (`log_lik`) for LOO-CV
- Posterior predictive draws (`y_rep`) for predictive checks

## Notes

A non-fatal warning appeared during sampling about sigma = 0, which occurs during the warmup phase as the sampler explores extreme regions. This is expected behavior during adaptation and does not affect the final posterior samples.

## Next Steps

This model is ready for:
1. Posterior predictive checks to assess model fit
2. LOO-CV comparison with alternative models (hierarchical, varying effects)
3. Model critique to identify potential improvements
