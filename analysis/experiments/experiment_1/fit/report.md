# A1-Baseline Model Fit Report

## Summary

The A1-Baseline model (log-log physical model: log(mpg) ~ log(weight)) was successfully fit to the Auto-MPG dataset. All convergence diagnostics passed and the model shows excellent sampling behavior.

## Data Preparation

- **Original dataset**: 398 observations
- **After removing missing horsepower**: 392 observations (6 rows removed)
- **Centering**: log(weight) centered at mean = 7.959 (corresponding to weight ~ 2866 lbs)
- **Response range**: log(mpg) in [2.197, 3.842], corresponding to mpg in [9, 46.6]

## Model Specification

```
log(mpg) ~ Normal(alpha + beta_weight * log_weight_c, sigma)

Priors:
  alpha ~ Normal(3.1, 0.3)        # log-MPG at mean weight
  beta_weight ~ Normal(-1, 0.3)   # elasticity (physics suggests ~-1)
  sigma ~ Exponential(5)          # residual SD
```

## Sampling Configuration

- Chains: 4
- Warmup: 1000 iterations per chain
- Sampling: 1000 iterations per chain
- Total posterior draws: 4000
- adapt_delta: 0.9

## Convergence Diagnostics

**Status: PASSED**

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Max R-hat | 1.000 | < 1.01 | Pass |
| Min ESS bulk | 3483 | > 400 | Pass |
| Min ESS tail | 2650 | > 400 | Pass |
| Divergences | 0 | = 0 | Pass |

All parameters show excellent mixing with ESS values well above the minimum threshold. The rank plots (`rank_plots.png`) show uniform distributions across chains, confirming good mixing.

## Parameter Estimates

| Parameter | Mean | SD | 94% HDI | ESS bulk | R-hat |
|-----------|------|-----|---------|----------|-------|
| alpha | 3.098 | 0.008 | [3.083, 3.114] | 4169 | 1.00 |
| beta_weight | -1.057 | 0.031 | [-1.114, -0.999] | 3483 | 1.00 |
| sigma | 0.166 | 0.006 | [0.154, 0.177] | 3807 | 1.00 |

**Interpretation:**
- **alpha = 3.098**: At mean log-weight (7.96), expected log(mpg) is 3.098, corresponding to exp(3.098) = 22.2 mpg
- **beta_weight = -1.057**: A 1% increase in weight corresponds to a 1.06% decrease in mpg (elasticity close to -1 as predicted by physics)
- **sigma = 0.166**: Residual SD on log scale; approximately 16.6% multiplicative error

## LOO-CV Results

| Metric | Value |
|--------|-------|
| ELPD | 147.6 +/- 16.4 |
| p_loo | 3.3 |
| Pareto k good (< 0.5) | 392 (100%) |
| Pareto k ok (0.5-0.7) | 0 |
| Pareto k bad (0.7-1.0) | 0 |
| Pareto k very bad (> 1.0) | 0 |

All Pareto k values are in the "good" category, indicating reliable LOO estimates. The effective number of parameters (p_loo = 3.3) matches the model's 3 parameters, suggesting no overfitting concerns.

## Diagnostic Plots

| Plot | Description | File |
|------|-------------|------|
| Trace plots | Chain trajectories and marginal densities | `trace_plots.png` |
| Rank plots | Chain mixing verification | `rank_plots.png` |
| Pair plot | Parameter correlations with divergences | `pair_plot.png` |
| Posterior plots | Marginal posteriors with 94% HDI | `posterior_plots.png` |
| Energy plot | HMC energy diagnostic (BFMI) | `energy_plot.png` |
| ESS evolution | Effective sample size growth | `ess_evolution.png` |

## Output Files

| File | Description |
|------|-------------|
| `posterior.nc` | ArviZ InferenceData (NetCDF) with posterior, log_likelihood, posterior_predictive |
| `convergence.json` | Convergence diagnostic summary |
| `loo.json` | LOO-CV results |
| `run_fit.py` | Fitting script |
| `plots.py` | Diagnostic plotting script |

## Conclusion

The A1-Baseline model converged successfully with no sampling issues. The weight elasticity estimate of -1.057 is remarkably close to the theoretical prediction of -1 from physics (fuel consumption proportional to weight). This simple model provides a strong baseline for comparison with more complex models that include additional predictors.
