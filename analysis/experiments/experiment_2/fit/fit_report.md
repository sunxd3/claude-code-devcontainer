# A2-Year Model Fit Report

## Model Overview

Model A2-Year extends the baseline log-log weight model by adding a year effect to capture technological drift in fuel efficiency over the 1970-1982 period.

**Model specification:**
```
log(mpg) ~ Normal(alpha + beta_weight * log_weight_c + beta_year * year_c, sigma)
```

where:
- `log_weight_c = log(weight) - 7.96` (centered log weight)
- `year_c = model_year - 76` (centered at midpoint)

## Data

- Original dataset: 398 observations
- After removing 6 rows with missing horsepower: **392 observations**
- Year range: 70-82 (centered: -6 to +6)

## Sampling Configuration

| Parameter | Value |
|-----------|-------|
| Chains | 4 |
| Warmup | 1000 |
| Sampling | 1000 |
| adapt_delta | 0.9 |
| Total draws | 4000 |

## Convergence Diagnostics

**Status: PASSED**

| Metric | Value | Threshold |
|--------|-------|-----------|
| Max R-hat | 1.000 | < 1.01 |
| Min ESS bulk | 2471 | > 400 |
| Min ESS tail | 2148 | > 400 |
| Divergences | 0 | 0 |

All parameters show excellent mixing with ESS well above the minimum threshold. No divergent transitions occurred, indicating the sampler explored the posterior efficiently.

## Parameter Estimates

| Parameter | Mean | SD | 95% HDI |
|-----------|------|-----|---------|
| alpha | 3.098 | 0.006 | [3.087, 3.110] |
| beta_weight | -0.935 | 0.021 | [-0.979, -0.893] |
| beta_year | 0.033 | 0.002 | [0.030, 0.036] |
| sigma | 0.118 | 0.004 | [0.110, 0.127] |

### Interpretation

- **alpha = 3.098**: Expected log(mpg) at centered weight and year is exp(3.098) = 22.2 mpg
- **beta_weight = -0.935**: A 1% increase in weight is associated with approximately 0.94% decrease in mpg (near unit elasticity)
- **beta_year = 0.033**: Each year corresponds to about 3.3% improvement in mpg, holding weight constant. This represents technological progress in engine efficiency.
- **sigma = 0.118**: Residual standard deviation on log scale (about 12% CV)

The year effect is strongly significant (95% HDI excludes zero), indicating real technological improvement in fuel efficiency over this period.

## LOO-CV Results

| Metric | Value |
|--------|-------|
| ELPD LOO | 279.7 |
| SE | 17.3 |
| p_loo | 4.3 |

**Pareto k diagnostics:**
- Good (k < 0.5): 392 (100%)
- OK (0.5 <= k < 0.7): 0
- Bad (0.7 <= k < 1.0): 0
- Very bad (k >= 1.0): 0

All Pareto k values are good, indicating reliable LOO estimates with no influential observations.

## Diagnostic Plots

All plots saved to `figures/`:

- `trace_plots.png` - Chain traces showing mixing
- `rank_plots.png` - Rank histograms (uniform = good)
- `posterior_distributions.png` - Marginal posteriors with HDI
- `pair_plot.png` - Parameter correlations
- `ess_evolution.png` - ESS growth over draws
- `energy_plot.png` - HMC energy diagnostics
- `autocorrelation.png` - Autocorrelation decay
- `loo_pit.png` - LOO-PIT calibration check

## Output Files

| File | Description |
|------|-------------|
| `posterior.nc` | ArviZ InferenceData (NetCDF) |
| `convergence.json` | Convergence diagnostics |
| `loo.json` | LOO-CV results |
| `figures/` | Diagnostic plots |

## Conclusion

The A2-Year model fit successfully with excellent convergence. The year effect (beta_year = 0.033, 95% HDI [0.030, 0.036]) is clearly non-zero, suggesting that technological improvements contributed meaningfully to fuel efficiency gains beyond what weight alone explains. This model should be compared to the baseline A1-Weight model using LOO-CV to quantify the predictive benefit of including year.
