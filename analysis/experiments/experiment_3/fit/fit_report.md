# A3-Robust Model Fit Report

**Model**: Log-log physical model with Student-t errors
**Data**: Auto-MPG dataset (N=392 after removing 6 rows with missing horsepower)
**Date**: 2026-01-04

## Model Specification

The A3-Robust model extends the A2-Year model by replacing Normal errors with Student-t errors to test robustness to outliers:

```
log(mpg) ~ Student-t(nu, mu, sigma)
mu = alpha + beta_weight * log_weight_c + beta_year * year_c
```

where:
- `log_weight_c = log(weight) - 7.96` (centered at ~2860 lbs)
- `year_c = model_year - 76` (centered at 1976)
- `nu` is the degrees of freedom parameter (estimated from data)

The key question: does `nu` favor heavy tails (nu < 15) or Normal-like behavior (nu > 30)?

## Convergence Diagnostics

**Status: PASSED**

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Max R-hat | 1.000 | < 1.01 | Pass |
| Min ESS bulk | 1622 | > 400 | Pass |
| Min ESS tail | 1826 | > 400 | Pass |
| Divergences | 0 | = 0 | Pass |

All chains mixed well with no pathological behavior. The trace plots show excellent mixing ("fat fuzzy caterpillars") and rank plots are uniform across chains. The energy plot shows good overlap between marginal and transition energies (BFMI satisfactory).

## Parameter Estimates

| Parameter | Mean | SD | 95% HDI | ESS bulk |
|-----------|------|-----|---------|----------|
| alpha | 3.098 | 0.006 | [3.087, 3.109] | 3482 |
| beta_weight | -0.934 | 0.020 | [-0.973, -0.897] | 2670 |
| beta_year | 0.031 | 0.002 | [0.028, 0.034] | 4071 |
| sigma | 0.100 | 0.007 | [0.087, 0.113] | 1622 |
| nu | 8.33 | 4.60 | [3.9, 20.6] | 1634 |

The regression coefficients are consistent with experiment 2:
- **Weight effect**: A 1% increase in weight corresponds to ~0.93% decrease in MPG (elasticity near -1)
- **Year effect**: ~3.1% MPG improvement per model year from technological advances
- **Scale**: The sigma is slightly lower (0.100 vs 0.101 in A2) due to robustness to outliers

## Key Finding: Degrees of Freedom (nu)

The posterior for `nu` provides critical insight into data quality:

| Statistic | Value |
|-----------|-------|
| Mean | 8.3 |
| Median | 7.1 |
| 95% CI | [3.9, 20.6] |

**Interpretation**: The posterior median of nu = 7.1 falls well below the threshold of 15, indicating **heavy tails are needed**. This suggests the Auto-MPG data contains outliers or measurement errors that the Student-t distribution handles better than Normal errors.

With nu ~ 7, the Student-t distribution has heavier tails than a Normal (which corresponds to nu = infinity). Extreme observations are down-weighted, making parameter estimates more robust.

The `nu_interpretation.png` plot visualizes this finding with reference thresholds at nu=15 and nu=30.

## LOO-CV Model Comparison

| Model | ELPD | SE | p_loo | Pareto k |
|-------|------|----|-------|----------|
| A2-Year (Normal) | 279.7 | 17.3 | 4.3 | All good |
| A3-Robust (Student-t) | 286.1 | 17.5 | 5.0 | All good |

**Difference (A3 - A2)**: +6.4 ELPD

The A3-Robust model shows a modest improvement of 6.4 points in expected log predictive density. However, the approximate z-score is only 0.26 (well below 2), so this difference is **not statistically significant**. The models are essentially equivalent in predictive performance despite the clear evidence for heavy tails.

This pattern (heavy tails detected but little predictive gain) suggests the outliers, while present, are not extreme enough to substantially distort predictions from the Normal model. The Student-t provides insurance against such outliers without meaningful cost.

## Diagnostic Plots

All plots saved to `figures/`:

- `trace_plots.png`: Chain trajectories and marginal densities
- `rank_plots.png`: Rank histograms (uniform = good mixing)
- `posterior_distributions.png`: Parameter posteriors with HDI
- `pair_plot.png`: Bivariate relationships between parameters
- `ess_evolution.png`: ESS accumulation over draws
- `energy_plot.png`: HMC energy diagnostics
- `autocorrelation.png`: Autocorrelation decay
- `loo_pit.png`: LOO probability integral transform
- `nu_interpretation.png`: Nu posterior with interpretation regions

## Conclusions

1. **Convergence**: Excellent. All diagnostics pass with large margins.

2. **Heavy tails needed**: The estimated nu ~ 7 indicates the Auto-MPG data has outliers or heavy-tailed residuals. The Student-t likelihood is appropriate.

3. **Predictive equivalence**: Despite heavy tails, LOO-CV shows no significant improvement over Normal errors (ELPD difference = 6.4, z = 0.26). The outliers affect tail behavior but not core prediction.

4. **Recommendation**: The A3-Robust model is preferred for inference because it provides robustness to outliers. For pure prediction, either model suffices.

## Files Generated

- `posterior.nc`: ArviZ InferenceData with posterior samples and log_lik
- `convergence.json`: Convergence diagnostics summary
- `loo.json`: LOO-CV results
- `figures/`: Diagnostic plots (9 figures)
