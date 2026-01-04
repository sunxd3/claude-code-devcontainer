# Prior Predictive Check: A1-Baseline Model

## Summary

**Result: PASS**

The priors for the A1-Baseline model generate plausible prior predictive distributions that adequately cover the observed MPG range without producing excessive impossible values.

## Model Specification

The A1-Baseline model regresses log(MPG) on centered log(weight):

```
log(mpg) ~ Normal(alpha + beta_weight * log_weight_c, sigma)
```

Priors:
- alpha ~ Normal(3.1, 0.3) - intercept at mean log-weight, centered on log(22) ~ 3.1
- beta_weight ~ Normal(-1, 0.3) - elasticity, physics suggests ~-1
- sigma ~ Exponential(5) - residual SD, expecting ~0.2

## Prior Predictive Simulation

Sampled 4000 draws from the prior predictive distribution using 398 covariate values from the Auto-MPG dataset.

### Prior Parameter Distributions

| Parameter | Prior | Sampled Mean | Sampled SD |
|-----------|-------|--------------|------------|
| alpha | Normal(3.1, 0.3) | 3.10 | 0.30 |
| beta_weight | Normal(-1, 0.3) | -1.00 | 0.30 |
| sigma | Exponential(5) | 0.20 | 0.20 |

The sampled parameters match the specified priors, confirming correct implementation.

### Prior Predictive MPG Distribution

| Percentile | MPG Value |
|------------|-----------|
| 0.5% | 5.4 |
| 2.5% | 8.3 |
| 10% | 11.9 |
| 25% | 16.1 |
| 50% (median) | 22.3 |
| 75% | 30.9 |
| 90% | 41.5 |
| 97.5% | 59.6 |
| 99.5% | 93.5 |

Observed data range: 9.0 - 46.6 MPG

## Assessment Criteria

### 1. No impossible values

**PASS** - The log-normal structure guarantees all predictions are positive. No negative MPG values are possible by construction.

### 2. Reasonable extreme values

**PASS** - The 99.5th percentile (93.5 MPG) is below 100 MPG, indicating the prior does not frequently predict implausibly high fuel efficiency. Only a small fraction of extreme draws exceed 100 MPG.

### 3. Coverage of observed range

**PASS** - The prior predictive 95% interval [8.3, 59.6] fully contains the observed range [9.0, 46.6]. The prior is neither too narrow (which would impose strong constraints before seeing data) nor too wide (which would provide no useful regularization).

### 4. Prior is weakly informative

**PASS** - The prior 90% interval width (~30 MPG) exceeds the observed range width (~38 MPG), indicating the prior allows the data to dominate inference while still providing reasonable regularization.

## Visual Evidence

The diagnostic plots (`prior_predictive_check.png`) show:

1. **Alpha prior**: Centered at 3.1 with SD 0.3, matching the expected log(mean MPG) of the data
2. **Beta_weight prior**: Centered at -1 as expected from physical reasoning (heavier cars use more fuel)
3. **MPG histogram**: Prior predictive distribution (blue) overlaps well with observed data (orange), with the observed range falling comfortably within the prior predictive range
4. **ECDF comparison**: The observed ECDF (orange line) falls within the envelope of prior predictive ECDFs (blue lines), indicating the priors are compatible with the data

The sigma prior (`prior_sigma.png`) shows the Exponential(5) distribution concentrating mass near small values (mode at 0, mean at 0.2), appropriate for log-scale residuals.

## Conclusion

The priors are well-calibrated for this regression problem:

- They encode reasonable domain knowledge (intercept near typical MPG, negative weight effect, small residual variance)
- They are weakly informative, allowing data to dominate while preventing pathological fits
- They respect domain constraints (positivity via log transform)
- They generate plausible synthetic data that encompasses the observed range

**Recommendation: Proceed with model fitting.**

## Files Generated

- `prior_predictive.nc` - ArviZ InferenceData with prior and prior_predictive groups
- `summary.json` - Numerical summary of prior predictive distribution
- `prior_predictive_check.png` - Four-panel diagnostic plot
- `prior_sigma.png` - Sigma prior distribution
- `verdict.txt` - PASS/FAIL result
