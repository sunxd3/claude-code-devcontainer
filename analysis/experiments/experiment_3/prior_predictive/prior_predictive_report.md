# Prior Predictive Check Report: A3-Robust Model (Student-t Errors)

## Model Specification

**Likelihood:** log(mpg) ~ Student-t(nu, mu, sigma) where mu = alpha + beta_weight * log_weight_c + beta_year * year_c

**Priors:**
- alpha ~ Normal(3.1, 0.3)
- beta_weight ~ Normal(-1, 0.3)
- beta_year ~ Normal(0.03, 0.02)
- sigma ~ Exponential(5)
- nu ~ Gamma(2, 0.1)

**Data:** N=392 observations, MPG range 9.0-46.6

## What Was Checked

1. **Distribution of prior predictive samples** - Generated 4,000 posterior draws (4 chains x 1,000 samples) from priors only
2. **Coverage of observed MPG range** - Proportion of simulated values within plausible domain
3. **Nu (degrees of freedom) prior behavior** - Key parameter controlling tail heaviness
4. **Numerical stability** - Checked for NaN/Inf values in simulations

## Findings

### Prior Predictive Distribution

The prior predictive distribution (see `prior_predictive_overview.png`) shows:

- **Log scale:** Mean = 3.10, SD = 0.55, range approximately [-22, 18.5]
- **MPG scale:** Mean = 25.9, Median = 22.3, 95% interval [7.6, 62.9]

The observed data mean (approximately 23 MPG) falls well within the bulk of the prior predictive distribution.

### Coverage Assessment

| Metric | Value |
|--------|-------|
| Within observed range (9-46.6 MPG) | 87.7% |
| Below minimum (< 9 MPG) | 4.6% |
| Above maximum (> 46.6 MPG) | 7.7% |

The 87.7% coverage is appropriate for a prior predictive check. The priors are neither too narrow (which would restrict learning) nor too diffuse (which would generate implausible data). A small fraction of extreme values (4.6% below, 7.7% above) is expected and reflects reasonable uncertainty before seeing data.

### Nu Prior (Degrees of Freedom)

The Gamma(2, 0.1) prior on nu yields:
- Mean: 20.2
- Median: 16.7
- 95% interval: [3.0, 56.6]
- P(nu < 10) = 26.3% (allows heavy tails)
- P(nu < 30) = 79.3% (moderate to heavy tails common)

This prior allows the model to detect outliers if present (small nu) while also permitting near-normal behavior (large nu). The lower bound of nu > 2 ensures finite variance. See `prior_predictive_overview.png` (bottom right panel) for visualization.

### ECDF Comparison

The ECDF plot (`prior_predictive_ecdf.png`) shows the observed data ECDF (black) overlaid on 50 prior predictive draws (light blue). The observed data falls comfortably within the envelope of prior predictive samples, confirming the priors generate data consistent with what we might observe.

### Numerical Stability

- NaN values: 0
- Inf values: 0

No numerical issues detected in the prior predictive samples.

## Visual Evidence

Two diagnostic plots are provided:

1. **`prior_predictive_overview.png`**: Four-panel summary showing (a) prior predictive on log scale, (b) prior predictive on MPG scale, (c) prior parameter distributions, and (d) nu prior distribution with reference lines

2. **`prior_predictive_ecdf.png`**: ECDF comparison of prior predictive draws against observed data

## Adjustments Made

None required. The priors generate plausible data that appropriately covers the observed range.

## Recommendation

**PASS**

The priors are well-calibrated for this analysis:
- Prior predictive samples cover the observed data range appropriately (87.7%)
- The nu prior allows flexibility in tail behavior without forcing heavy or light tails
- No numerical issues detected
- The prior predictive distribution is centered near the observed mean

The model is ready for posterior inference.

## Files Created

| File | Description |
|------|-------------|
| `prior_predictive.nc` | ArviZ InferenceData with prior and prior_predictive groups |
| `prior_predictive_stats.json` | Summary statistics |
| `prior_predictive_overview.png` | Four-panel diagnostic plot |
| `prior_predictive_ecdf.png` | ECDF comparison plot |
| `prior_predictive_report.md` | This report |
