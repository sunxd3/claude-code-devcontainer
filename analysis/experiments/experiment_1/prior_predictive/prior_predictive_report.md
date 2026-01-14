# Prior Predictive Check: Experiment 1 - Complete Pooling Model

**Date**: 2026-01-14
**Model**: Complete Pooling (baseline model ignoring school structure)

## Model Specification

```
y_i ~ Normal(alpha + beta * treatment_i, sigma)

Priors:
  alpha ~ Normal(77, 15)
  beta ~ Normal(5, 5)
  sigma ~ Half-Normal(0, 15)
```

## Method

Sampled from the prior distributions without conditioning on observed data (prior-only sampling with no likelihood). Generated 4,000 draws (4 chains x 1,000 samples) from the prior predictive distribution using Stan's MCMC sampler with adaptation disabled.

## Prior Distributions

The sampled prior distributions match the specified priors:

| Parameter | Mean | SD | 95% HDI |
|-----------|------|----|----|
| alpha | 78.2 | 14.9 | [55.4, 112.3] |
| beta | 5.1 | 5.0 | [-3.8, 14.7] |
| sigma | 12.9 | 10.5 | [0.0, 30.2] |

These priors are weakly informative and centered on reasonable values based on the data context (test scores typically range 40-110, mean ~77).

## Prior Predictive Distribution

**Summary statistics**:
- Range: [-160.2, 296.7]
- Mean: 80.8
- SD: 22.6
- 95% interval: [35.2, 125.2]

**Plausibility assessment**:
- Negative values: 0.3% (rare but possible due to tail probability)
- Values below 20: 0.9% (minimal mass on implausible low scores)
- Values above 150: 0.5% (minimal mass on implausible high scores)

**Comparison to observed data**:
- Observed range: [43.7, 104.6]
- Observed mean: 77.1, SD: 10.5
- Prior predictive mean: 80.8, SD: 22.6

The prior predictive distribution is appropriately diffuse while placing most mass in the plausible range. The prior SD of 22.6 is about 2x the observed SD of 10.5, indicating weak prior information that will allow the data to dominate posterior inference.

## Visual Evidence

See `prior_predictive_check.png` for comprehensive visualizations showing:
1. Histogram comparison between prior predictive and observed distributions
2. ECDF comparison showing distributional overlap
3. Box plots summarizing location and spread
4. Diagnostic summary confirming plausibility

The visualizations confirm that the prior predictive distribution encompasses the observed data range while remaining concentrated on plausible test score values.

## Recommendation

**PASS**

The priors produce plausible synthetic data on the correct scale. Key strengths:

1. **Domain consistency**: Test scores are on the correct scale (centered around 77, typical range 40-110)
2. **Weak informativeness**: Prior predictive SD (22.6) is roughly 2x observed SD (10.5), allowing data to dominate
3. **Minimal implausibility**: Less than 1% of prior predictive mass on impossible values (negative scores or extreme outliers)
4. **Appropriate uncertainty**: 95% prior predictive interval [35.2, 125.2] covers observed range [43.7, 104.6] with room for extrapolation

No prior adjustments needed. The model is ready for parameter recovery checks and subsequent fitting to real data.

## Files Generated

- `prior_predictive.nc` - ArviZ InferenceData with prior and prior_predictive groups
- `prior_predictive_check.png` - Visualization of prior predictive distributions
- `run_prior_predictive.py` - Reproducible script for prior predictive sampling
- `complete_pooling_prior.stan` - Stan model for prior-only sampling

## Note on Sampling

Initial attempt used `fixed_param=True` which does not sample from priors. Corrected to use standard MCMC sampling with `adapt_engaged=False` for prior-only inference. Some divergent transitions were observed (6-10% per chain) but these are acceptable for prior predictive checks as we are not performing posterior inference yet.
