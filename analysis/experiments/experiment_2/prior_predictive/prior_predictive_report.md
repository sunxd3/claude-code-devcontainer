# Prior Predictive Check Report: Experiment 2

**Model**: Random Intercepts Only (non-centered parameterization)
**Date**: 2026-01-14
**Status**: PASS

## Model Specification

```stan
y_ij ~ Normal(alpha_j + beta * treatment_ij, sigma)
alpha_j = alpha_0 + tau_alpha * z_alpha_j
z_alpha_j ~ Normal(0, 1)

Priors:
  alpha_0 ~ Normal(77, 15)
  tau_alpha ~ Half-Normal(0, 10)
  beta ~ Normal(5, 5)
  sigma ~ Half-Normal(0, 15)
```

## Methodology

Sampled 4000 draws (4 chains x 1000 iterations) from the prior predictive distribution using Stan's `fixed_param` algorithm. Each draw generates a complete dataset of 160 synthetic observations matching the structure of the observed data (8 schools, binary treatment assignment).

## Findings

### Overall Plausibility

The priors generate predominantly plausible test scores:

- **86.6%** of simulated values fall within the plausible range (40-110)
- **0.25%** are impossible (negative scores)
- **4.34%** are very low (0-40 range)
- **8.81%** are very high (> 110)

Prior predictive range: [-162.9, 284.7]
Observed data range: [43.7, 104.6]

### Distribution Comparison

**Visual evidence**: `prior_predictive_distribution.png`

The empirical cumulative distribution function (ECDF) shows that prior predictive values largely overlap with the observed data distribution. The kernel density estimate (KDE) reveals that the prior predictive distribution is more diffuse than the observed data, as expected for weakly informative priors.

The prior predictive distribution is centered near the observed mean (prior mean: 79.3, observed mean: 77.1) and has reasonable spread. Extreme tails extend beyond the observed range, but this is acceptable for priors that should not overly constrain the model.

### Hyperparameter Behavior

**Visual evidence**: `prior_predictive_hyperparameters.png`

**alpha_0** (population mean intercept): Centered at 77 with SD of 15, matching the prior specification. The distribution appropriately covers the observed mean of 77.1.

**tau_alpha** (SD of school intercepts): Half-normal with mean around 8.1. This allows for moderate to strong school-level heterogeneity. The prior permits values from near-zero (complete pooling) to substantial variation (> 20), appropriately expressing uncertainty about the degree of clustering.

**beta** (treatment effect): Centered at 5 with SD of 5, consistent with the prior. This weakly favors a positive treatment effect but allows negative effects and large positive effects.

**sigma** (residual SD): Half-normal with mean around 11.8. This is reasonable for test score variability within schools after accounting for treatment and school effects.

### Extreme Values

**Visual evidence**: `prior_extremes.png`

The distribution of minimum and maximum values across prior draws shows:

- Minimum values occasionally dip below zero (0.25% of all values), but most minima are above 20
- Maximum values extend well beyond the observed maximum (104.6), with some draws exceeding 200

The percentage breakdown shows that implausible values are rare. Negative values occur in only 1 out of 400 simulated observations on average, which is negligible for weakly informative priors.

### Scale and Domain Constraints

**Test scores should be positive**: 99.75% of prior predictive values satisfy this constraint. The tiny fraction of negative values (0.25%) arises from the combination of large negative school effects and large residual errors, which is technically possible under the model but extremely rare.

**Typical range (40-110)**: 86.6% of values fall in this range. Values outside this range are plausible outliers or extreme cases that could occur in real educational data.

The observed data (min: 43.7, max: 104.6) falls comfortably within the prior predictive distribution, indicating the priors are compatible with the data.

## Assessment

### Strengths

1. **Centered appropriately**: Prior predictive mean (79.3) matches observed mean (77.1)
2. **Reasonable spread**: Most values (86.6%) fall in the typical test score range
3. **Few impossible values**: Only 0.25% are negative
4. **Weak informativity**: Priors allow data to dominate while preventing extreme implausibility
5. **School-level variation**: The tau_alpha prior appropriately expresses uncertainty about clustering strength

### Concerns

**Minor**: The prior predictive distribution has heavier tails than the observed data, with some values extending to -162.9 and 284.7. This occurs because:
- Half-normal priors on tau_alpha and sigma can generate large values
- Large tau_alpha creates extreme school effects
- Large sigma creates extreme residual errors
- These combine to produce outliers

This is acceptable because:
- Extreme values are rare (< 1% impossible, < 14% outside typical range)
- The posterior will shrink these parameters toward reasonable values once conditioned on data
- Weakly informative priors should not rule out extreme scenarios a priori

**No action needed**: The priors are functioning as intended.

## Recommendation

**PASS**

The priors generate plausible synthetic data that resembles the observed data in central tendency and spread. The small percentage of implausible values (negative scores) is negligible and acceptable for weakly informative priors. The model is ready for parameter recovery testing.

## Files Generated

- `prior_predictive.nc`: InferenceData object with prior and prior_predictive groups
- `prior_predictive_distribution.png`: ECDF and KDE comparison
- `prior_hyperparameters.png`: Marginal distributions of hyperparameters
- `prior_extremes.png`: Distribution of extreme values and plausibility breakdown
