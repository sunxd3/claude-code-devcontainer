# Prior Predictive Check: Experiment 3

**Date**: 2026-01-14
**Model**: Random Intercepts + Random Slopes (Non-centered parameterization)
**Recommendation**: PASS

## Model Specification

Random intercepts and slopes model with non-centered parameterization:

```stan
y_ij ~ Normal(alpha_j + beta_j * treatment_ij, sigma)
alpha_j = alpha_0 + tau_alpha * z_alpha_j
beta_j = beta_0 + tau_beta * z_beta_j
z_alpha_j ~ Normal(0, 1)
z_beta_j ~ Normal(0, 1)

Priors:
  alpha_0 ~ Normal(77, 15)
  tau_alpha ~ Half-Normal(0, 10)
  beta_0 ~ Normal(5, 5)
  tau_beta ~ Half-Normal(0, 5)
  sigma ~ Half-Normal(0, 15)
```

## Data Context

- Outcome: Student test scores (typical range 40-110, mean ~77)
- Treatment: Binary (0/1)
- Structure: J=8 schools, N=160 students
- Observed range: [43.7, 104.6]
- Observed mean: 77.1 (SD: 10.5)

## Method

Generated 4000 prior predictive samples (1000 per chain) using CmdStanPy with fixed_param sampling. The model samples from the priors in the generated quantities block and produces replicated test scores y_rep for each of the 160 students.

## Findings

### Prior Distributions

The priors generate reasonable parameter values centered near expected values:

- **alpha_0** (population mean intercept): 76.5 ± 15.2
- **tau_alpha** (school intercept SD): 8.1 ± 6.2
- **beta_0** (population treatment effect): 5.1 ± 4.9
- **tau_beta** (school slope SD): 4.0 ± 3.1
- **sigma** (residual SD): 12.0 ± 9.1

All prior means are close to scientifically plausible values, with sufficient spread to be weakly informative.

### Prior Predictive Distribution

**Central tendency:**
- Prior predictive mean: 79.1 (close to observed 77.1)
- Prior predictive SD: 24.2 (larger than observed 10.5)

**Range:**
- Prior predictive: [-85.0, 252.2]
- Observed: [43.7, 104.6]
- 5th-95th percentile: [40.0, 118.5]

**Implausible values:**
- Negative values: 1586 samples (0.25%)
- Very low (< 40): 32,070 samples (5.01%)
- Very high (> 110): 58,612 samples (9.16%)
- Extreme (> 150): 2923 samples (0.46%)

### Visual Assessment

See `prior_predictive_check.png` for detailed comparisons:

1. **ECDF comparison**: Prior predictive distribution is more dispersed but covers the observed range
2. **Histogram**: Prior predictive has heavier tails than observed
3. **Range visualization**: Prior allows wider range, with 90% credible interval [40, 118] versus observed [44, 105]
4. **Proportion of extremes**: Small but non-zero proportions of impossible/implausible values

## Assessment

### Strengths

1. **Central tendency**: Prior predictive mean (79.1) matches observed mean (77.1) well
2. **Coverage**: The prior encompasses the entire observed range
3. **Weak informativity**: Priors are diffuse enough to let data dominate, but centered near plausible values
4. **Small proportion of impossible values**: Only 0.25% of samples are negative

### Concerns

1. **Scale mismatch**: Prior predictive SD (24.2) is more than 2x observed SD (10.5), indicating quite diffuse priors
2. **Tail behavior**: ~9% of samples exceed 110 (high but plausible for test scores)
3. **Impossible values**: Small number of negative test scores (0.25%)

### Domain Constraints

Test scores are inherently bounded (cannot be negative, unlikely to exceed ~150). The priors do not enforce hard constraints, leading to:
- 0.71% of samples outside reasonable bounds (<0 or >150)
- 9.16% above typical maximum (>110)

However, these proportions are small enough that the likelihood will quickly overwhelm the prior once we fit to data.

## Recommendation: PASS

The priors are appropriate for this model and should produce valid posterior inference. Rationale:

1. **Central tendency is correct**: Prior means match expected values from data context
2. **Weak informativity is intentional**: Diffuse priors ensure data dominates inference
3. **Small proportion of extremes**: Less than 1% of samples are truly impossible (<0 or >150)
4. **Likelihood will dominate**: With N=160 observations, the data will quickly constrain the posterior

The priors successfully encode weak domain knowledge (test scores around 77, treatment effects around 5, school variation around 8-10) while remaining open to what the data reveal. The small proportion of impossible values (0.25% negative) is acceptable for weakly informative priors and will be eliminated by the likelihood during posterior inference.

## Files Generated

- `model.stan` - Stan model with non-centered parameterization
- `prior_predictive_model.stan` - Fixed-param model for prior predictive sampling
- `prior_predictive.nc` - ArviZ InferenceData with prior samples
- `prior_predictive_check.png` - Visual comparison of prior predictive vs observed
- `prior_predictive_check_report.md` - This report
