# Posterior Predictive Check Report
## Experiment 2: Random Intercepts Only Model

**Date:** 2026-01-14
**Model:** Random intercepts for schools, fixed treatment effect
**Data:** 160 students across 8 schools

## Summary

The Random Intercepts Only model provides adequate fit for several aspects of the data but reveals a key limitation: it cannot capture treatment effect heterogeneity across schools. While the model successfully reproduces overall score distributions and school-level baseline differences, the assumption of a constant treatment effect appears violated in the observed data.

## Model Adequacy: What Works Well

### Overall Score Distribution

The model captures the marginal distribution of test scores effectively (see `ppc_ecdf.png` and `ppc_kde.png`). The observed data falls comfortably within the posterior predictive distribution, with no systematic deviations in location, spread, or shape.

**Summary statistics** (see `ppc_summary_stats.png`):
- Mean: Observed = 77.09, Posterior predictive = 77.11 [75.35, 78.83], p = 0.965
- SD: Observed = 10.47, Posterior predictive = 10.55 [9.29, 11.96], p = 0.948
- Min: Observed = 43.70, Posterior predictive = 49.01 [40.91, 55.59], p = 0.118
- Max: Observed = 104.60, Posterior predictive = 105.48 [98.80, 113.81], p = 0.543

All p-values indicate good agreement, with no extreme departures.

### School-Level Baseline Differences

The model successfully captures between-school variation in average scores (see `ppc_school_means.png`). All eight observed school means fall within the 90% posterior predictive intervals:

- School 1: 75.70 within [70.05, 80.90]
- School 2: 72.36 within [68.87, 79.12]
- School 3: 77.34 within [72.99, 85.38]
- School 4: 82.02 within [70.59, 85.66]
- School 5: 73.95 within [70.38, 80.24]
- School 6: 74.45 within [70.33, 79.81]
- School 7: 85.04 within [70.44, 88.99]
- School 8: 76.38 within [69.96, 77.84]

The random intercepts structure appropriately models this school-level clustering.

## Model Limitation: Treatment Effect Heterogeneity

### Evidence of Heterogeneity

The posterior predictive check reveals substantial variation in observed treatment effects across schools that the model cannot reproduce (see `ppc_treatment_effects.png`):

**Observed treatment effects by school:**
- School 1: +15.00 (very large positive effect)
- School 2: +3.99
- School 3: -0.45 (negative effect)
- School 4: +10.00
- School 5: +10.63
- School 6: +7.84
- School 7: +1.43
- School 8: +9.00

**Model assumption:** Constant effect β ≈ 6.58 across all schools

The model assumes all schools share the same treatment effect, constraining posterior predictive treatment effects to cluster around the pooled estimate. While the overall variance of treatment effects (observed = 23.29, predicted = 17.45 [5.13, 35.63]) falls within the posterior predictive range, this masks the fact that specific schools show effects far from the pooled estimate.

### Residual Patterns

Residual analysis supports this finding (see `residual_patterns.png`):

**By school:** Residuals show variable patterns across schools, with some schools (e.g., School 1, School 7) showing systematic deviations suggesting the fixed treatment effect doesn't fit all contexts equally.

**By treatment group:** The treatment group shows clustering of positive residuals in the upper range, while control group residuals are more dispersed. This pattern suggests the model systematically under-predicts scores for some treated students and over-predicts for others, consistent with treatment effect heterogeneity the model cannot express.

## Implications

The Random Intercepts Only model is adequate for:
- Estimating the average treatment effect pooled across schools (β ≈ 6.58)
- Capturing school-level baseline differences
- Predicting overall score distributions

The model is inadequate for:
- Estimating school-specific treatment effects
- Understanding which schools benefit most from treatment
- Assessing whether treatment effectiveness varies by context

## Recommendations

The evidence of treatment effect heterogeneity suggests exploring a **Random Intercepts and Slopes** model that allows treatment effects to vary across schools. This would test whether observed variation in school-specific effects exceeds what we'd expect from sampling variability alone.

Key questions for next iteration:
1. Is the treatment effect heterogeneity substantively significant, or could it arise from small sample sizes within schools?
2. Are there school-level characteristics (size, baseline performance) that predict treatment effect magnitude?
3. Would random slopes improve predictive performance enough to justify the additional complexity?

## Diagnostic Files

All diagnostic visualizations saved to:
`/home/user/claude-code-devcontainer/analysis/experiments/experiment_2/posterior_predictive/`

- `ppc_ecdf.png` - Empirical CDF comparison
- `ppc_kde.png` - Kernel density comparison
- `ppc_summary_stats.png` - Test statistics (mean, SD, min, max)
- `ppc_school_means.png` - School-level mean scores
- `ppc_treatment_effects.png` - Treatment effects by school
- `residual_patterns.png` - Residuals by school and treatment group
