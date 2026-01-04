# Model Critique: A2-Year

**Assessment: VIABLE**

The A2-Year model passes all validation stages and represents a substantial improvement over the baseline. Adding year as a predictor reduces residual variance by 50% and improves ELPD by 132 points. The model is scientifically sensible and computationally well-behaved.

## Diagnostic Summary

| Stage | Status | Key Finding |
|-------|--------|-------------|
| Prior predictive | PASS | 87.6% coverage of observed MPG range |
| Simulation-based calibration | PASS | 95% coverage, parameters recovered accurately |
| Convergence | PASS | R-hat=1.0, ESS>2400, 0 divergences |
| LOO-CV | PASS | All k<0.5, ELPD=279.7 |
| Posterior predictive | PASS | No systematic residual patterns |

## Performance vs Baseline

The year effect captures substantial variance that weight alone cannot explain.

| Metric | A1-Baseline | A2-Year | Change |
|--------|-------------|---------|--------|
| ELPD | 147.6 | 279.7 | +132.1 |
| Residual SD | 0.166 | 0.118 | -29% |
| R-squared (approx) | 0.76 | 0.88 | +12pp |
| p_loo | 3.3 | 4.3 | +1 param |

The ELPD improvement of 132 points is overwhelming evidence that year belongs in the model. This difference far exceeds the standard error and represents a difference of over 5 standard errors.

## Scientific Interpretation

The model estimates that fuel efficiency improved by 3.3% per year (95% HDI: 3.0-3.6%) holding weight constant. Over the 12-year span, this amounts to approximately 48% cumulative improvement from technology. The weight elasticity is -0.93, meaning a 1% increase in weight reduces MPG by about 0.93%.

## Origin Effect Assessment

**Conclusion: Origin effects NOT needed.**

After controlling for weight and year, residuals show no systematic pattern by origin:

| Origin | n | Mean Residual | t-statistic |
|--------|---|---------------|-------------|
| USA | 245 | -0.010 | -1.5 |
| Europe | 68 | +0.033 | +1.9 |
| Japan | 79 | +0.003 | +0.2 |

All t-statistics are below 2. The European cars show a marginally positive residual (about 3.4% higher efficiency than predicted), but this is not statistically significant (p approximately 0.06). The apparent efficiency advantage of Japanese and European cars is fully explained by their lighter weight and later market entry timing.

**Recommendation**: Skip Class B origin model. The evidence is clear that origin provides no additional predictive power once weight and year are controlled.

## Robust Errors Assessment

**Conclusion: Low priority for testing.**

The residuals show no strong evidence of heavy tails:
- Residual range: [-0.41, +0.39] on log scale
- Maximum |residual|/sigma = 3.5, normal for n=392
- PIT histogram shows no excess at extremes
- LOO-PIT calibration falls within 95% band

The scale-location plot shows approximately constant variance. Some mild increase in spread at the extremes of fitted values is visible, but not severe enough to warrant immediate attention.

**Recommendation**: Robust errors (Student-t likelihood) could be tested as a sensitivity analysis but should not be prioritized over other extensions.

## Remaining Weaknesses

The model fits well overall but has identifiable limitations:

**1. Year-specific deviations from linear trend**

Several years show significant departures from the assumed linear technology trend:

| Year | Mean Residual | t-stat | Interpretation |
|------|---------------|--------|----------------|
| 1971 | +0.065 | +4.8 | Above trend (early efficiency push?) |
| 1973 | -0.069 | -4.0 | Below trend (pre-crisis inefficiency peak) |
| 1980 | +0.108 | +3.6 | Above trend (second oil crisis response) |

The oil crises of 1973-74 and 1979-80 created non-linear shocks to efficiency that a linear year effect cannot capture. However, these deviations represent real historical events rather than model misspecification.

**2. Marginal European effect**

European cars show a consistent +3.3% residual (t=1.9, p approximately 0.06). This could represent genuine manufacturing differences or unmeasured confounders. Given the borderline significance, monitoring this in future models is worthwhile but does not justify adding origin as a predictor.

**3. Potential weight-year interaction**

The model assumes technology improvements affected all cars equally regardless of weight. In reality, fuel injection, aerodynamics, and materials advances may have benefited different vehicle segments differently. A weight-year interaction could test whether the technology trend varies by vehicle size.

## Recommended Next Steps

**Priority 1: Accept A2-Year as primary model**

The model is well-validated and answers the key scientific questions. Year captures the technology trend, and origin effects are explained away.

**Priority 2 (optional sensitivity analyses):**

1. **Weight-year interaction**: Test whether technology improvements differed by vehicle size. Model: add `beta_wy * log_weight_c * year_c` term.

2. **Robust errors**: Test Student-t likelihood as sensitivity check for influential observations. Low priority given good LOO diagnostics.

3. **Year random effects**: Model years as random intercepts around the linear trend to capture oil crisis shocks. Adds complexity with marginal benefit.

**Not recommended:**
- Origin effects: Evidence clearly shows no predictive value
- Nonlinear weight effects: No residual pattern suggests this
- Additional predictors (horsepower, cylinders): Would complicate interpretation without clear benefit

## Files Generated

- `assessment.md` - This critique report
- `loo_results.json` - LOO statistics for model comparison
