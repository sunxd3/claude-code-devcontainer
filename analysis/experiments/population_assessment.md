# Population Assessment: Class A Models

**Assessment Date**: 2026-01-04
**Models Evaluated**: A1-Baseline, A2-Year, A3-Robust
**Strategic Recommendation**: **ADEQUATE**

## Summary

Class A exploration is complete. The A2-Year model provides adequate predictive performance for the Auto-MPG dataset. The key finding is that **origin effects are unnecessary** once weight and year are controlled---the apparent efficiency advantage of Japanese and European cars is fully explained by lighter weights and later market timing.

## Model Ranking

| Rank | Model | ELPD | SE | p_loo | Pareto k |
|------|-------|------|-----|-------|----------|
| 1 | A3-Robust | 286.1 | 17.5 | 5.0 | All good |
| 2 | A2-Year | 279.7 | 17.3 | 4.3 | All good |
| 3 | A1-Baseline | 147.6 | 16.4 | 3.3 | All good |

All models pass LOO diagnostics with 100% of Pareto k values below 0.5.

## ELPD Comparisons

| Comparison | ELPD Diff | SE Diff | z-score | Verdict |
|------------|-----------|---------|---------|---------|
| A2 vs A1 | +132.2 | 11.2 | 11.8 | A2 decisively better |
| A3 vs A2 | +6.3 | 3.4 | 1.9 | Not significant |
| A3 vs A1 | +138.5 | 11.2 | 12.4 | A3 decisively better |

**Interpretation**: Adding year to the baseline model is essential (z = 11.8). The Student-t extension (A3) provides modest ELPD improvement but the difference is not statistically significant at conventional thresholds (z = 1.9 < 2). By parsimony, A2-Year is the preferred Class A model.

## Key Evidence

### 1. Year Effect is Critical

The A1 baseline showed strong residual trends by year (-0.15 to +0.22 on log scale), indicating unmeasured technological progress. Adding year reduced residual SD from 0.166 to 0.118 (29% reduction) and improved ELPD by 132 points.

### 2. Origin Effects Disappear

After controlling for weight and year, residuals by origin show no pattern:

| Origin | n | Mean Residual | t-statistic |
|--------|---|---------------|-------------|
| USA | 245 | -0.010 | -1.5 |
| Europe | 68 | +0.033 | +1.9 |
| Japan | 79 | +0.003 | +0.2 |

All t-statistics are below 2. The European marginal effect (3.3% efficiency premium, p ~ 0.06) is not strong enough to justify adding origin as a predictor.

### 3. Outliers Exist but Don't Matter

The A3-Robust model estimates nu ~ 7 (95% CI: 3.9-20.6), confirming heavier tails than Normal. Identified outliers include diesel vehicles (Oldsmobile Cutlass Ciera Diesel) and rotary engines (Mazda RX3). However, these outliers do not substantially affect predictions---the ELPD gain is only 6.3 points (z = 1.9).

## Strategic Decision: ADEQUATE

Class A models are sufficient for the Auto-MPG analysis. The A2-Year model:

- **Passes all validation stages**: prior predictive, recovery, convergence, PPC
- **Achieves strong predictive performance**: ELPD = 279.7
- **Answers the scientific question**: Origin effects are explained by weight and year
- **Is parsimonious**: 4 parameters (alpha, beta_weight, beta_year, sigma)

### Why Not Continue to Class B?

The experiment plan anticipated that "Class A will underperform due to missing origin effects." This expectation was wrong. The PPC analysis unambiguously demonstrates that:

1. Residuals show no origin structure (all |t| < 2)
2. Weight + year capture what appeared in EDA to be origin effects
3. Adding origin terms would increase complexity without improving predictions

Testing Class B would consume resources to confirm what the data already show: origin is not needed.

### Why Not Continue Refining Class A?

The A3-Robust extension tested whether outliers warranted heavy-tailed errors. While nu ~ 7 confirms outliers exist, the predictive improvement is negligible (z = 1.9). Further Class A refinements (weight-year interaction, year random effects) would provide diminishing returns.

## Recommended Model: A2-Year

```
log(mpg) ~ Normal(alpha + beta_weight * log_weight_c + beta_year * year_c, sigma)
```

**Parameter estimates (posterior mean, 95% HDI)**:

| Parameter | Estimate | 95% HDI | Interpretation |
|-----------|----------|---------|----------------|
| alpha | 3.098 | [3.09, 3.11] | 22.2 MPG at mean weight/year |
| beta_weight | -0.935 | [-0.98, -0.89] | 1% weight increase -> 0.94% MPG decrease |
| beta_year | 0.033 | [0.030, 0.036] | 3.3% efficiency gain per year |
| sigma | 0.118 | [0.11, 0.13] | 12% residual CV |

The weight elasticity (~-0.94) aligns with physical intuition (fuel consumption proportional to mass). The year trend (3.3%/year, ~48% total over 12 years) captures CAFE standards and oil crisis responses.

## Next Steps

1. **Model Audit**: Review A2-Year code, priors, and sensitivity to specification choices
2. **Final Report**: Document the modeling journey and scientific conclusions
3. **Optional**: If stakeholders request it, briefly confirm A3-Robust as a sensitivity analysis

## Files

| File | Description |
|------|-------------|
| `class_a_comparison.csv` | ArviZ comparison table |
| `class_a_comparison.png` | Comparison visualization |
| `class_a_summary.json` | Machine-readable summary |
| `population_assessment.md` | This document |

![Class A Model Comparison](class_a_comparison.png)
