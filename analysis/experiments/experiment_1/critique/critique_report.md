# Model Critique: A1-Baseline

## Assessment: VIABLE

The A1-Baseline model (log_mpg ~ log_weight) is computationally sound and captures the dominant physical relationship in the data. However, posterior predictive checks reveal systematic misfit that requires model extension.

## Diagnostic Summary

### Prior Predictive Check: PASS

The priors generate plausible predictions covering the observed MPG range [9, 47] without producing impossible values. The prior predictive 95% interval [8.3, 59.6] appropriately encompasses the data while providing useful regularization.

### Simulation-Based Calibration: PASS

Parameter recovery tests achieved 93% coverage (14/15 parameters within 90% HDI), consistent with nominal rates. The model is identifiable and unbiased:
- alpha bias: -0.009 (z = 0.73)
- beta_weight bias: +0.002 (z = 0.36)
- sigma bias: +0.003 (z = 0.66)

### Convergence: PASS

All chains converged with excellent mixing:
- R-hat: 1.00 for all parameters
- ESS bulk: 3483-4169 (minimum threshold: 400)
- ESS tail: 2650+ (minimum threshold: 400)
- Divergences: 0

### LOO-CV Diagnostics: PASS

| Metric | Value |
|--------|-------|
| ELPD | 147.6 +/- 16.4 |
| p_loo | 3.28 |
| Pareto k (all < 0.5) | 392/392 (100%) |
| Max k | 0.15 |

All Pareto k values are well below the 0.5 threshold (see `loo_khat.png`), indicating reliable importance sampling and no influential outliers. The effective number of parameters (p_loo = 3.3) matches the model's 3 parameters exactly.

### Posterior Predictive Check: MARGINAL PASS, CONDITIONAL FAIL

**Marginal distribution**: The model reproduces the overall distribution of log(MPG) adequately. Test statistic p-values are well-calibrated (0.23-0.87).

**Conditional structure**: Critical failure. Residuals show strong monotonic trend by model year:
- Early years (1970-1973): mean residual -0.13 to -0.15 (underprediction)
- Late years (1979-1982): mean residual +0.12 to +0.22 (overprediction)
- Effect magnitude: ~0.35 log-units (~40% on original scale)

The PIT histogram shows excess density near 0.4 and deficit near 0.9, consistent with unmeasured heterogeneity across years.

**Origin residuals**: No systematic pattern (all t-statistics < 1). Weight adequately accounts for origin-related MPG differences.

## Parameter Estimates

| Parameter | Estimate | 94% HDI | Interpretation |
|-----------|----------|---------|----------------|
| alpha | 3.098 | [3.08, 3.11] | log(MPG) = 3.10 at mean weight, i.e., 22.2 MPG |
| beta_weight | -1.057 | [-1.11, -1.00] | 1% weight increase -> 1.06% MPG decrease |
| sigma | 0.166 | [0.15, 0.18] | ~17% multiplicative residual error |

**Physical interpretation**: The weight elasticity of -1.06 closely matches the theoretical prediction from physics (fuel consumption proportional to mass). This validates the log-log specification and confirms weight as the dominant physical driver of fuel efficiency.

## Identified Deficiencies

1. **Missing temporal trend**: The strongest signal in residuals. Fuel efficiency improved ~1.2 MPG/year due to oil crisis responses and CAFE standards, independent of weight changes. This omission biases predictions by era.

2. **Calibration imperfection**: PIT histogram shows mild miscalibration - the model's predictive intervals are not optimally calibrated due to within-year structure it cannot capture.

3. **Residual variance overestimated**: sigma = 0.166 includes both measurement noise and systematic year variation. Adding year should reduce sigma.

## Recommendations for Next Iteration

### Priority 1: Add Year (Strong Evidence)

The residual analysis provides unambiguous evidence for a year effect. Recommended extension:

```
log(mpg) ~ log(weight) + year_centered
```

Expected improvements:
- Eliminate systematic residual pattern
- Reduce sigma from ~0.17 to ~0.12-0.14
- Improve ELPD by substantial margin (estimated +30-50 points)

Year can be included as linear (simple, interpretable) or with a changepoint around 1980 (captures the sharp efficiency jump visible in residuals).

### Priority 2: Re-evaluate Origin After Adding Year (Uncertain)

Currently, weight explains origin differences. However, Japanese/European manufacturers may have adopted efficiency technology earlier. After adding year:
- Check residuals by origin again
- If pattern emerges, consider origin intercepts or year x origin interaction
- If no pattern, exclude origin (parsimony)

### Priority 3: Consider Robust Errors (Low Priority)

Some residual outliers exist (range: -0.52 to +0.59). Student-t errors could provide robustness, but this is lower priority than structural improvements.

### What NOT to Add

- **Horsepower/displacement**: Highly collinear with weight (r > 0.87). Would cause identifiability issues without improving fit.
- **Cylinders**: Confounded with origin and weight. Structural zeros (all 8-cylinders are American) make interpretation problematic.
- **Nonlinear weight terms**: The log-log specification already handles the curvature. Additional polynomial terms are unnecessary.

## Files Generated

| File | Description |
|------|-------------|
| `critique_report.md` | This assessment |
| `loo_khat.png` | Pareto k diagnostic plot |
| `loo_extended.json` | LOO results for model comparison |

## Conclusion

The A1-Baseline model successfully validates the physical hypothesis that weight dominates fuel efficiency (elasticity ~ -1). However, it ignores technological progress over time, leading to systematic prediction errors by year. The model is VIABLE as a baseline but should be extended with a year term before substantive interpretation or prediction.

**Next model**: A2 with year effect (linear or changepoint specification).
