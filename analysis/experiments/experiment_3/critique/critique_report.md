# Model Critique: A3-Robust (Student-t Errors)

**Assessment: VIABLE**

**Recommendation: Use A2-Year as the Class A representative model. The Student-t extension in A3 is justified but provides negligible predictive benefit. Proceed to Class B to test origin effects.**

## Diagnostic Summary

### Prior Predictive Check: PASS

The Gamma(2, 0.1) prior on nu yields mean ~20 with support from 3 to 50+, allowing both heavy-tailed and Normal-like behavior. Prior predictive samples cover 87.7% of the observed MPG range (9-46.6), which is appropriate calibration. No numerical issues detected.

### Simulation-Based Calibration: PASS

Parameter recovery across 5 simulations achieved 96% overall coverage at the 90% CI level. All parameters recovered well:
- Regression coefficients (alpha, beta_weight, beta_year): minimal bias, excellent coverage
- Scale (sigma): accurate recovery
- Degrees of freedom (nu): wide posteriors but 100% coverage, reflecting inherent difficulty in identifying tail behavior

The wide nu posteriors (mean SD = 10.8 when true nu = 15) are expected behavior, not a defect. When nu is moderate, distinguishing Student-t from Normal requires extreme observations, which may be sparse.

### Convergence: PASS

All diagnostics excellent:
- R-hat: 1.000 (threshold < 1.01)
- ESS bulk: 1622 minimum (threshold > 400)
- ESS tail: 1826 minimum
- Divergences: 0

### Posterior Predictive Check: PASS

The model captures the observed data distribution well:
- Density and ECDF overlays show close agreement
- LOO-PIT histogram approximately uniform, indicating good calibration
- Test statistics (median, MAD, skewness) all have acceptable p-values
- Residuals show no systematic patterns vs fitted values or year

### LOO Diagnostics: PASS

| Metric | Value |
|--------|-------|
| ELPD | 286.1 |
| SE | 17.5 |
| p_loo | 5.0 |
| Pareto k > 0.7 | 0 |

All 392 Pareto k values are good (< 0.5), indicating reliable LOO estimates with no influential observations. The effective number of parameters (p_loo = 5.0) is slightly higher than the 4 structural parameters due to the flexibility of nu.

## Key Findings

### 1. Heavy Tails Are Present (nu ~ 7)

The posterior median nu = 7.1 with 95% CI [3.9, 20.6] strongly favors heavy tails. This indicates the Auto-MPG data contains outliers or measurement errors that benefit from down-weighting. The nu posterior is concentrated well below the nu=15 threshold typically used to distinguish heavy-tailed from Normal-like behavior.

Identified outliers include:
- **Diesel vehicles** (Oldsmobile Cutlass Ciera Diesel, Audi 5000S Diesel): systematically under-predicted due to diesel's inherent efficiency advantage
- **Rotary engine vehicles** (Mazda RX3): over-predicted due to rotary engines' lower efficiency
- **Unusual variants** (Ford Mustang II performance variant): unexplained by weight/year alone

### 2. ELPD Difference Not Significant

Despite clear evidence for heavy tails, the predictive improvement is modest:

| Model | ELPD | SE |
|-------|------|----|
| A2-Year (Normal) | 279.7 | 17.3 |
| A3-Robust (Student-t) | 286.1 | 17.5 |
| **Difference** | +6.4 | ~24.6 (combined) |

The z-score of 0.26 is far below 2, meaning the difference is not statistically significant. The outliers affect tail behavior but not core prediction accuracy.

### 3. Residuals by Origin: Minimal Structure

The residuals-by-origin plot shows mean residuals very close to zero for all three origins:
- USA (n=245): mean = 0.003
- Europe (n=68): mean = -0.003
- Japan (n=79): mean = 0.001

The confidence intervals all overlap zero substantially. This suggests that weight and year capture the dominant effects, and origin adds little conditional on these variables. However, Europe shows slightly wider residual spread, which may reflect greater heterogeneity in European vehicle types.

### 4. Residuals by Year: Slight Late-Period Positive Bias

Year-by-year residuals hover around zero through most of the range, but years 1980-1982 show a small positive bias (model under-predicts MPG). This coincides with:
- The second oil crisis (1979) accelerating efficiency improvements
- Introduction of more fuel-efficient designs
- Possible non-linearity in the year effect

The linear year term may not fully capture accelerating efficiency gains in the early 1980s.

## A2 vs A3: Final Comparison

| Criterion | A2-Year | A3-Robust | Winner |
|-----------|---------|-----------|--------|
| ELPD | 279.7 | 286.1 | A3 (marginal) |
| Parameters | 4 | 5 | A2 |
| sigma | 0.118 | 0.100 | A3 |
| Tail handling | Fixed | Adaptive | A3 |
| Complexity | Simple | Moderate | A2 |
| Predictive equivalence | - | z=0.26 | Tie |

**Verdict**: A2-Year is the preferred Class A model for parsimony. The Student-t extension is scientifically justified (nu ~ 7 confirms outliers exist) but provides no meaningful predictive improvement. For inference tasks where outlier influence is a concern, A3-Robust provides insurance at negligible computational cost.

## Should We Proceed to Class B/C?

**Yes, proceed to Class B (origin effects).**

Rationale:
1. The Class A models (A2 and A3) achieve good fit but were designed to test whether origin effects are unnecessary. The residual analysis shows origin differences are small *conditional on weight and year*, but this does not prove origin is uninformative.

2. The experiment plan's expected outcome was that "Class A will underperform due to missing origin effects." The A2/A3 ELPD of ~280-286 provides a baseline. Class B models (B1-Additive, B2-Interaction) will reveal whether explicit origin terms improve prediction.

3. Origin may matter through mechanisms not captured by weight alone:
   - Different engine technologies (Japanese inline-4 vs American V8)
   - Different design philosophies (European performance vs Japanese economy)
   - Regulatory differences across markets

4. The EDA-recommended model is B1 (Full Additive). Testing this is essential to determine if the simpler physical models suffice.

**Class C (hierarchical) is lower priority.** With only 3 groups (USA, Europe, Japan) and large sample sizes per group (245, 68, 79), partial pooling offers limited benefit. Test Class B first; consider Class C only if origin effects prove important.

## Remaining Model Deficiencies

1. **Fuel type not modeled**: Diesel vehicles are systematically under-predicted. The dataset likely contains few diesels, making this a minor issue, but a fuel_type indicator would improve fit for those observations.

2. **Engine type not modeled**: Rotary engines (Mazda RX3) behave differently from reciprocating engines. Again, likely few observations affected.

3. **Possible year non-linearity**: The late-period positive residual bias suggests the linear year term may under-capture accelerating efficiency gains after 1979. A quadratic term or change-point model could address this, but the effect is small.

4. **Displacement/cylinders omitted**: These are strongly correlated with weight but contain some independent information. The physical model treats weight as a sufficient summary, which is a reasonable simplification.

## Files Generated

| File | Description |
|------|-------------|
| `critique_report.md` | This assessment |
| `class_a_summary.json` | Machine-readable Class A comparison |

## Final Class A Recommendations

1. **Representative model**: A2-Year (simpler, equivalent predictive performance)
2. **Robust alternative**: A3-Robust (use when outlier influence is a concern)
3. **Next step**: Proceed to Class B to test origin effects
4. **Skip**: Class C hierarchical models (unlikely to add value with 3 large groups)
