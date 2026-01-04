# Posterior Predictive Check: A1-Baseline Model

## Model Summary

**Model**: log(mpg) ~ log(weight) only
**Observations**: 392 (after removing 6 rows with missing horsepower)
**Posterior draws**: 4,000 (4 chains x 1,000 draws)

**Posterior parameter estimates**:
- alpha (intercept): 3.098
- beta_weight (elasticity): -1.057
- sigma (residual SD): 0.166

## Overall Assessment

**VERDICT: POOR FIT** - The model exhibits systematic misfit that is substantively important.

The weight-only model captures the marginal distribution of log(MPG) reasonably well, and standard test statistics pass posterior predictive checks. However, residual analysis reveals a critical deficiency: **strong temporal trends in residuals indicate the model is missing important structure**.

## Key Findings

### 1. Marginal Distribution: Adequate

The posterior predictive distribution matches the observed distribution of log(MPG) well. The ECDF comparison shows the observed data falls within the 95% predictive interval across the full range. All test statistic p-values are well-calibrated (0.23 to 0.87), indicating the model captures location, spread, and shape of the marginal distribution.

### 2. Residuals vs Fitted: Acceptable

The residuals-vs-fitted plot shows no obvious heteroscedasticity. The scale-location plot confirms roughly constant variance across the range of fitted values. This validates the assumption of constant sigma on the log scale.

### 3. Residuals vs Year: SYSTEMATIC MISFIT

This is the critical finding. Residuals show a strong monotonic trend by model year:

| Year | n | Mean Residual | t-statistic |
|------|---|---------------|-------------|
| 70 | 29 | -0.132 | -5.9 |
| 71 | 27 | -0.098 | -5.8 |
| 72 | 28 | -0.127 | -7.2 |
| 73 | 40 | -0.150 | -7.6 |
| 74 | 26 | -0.065 | -3.8 |
| 75 | 30 | -0.037 | -1.6 |
| 76 | 34 | -0.027 | -1.3 |
| 77 | 28 | +0.015 | +1.0 |
| 78 | 36 | +0.021 | +1.3 |
| 79 | 29 | +0.125 | +6.0 |
| 80 | 27 | +0.219 | +7.0 |
| 81 | 28 | +0.138 | +5.8 |
| 82 | 30 | +0.173 | +7.8 |

The pattern is unambiguous: early-year cars (1970-1973) have **negative residuals** (observed MPG lower than predicted), while late-year cars (1979-1982) have **positive residuals** (observed MPG higher than predicted). The effect size is substantial - about 0.35 log-units difference between early and late years, corresponding to approximately 40% improvement in MPG at equivalent weight.

This reflects the historical reality: the 1973 oil crisis and subsequent CAFE standards drove substantial improvements in fuel efficiency independent of weight reduction. A weight-only model cannot capture this temporal trend.

### 4. Residuals vs Origin: No Systematic Misfit

Unlike year, residuals by car origin show no systematic pattern:

| Origin | n | Mean Residual | t-statistic |
|--------|---|---------------|-------------|
| USA | 245 | -0.006 | -0.7 |
| Europe | 68 | +0.001 | +0.1 |
| Japan | 79 | +0.019 | +1.0 |

All t-statistics are small (<1), indicating origin is not a confounded predictor in this model. The weight variable adequately accounts for origin-related differences in MPG.

### 5. PIT Calibration: Minor Concerns

The PIT histogram shows some deviation from uniformity, with excess density around 0.4-0.5 and a deficit near 0.9. This suggests the model's predictive intervals are not perfectly calibrated - some observations are more predictable than the model implies. This is consistent with the missing year effect: within-year predictions are more precise than the aggregate model suggests.

## Implications

The strong year effect has important scientific implications:

1. **Causal interpretation**: Weight alone does not explain MPG variation. Technological progress (engine efficiency, aerodynamics, fuel injection) contributes independently.

2. **Prediction quality**: Using this model to predict MPG for a new car would be biased. A 1982 car would have its MPG underpredicted by ~0.17 log-units (about 19% on original scale).

3. **Model comparison**: Any model comparison should account for year. The baseline model serves as a useful reference but is clearly deficient for substantive analysis.

## Recommendations

1. **Include year as predictor**: Add model_year (linear or categorical) to capture temporal trends in fuel efficiency.

2. **Consider structural interpretation**: The year effect likely proxies for technological improvements. If the goal is understanding physical relationships, year captures confounded effects.

3. **Origin may still matter**: Although residuals-by-origin show no pattern, this could change once year is included. European and Japanese cars may have been early adopters of efficiency technology.

## Output Files

- `ppc_density_overlay.png` - Posterior predictive density comparison
- `ppc_ecdf.png` - ECDF comparison with uncertainty bands
- `residuals_vs_fitted.png` - Residuals and scale-location plots
- `residuals_vs_omitted.png` - Residuals by year and origin
- `ppc_test_statistics.png` - Test statistic posterior predictive distributions
- `loo_pit.png` - PIT calibration diagnostics
- `ppc_results.json` - Numerical results
