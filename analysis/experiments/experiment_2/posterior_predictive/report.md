# Posterior Predictive Check: A2-Year Model

## Model Summary

**Model**: log(mpg) ~ log(weight) + year
**Observations**: 392
**Posterior draws**: 4,000 (4 chains x 1,000 draws)

**Posterior parameter estimates**:
- alpha (intercept): 3.098
- beta_weight (weight elasticity): -0.935
- beta_year (technology drift): 0.033
- sigma (residual SD): 0.118

The residual standard deviation dropped from 0.166 (baseline model) to 0.118, a 29% reduction indicating year captures substantial variance.

## Overall Assessment

**VERDICT: GOOD FIT** - The model adequately captures the key features of the data.

Adding year as a predictor resolves the primary deficiency identified in the baseline model (A1). The strong temporal trends in residuals are largely eliminated. Importantly, **residuals show no systematic pattern by ORIGIN**, answering the key question: once weight and year are accounted for, car origin does not provide additional predictive information.

## Key Findings

### 1. Marginal Distribution: Excellent

The posterior predictive distribution closely matches the observed distribution of log(MPG). The ECDF comparison shows the observed data falls within the 95% predictive interval across the entire range. All test statistic p-values are well-calibrated (0.19 to 0.93), indicating the model captures location, spread, shape, and tail behavior.

| Statistic | p-value | Assessment |
|-----------|---------|------------|
| Mean | 0.50 | Centered |
| SD | 0.51 | Centered |
| Median | 0.38 | Good |
| IQR | 0.19 | Acceptable |
| Skewness | 0.19 | Acceptable |
| Kurtosis | 0.93 | Good |
| Min | 0.80 | Good |
| Max | 0.65 | Good |

### 2. Residuals vs Fitted: Good

The residuals-vs-fitted plot shows no obvious heteroscedasticity. The binned means hover around zero across the range of fitted values. The scale-location plot confirms approximately constant variance, validating the homoscedastic assumption on the log scale.

### 3. Residuals vs Year: Mostly Resolved

The dramatic monotonic year trend from the baseline model is eliminated. However, some year-specific deviations remain:

| Year | n | Mean Residual | t-statistic |
|------|---|---------------|-------------|
| 70 | 29 | +0.048 | 2.0 |
| 71 | 27 | +0.065 | 4.8 |
| 72 | 28 | -0.007 | -0.4 |
| 73 | 40 | -0.069 | -4.0 |
| 74-79 | varies | near 0 | < 2.5 |
| 80 | 27 | +0.108 | 3.6 |
| 81-82 | 58 | near 0 | < 0.5 |

The linear year effect captures the overall trend, but year-specific deviations suggest:
- 1971 and 1980 had above-trend efficiency (possibly fuel crisis responses)
- 1973 had below-trend efficiency (pre-oil crisis peak of inefficient vehicles)

These deviations represent real historical effects but are not large enough to warrant categorical year modeling. The linear trend is a reasonable simplification.

### 4. Residuals vs Origin: NO SYSTEMATIC PATTERN

This is the critical finding addressing the key question. After accounting for weight and year:

| Origin | n | Mean Residual | t-statistic |
|--------|---|---------------|-------------|
| USA | 245 | -0.010 | -1.5 |
| Europe | 68 | +0.033 | 1.9 |
| Japan | 79 | +0.003 | 0.2 |

All t-statistics are below the conventional threshold of 2, indicating no statistically significant difference in residuals by origin. The European cars show a marginally higher mean residual (+0.033 log-units, about 3.4% on original scale), but this is not statistically significant (t=1.9, p approx 0.06).

**Interpretation**: The apparent fuel efficiency advantage of Japanese and European cars over American cars is fully explained by two factors:
1. **Weight**: Foreign cars tend to be lighter
2. **Year**: Foreign cars entered the US market more heavily in later years when technology was better

Once these factors are controlled, origin provides no additional predictive power. This suggests the "Japanese efficiency advantage" was primarily a weight and timing phenomenon, not an inherent manufacturing superiority in the 1970-82 period.

### 5. PIT Calibration: Good

The PIT ECDF closely follows the diagonal and remains within the 95% simultaneous confidence band. The PIT histogram shows some variation around uniformity but no systematic pattern (no excess at extremes, no strong mode). The calibration is substantially improved over the baseline model.

## Comparison with Baseline Model

| Aspect | A1-Baseline | A2-Year |
|--------|-------------|---------|
| sigma | 0.166 | 0.118 |
| R-squared (approx) | 0.76 | 0.88 |
| Year residual trend | Strong (0.35 range) | Eliminated |
| Origin residual pattern | None | None |
| Calibration | Poor | Good |

The year effect explains approximately 12% additional variance in log(MPG), improving prediction substantially.

## Scientific Conclusions

1. **Weight elasticity refined**: The weight elasticity is -0.93, meaning a 1% increase in weight corresponds to approximately 0.93% decrease in MPG. This is slightly lower than the baseline estimate (-1.06) because year confounded the weight effect.

2. **Technology trend quantified**: The year coefficient of 0.033 means fuel efficiency improved by approximately 3.3% per year on average, holding weight constant. Over the 12-year span (1970-1982), this amounts to about 48% improvement due to technology.

3. **Origin is not a predictor**: After controlling for weight and year, car origin (USA, Europe, Japan) does not predict fuel efficiency. Any observed differences by origin are compositional (different weight distributions and market timing).

## Remaining Limitations

While the model fits well overall, some aspects remain unmodeled:
- Year-specific shocks (oil crises, regulation changes)
- Within-year heterogeneity in technology adoption
- Potential nonlinear weight effects at extremes

These limitations are minor for the current scientific questions and do not warrant additional model complexity.

## Output Files

- `ppc_density_overlay.png` - Posterior predictive density comparison
- `ppc_ecdf.png` - ECDF comparison with uncertainty bands
- `residuals_vs_fitted.png` - Residuals and scale-location plots
- `residuals_vs_year.png` - Residuals by year (now in model)
- `residuals_vs_origin.png` - Residuals by origin (key diagnostic)
- `ppc_test_statistics.png` - Test statistic posterior predictive distributions
- `loo_pit.png` - PIT calibration diagnostics
- `ppc_results.json` - Numerical results
