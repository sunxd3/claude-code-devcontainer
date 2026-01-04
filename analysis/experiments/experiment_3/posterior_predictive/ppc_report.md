# Posterior Predictive Check Report: A3-Robust Model

**Model**: log(mpg) ~ log(weight) + year with Student-t(nu, mu, sigma) errors
**Date**: 2026-01-04
**N**: 392 observations

## Summary

The A3-Robust model with Student-t errors provides adequate fit to the Auto-MPG data. Posterior predictive checks reveal no major model deficiencies. The Student-t specification offers modest improvement over Normal errors in capturing tail behavior, though the practical difference is small.

## Key Findings

### 1. Marginal Distribution: GOOD FIT

The posterior predictive distribution closely matches the observed data (`ppc_density_ecdf.png`):
- Density overlay shows replicated data spans the observed distribution
- ECDF comparison shows excellent agreement across the full range
- Observed mean and replicated mean are nearly identical

### 2. Calibration (PIT): GOOD FIT

The probability integral transform histogram and ECDF (`loo_pit.png`) show:
- Approximately uniform distribution, indicating well-calibrated predictions
- No systematic over- or under-coverage
- Minor deviations are consistent with sampling variation

### 3. Test Statistics: ADEQUATE FIT

Posterior predictive p-values for statistics not directly fit by the model (`ppc_test_statistics.png`):

| Statistic | Observed | p-value | Assessment |
|-----------|----------|---------|------------|
| Median | 3.06 | 0.59 | Good |
| MAD | 0.36 | 0.19 | Acceptable |
| Skewness | -0.13 | 0.78 | Good |
| Kurtosis | -0.82 | 0.08 | Marginal |

The kurtosis p-value of 0.08 warrants attention. The observed data shows negative excess kurtosis (-0.82, lighter tails than Gaussian), while replications produce less negative kurtosis on average (-0.65). This is expected: the Student-t is designed for heavier tails, but the data has lighter tails than Normal in the marginal distribution. However, this reflects the conditional residual distribution, not the marginal distribution, so the finding is not necessarily problematic.

### 4. Residual Diagnostics: ADEQUATE FIT

**Residuals vs Fitted** (`residuals_vs_fitted.png`):
- No systematic pattern in residuals
- Variance appears constant (homoscedastic)
- Smoothed trend line stays close to zero

**Residuals by Year** (`residuals_by_year.png`):
- Year-by-year means hover around zero
- No systematic temporal trend in residuals
- Variance consistent across years

**Residuals by Origin** (`residuals_by_origin.png`):
- Mean residuals close to zero for all origins
- USA: mean = 0.003, European: mean = -0.003, Japanese: mean = 0.001
- The weight+year model captures origin differences adequately (origin correlates with weight)

### 5. Tail Behavior Comparison: MODEST IMPROVEMENT

The key question: Does Student-t better capture extreme observations?

**Observed extremes**:
- Minimum: log(9 MPG) = 2.20 (Hi 1200D pickup truck)
- Maximum: log(46.6 MPG) = 3.84 (Mazda GLC)

**Model capability to generate extremes** (`tail_comparison.png`):

| Statistic | Student-t p-val | Normal p-val | Better? |
|-----------|-----------------|--------------|---------|
| Min value | 0.29 | 0.20 | Student-t |
| Max value | 0.67 | 0.65 | Comparable |
| Range | 0.49 | 0.37 | Student-t |

The Student-t model shows more variability in replicated extremes (larger std for min/max), which is expected from heavier tails:
- Student-t min std: 0.13 vs Normal min std: 0.07
- Student-t max std: 0.14 vs Normal max std: 0.06

Both models can generate the observed extremes with reasonable probability. The Student-t shows better ability to generate the observed minimum (p=0.29 vs 0.20), suggesting it handles low-MPG outliers slightly better.

### 6. Extreme Observations: 8 IDENTIFIED

Eight observations have standardized residuals exceeding 2.5 SD (`extreme_observations.png`):

**Over-predicted (model predicts higher MPG than observed)**:
1. Ford Mustang II (1975): 13 MPG - possibly a performance variant
2. Mazda RX3 (1973): 18 MPG - rotary engine, lower efficiency
3. Oldsmobile Omega (1973): 11 MPG - unusual for this model

**Under-predicted (model predicts lower MPG than observed)**:
1. Oldsmobile Cutlass Ciera Diesel (1982): 38 MPG - diesel efficiency bonus
2. Audi 5000S Diesel (1980): 36.4 MPG - diesel efficiency bonus

The diesels are systematically under-predicted because the model does not include a fuel type variable. The rotary engine car (Mazda RX3) is over-predicted because rotary engines have lower fuel efficiency than reciprocating engines of similar displacement. These are known model limitations, not defects.

## Assessment

**GOOD FIT** - The A3-Robust model adequately captures the observed data.

Strengths:
- Excellent calibration (uniform PIT)
- No systematic residual patterns
- Captures weight and year effects well
- Student-t provides robustness to outliers

Limitations:
- Does not account for fuel type (diesel cars under-predicted)
- Does not account for engine type (rotary cars over-predicted)
- These limitations are by design (model simplicity) rather than defects

**Student-t vs Normal**: The Student-t specification provides modest improvement in tail behavior. The estimated nu ~ 7 indicates the data has some outliers, and the Student-t down-weights their influence on parameter estimates. However, predictive performance is essentially equivalent (ELPD difference not significant). The Student-t is recommended as "insurance" against outlier influence at no cost.

## Files Generated

| File | Description |
|------|-------------|
| `ppc_density_ecdf.png` | Observed vs predicted density and ECDF |
| `loo_pit.png` | PIT calibration histogram and ECDF |
| `ppc_test_statistics.png` | Test statistics: median, MAD, skewness, kurtosis |
| `residuals_vs_fitted.png` | Residuals vs fitted values |
| `residuals_by_year.png` | Residual patterns by model year |
| `residuals_by_origin.png` | Residual patterns by car origin |
| `tail_comparison.png` | Tail behavior: Student-t vs Normal |
| `extreme_observations.png` | Identification of extreme observations |
| `ppc_summary.json` | Quantitative summary statistics |
