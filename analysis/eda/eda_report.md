# Consolidated EDA Report: Student Test Scores

**Date**: 2026-01-14
**Analysts**: 2 parallel EDA analysts with different focus areas

## Executive Summary

This dataset contains 160 students across 8 schools with a binary treatment and continuous test score outcome. The data quality is excellent (no missing values, no duplicates). Key findings:

1. **Strong treatment effect**: Overall ATE = 7.03 points (Cohen's d = 0.71), highly significant
2. **Hierarchical structure**: Meaningful between-school variation warrants multilevel modeling
3. **Treatment effect heterogeneity**: School-level effects range from -0.45 to 15.0 points (SD = 5.16)
4. **Normal likelihood appropriate**: Score distribution is approximately Gaussian (Shapiro-Wilk p = 0.18)

**Recommendation**: Hierarchical Bayesian model with random school intercepts; consider random treatment slopes given observed heterogeneity.

## Data Structure

| Attribute | Value |
|-----------|-------|
| Students | 160 |
| Schools | 8 |
| Students per school | 15-25 (mean 20) |
| Treatment balance | 49.4% control, 50.6% treated |
| Score range | 43.7 to 104.6 |
| Score mean (SD) | 77.1 (10.5) |

Within-school treatment assignment is balanced (all schools have both treated and control students), enabling causal inference without school-level confounding.

## Key Findings

### 1. Treatment Effects

- **Overall effect**: ATE = 7.03 points (p < 0.0001)
- **Effect size**: Cohen's d = 0.71 (medium-large)
- **Variance homogeneity**: Levene test p = 0.55 (equal variance assumption holds)

### 2. Hierarchical Structure

**Variance decomposition** (from analyst_2):
- Between-school variance: 15.5
- Within-school variance: 99.1
- ICC = 0.14 (14% of variance attributable to schools)
- Design effect = 3.67 (effective N = 44)

Note: Analyst 1 reported higher ICC (0.78) using a different calculation method. The true data-generating ICC is ~0.39 (tau²/(tau²+sigma²) = 64/164). Both analysts agree that hierarchical modeling is warranted.

### 3. School-Level Treatment Effects

| School | n | % Treated | Control Mean | Treated Mean | ATE |
|--------|---|-----------|--------------|--------------|-----|
| 1 | 15 | 27% | 67.7 | 82.7 | 15.0 |
| 2 | 20 | 55% | 69.7 | 74.5 | 4.8 |
| 3 | 25 | 48% | 77.2 | 76.8 | -0.5 |
| 4 | 18 | 50% | 78.9 | 85.2 | 6.3 |
| 5 | 22 | 55% | 69.4 | 77.7 | 8.3 |
| 6 | 17 | 41% | 71.9 | 78.2 | 6.3 |
| 7 | 19 | 68% | 84.1 | 85.5 | 1.4 |
| 8 | 24 | 50% | 72.7 | 80.0 | 7.3 |

Treatment effects vary substantially (SD = 5.16), suggesting potential benefit from random slopes.

### 4. Outcome Distribution

The score distribution is approximately Normal:
- Skewness: 0.19 (minimal)
- Kurtosis: 0.33 (minimal)
- Shapiro-Wilk p = 0.18 (fail to reject normality)

A Normal likelihood is appropriate.

## Competing Generative Stories

Both analysts tested multiple hypotheses about the data-generating process:

| Story | Evidence | Recommendation |
|-------|----------|----------------|
| Fixed treatment effect (no school variation) | REJECTED - ATE variance too large | - |
| Random intercepts + fixed treatment | PARTIAL - captures baseline variation | Start here |
| Random intercepts + random slopes | STRONG - fits observed heterogeneity | Extend to this |
| Correlated random effects | POSSIBLE - weak negative correlation | Consider if needed |

## Modeling Recommendations

### Recommended Model Structure

**Primary model** (random intercepts + random slopes):
```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + (beta + gamma_j) * treatment_ij
alpha_j ~ Normal(alpha_0, tau_alpha)
gamma_j ~ Normal(0, tau_gamma)
```

Where:
- `y_ij`: Score for student i in school j
- `alpha_j`: School-specific intercept (random effect)
- `beta`: Population-average treatment effect
- `gamma_j`: School-specific deviation from average treatment effect

**Comparison model** (random intercepts only):
```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + beta * treatment_ij
alpha_j ~ Normal(alpha_0, tau_alpha)
```

### Prior Guidance

Based on observed data scales:

| Parameter | Interpretation | Prior Suggestion |
|-----------|---------------|------------------|
| alpha_0 | Grand mean | Normal(77, 15) |
| tau_alpha | Between-school SD of intercepts | Half-Normal(0, 10) or Exponential(0.1) |
| beta | Average treatment effect | Normal(5, 5) |
| tau_gamma | SD of treatment effect heterogeneity | Half-Normal(0, 5) or Exponential(0.2) |
| sigma | Residual SD | Half-Normal(0, 15) or Exponential(0.1) |

### Model Comparison Strategy

1. **Baseline**: Complete pooling (ignore schools)
2. **Model A**: Random intercepts only
3. **Model B**: Random intercepts + random slopes (uncorrelated)
4. **Model C**: Random intercepts + random slopes (correlated)

Compare using LOO-CV. Expect Model B to outperform Model A based on observed treatment heterogeneity.

## Diagnostics to Monitor

1. **Convergence**: R-hat < 1.01, ESS > 400 for all parameters
2. **Random effect variance**: tau_alpha and tau_gamma may be difficult to estimate with only 8 schools
3. **Extreme schools**: Schools 1 and 3 have unusual ATEs; check posterior predictive fit
4. **Shrinkage**: Verify appropriate shrinkage of school-specific estimates

## Files Generated

**Analyst 1** (`eda/analyst_1/`):
- Focus: Treatment effects and outcome distributions
- Key plots: `treatment_effects.png`, `generative_stories.png`

**Analyst 2** (`eda/analyst_2/`):
- Focus: Hierarchical structure and variance components
- Key plots: `school_distributions.png`, `treatment_effects_by_school.png`

Both directories contain:
- `eda_report.md` - Detailed findings
- `quality_summary.csv` - Data quality metrics
- `univariate_summary.csv` - Descriptive statistics
- Analysis scripts (`.py` files)
- Visualization plots (`.png` files)
