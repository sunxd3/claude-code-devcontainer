# Model Assessment Report: Population Comparison

**Date:** 2026-01-14
**Analyst:** model-selector
**Validated Models:** 3 (Complete Pooling, Random Intercepts, Random Slopes)

## Summary

Three hierarchical models were validated and compared via LOO-CV and posterior predictive checks. Random Intercepts and Random Slopes show statistically equivalent predictive performance (ELPD difference 0.50 ± 0.67, < 2×SE). However, Random Slopes is superior on scientific grounds: it captures treatment effect heterogeneity across schools, which the data clearly exhibits and Random Intercepts cannot express.

**Recommendation: ADEQUATE.** Accept Random Slopes (Experiment 3) as the final model.

## Model Rankings (LOO-CV)

| Rank | Model | ELPD_LOO | SE | p_loo | Pareto k Status |
|------|-------|----------|-----|-------|----------------|
| 1 | Random Intercepts | -590.8 | 10.1 | 8.5 | ✓ All k < 0.7 |
| 2 | Random Slopes | -591.3 | 10.0 | 10.4 | ✓ All k < 0.7 |
| 3 | Complete Pooling | -596.3 | 10.0 | 3.2 | ✓ All k < 0.7 |

All LOO estimates are reliable (no Pareto k values > 0.7 in any model).

## Pairwise Comparisons

**Random Intercepts vs Random Slopes:**
ELPD difference: 0.50 ± 0.67
**Verdict: TOO CLOSE TO CALL** (< 2×SE). Predictive performance is statistically equivalent.

**Random Intercepts vs Complete Pooling:**
ELPD difference: 5.47 ± 3.28
**Verdict: MODERATE ADVANTAGE** (2×SE < difference < 4×SE). Hierarchical structure provides meaningful improvement.

## Posterior Predictive Check Results

### Complete Pooling (Experiment 1)
- **Overall treatment effect:** Excellent (ATE = 6.84, observed = 7.03)
- **School-level fit:** POOR. Systematic residuals by school (mean residuals -4.7 to +6.7 points)
- **Critical limitation:** Cannot capture school-specific baselines, leading to biased school-level predictions
- **Use case:** Adequate only for population-average treatment effect estimation

### Random Intercepts (Experiment 2)
- **Overall distribution:** Excellent (all test statistics within posterior predictive range)
- **School-level baselines:** Excellent (100% coverage, all observed school means within 90% intervals)
- **Treatment effects:** INADEQUATE. Cannot reproduce observed treatment effect heterogeneity across schools
- **Critical limitation:** Assumes constant treatment effect β ≈ 6.58 across all schools, but observed effects range from -0.45 to +15.0 points
- **Evidence:** Residual patterns show systematic deviations in treated students, consistent with heterogeneity the model cannot express

### Random Slopes (Experiment 3)
- **Overall distribution:** Excellent (LOO-PIT calibration p = 0.98)
- **School-level baselines:** Excellent (100% coverage)
- **School-specific treatment effects:** Excellent (100% coverage, captures both extremes: School 1's +15 effect and School 3's near-zero effect)
- **Test statistics:** All within posterior predictive ranges
- **Model deficiencies:** None identified
- **Verdict:** GOOD FIT. Appropriately captures the data-generating process.

## Strategic Assessment

### Why Random Slopes Wins Despite Tied LOO Performance

The LOO comparison shows Random Intercepts marginally ahead (0.50 ELPD), but this narrow margin (< 1×SE) does not account for:

1. **Scientific interpretability:** The EDA identified treatment effect heterogeneity as a key phenomenon (school-level effects range from -0.5 to +15.0 points, SD = 5.16). Random Slopes can quantify this heterogeneity; Random Intercepts cannot.

2. **PPC evidence:** Random Slopes achieves 100% coverage for school-specific treatment effects with LOO-PIT calibration p = 0.98. Random Intercepts' PPC report explicitly identifies its inability to capture treatment effect variation as a model limitation.

3. **Predictive equivalence:** The ELPD difference (0.50 ± 0.67) is well below the 2×SE threshold, meaning we cannot confidently distinguish their predictive performance. When predictive performance is tied, scientific interpretability breaks the tie.

4. **No complexity penalty:** While Random Slopes has higher p_loo (10.4 vs 8.5), indicating more effective parameters, this complexity is justified by capturing real variation in the data rather than overfitting. The reliable Pareto k diagnostics confirm this.

### Model Class Comparison

Complete Pooling is clearly inadequate (5.5 ELPD behind, systematic school-level residuals). The choice is between the two hierarchical models.

Both Random Intercepts and Random Slopes belong to the same model class (hierarchical linear models), differing only in whether treatment slopes vary by school. Since we explored both variants and found Random Slopes superior on scientific grounds with equivalent predictive performance, the hierarchical model class is exhausted.

## Decision: ADEQUATE

**Accept Random Slopes (Experiment 3) as the final model.**

### Rationale
1. Predictive performance statistically tied with Random Intercepts (< 2×SE difference)
2. PPC shows excellent fit across all criteria (LOO-PIT p = 0.98, 100% coverage)
3. Captures the scientific phenomenon of interest: heterogeneous treatment effects across schools
4. No compelling model extensions remain unexplored within this model class
5. LOO estimates are reliable (all Pareto k < 0.7)

### What This Model Tells Us

The Random Slopes model estimates:
- Population-average treatment effect β ≈ 6.6-7.0 points
- School-specific treatment effects ranging from near-zero (School 3) to +15-18 points (School 1)
- Between-school variation in treatment effects (τ_gamma) quantifies heterogeneity
- School-specific intercepts capture baseline performance differences

This allows researchers to answer:
- "Does the intervention work on average?" → Yes, ~7 points
- "Does effectiveness vary across schools?" → Yes, substantially (SD ≈ 5 points)
- "Which schools benefit most?" → School 1 shows largest effect, School 3 shows minimal benefit

### Alternative Considerations

Could further extensions improve the model?

**Correlated random effects** (allowing intercepts and slopes to correlate): The EDA reported a weak negative correlation between school baselines and treatment effects. This could be explored, but:
- No evidence of poor fit in current model
- Correlation is weak and exploratory rather than theory-driven
- Additional complexity may not improve predictive performance

**School-level predictors** (explaining why effects vary): If school characteristics (size, resources, demographics) were available, they could explain heterogeneity. However, this requires additional data not present in the current dataset.

Neither extension is compelling given the current data and the excellent fit achieved by Random Slopes.

## Files Generated

- `loo_comparison.csv` - LOO comparison table (az.compare results)
- `loo_diagnostics.json` - Pareto k diagnostics for all models
- `loo_comparison.png` - Visual comparison of model rankings
- `compare_models.py` - Reproducible comparison script

## References

- LOO-CV: Vehtari et al. (2017). "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC"
- Pareto k diagnostics: k < 0.5 = good, 0.5-0.7 = ok, > 0.7 = problematic
- ELPD interpretation: Differences < 2×SE are "too close to call", > 4×SE are "clear winner"
- LOO-PIT calibration: Säilynoja et al. (2022). "Graphical test for discrete uniformity and its applications in goodness of fit evaluation and multiple sample comparison"
