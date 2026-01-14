# Posterior Predictive Check Assessment: Complete Pooling Model

**Model**: Complete Pooling (single global intercept + treatment effect)

**Date**: 2026-01-14

## Summary

The Complete Pooling model captures the overall treatment effect and marginal score distribution adequately but exhibits systematic residual patterns across schools. This indicates a critical model deficiency: ignoring school-level heterogeneity leads to biased predictions for individual schools. While the model is acceptable for estimating the average treatment effect, it is inappropriate for making school-specific inferences or predictions.

## Assessment by Component

### 1. Overall Distribution

**Figures**: `distribution_checks.png`

The ECDF and KDE comparisons show that replicated data envelopes the observed distribution reasonably well. The model can reproduce the overall shape, location, and spread of test scores.

**Finding**: ADEQUATE. The marginal distribution is well-captured.

### 2. Calibration (LOO-PIT)

**Figure**: `loo_pit_calibration.png`

The LOO-PIT ECDF remains mostly within the 94% credible interval, indicating that the leave-one-out predictive distributions are reasonably calibrated. No strong systematic deviations suggest the model provides appropriately uncertain predictions on average.

**Finding**: ADEQUATE. Predictive calibration is acceptable at the population level.

### 3. Test Statistics

**Figure**: `test_statistics.png`

| Statistic | Observed | P-value | Assessment |
|-----------|----------|---------|------------|
| Median | 76.97 | 0.964 | Excellent |
| MAD | 5.93 | 0.072 | Marginal |
| IQR | 12.48 | 0.175 | Good |
| Minimum | 43.70 | 0.232 | Good |
| Maximum | 104.60 | 0.902 | Good |

The median, IQR, and extremes are well-reproduced. The MAD (median absolute deviation) shows a marginal p-value of 0.072, with the observed MAD falling in the lower tail of the predictive distribution. This suggests the model may slightly overestimate within-group variability, though not severely.

**Finding**: ADEQUATE overall, with a minor tendency to overestimate dispersion.

### 4. School-Level Residuals (CRITICAL)

**Figure**: `school_level_residuals.png`

**Mean residuals by school**:

| School ID | Mean Residual | RMSE | N Students | Mean Score |
|-----------|---------------|------|------------|------------|
| 1 | +0.30 | 8.1 | 15 | 75.7 |
| 2 | -4.71 | 9.3 | 20 | 72.4 |
| 3 | +0.42 | 9.7 | 25 | 77.3 |
| 4 | +4.96 | 10.0 | 18 | 82.0 |
| 5 | -3.71 | 10.4 | 22 | 73.9 |
| 6 | -2.80 | 8.6 | 17 | 74.5 |
| 7 | +6.71 | 11.4 | 19 | 85.0 |
| 8 | -0.38 | 10.3 | 24 | 76.4 |

**Critical observations**:

1. **Systematic bias**: Five schools (2, 4, 5, 6, 7) exhibit mean residuals with |residual| > 2, indicating systematic under- or over-prediction.

2. **Large biases**: School 7 is under-predicted by 6.7 points on average (actual mean: 85.0, model struggles to predict this high baseline). School 4 is under-predicted by 5.0 points. Schools 2 and 5 are over-predicted by 4.7 and 3.7 points respectively.

3. **Pattern**: The boxplots show that residuals cluster by school rather than centering uniformly around zero. This is direct evidence that schools differ in baseline score beyond what treatment assignment can explain.

4. **RMSE variation**: RMSE ranges from 8.1 to 11.4 across schools, with higher-performing schools (4, 7) showing larger prediction errors.

5. **Q-Q plot**: Residuals follow a roughly normal distribution (good for model assumptions), but the school-level clustering indicates structural misspecification.

**Finding**: POOR FIT at the school level. The complete pooling assumption is violated. Schools have distinct baseline scores that the model cannot capture with a single global intercept.

### 5. Treatment Effect

**Figure**: `treatment_effect.png`

- **Posterior mean**: 6.84 points
- **Observed effect**: 7.03 points
- **Agreement**: Excellent

The model accurately captures the average treatment effect. The observed effect falls near the center of the posterior distribution, indicating the model's primary inferential target (treatment effect) is well-estimated.

**Treatment group residuals**:
- Control: mean residual = -0.11 (SD = 9.68)
- Treatment: mean residual = +0.10 (SD = 10.13)

No systematic bias by treatment group, confirming that pooling across treatment is appropriate.

**Finding**: EXCELLENT for treatment effect estimation.

## Overall Assessment

### What the Model Does Well

1. **Treatment effect estimation**: The model accurately estimates the average treatment effect across all schools.
2. **Marginal distribution**: Can reproduce overall score distribution, central tendency, and range.
3. **Population-level calibration**: Predictive distributions are appropriately uncertain on average.

### Critical Limitation

**School-level heterogeneity is ignored**. By using a single global intercept, the model assumes all schools have the same baseline score (after accounting for treatment). This assumption is clearly violated:

- Schools differ substantially in baseline performance (observed means range from 72.4 to 85.0)
- Systematic residual patterns by school indicate structural misfit
- The model cannot make valid school-specific predictions or inferences

### Substantive Implications

**For population-level questions**: "What is the average treatment effect across schools?" - this model is adequate.

**For school-level questions**: "What is the treatment effect in School 7?" or "Which schools would benefit most from the intervention?" - this model is inappropriate. The systematic biases mean school-specific estimates would be unreliable.

**For prediction**: If asked to predict a new student's score, knowing their school provides important information that this model discards. Predictions will be biased for students from high- or low-performing schools.

## Recommended Next Steps

1. **Hierarchical model with school-specific intercepts**: Allow each school to have its own baseline score, with schools drawn from a common distribution. This would eliminate the systematic residual patterns while maintaining partial pooling benefits.

2. **Consider school-specific treatment effects**: If treatment effectiveness varies by school characteristics (e.g., baseline performance), a hierarchical model with varying slopes would be appropriate.

3. **Investigate school characteristics**: If available, school-level predictors (size, resources, demographics) could help explain the observed heterogeneity.

## Technical Details

- **Observations**: 160 students across 8 schools
- **Posterior draws**: 4,000 (4 chains × 1,000 draws each)
- **Replications analyzed**: 50 randomly sampled for visualization
- **Diagnostic code**: `run_checks.py`

## Conclusion

The Complete Pooling model is **adequate for estimating the average treatment effect** but **inadequate for capturing school-level variation**. The systematic residual patterns by school constitute a model deficiency that must be addressed. A hierarchical model allowing school-specific intercepts is strongly recommended for the next iteration.
