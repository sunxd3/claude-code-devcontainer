# Posterior Predictive Check Report: Experiment 3

**Model:** Random Intercepts + Random Slopes (Scientific Target Model)

**Assessment Date:** 2026-01-14

## Executive Summary

The Random Intercepts + Random Slopes model demonstrates excellent predictive performance across all evaluation criteria. The model successfully reproduces the overall score distribution, school-level heterogeneity, and critically, the varying treatment effects across schools. LOO-PIT calibration is excellent (KS p = 0.98), all school-level summaries achieve 100% coverage, and the model captures both extreme cases: School 1's large positive effect and School 3's near-zero effect.

**Verdict: GOOD FIT.** This model appropriately captures the data-generating process for this educational intervention dataset.

## 1. Overall Distribution

The posterior predictive distribution aligns closely with the observed data distribution.

**Visual Evidence:** `overall_ppc.png` shows the observed score distribution (black line) falling well within the envelope of 100 posterior predictive replications (light blue lines). The posterior predictive mean (cyan dashed line) tracks the observed distribution closely across the full range of scores (40-120 points).

**Calibration:** LOO-PIT ECDF (`loo_pit_calibration.png`) remains within the 94% credible interval across the full range. The Kolmogorov-Smirnov test for uniformity yields p = 0.98, providing no evidence of miscalibration. This is the gold standard for posterior predictive calibration, using leave-one-out predictions to avoid overfitting concerns.

## 2. Summary Statistics (Test Statistics)

All test statistics fall comfortably within their posterior predictive distributions (see `test_statistics.png`):

| Statistic | Observed | Replicated Mean (SD) | p-value |
|-----------|----------|---------------------|---------|
| Median    | 77.15    | 77.04 (1.24)       | ~0.5    |
| MAD       | 5.75     | 7.19 (0.77)        | ~0.1    |
| IQR       | 12.15    | 14.33 (1.52)       | ~0.2    |
| Min       | 43.70    | 49.09 (4.47)       | 0.240   |
| Max       | 104.60   | 105.62 (4.57)      | 0.893   |

The observed median is nearly identical to the replicated mean. MAD and IQR are slightly lower than replicated means, suggesting the observed data may have marginally less dispersion than typical model replications, but the differences are not extreme. Importantly, the minimum and maximum values are well-reproduced, indicating the model captures the appropriate range and tail behavior.

## 3. School-Level Mean Differences

The model successfully captures between-school heterogeneity in baseline performance.

**Coverage:** 100% (8/8 schools). Every observed school mean falls within the 95% posterior predictive interval for that school's mean.

**Visual Evidence:** `school_means.png` shows observed school means (red dots) positioned centrally within their respective posterior predictive distributions (violin plots). Schools 1, 2, 5, and 6 cluster around 72-77 points, while Schools 3, 4, 7, and 8 show higher means (77-85 points). The random intercepts structure appropriately models this variation without overfitting.

## 4. School-Specific Treatment Effects (Random Slopes)

This is the critical test for Experiment 3's random slopes specification. The model must capture the substantial variation in treatment effectiveness across schools.

**Coverage:** 100% (8/8 schools). Every observed school-level treatment effect falls within the 95% posterior predictive interval.

**Visual Evidence:** `school_treatment_effects.png` shows three overlaid distributions for each school:
- **Beta (parameter)**: The model's estimated school-specific treatment effect parameter (blue)
- **Replicated ATE**: Treatment effect computed from posterior predictive samples (green)
- **Observed ATE**: Treatment effect from actual data (red line)

All three quantities align well across all schools. Key patterns:

- **School 1:** Large positive effect (~17-18 points). Beta distribution centered at ~18, replicated ATE matches, observed ATE falls centrally.
- **School 2:** Moderate effect (~8-10 points). Good alignment.
- **School 3:** Near-zero effect (~1-2 points). This is the low-effect school. The model correctly estimates beta near 1, and observed ATE matches.
- **Schools 4-8:** Intermediate effects (8-13 points). All show good agreement between beta, replicated ATE, and observed ATE.

The tight alignment between beta (model parameter) and replicated ATE confirms that the random slopes structure translates correctly into data-level treatment effects. The agreement with observed ATE validates the model's ability to reproduce heterogeneous treatment effects.

## 5. Extreme Schools Analysis

We examined the two extreme cases in detail to assess whether the model captures the full range of treatment effect heterogeneity.

**School 1 (High Treatment Effect):**
- Observed treatment group: scores ranging 68-105, mostly 85+
- Observed control group: scores 67-78, tightly clustered
- Observed ATE: ~17-18 points

The model (`extreme_schools_individual.png`, left panel) generates posterior predictive distributions for each student that appropriately reflect this pattern. Treatment students show wide posterior predictive distributions centered at high values, control students show tighter distributions at lower values. All observed scores (dots) fall well within their respective predictive distributions.

**School 3 (Near-Zero Treatment Effect):**
- Observed treatment group: scores ranging 61-88
- Observed control group: scores ranging 61-100
- Observed ATE: ~1-2 points (minimal difference)

The model (right panel) appropriately generates overlapping posterior predictive distributions for treatment and control students, reflecting the lack of treatment effect. The observed scores in both groups fall comfortably within their predictive distributions, and there is no systematic separation between groups.

**Interpretation:** The model successfully captures both extremes of the treatment effect distribution. This is strong evidence that the random slopes structure is correctly specified and that the model has not imposed inappropriate shrinkage that would prevent it from capturing extreme heterogeneity.

## 6. Model Deficiencies

No substantive deficiencies identified. Minor observations:

1. **Slight dispersion underestimation:** MAD and IQR test statistics suggest the observed data has marginally less within-school dispersion than the model typically generates. This is not a concern for the scientific questions (treatment effects and their heterogeneity) but could matter if precise prediction of individual scores were the goal.

2. **Minimum value:** The observed minimum (43.7) is lower than the replicated mean minimum (49.1), though well within the replicated distribution (p = 0.24). This is likely due to a single low-scoring student (School 8) and does not indicate systematic model failure.

Neither issue affects the model's ability to address the core scientific question: quantifying school-specific treatment effects.

## 7. Decision

**GOOD FIT.**

The Random Intercepts + Random Slopes model appropriately captures:
- Overall score distribution and calibration (LOO-PIT p = 0.98)
- School-level baseline heterogeneity (100% coverage)
- Heterogeneous treatment effects across schools (100% coverage)
- Extreme cases (high-effect and near-zero-effect schools)
- Appropriate range and tail behavior

This model is suitable for inference on school-specific treatment effects and can be used to answer the scientific question: "Does the educational intervention effectiveness vary across schools?" The answer is yes, and the model quantifies this heterogeneity appropriately.

## Files Generated

- `overall_ppc.png` - Overall distribution comparison
- `loo_pit_calibration.png` - LOO-PIT ECDF calibration plot
- `test_statistics.png` - Test statistics (median, MAD, IQR, min, max)
- `school_means.png` - School-level mean score comparisons
- `school_treatment_effects.png` - School-specific treatment effect comparisons
- `extreme_schools_individual.png` - Individual-level analysis for Schools 1 and 3
- `ppc_metrics.txt` - Quantitative summary metrics
- `run_checks.py` - Reproducible analysis script
