# EDA Report: Hierarchical Structure Analysis
## Student Test Scores Dataset

**Analyst**: analyst_2
**Focus**: School-level hierarchical structure and variance components
**Date**: 2026-01-14

## Executive Summary

This dataset exhibits moderate hierarchical clustering with 14.1% of variance attributable to between-school differences (ICC = 0.141). Treatment is assigned within schools, not at the school level, enabling estimation of within-school treatment effects. However, substantial heterogeneity exists across schools (treatment effects range from -0.45 to 15.0 points), suggesting hierarchical modeling with partial pooling is warranted. The design effect of 3.67 reduces the effective sample size from 160 to 44 students, highlighting the importance of accounting for clustering when estimating standard errors.

## Data Structure and Quality

The dataset contains 160 students nested within 8 schools. Data quality is excellent: no missing values, no duplicates, all values within plausible ranges (scores 43.7 to 104.6). Treatment is balanced overall (81 treated, 79 control) and varies within every school.

School sample sizes range from 15 to 25 students (mean 20.0, median 19.5, std 3.5), with coefficient of variation 0.173 indicating well-balanced cluster sizes. This balance is favorable for hierarchical modeling as it reduces the influence of small-sample schools and improves estimation efficiency.

## Variance Decomposition

Total score variance is 110.21, decomposing into between-school variance of 15.49 and within-school variance of 99.08. The intraclass correlation coefficient (ICC) is 0.1406, meaning 14.1% of variance is between schools and 85.9% is within schools.

This moderate ICC falls into the range where hierarchical modeling is strongly recommended. Ignoring school structure would bias standard errors and inflate Type I error rates. The design effect (DEFF = 3.67) quantifies this: ignoring clustering inflates precision by a factor of 1.92, or equivalently, reduces effective sample size to 44 students.

School membership alone (ignoring treatment) explains 14.1% of score variance (R² = 0.141). This substantial school-level variation suggests unobserved school characteristics (resources, teaching quality, student demographics) meaningfully influence outcomes.

As shown in `school_distributions.png`, school mean scores range from 72.4 (School 2) to 85.0 (School 7), a span of 12.7 points. The standard deviation of school means is 4.30, which is non-trivial relative to the grand mean of 77.1.

## Treatment Assignment and Within-School Variation

Treatment varies within all 8 schools, with the proportion treated ranging from 26.7% (School 1) to 68.4% (School 7). This within-school variation is critical: it enables estimation of treatment effects controlling for school-level confounders. Had treatment been assigned at the school level, treatment and school effects would be confounded.

The overall treatment effect (ignoring school structure) is 7.03 points (control mean 73.5, treated mean 80.6). However, within-school effects vary substantially. As shown in `school_treatment_effects.csv`, effects range from -0.45 (School 3) to 15.0 (School 1), with mean 7.18 and standard deviation 5.16. One school shows a negative effect, though its confidence interval likely includes zero.

This heterogeneity raises two modeling questions. First, is the variation in treatment effects due to sampling variability (small school samples) or genuine effect modification by school characteristics? A hierarchical model with partial pooling can regularize school-specific estimates and shrink extreme values toward the population mean. Second, should we model treatment-by-school interactions, or assume a common treatment effect with school-specific intercepts?

The plot `treatment_effects_by_school.png` shows school-specific effects with 95% confidence intervals. Most CIs overlap with the overall effect, but School 1 shows a notably larger effect (15.0) and School 3 shows near-zero effect. This pattern is consistent with sampling variability in small clusters, but could also reflect true effect heterogeneity.

## Visual Evidence

`school_distributions.png` displays box plots of scores by school, sorted by mean. School 7 has the highest mean, School 2 the lowest. Within-school variation is substantial in all schools, with most showing similar spreads. No obvious outliers or heavy tails are apparent.

`school_violin_plots.png` adds density information via violin plots. Distributions are roughly symmetric within schools, with no strong evidence of bimodality or unusual shapes. This supports a Normal likelihood for the outcome.

`school_means.png` plots school means with 95% confidence intervals. All CIs overlap, suggesting differences in school means may be within sampling variability, but the spread is non-trivial. Sample sizes are annotated above each point.

`treatment_by_school.png` shows side-by-side bars for control and treatment means within each school. Treatment groups score higher in 7 of 8 schools. School 3 is the exception, showing nearly identical means.

`treatment_scatter_by_school.png` displays individual student scores with school-specific colors, connecting control and treatment means with lines. The lines slope upward for most schools (positive treatment effects), but School 3's line is flat and School 1's line has the steepest slope. This visualization reveals that while the overall effect is positive, school-specific effects vary meaningfully.

## Implications for Modeling

The evidence strongly supports a hierarchical (multilevel) Bayesian model with students nested in schools. Key design choices:

**Likelihood**: Normal for the continuous, symmetric outcome. No evidence of heavy tails, zero-inflation, or boundedness that would require alternative likelihoods.

**School-level effects**: Model school intercepts as random effects drawn from a Normal distribution. This enables partial pooling: school-specific estimates shrink toward the population mean, with shrinkage proportional to school sample size and between-school variance.

**Treatment effect**: Start with a fixed treatment effect (common across schools). If model checking reveals misfit, consider random slopes (school-specific treatment effects) with hierarchical structure.

**Priors**: Score scale is approximately 43 to 105 (range 62 points, mean 77.1, std 10.5). For school intercepts, center at the grand mean with prior standard deviation informed by the observed between-school std (4.30). For treatment effect, the observed overall effect of 7.03 provides a reasonable location, but use weakly informative priors to avoid overconfidence.

**ICC interpretation**: With ICC = 0.14, approximately 14% of variance requires school-level modeling. This is substantial enough to matter for inference but not so large as to dominate within-school variation. Partial pooling will be effective: enough schools (8) to estimate population-level parameters, enough within-school data (15-25 per school) to inform school-specific estimates.

**Design effect**: The DEFF of 3.67 implies that a naive model ignoring clustering would underestimate standard errors by a factor of √3.67 = 1.92. This is not catastrophic but would lead to overconfident inference and inflated Type I error rates. A hierarchical model naturally accounts for this by modeling between-school variance explicitly.

**Complete pooling vs no pooling vs partial pooling**: Complete pooling (ignoring schools) would bias SEs downward. No pooling (separate models per school) would produce unstable school-specific estimates due to small samples. Partial pooling via hierarchical modeling balances these extremes, borrowing strength across schools while allowing school-specific deviations.

## Data Generation Story

Scores are likely generated by a process where students belong to schools with persistent characteristics (quality, resources, demographics) that shift baseline achievement. Within schools, students vary due to individual characteristics and measurement error. Treatment is assigned to individual students within schools, with effects that may be common across schools or vary due to implementation differences or effect modifiers.

The observed heterogeneity in treatment effects is consistent with either sampling variability (more likely) or genuine treatment-by-school interactions (possible but requires more evidence). A hierarchical model can adjudicate this by estimating the variance of school-specific effects. If this variance is small after accounting for sampling uncertainty, effects are likely common; if large, true heterogeneity exists.

## Recommendations for Modeling

1. Use a two-level hierarchical model: students (level 1) nested in schools (level 2).

2. Model school intercepts as random effects: α_j ~ Normal(μ_α, σ_α), where j indexes schools. This captures between-school variation and enables partial pooling.

3. Start with a fixed treatment effect β_treat (common across schools). Assess model fit via posterior predictive checks. If systematic misfit appears for specific schools, extend to random slopes.

4. Use weakly informative priors:
   - Population mean μ_α: Normal(77, 20) reflecting score scale
   - Between-school std σ_α: Exponential(1/4) weakly centered near observed 4.30
   - Treatment effect β_treat: Normal(7, 10) weakly informed by observed 7.03
   - Within-school std σ: Exponential(1/10) weakly centered near observed 9.95

5. After fitting, compute school-specific shrinkage: compare no-pooling estimates (school-specific means) to partial-pooling posterior means. Small schools should show more shrinkage.

6. Report both population-level treatment effect (averaged across schools) and school-specific deviations. The former estimates the average causal effect; the latter quantifies heterogeneity.

7. If extending to random slopes, compare models via LOO or WAIC. Random slopes add complexity (8 additional parameters) and require stronger shrinkage priors to avoid overfitting. Only adopt if predictive performance improves.

## Summary Statistics

Key quantities for prior specification:

- Score range: 43.7 to 104.6 (span 60.9)
- Score mean: 77.1, std: 10.5
- School means: range 72.4 to 85.0 (span 12.6), std: 4.30
- Within-school std: approximately 9.95 (pooled)
- Treatment effect: overall 7.03, school-specific range -0.45 to 15.0, std 5.16
- ICC: 0.1406
- Design effect: 3.67

## Files Generated

- `quality_summary.csv`: Data quality checks summary
- `univariate_summary.csv`: Per-column statistics
- `school_summary.csv`: School-level sample sizes, treatment balance, score statistics
- `variance_components.csv`: Variance decomposition and ICC
- `school_treatment_effects.csv`: Within-school treatment effects
- `school_distributions.png`: Box plots of scores by school
- `school_violin_plots.png`: Violin plots showing density by school
- `school_means.png`: School means with 95% confidence intervals
- `treatment_by_school.png`: Side-by-side bar chart of treatment vs control by school
- `treatment_effects_by_school.png`: Within-school treatment effects with 95% CI
- `treatment_scatter_by_school.png`: Individual scores colored by school with treatment effects
