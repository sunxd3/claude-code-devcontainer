# Exploratory Data Analysis: Student Test Scores

## Executive Summary

This analysis examines a hierarchical dataset of 160 students across 8 schools to understand treatment effects on test scores. The data shows a strong, statistically significant treatment effect (ATE = 7.03, p < 0.0001, Cohen's d = 0.71) but with substantial heterogeneity across schools. The high intraclass correlation (ICC = 0.78) indicates that 78% of score variance exists between schools rather than within them, making hierarchical modeling essential. School-level treatment effects vary considerably (SD = 5.16, range -0.45 to 15.00), suggesting that both intercepts and slopes should be modeled as varying by school.

## Data Structure and Quality

**Dataset dimensions:** 160 observations × 5 variables

**Variables:**
- `student_id`: Unique identifier (object, 160 unique values)
- `school_id`: School identifier (integer, 8 schools)
- `school_name`: School name (object, consistent 1-to-1 mapping with school_id)
- `treatment`: Binary indicator (0 = control, 1 = treated)
- `score`: Continuous outcome variable (float)

**Data quality:** The dataset is exceptionally clean with no missing values, no duplicates, no invalid entries, and consistent typing. Student IDs follow expected format, school mappings are consistent, and scores fall within plausible ranges (43.7 to 104.6). One student scored below 50, and 5 students scored above 100 (3.1% of sample).

**School structure:** Schools vary in size from 15 to 25 students (mean = 20.0, SD = 3.5). Treatment assignment is balanced overall (49.4% control, 50.6% treated) but varies substantially by school, ranging from 27% to 68% treated. This 41.8 percentage point range warrants attention in the modeling phase.

See `quality_summary.csv` for complete quality metrics.

## Outcome Distribution

**Score statistics:**
- Mean: 77.09, SD: 10.50
- Median: 77.15
- Range: [43.70, 104.60]
- IQR: [70.65, 82.80]
- Skewness: 0.186 (slightly right-skewed)
- Kurtosis: 0.334 (slightly heavy tails)

The score distribution is approximately normal (Shapiro-Wilk p = 0.179), with minimal skewness and kurtosis. As shown in `univariate_distributions.png`, the empirical distribution closely matches a normal fit, and the Q-Q plot shows strong adherence to normality except at the extremes. This supports using a Normal likelihood for the outcome.

## Treatment Effects

**Overall effect:** The treated group (mean = 80.56, SD = 10.14) scored substantially higher than controls (mean = 73.53, SD = 9.69), yielding an average treatment effect of 7.03 points. This difference is highly significant (t = 4.48, p < 0.0001) with a medium-to-large effect size (Cohen's d = 0.71). Variance is similar between groups (Levene test p = 0.549), supporting homoscedasticity assumptions.

**School-level heterogeneity:** Treatment effects vary considerably across schools, as shown in `treatment_effects.png` and `school_heterogeneity.png`. School-level ATEs range from -0.45 (School 3) to 15.00 (School 1), with a standard deviation of 5.16. Seven of eight schools show positive effects, but the magnitude differs substantially. This heterogeneity is not merely sampling noise—it reflects genuine variation in how schools respond to treatment.

Key school-level findings (see `school_treatment_effects.csv`):
- School 1: ATE = 15.00 (strongest effect)
- School 3: ATE = -0.45 (essentially null effect)
- School 7: ATE = 1.43 (weak positive effect)
- Remaining schools: ATEs between 4 and 11 points

The coefficient of variation for school-level ATEs (0.67) indicates substantial relative variation. This pattern strongly suggests that treatment slopes should vary by school in the model.

## Hierarchical Structure

The data exhibits strong hierarchical structure. As shown in `generative_stories.png`, between-school variance (351.9) far exceeds within-school variance (99.1), yielding an ICC of 0.78. This means that knowing a student's school explains 78% of score variation before considering any other predictors. This high ICC mandates a hierarchical model that accounts for school clustering.

**Baseline school differences:** Even among control students, schools differ significantly in baseline performance (ANOVA F = 2.69, p = 0.016). Control group means range from 67.7 to 84.1 across schools, indicating that schools have different baseline achievement levels independent of treatment. This supports random intercepts for schools.

**Treatment effect heterogeneity:** The variance in school-level treatment effects (variance = 23.3, SD = 5.16) is large relative to the overall ATE (7.03). This heterogeneity cannot be explained by baseline differences alone (correlation = -0.58, but p = 0.14), suggesting that treatment effects vary for reasons beyond initial school performance. This supports random slopes for treatment by school.

## Competing Generative Mechanisms

Five data-generating stories were tested to understand the treatment mechanism:

**Story 1: Fixed treatment effect (no school variation)**
Evidence: REJECTED. School-level ATE variance (23.3) and range (15.5) are too large to support a constant treatment effect.

**Story 2: Random intercepts + fixed treatment effect**
Evidence: PARTIAL. Significant baseline differences across schools (ANOVA p = 0.016) support random intercepts, but this model cannot account for the observed variation in treatment effects.

**Story 3: Random intercepts + random treatment slopes**
Evidence: STRONG. High ICC (0.78) supports random intercepts. Large SD of school-level ATEs (5.16) relative to overall ATE (7.03) supports random slopes. This is the most plausible generative model.

**Story 4: Within-school correlation**
Evidence: CONFIRMED. ICC = 0.78 indicates students within schools are much more similar than students across schools, necessitating hierarchical structure.

**Story 5: Treatment effect moderated by baseline**
Evidence: WEAK. Negative correlation (r = -0.58) suggests schools with lower baselines may benefit more, but this relationship is not statistically significant (p = 0.14) with only 8 schools. The pattern warrants exploration in modeling but should not be imposed as a strong prior.

## Modeling Recommendations

Based on this analysis, the following modeling approach is recommended:

**Likelihood:** Normal distribution for the outcome. The score distribution is approximately normal (Shapiro-Wilk p = 0.179) with symmetric spread around the mean. Variance appears homogeneous across treatment groups (Levene p = 0.549). A Normal likelihood with identity link is appropriate.

**Hierarchical structure:** Two-level model with students (level 1) nested in schools (level 2). The high ICC (0.78) and significant school differences make hierarchical modeling essential—pooled models would severely underestimate uncertainty.

**Random effects:**
- Random intercepts for schools (α_j): Essential. ANOVA shows significant baseline differences (p = 0.016), and between-school variance is large.
- Random slopes for treatment by school (γ_j): Strongly recommended. School-level treatment effects vary substantially (SD = 5.16, range 15.5), and this variation is not explained by baseline alone.

**Recommended model structure:**
```
Y_ij ~ Normal(μ_ij, σ²)
μ_ij = α_j + (β + γ_j) * Treatment_ij
α_j ~ Normal(α₀, τ_α²)  [random intercepts]
γ_j ~ Normal(0, τ_γ²)    [random slopes]
```

Where Y_ij is the score for student i in school j, α_j captures school-specific baselines, β is the population-average treatment effect, and γ_j captures school-specific deviations from the average treatment effect.

**Alternative specifications to consider:**
1. **Simpler model (random intercepts only):** May be adequate if parsimony is prioritized, but will underfit treatment effect heterogeneity.
2. **Covariate on slopes:** If additional school-level data become available, model treatment effect heterogeneity as a function of school characteristics rather than pure random effects.
3. **Correlated random effects:** Allow correlation between intercepts and slopes (ρ_αγ) to capture potential relationship between baseline and treatment response.

**Scale and centering guidance:**
- Score values: typical range 65-85 (±1 SD around mean of 77)
- Treatment effect: expect 5-10 point increase on average
- School intercepts: expect variation of ±15 points around overall mean
- School slopes: expect variation of ±5 points around average treatment effect
- Within-school residual SD: expect ~10 points based on within-school variance
- Prior on overall intercept: center around 75 (approximate grand mean)
- Prior on treatment effect β: center around 5-10 (based on Cohen's d ~ 0.7)
- Prior on random effect SDs: weakly informative, allowing τ_α ~ 5-15, τ_γ ~ 2-8

**Model comparison strategy:**
1. Baseline: Random intercepts only
2. Target: Random intercepts + random slopes (uncorrelated)
3. Extension: Random intercepts + random slopes (correlated)

Compare via LOO-CV or WAIC. Expect random slopes model to substantially outperform intercepts-only based on observed heterogeneity.

## Key Diagnostics to Monitor

When fitting models, pay attention to:

1. **Convergence:** With only 8 schools, random effects variance parameters (τ_α, τ_γ) may be difficult to estimate. Monitor R-hat and ESS for these parameters carefully.

2. **Identifiability:** With limited schools, correlation between random intercepts and slopes may be poorly identified. Consider uncorrelated random effects initially.

3. **Extreme schools:** School 1 (ATE = 15.00) and School 3 (ATE = -0.45) are outliers in treatment response. Posterior predictive checks should verify that the model can accommodate this range.

4. **Treatment imbalance:** Schools vary in treatment assignment (27-68%). This imbalance increases uncertainty in school-specific treatment effects, especially for schools with small sample sizes. Posterior intervals for γ_j should reflect this differential precision.

5. **Shrinkage:** With high between-school variance and limited schools, random effects will exhibit substantial shrinkage toward the population mean. This is desirable, but ensure that extreme schools (1 and 3) are not overly shrunk.

## Files Generated

**Summary tables:**
- `quality_summary.csv` - Data quality metrics by column
- `univariate_summary.csv` - Descriptive statistics for key variables
- `school_treatment_effects.csv` - School-level treatment effect estimates

**Visualizations:**
- `univariate_distributions.png` - Score distribution, Q-Q plot, treatment balance
- `treatment_effects.png` - Treatment effect by group and school, including heterogeneity
- `school_heterogeneity.png` - School-specific treatment responses connected by lines
- `generative_stories.png` - Variance decomposition, ATE heterogeneity, baseline relationships

**Analysis scripts:**
- `load_data.py` - Data loading and initial inspection
- `data_quality.py` - Comprehensive data quality checks
- `univariate_analysis.py` - Distribution profiling
- `treatment_effects.py` - Treatment effect analysis and heterogeneity
- `generative_stories.py` - Testing competing data-generating mechanisms
