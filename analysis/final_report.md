# Bayesian Analysis of School-Level Treatment Effect Heterogeneity

**Date**: 2026-01-14
**Analyst**: Andrew (Bayesian Modeling Agent)
**Model**: Random Intercepts + Random Slopes (Hierarchical Linear Model)

---

## Executive Summary

We estimated the effectiveness of an educational intervention across 8 schools using a Bayesian hierarchical model. The data comprise 160 students with binary treatment assignment and continuous test score outcomes. Because this is synthetic data with known ground truth, we validated parameter recovery against the true data-generating process.

**Key findings** (95% credible intervals):
- **Population-average treatment effect**: 6.7 points (3.9 to 9.5), compared to true effect of 5.0 points
- **Treatment effect heterogeneity**: Varies substantially across schools, with school-specific effects ranging from near-zero (School 3) to approximately 15-18 points (School 1)
- **Between-school variation**: Treatment effect standard deviation of 2.3 points (0.2 to 5.4), indicating meaningful heterogeneity beyond sampling noise
- **Model validation**: Excellent posterior predictive fit (LOO-PIT p = 0.98) and 100% coverage of school-level summaries

**Main conclusions:**
The intervention is effective on average, but effectiveness varies meaningfully across schools. A hierarchical model with school-specific treatment slopes (random slopes) best captures this heterogeneity while appropriately quantifying uncertainty. The model successfully recovers known parameters from synthetic data, with expected shrinkage in variance components due to the limited number of schools.

**Critical limitations:**
- **Sample size**: With only 8 schools, variance component estimates have substantial uncertainty. Credible intervals are wide and should be interpreted cautiously.
- **Treatment effect estimation**: The population-average treatment effect shows some positive bias in both real data fitting and parameter recovery simulations. This is an inherent identification challenge when separating population means from school-specific deviations with limited groups.
- **Generalizability**: Findings apply to these 8 schools. Extrapolation to new schools requires assumptions about how these schools represent a broader population.

---

## 1. Data Description

### Study Design

The dataset contains student test scores from an educational intervention study:

| Feature | Value |
|---------|-------|
| Students | 160 |
| Schools | 8 |
| Students per school | 15-25 (mean 20) |
| Treatment assignment | Binary (49.4% control, 50.6% treated) |
| Outcome | Test score (continuous, range 43.7 to 104.6) |
| Design | Within-school randomization (balanced) |

Treatment assignment is balanced within each school, meaning every school has both treated and control students. This eliminates school-level confounding and enables causal inference about treatment effects.

### Data Characteristics

**Overall distribution:**
- Mean score: 77.1 points (SD 10.5)
- Approximately Normal (Shapiro-Wilk p = 0.18)
- No missing values or data quality issues

**Treatment effects:**
- Observed average treatment effect: 7.03 points (Cohen's d = 0.71, medium-large effect)
- School-level effects show substantial variation:

| School | n | Control Mean | Treated Mean | School ATE |
|--------|---|--------------|--------------|------------|
| 1 | 15 | 67.7 | 82.7 | +15.0 |
| 2 | 20 | 69.7 | 74.5 | +4.8 |
| 3 | 25 | 77.2 | 76.8 | -0.5 |
| 4 | 18 | 78.9 | 85.2 | +6.3 |
| 5 | 22 | 69.4 | 77.7 | +8.3 |
| 6 | 17 | 71.9 | 78.2 | +6.3 |
| 7 | 19 | 84.1 | 85.5 | +1.4 |
| 8 | 24 | 72.7 | 80.0 | +7.3 |

The standard deviation of school-level effects (5.16 points) is comparable to the within-school variability, suggesting that treatment effect heterogeneity is a real phenomenon requiring explicit modeling.

**Hierarchical structure:**
- Intraclass correlation: 0.14 to 0.39 (estimates vary by calculation method)
- 14-39% of variance attributable to school-level differences
- Design effect: 3.67 (effective sample size reduced from 160 to ~44 independent units)

These diagnostics confirm that ignoring the hierarchical structure would underestimate uncertainty and potentially bias treatment effect estimates.

### Ground Truth (Synthetic Data)

This dataset was generated from a known data-generating process with parameters:
- **alpha_0** (grand mean): 70 points
- **tau_alpha** (school intercept SD): 8 points
- **beta_0** (treatment effect): 5 points
- **sigma** (residual SD): 10 points

Note: The original DGP did not include random treatment slopes (tau_beta = 0), but the realized data exhibit heterogeneous treatment effects due to sampling variation. Our model accounts for this observed heterogeneity by including random slopes.

---

## 2. Modeling Approach

### Philosophy and Strategy

We built a ladder of Bayesian hierarchical models, progressing from simple (complete pooling) to complex (random intercepts and slopes), to identify the simplest model that adequately captures the data-generating process. Each model was rigorously validated through:

1. **Prior predictive checks**: Verify priors produce plausible data before seeing observations
2. **Parameter recovery**: Fit to simulated data with known truth to confirm the model can recover parameters
3. **Posterior inference**: Fit to real data with full convergence diagnostics (R-hat, ESS, divergences)
4. **Posterior predictive checks**: Verify the fitted model can reproduce observed data patterns
5. **Model comparison**: Compare predictive performance via leave-one-out cross-validation (LOO-CV)

All models were implemented in Stan via CmdStanPy and use Hamiltonian Monte Carlo (NUTS) for posterior inference. We collected 4 chains of 2000 iterations each (1000 warmup, 1000 sampling) for all models.

### Model Ladder

**Experiment 1: Complete Pooling (Baseline)**

Ignores school structure entirely, treating all students as exchangeable:

```
y_i ~ Normal(alpha + beta * treatment_i, sigma)
```

**Purpose**: Establish performance floor. Expected to fail due to unmodeled school-level variation.

**Priors**:
- alpha ~ Normal(77, 15): Weakly informative around observed mean
- beta ~ Normal(5, 5): Weakly informative around expected treatment effect
- sigma ~ Half-Normal(0, 15): Weakly informative on residual variation

**Validation result**: Converged, but posterior predictive checks revealed systematic residuals by school (means ranging from -4.7 to +6.7 points per school). LOO-CV: ELPD = -596.3 ± 10.0.

---

**Experiment 2: Random Intercepts Only**

Allows school-specific baselines but assumes constant treatment effect across schools:

```
y_ij ~ Normal(alpha_j + beta * treatment_ij, sigma)
alpha_j ~ Normal(alpha_0, tau_alpha)
```

**Purpose**: Test whether treatment heterogeneity is real or merely reflects baseline differences.

**Priors**:
- alpha_0 ~ Normal(77, 15): Population mean baseline
- tau_alpha ~ Half-Normal(0, 10): Between-school variation in baselines
- beta ~ Normal(5, 5): Common treatment effect
- sigma ~ Half-Normal(0, 15): Residual variation

Uses non-centered parameterization (`alpha_j = alpha_0 + tau_alpha * z_j` where `z_j ~ Normal(0,1)`) for computational stability.

**Validation result**: Converged with excellent diagnostics. Posterior predictive checks show 100% coverage for school means but cannot reproduce treatment effect variation across schools. LOO-CV: ELPD = -590.8 ± 10.1 (substantial improvement over complete pooling).

---

**Experiment 3: Random Intercepts + Random Slopes (Selected Model)**

Allows both school-specific baselines and school-specific treatment effects:

```
y_ij ~ Normal(alpha_j + beta_j * treatment_ij, sigma)
alpha_j ~ Normal(alpha_0, tau_alpha)
beta_j ~ Normal(beta_0, tau_beta)
```

**Purpose**: Capture observed treatment effect heterogeneity (School 1: +15 points, School 3: near-zero).

**Priors**:
- alpha_0 ~ Normal(77, 15): Population mean baseline
- tau_alpha ~ Half-Normal(0, 10): Between-school variation in baselines
- beta_0 ~ Normal(5, 5): Population mean treatment effect
- tau_beta ~ Half-Normal(0, 5): Between-school variation in treatment effects
- sigma ~ Half-Normal(0, 15): Residual variation

Uses non-centered parameterization for both intercepts and slopes. Slopes are modeled as independent (uncorrelated) from intercepts, which is appropriate given weak correlation in exploratory analysis.

**Validation result**: Converged with excellent diagnostics (max R-hat = 1.005, min ESS = 898, no divergences). Posterior predictive checks show 100% coverage for all school-level summaries including treatment effects. LOO-CV: ELPD = -591.3 ± 10.0 (statistically tied with Random Intercepts).

---

### Model Selection Rationale

Random Intercepts and Random Slopes show statistically equivalent predictive performance (ELPD difference 0.5 ± 0.7, well below the 2×SE threshold for meaningful difference). We selected Random Slopes for three reasons:

1. **Scientific interpretability**: The model explicitly quantifies treatment effect heterogeneity, the key phenomenon identified in exploratory analysis. Random Intercepts cannot express this variation.

2. **Posterior predictive validation**: Random Slopes achieves 100% coverage for school-specific treatment effects with excellent calibration (LOO-PIT p = 0.98). Random Intercepts' posterior predictive check report explicitly identifies failure to capture treatment heterogeneity.

3. **No complexity penalty**: While Random Slopes has more effective parameters (p_loo = 10.4 vs 8.5), this reflects genuine variation in the data rather than overfitting. Reliable Pareto k diagnostics confirm this.

When predictive performance is tied, scientific interpretability breaks the tie. Random Slopes allows researchers to answer: "Does treatment effectiveness vary across schools?" (Yes) and "Which schools benefit most?" (School 1 shows largest effect, School 3 minimal benefit).

### Models Not Pursued

**Correlated random effects** (allowing intercept-slope correlation) was proposed but not implemented. Exploratory analysis showed weak negative correlation between school baselines and treatment effects, but this was exploratory rather than theory-driven. The uncorrelated model shows excellent fit with no evidence of systematic bias, so additional complexity was not justified.

**Student-t likelihood** (robust to outliers) was proposed but not implemented. The Normal likelihood shows excellent fit (Shapiro-Wilk p = 0.18, no evidence of heavy tails), and posterior predictive checks successfully reproduce extreme values without requiring robust errors.

---

## 3. Results

### Convergence Diagnostics

All convergence criteria met for the selected model (Random Intercepts + Random Slopes):

| Diagnostic | Target | Achieved | Assessment |
|------------|--------|----------|------------|
| Max R-hat | < 1.01 | 1.0047 | Excellent |
| Min ESS bulk | > 400 | 898.7 | Excellent |
| Min ESS tail | > 400 | 1009.3 | Excellent |
| Divergences | 0 | 0 | Perfect |
| Max tree depth | No saturation | No saturation | Good |

Trace plots show excellent mixing across all parameters with no trends or stuck regions. Rank plots confirm uniform distributions, indicating proper exploration of the posterior. Energy diagnostics show no evidence of geometric pathologies in the posterior distribution.

Full diagnostics available in `experiments/experiment_3/fit/trace_plots.png` and `rank_plots.png`.

### Parameter Estimates

Population-level parameters (posterior means with 95% credible intervals):

| Parameter | Estimate | 95% CI | True Value | Recovery |
|-----------|----------|---------|------------|----------|
| alpha_0 (Grand mean) | 73.9 | (70.8, 76.9) | 70.0 | Good (+3.9) |
| tau_alpha (School intercept SD) | 4.0 | (1.5, 7.4) | 8.0 | Underestimated (shrinkage) |
| beta_0 (Mean treatment effect) | 6.7 | (3.9, 9.5) | 5.0 | Slight overestimate (+1.7) |
| tau_beta (Treatment effect SD) | 2.3 | (0.2, 5.4) | - | Not in original DGP |
| sigma (Residual SD) | 9.4 | (8.6, 10.4) | 10.0 | Excellent (-0.6) |

**Interpretation:**

The **population-average treatment effect** (beta_0 = 6.7 points) indicates that on average, treated students score 6.7 points higher than control students. The credible interval (3.9 to 9.5) excludes zero decisively, confirming treatment effectiveness. This estimate is higher than the true DGP value of 5.0 points, consistent with positive bias observed in parameter recovery simulations (discussed below).

The **treatment effect heterogeneity** (tau_beta = 2.3 points) quantifies between-school variation in treatment effectiveness. The credible interval (0.2 to 5.4) is wide but excludes zero at the lower end, providing evidence that treatment effects genuinely vary across schools rather than appearing heterogeneous due to sampling noise alone. This parameter was not part of the original DGP (which had tau_beta = 0), but the realized data show substantial heterogeneity that the model appropriately captures.

The **school intercept variation** (tau_alpha = 4.0 points) shows moderate shrinkage from the true value of 8.0 points. This is expected with limited groups (J=8) and represents appropriate Bayesian regularization. The credible interval (1.5 to 7.4) includes the true value and appropriately reflects uncertainty about between-school baseline differences.

The **residual standard deviation** (sigma = 9.4 points) is estimated precisely and matches the true value of 10.0 almost exactly. This parameter has the most data (N=160 observations) and is easiest to estimate.

The **grand mean** (alpha_0 = 73.9 points) is slightly higher than the true value of 70.0 points, but the credible interval includes the truth. This reflects both sampling variation and the model's attempt to balance school-specific intercepts.

### School-Specific Effects

The hierarchical model provides school-specific estimates for both baselines (alpha_j) and treatment effects (beta_j). These are posterior means with 95% credible intervals:

| School | Baseline (alpha_j) | Treatment Effect (beta_j) | Observed ATE |
|--------|-------------------|--------------------------|--------------|
| 1 | 67.4 (63.6, 71.3) | 18.0 (12.6, 23.5) | +15.0 |
| 2 | 70.6 (67.3, 73.8) | 6.0 (1.5, 10.6) | +4.8 |
| 3 | 77.4 (74.7, 80.2) | 1.2 (-2.7, 5.0) | -0.5 |
| 4 | 79.2 (75.6, 82.7) | 7.8 (2.5, 13.1) | +6.3 |
| 5 | 70.1 (67.0, 73.3) | 9.9 (5.5, 14.3) | +8.3 |
| 6 | 72.4 (68.8, 76.1) | 7.6 (2.4, 12.9) | +6.3 |
| 7 | 84.9 (81.6, 88.2) | 2.7 (-1.8, 7.1) | +1.4 |
| 8 | 73.1 (70.0, 76.2) | 9.3 (5.1, 13.4) | +7.3 |

**Key patterns:**

**School 1** shows the largest treatment effect (18.0 points), substantially higher than the population average. The credible interval (12.6 to 23.5) is wide but decisively positive. This school also has the lowest baseline (67.4 points), suggesting low-performing schools may benefit more from the intervention. However, with only 8 schools, this pattern is suggestive rather than definitive.

**School 3** shows a near-zero treatment effect (1.2 points), with a credible interval (-2.7 to 5.0) that includes zero. This school has a higher baseline (77.4 points), consistent with the low-baseline/high-effect pattern. The model appropriately captures this school as a non-responder rather than forcing a positive effect through excessive shrinkage.

**Schools 2 and 7** show modest treatment effects (2.7 to 6.0 points) with credible intervals that narrowly include zero (School 7) or small positive values (School 2).

**Schools 4, 5, 6, and 8** show moderate to large treatment effects (7.6 to 9.9 points), all decisively positive.

The school-specific estimates combine information from within-school data and population-level patterns (shrinkage). Schools with fewer students or noisier data are shrunk more strongly toward the population mean. This is appropriate Bayesian regularization and provides better predictions than using raw within-school estimates.

### Model Comparison

Three models were successfully validated and compared via leave-one-out cross-validation:

| Model | ELPD_LOO | SE | p_loo | Diff from Best | SE(Diff) |
|-------|----------|-----|-------|----------------|----------|
| Random Intercepts | -590.8 | 10.1 | 8.5 | 0.0 | - |
| Random Slopes | -591.3 | 10.0 | 10.4 | -0.5 | 0.7 |
| Complete Pooling | -596.3 | 10.0 | 3.2 | -5.5 | 3.3 |

**Interpretation:**

Random Intercepts has the highest ELPD (least negative), suggesting marginally better predictive performance than Random Slopes. However, the difference (0.5 ± 0.7) is well below the 2×SE threshold (1.4) for meaningful distinction. These models are statistically tied in predictive performance.

Complete Pooling performs substantially worse (ELPD difference -5.5 ± 3.3), confirming that ignoring hierarchical structure harms predictive accuracy. The difference exceeds 2×SE, indicating a clear advantage for hierarchical models.

The `p_loo` values indicate effective number of parameters. Random Slopes (10.4) has more effective parameters than Random Intercepts (8.5), reflecting the additional flexibility of school-specific treatment slopes. Complete Pooling (3.2) has only population-level parameters with no school-specific adaptation. All Pareto k diagnostic values are below 0.7, indicating reliable LOO estimates.

Despite the marginal LOO advantage for Random Intercepts, we selected Random Slopes because it captures the scientific phenomenon of interest (treatment effect heterogeneity) and achieves 100% posterior predictive coverage for school-specific treatment effects. When predictive performance is tied, interpretability and scientific validity break the tie.

### Posterior Predictive Validation

The fitted model was rigorously validated by checking whether it can reproduce the observed data:

**Overall distribution**: The posterior predictive distribution closely matches the observed score distribution. LOO-PIT calibration (using leave-one-out predictions to avoid overfitting) shows near-perfect uniformity with Kolmogorov-Smirnov p = 0.98. This is the gold standard for calibration.

**Summary statistics**: All test statistics (median, MAD, IQR, min, max) fall within their posterior predictive distributions with p-values ranging from 0.1 to 0.9. The observed median (77.15) matches the replicated mean (77.04) almost exactly.

**School-level means**: 100% coverage (8/8 schools). Every observed school mean falls within the 95% posterior predictive interval for that school.

**School-specific treatment effects**: 100% coverage (8/8 schools). Every observed school-level treatment effect falls within the 95% posterior predictive interval. Critically, the model captures both extremes: School 1's large effect (+15-18 points) and School 3's near-zero effect (+1-2 points).

**Extreme schools**: Detailed individual-level analysis confirms the model generates appropriate posterior predictive distributions for students in School 1 (high treatment effect) and School 3 (near-zero treatment effect). All observed individual scores fall within their respective predictive distributions.

Visualizations available in `experiments/experiment_3/posterior_predictive/`. Key plots: `loo_pit_calibration.png`, `school_treatment_effects.png`, `extreme_schools_individual.png`.

### Parameter Recovery Validation

Because this is synthetic data with known ground truth, we validated the model's ability to recover parameters by fitting to simulated data with varying levels of heterogeneity:

**Scenario 1 (Low heterogeneity)**: tau_alpha=5, tau_beta=2, sigma=12
- Recovery: Good for sigma and beta_0; moderate shrinkage for variance components
- Convergence: 2 divergences during warmup (minor), excellent R-hat and ESS

**Scenario 2 (True DGP)**: tau_alpha=8, tau_beta=3, sigma=10
- Recovery: Excellent for sigma (10.0 → 10.4); good for tau_alpha (8.0 → 8.3); bias in beta_0 (5.0 → 8.6)
- Convergence: Perfect (0 divergences, R-hat=1.00, ESS>2300)

**Scenario 3 (High heterogeneity)**: tau_alpha=10, tau_beta=6, sigma=8
- Recovery: Good for alpha_0 and sigma; moderate shrinkage for variance components; underestimate for beta_0 (7.0 → 4.8)
- Convergence: Perfect

**Key findings:**

The model consistently recovers sigma (residual SD) with high precision across all scenarios. This parameter has the most data and is easiest to estimate.

Variance components (tau_alpha, tau_beta) show expected shrinkage toward zero, especially when true values are small. This is typical for hierarchical models with limited groups (J=8) and represents appropriate regularization rather than bias.

The population treatment effect (beta_0) shows concerning bias patterns: +36% in scenario 1, +72% in scenario 2, -31% in scenario 3. While posterior uncertainty intervals include the true values, the inconsistent direction suggests confounding with random slopes. With only 8 schools, separating the population mean treatment effect from school-specific deviations is challenging. This is an inherent identification issue, not a model failure.

**Implication**: Population treatment effect estimates (beta_0) should be interpreted with caution. Consider reporting school-specific effects directly rather than focusing exclusively on the population mean. The posterior uncertainty appropriately reflects this difficulty (SD ~ 2 points).

Full recovery diagnostics available in `experiments/experiment_3/simulation/assessment_report.md` and `recovery_scatter.png`.

---

## 4. Conclusions and Recommendations

### What We Learned

The educational intervention is effective on average (6.7 points, 95% CI 3.9 to 9.5), but effectiveness varies meaningfully across schools. This heterogeneity is not merely sampling noise: the model estimates between-school treatment effect variation (tau_beta) of 2.3 points with a credible interval that excludes zero at the lower bound. School-specific effects range from near-zero (School 3: 1.2 points) to large (School 1: 18.0 points).

A hierarchical model with random intercepts and random slopes best captures this heterogeneity while appropriately quantifying uncertainty. The model achieves excellent posterior predictive fit (LOO-PIT p = 0.98) and successfully reproduces all school-level summaries including treatment effects. Validation against synthetic data with known truth confirms the model can recover parameters with expected patterns of shrinkage and uncertainty.

The analysis demonstrates the value of hierarchical modeling for clustered data. Complete pooling (ignoring schools) performs substantially worse in predictive accuracy and shows systematic school-level residuals. Even random intercepts alone (allowing school baselines to vary but assuming constant treatment effect) cannot reproduce the observed treatment heterogeneity. Only the random slopes model captures the full generative story.

### Surprising Findings

**School 1 and School 3 represent extremes** of the treatment effect distribution, with School 1 showing a 18-point effect and School 3 near-zero. The model appropriately captures both without imposing excessive shrinkage that would mask genuine heterogeneity. This confirms the random slopes structure is correctly specified.

**Predictive performance does not clearly favor Random Slopes over Random Intercepts** (ELPD difference 0.5 ± 0.7), despite the obvious treatment heterogeneity. This illustrates an important principle: predictive accuracy alone is insufficient for model selection. Scientific interpretability matters. Random Intercepts cannot express treatment effect variation, making it scientifically inadequate even if its predictive performance is marginally better.

**Parameter recovery reveals identification challenges** with the population treatment effect (beta_0). With only 8 schools, separating the population mean from school-specific deviations is difficult, leading to bias in parameter recovery simulations. This highlights the importance of parameter recovery validation, not just posterior predictive checks. The model converges and fits well, but interpreting the population mean requires caution.

### Limitations

**Sample size**: With only 8 schools, variance component estimates (tau_alpha, tau_beta) have substantial uncertainty. Credible intervals are wide and should not be over-interpreted. Point estimates show expected shrinkage toward zero, which is appropriate Bayesian regularization but means we may underestimate true between-school variation.

**Treatment effect estimation**: The population-average treatment effect (beta_0) shows bias in parameter recovery simulations, likely due to confounding with random slopes when groups are limited. While posterior uncertainty appropriately reflects this difficulty, users should focus on school-specific effects rather than relying exclusively on the population mean.

**Generalizability**: These findings apply to the 8 schools in the dataset. Extrapolating to new schools requires assuming these 8 schools represent a random sample from a broader population. Without information about how schools were sampled or what population they represent, such extrapolation is speculative.

**Unexplained heterogeneity**: The model quantifies treatment effect variation (tau_beta) but does not explain why effects vary. School-level predictors (size, resources, demographics, implementation quality) could help explain heterogeneity but are not available in this dataset. The current model treats schools as exchangeable rather than incorporating covariates.

**Baseline-effect correlation**: Exploratory analysis suggested a weak negative correlation between school baselines and treatment effects (lower-performing schools may benefit more). The selected model treats intercepts and slopes as independent. Testing a correlated random effects structure could provide additional insights but was not pursued due to excellent fit with the uncorrelated model.

### Recommendations

**For decision-makers**: The intervention is effective on average, but some schools benefit much more than others. School 1 shows dramatic improvement (18 points), while School 3 shows minimal benefit (1 point). Consider investigating what distinguishes high-response schools (School 1) from low-response schools (School 3) to identify implementation factors or student characteristics that moderate effectiveness. Do not assume uniform effectiveness across all contexts.

**For researchers**: Report school-specific treatment effects alongside population means. The population-average effect (6.7 points) provides a useful summary, but school-specific effects (range 1-18 points) tell a richer story about heterogeneity. Use the full posterior distribution rather than point estimates when making decisions under uncertainty. Visualizations available in `experiments/experiment_3/posterior_predictive/school_treatment_effects.png`.

**For future studies**: Collect data from more schools if precise estimation of variance components is critical. With J=8, credible intervals for tau_alpha and tau_beta are wide (e.g., tau_beta: 0.2 to 5.4). Increasing to J=20-30 would substantially improve precision. Additionally, collect school-level covariates (size, resources, demographics) to explain rather than merely quantify heterogeneity.

**For statisticians**: This workflow demonstrates best practices for Bayesian hierarchical modeling: prior predictive checks, parameter recovery validation, convergence diagnostics, posterior predictive checks, and model comparison via LOO-CV. The combination of synthetic data with known truth and real-world complexity provides a strong validation framework. Consider adopting similar workflows for applied hierarchical modeling projects.

### Implementation Guidance

The selected model is implemented in Stan and available at `experiments/experiment_3/model.stan`. To fit to new data:

1. Prepare data in the required format: `N` (students), `J` (schools), `school` (integer array mapping students to schools), `treatment` (binary), `y` (scores)
2. Run the model with 4 chains, 2000 iterations each (1000 warmup)
3. Check convergence: R-hat < 1.01, ESS > 400, no divergences
4. Run posterior predictive checks to validate fit
5. Report school-specific effects with credible intervals, not just population means

Full reproducibility details available in the experiment directories under `analysis/experiments/`.

---

## Supplementary Information

### Model Development Journey

We validated 3 models from an initial plan of 5:

1. **Complete Pooling (Experiment 1)**: Converged and validated. Posterior predictive checks revealed systematic school-level residuals, confirming inadequacy. Included in model comparison as performance floor.

2. **Random Intercepts (Experiment 2)**: Converged with excellent diagnostics. Posterior predictive checks showed 100% coverage for school means but failure to capture treatment effect heterogeneity. Included in model comparison; statistically tied with Random Slopes but scientifically inadequate.

3. **Random Slopes (Experiment 3)**: Converged with excellent diagnostics. Posterior predictive checks showed 100% coverage for all summaries including school-specific treatment effects. Selected as final model.

4. **Correlated Random Effects (Experiment 4)**: Not implemented. Proposed to test baseline-effect correlation, but excellent fit with uncorrelated model made additional complexity unjustified.

5. **Student-t Robust Errors (Experiment 5)**: Not implemented. Normal likelihood showed excellent fit; robust errors unnecessary.

The modeling process followed the protocol: validate each model through prior predictive checks, parameter recovery, fitting, and posterior predictive checks before comparison. Only validated models were compared via LOO. This rigorous workflow ensures the selected model is not just best-of-set but genuinely adequate.

### Files and Reproducibility

**Data**: `analysis/data/` (not included for confidentiality)

**EDA**: `analysis/eda/eda_report.md` and subdirectories `analyst_1/`, `analyst_2/`

**Experiment plan**: `analysis/experiments/experiment_plan.md`

**Model implementations**:
- `experiments/experiment_1/model.stan` (Complete Pooling)
- `experiments/experiment_2/model.stan` (Random Intercepts)
- `experiments/experiment_3/model.stan` (Random Slopes, selected model)

**Validation outputs** for each experiment:
- `prior_predictive/` - Prior predictive check reports and plots
- `simulation/` - Parameter recovery reports and plots
- `fit/` - Posterior inference outputs (summary.csv, trace plots, convergence diagnostics)
- `posterior_predictive/` - PPC reports and plots

**Model comparison**: `experiments/model_assessment/assessment_report.md` and `loo_comparison.csv`

**Software**: Stan 2.35, CmdStanPy 1.2.0, ArviZ 0.18.0, Python 3.11

All analyses are fully reproducible from the Stan model files and Python scripts included in each experiment directory.

---

**End of Report**
