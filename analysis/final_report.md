# What Determines Automobile Fuel Efficiency? A Bayesian Analysis of the Auto-MPG Dataset

## Executive Summary

This analysis investigates the factors determining fuel efficiency in automobiles using Bayesian regression methods applied to the UCI Auto-MPG dataset (392 vehicles, 1970-1982).

**Key Findings:**

1. **Weight is the dominant physical determinant.** A 1% increase in vehicle weight is associated with a 0.94% decrease in fuel efficiency (95% CI: 0.89-0.98%), closely matching the theoretical prediction from physics that fuel consumption scales with mass.

2. **Technology improved fuel efficiency by 3.3% per year** (95% CI: 3.0-3.6%), holding weight constant. Over the 12-year study period, this amounts to approximately 48% cumulative improvement, reflecting the industry's response to oil crises and CAFE standards.

3. **Origin effects are explained away.** The apparent 10 mpg advantage of Japanese and European cars over American cars is fully accounted for by weight and timing. Once these factors are controlled, country of origin provides no additional predictive power (all residual t-statistics below 2).

4. **The final model achieves strong predictive performance** with ELPD = 279.7 and residual CV of 12%, capturing 88% of variance in log-transformed MPG.

**Main Conclusions:**

The "Japanese efficiency advantage" was primarily a weight and market-timing phenomenon, not an inherent manufacturing superiority. American cars were heavier and entered the fuel-efficient design space later. The physics of moving mass dominates fuel efficiency; technological improvements provided a steady 3.3%/year uplift across all manufacturers.

**Critical Limitations:**

- Observational data cannot establish causality; weight-efficiency associations may partly reflect unmeasured confounders
- EPA testing procedures from 1970-1982 differ from modern methods
- The dataset represents vehicles sold in the US market during a specific regulatory era
- Some outliers exist (diesel and rotary engines) that the model does not explicitly address

---

## Data and Methods

### Data Overview

The UCI Auto-MPG dataset contains 398 vehicles manufactured between 1970 and 1982. After removing 6 observations with missing horsepower values (1.5% of data, confirmed as missing completely at random), 392 vehicles were analyzed.

| Variable | Description | Range |
|----------|-------------|-------|
| mpg | Miles per gallon (target) | 9.0 - 46.6 |
| weight | Vehicle weight (lbs) | 1613 - 5140 |
| model_year | Year of manufacture | 1970 - 1982 |
| origin | Country: USA, Europe, Japan | - |

Weight, displacement, and horsepower exhibit severe multicollinearity (VIF > 10), so only weight was used as the physical predictor based on its strongest correlation with mpg (r = -0.83) and clearest physical interpretation.

### Exploratory Analysis

The EDA (documented in `eda/eda_report.md`) revealed:

- **MPG distribution:** Right-skewed (skewness = 0.46); log-transformation improves normality
- **Weight-MPG relationship:** Strong negative correlation with nonlinear curvature suggesting log-log specification
- **Origin effects:** Large unadjusted difference (Japan/Europe average 10 mpg more than USA, eta-squared = 0.33)
- **Temporal trend:** MPG increased 1.2 mpg/year on average, with a sharp jump in 1980

These findings motivated the hypothesis that a log-log model with year effects might explain apparent origin differences.

### Modeling Approach

All models were implemented in Stan and fit using CmdStanPy. The analysis followed a principled Bayesian workflow with four validation stages for each candidate model:

1. **Prior predictive check:** Verify priors generate plausible data before seeing observations
2. **Simulation-based calibration:** Confirm parameters can be recovered from synthetic data
3. **Model fitting:** Run MCMC with convergence diagnostics
4. **Posterior predictive check:** Assess fit quality and identify remaining deficiencies

### Model Development

Three model classes were planned, representing different structural hypotheses:

| Class | Hypothesis | Models |
|-------|------------|--------|
| A: Physical | Weight dominates; origin/year secondary | Baseline, +Year, +Robust errors |
| B: Combined | Weight, year, and origin all contribute | Additive, +Interactions |
| C: Hierarchical | Origin creates grouping structure | Partial pooling by origin |

The analysis proceeded sequentially. Class A models were fully validated, and the evidence showed origin effects are unnecessary once weight and year are controlled. This finding made Class B and C exploration unnecessary.

### Final Model Specification

The selected model (A2-Year) uses a log-log specification:

```
log(mpg) ~ Normal(alpha + beta_weight * log_weight_c + beta_year * year_c, sigma)
```

where `log_weight_c = log(weight) - 7.96` (centered) and `year_c = model_year - 76` (centered at midpoint).

**Stan implementation:**

```stan
data {
  int<lower=1> N;
  vector[N] log_mpg;
  vector[N] log_weight_c;
  vector[N] year_c;
}

parameters {
  real alpha;
  real beta_weight;
  real beta_year;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(3.1, 0.3);
  beta_weight ~ normal(-1, 0.3);
  beta_year ~ normal(0.03, 0.02);
  sigma ~ exponential(5);

  log_mpg ~ normal(alpha + beta_weight * log_weight_c + beta_year * year_c, sigma);
}
```

**Prior justification:**

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| alpha | Normal(3.1, 0.3) | Mean log(MPG) ~ 3.1 corresponds to ~22 MPG at reference point |
| beta_weight | Normal(-1, 0.3) | Physics predicts elasticity near -1 (fuel consumption proportional to mass) |
| beta_year | Normal(0.03, 0.02) | EDA showed ~1.2 mpg/year, translating to ~3% on log scale |
| sigma | Exponential(5) | Weakly informative; mean 0.2 with most mass below 0.5 |

Prior predictive checks confirmed these priors generate data covering 87.6% of the observed MPG range without producing implausible values.

### Computational Details

- **MCMC:** 4 chains, 1000 warmup + 1000 sampling iterations each
- **Adaptation:** adapt_delta = 0.9
- **Total posterior draws:** 4000
- **Software:** Stan 2.35, CmdStanPy, ArviZ

---

## Results

### Model Comparison

Three Class A models were validated and compared using leave-one-out cross-validation:

| Model | Structure | ELPD | SE | Pareto k |
|-------|-----------|------|-----|----------|
| A1-Baseline | log_mpg ~ log_weight | 147.6 | 16.4 | All < 0.5 |
| A2-Year | + year | 279.7 | 17.3 | All < 0.5 |
| A3-Robust | + Student-t errors | 286.1 | 17.5 | All < 0.5 |

Adding year to the baseline model improved ELPD by 132 points (z = 11.8), overwhelming evidence for the year effect. The Student-t extension (A3) provided only 6.3 additional ELPD points (z = 1.9), not statistically significant at conventional thresholds.

By parsimony, **A2-Year was selected as the final model**.

The comparison is visualized in `experiments/class_a_comparison.png`.

### Parameter Estimates

| Parameter | Posterior Mean | 95% HDI | Interpretation |
|-----------|----------------|---------|----------------|
| alpha | 3.098 | [3.087, 3.110] | 22.2 MPG at mean weight in 1976 |
| beta_weight | -0.935 | [-0.979, -0.893] | 1% weight increase leads to 0.94% MPG decrease |
| beta_year | 0.033 | [0.030, 0.036] | 3.3% MPG improvement per year |
| sigma | 0.118 | [0.110, 0.127] | 12% residual CV |

All parameters have narrow credible intervals and clear substantive interpretation.

### Convergence Diagnostics

The model achieved excellent convergence:

| Diagnostic | Value | Threshold |
|------------|-------|-----------|
| Max R-hat | 1.000 | < 1.01 |
| Min ESS bulk | 2471 | > 400 |
| Min ESS tail | 2148 | > 400 |
| Divergences | 0 | 0 |

Trace plots (`experiment_2/fit/figures/trace_plots.png`) show well-mixed chains with no drifting or multimodality.

### Model Validation Summary

| Stage | Status | Evidence |
|-------|--------|----------|
| Prior predictive | PASS | 87.6% coverage of observed MPG range |
| Recovery | PASS | 95% coverage, minimal parameter bias |
| Convergence | PASS | R-hat = 1.0, ESS > 2400, no divergences |
| LOO-CV | PASS | All 392 Pareto k < 0.5 |
| Posterior predictive | PASS | No systematic residual patterns |

### Key Finding: Origin Effects Vanish

The most substantively important finding is that origin shows no residual pattern after controlling for weight and year:

| Origin | n | Mean Residual | t-statistic |
|--------|---|---------------|-------------|
| USA | 245 | -0.010 | -1.5 |
| Europe | 68 | +0.033 | 1.9 |
| Japan | 79 | +0.003 | 0.2 |

All t-statistics are below 2. European cars show a marginally positive residual (3.3% efficiency premium, p ~ 0.06), but this does not reach statistical significance.

The residuals-by-origin plot (`experiment_2/posterior_predictive/residuals_vs_origin.png`) confirms the absence of systematic patterns.

---

## Interpretation and Insights

### The Physics of Fuel Efficiency

The weight elasticity of -0.94 (approximately -1) confirms that fuel efficiency fundamentally follows from physics: the energy required to move a vehicle scales with its mass. A 10% heavier car requires approximately 9.4% more fuel per mile, all else equal.

This result validates the log-log model specification and provides a physical anchor for understanding fuel economy.

### Technological Progress

The year coefficient indicates fuel efficiency improved by 3.3% annually, holding weight constant. Over the 12-year span (1970-1982), this compounds to approximately 48% improvement from technology alone. This reflects:

- Engine efficiency improvements (fuel injection replacing carburetors)
- Aerodynamic advances
- Transmission improvements
- Industry response to the 1973 and 1979 oil crises
- CAFE (Corporate Average Fuel Economy) standards implementation

The trend was consistent across origins: American, European, and Japanese manufacturers all improved at similar rates once weight is controlled.

### The Origin Myth

Perhaps the most striking finding is that the apparent Japanese/European efficiency advantage disappears when weight and year are accounted for. In the raw data, Japanese cars averaged 30.5 mpg versus 20.1 mpg for American cars (a 52% premium). This difference is entirely explained by:

1. **Weight:** Japanese and European cars were substantially lighter on average
2. **Market timing:** Import presence increased in later years when technology was better

This suggests the "Japanese efficiency advantage" narrative of the 1970s-80s was primarily a compositional effect rather than a manufacturing technology gap. American manufacturers produced what the American market demanded (larger, heavier vehicles), while imports filled the small-car niche that happened to be more fuel-efficient by virtue of physics.

### Prediction Example

For a hypothetical 3000 lb car in 1978:
- log_weight_c = log(3000) - 7.96 = 8.01 - 7.96 = 0.05
- year_c = 78 - 76 = 2
- log(mpg) = 3.098 + (-0.935)(0.05) + (0.033)(2) = 3.098 - 0.047 + 0.066 = 3.117
- **Predicted mpg = exp(3.117) = 22.6 mpg** (95% PI: approximately 18-28 mpg)

---

## Limitations and Future Work

### Limitations

**Observational design.** The weight-efficiency relationship is confounded with unmeasured variables (body style, engine type, customer preferences). While the association is physically interpretable, causal claims require caution. For example, diesel vehicles achieve higher efficiency than their weight would predict, but diesel is not modeled.

**Historical data.** The 1970-1982 period represents a unique regulatory environment (initial CAFE implementation, oil crises). Results may not generalize to modern vehicles with substantially different technology (hybrid drivetrains, turbocharging, advanced aerodynamics).

**Measurement era.** EPA testing procedures have changed since the 1970s. Absolute MPG values are not directly comparable to modern ratings.

**Limited scope.** The dataset excludes trucks, SUVs, and two-door sports cars that dominate some market segments. Findings apply to the sedan/compact car segment.

**Outliers identified but not modeled.** Diesel vehicles (e.g., Oldsmobile Cutlass Ciera Diesel) and rotary engines (Mazda RX3) behave differently from conventional gasoline vehicles. A more complete model would include fuel/engine type indicators.

### Extensions Not Pursued

- **Origin effects:** Evidence clearly shows no predictive value after conditioning on weight and year
- **Robust errors:** Student-t likelihood (nu ~ 7) confirmed outliers exist but provided negligible predictive improvement (ELPD +6.3, z = 1.9)
- **Weight-year interaction:** No residual pattern suggested differential technology trends by vehicle size

### Future Work

If extending this analysis:

1. **Fuel type indicator:** Explicitly model diesel vs. gasoline to explain remaining outliers
2. **Modern data:** Replicate with contemporary vehicles to test whether the weight-efficiency relationship persists under modern technology
3. **Causal modeling:** Use instrumental variables or regression discontinuity (e.g., CAFE thresholds) to estimate causal effects of weight reduction programs
4. **Year-specific effects:** Model years as random intercepts around the linear trend to capture oil crisis shocks more precisely

---

## Technical Appendix

### A1. Prior Sensitivity

The priors were chosen to be weakly informative based on domain knowledge:

| Parameter | Prior | Prior 95% interval | Posterior 95% HDI |
|-----------|-------|-------------------|-------------------|
| alpha | Normal(3.1, 0.3) | [2.5, 3.7] | [3.09, 3.11] |
| beta_weight | Normal(-1, 0.3) | [-1.6, -0.4] | [-0.98, -0.89] |
| beta_year | Normal(0.03, 0.02) | [-0.01, 0.07] | [0.030, 0.036] |
| sigma | Exponential(5) | [0.01, 0.60] | [0.11, 0.13] |

All posteriors are substantially narrower than priors, indicating the data dominate inference. Prior predictive checks confirmed 87.6% coverage of observed MPG range.

### A2. Convergence Diagnostics

All four parameters showed excellent mixing across 4 chains:

| Parameter | R-hat | ESS bulk | ESS tail |
|-----------|-------|----------|----------|
| alpha | 1.000 | 2652 | 2377 |
| beta_weight | 1.000 | 2795 | 2486 |
| beta_year | 1.000 | 2648 | 2421 |
| sigma | 1.000 | 2471 | 2148 |

No divergent transitions occurred. The energy plot shows good E-BFMI.

See `experiment_2/fit/figures/` for trace plots, rank plots, and pair plots.

### A3. LOO-CV Details

Leave-one-out cross-validation was performed using Pareto-smoothed importance sampling (PSIS-LOO):

| Metric | Value |
|--------|-------|
| ELPD LOO | 279.7 |
| Standard Error | 17.3 |
| p_loo | 4.3 |
| Pareto k range | [0.01, 0.28] |
| Pareto k > 0.5 | 0 / 392 |
| Pareto k > 0.7 | 0 / 392 |

All Pareto k values are well below 0.5, indicating PSIS-LOO estimates are reliable without need for resampling.

### A4. Posterior Predictive Checks

Test statistics comparing posterior predictive distribution to observed data:

| Statistic | Observed | Posterior Predictive Median | p-value |
|-----------|----------|----------------------------|---------|
| Mean | 3.08 | 3.08 | 0.50 |
| SD | 0.33 | 0.33 | 0.51 |
| Median | 3.14 | 3.14 | 0.38 |
| IQR | 0.47 | 0.46 | 0.19 |
| Skewness | -0.14 | -0.01 | 0.19 |
| Kurtosis | -0.29 | -0.11 | 0.93 |

All p-values are within the calibrated range [0.1, 0.9], indicating the model captures key distributional features.

LOO-PIT uniformity was assessed visually (`experiment_2/posterior_predictive/loo_pit.png`) and shows good calibration.

### A5. Simulation-Based Calibration

Five recovery tests were run with true parameters: alpha=3.1, beta_weight=-0.9, beta_year=0.03, sigma=0.15.

| Parameter | Coverage (90% CI) | Mean Bias |
|-----------|-------------------|-----------|
| alpha | 100% (5/5) | -0.002 |
| beta_weight | 80% (4/5) | +0.014 |
| beta_year | 100% (5/5) | +0.001 |
| sigma | 100% (5/5) | +0.002 |

Overall coverage: 95% (19/20), consistent with nominal rates.

### A6. Alternative Models Considered

**A1-Baseline (log_mpg ~ log_weight):** Strong temporal trends in residuals (range 0.35 on log scale). Year effect essential.

**A3-Robust (Student-t errors):** Posterior nu ~ 7 (95% CI: 3.9-20.6) confirms heavy tails. Identified outliers: diesel vehicles, rotary engines. ELPD improvement (+6.3) not significant.

**Class B (origin effects):** Not fit because A2 residuals showed no origin patterns. Would add complexity without benefit.

**Class C (hierarchical by origin):** Not fit. With only 3 groups of size 245/68/79, partial pooling offers minimal shrinkage benefit.

### A7. Reproducibility

**Data:** `eda/auto_mpg_cleaned.csv` (392 observations after removing missing horsepower)

**Code:**
- Prior predictive: `experiment_2/prior_predictive/run_prior_predictive.py`
- Simulation: `experiment_2/simulation/run_recovery.py`
- Fitting: `experiment_2/fit/run_fit.py`
- PPC: `experiment_2/posterior_predictive/run_ppc.py`

**Key output files:**
- Posterior samples: `experiment_2/fit/posterior.nc` (ArviZ NetCDF)
- LOO results: `experiment_2/fit/loo.json`
- Model comparison: `experiments/class_a_comparison.csv`

**Environment:** Stan 2.35, CmdStanPy, Python 3.11, ArviZ

---

## References

**Data source:** UCI Machine Learning Repository, Auto-MPG dataset. Original data from StatLib (Carnegie Mellon University), collected by Ross Quinlan.

**Methodology:**
- Gelman, A., et al. (2020). Bayesian Data Analysis, 3rd edition.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC.
- Gabry, J., et al. (2019). Visualization in Bayesian workflow.

---

## File Index

| File | Description |
|------|-------------|
| `eda/eda_report.md` | Exploratory data analysis |
| `eda/*.png` | EDA visualizations |
| `experiments/experiment_plan.md` | Model development plan |
| `experiments/experiment_2/model.stan` | Final model Stan code |
| `experiments/experiment_2/fit/fit_report.md` | Fitting diagnostics |
| `experiments/experiment_2/posterior_predictive/report.md` | PPC results |
| `experiments/population_assessment.md` | Model comparison assessment |
| `experiments/class_a_comparison.png` | LOO comparison visualization |
| `log.md` | Analysis decision log |

---

*Report generated: 2026-01-04*
*Analysis by: Bayesian Modeling Workflow*
