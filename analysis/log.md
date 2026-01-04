# Auto-MPG Bayesian Modeling Log

## Project Overview
- **Dataset**: Auto-MPG (UCI ML Repository classic)
- **Goal**: Build Bayesian models to understand/predict fuel efficiency (MPG)
- **Started**: 2026-01-04

## Data Summary (Initial)
- 398 observations (1 blank line at end)
- Columns: mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, car_name
- Missing values observed in horsepower (marked as "?")
- Target variable: MPG (miles per gallon)

---

## Phase 1: Exploratory Data Analysis

**Status**: COMPLETE
**Approach**: Single EDA analyst to explore the dataset comprehensively

### Key Findings:
1. **Target (MPG)**: Right-skewed (skewness=0.46), log-transform improves normality
2. **Primary predictor**: Weight has strongest correlation (r=-0.83), nonlinear relationship
3. **Multicollinearity**: Severe among weight, displacement, horsepower (VIF > 10) - don't use together
4. **Grouping effects**:
   - Origin: Large effect (eta²=0.33), Japan/Europe avg 10 mpg more than USA
   - Cylinders: Strong segmentation but confounded with origin (all 8-cyl are American)
5. **Temporal trend**: +1.22 mpg/year, sharp jump in 1980
6. **Missing data**: Only 6 horsepower values (1.5%), MCAR pattern
7. **Outliers**: Few and genuine (VW Diesel at 46.6 mpg, some large V8s)

### Data-Generating Process Hypotheses:
- **H1 Physical**: MPG ≈ f(weight) via physics (energy to move mass)
- **H2 Market Segmentation**: Origin/cylinders define market segments with different baselines
- **H3 Technological Progress**: Time trend independent of vehicle characteristics

### Modeling Recommendations:
- Likelihood: Normal on log(MPG) or lognormal
- Primary predictor: log(weight) (interpretable as elasticity)
- Additional: model_year (centered at 76), origin (categorical or hierarchical)
- Avoid: Including weight + displacement + horsepower together

### Files Created:
- `eda/eda_report.md` - Comprehensive report
- `eda/auto_mpg_cleaned.csv` - Cleaned dataset
- Various diagnostic plots (mpg_distribution.png, scatterplot_matrix.png, etc.)

---

## Phase 2: Model Design

**Status**: COMPLETE
**Approach**: 3 parallel model designers with different structural perspectives

### Designer Perspectives:

**Designer 1 (Physical/Mechanistic)**:
- Log-log baseline (log_mpg ~ log_weight)
- Power law with technological drift (+year)
- Robust errors with Student-t

**Designer 2 (Hierarchical Grouping)**:
- Fixed effects for origin
- Hierarchical partial pooling
- Varying slopes by origin

**Designer 3 (Combined Mechanisms)**:
- Full additive model (weight + year + origin)
- Year × origin interaction
- Student-t errors

### Synthesized Experiment Plan:

| Class | Hypothesis | Experiments |
|-------|------------|-------------|
| A: Physical | Weight dominates | exp_1 (baseline), exp_2 (+year), exp_3 (robust) |
| B: Combined | Additive effects | exp_4 (additive), exp_5 (+interaction), exp_6 (robust) |
| C: Hierarchical | Origin grouping | exp_7 (fixed), exp_8 (pooling), exp_9 (varying slopes) |

### Expected Winner:
Based on EDA, Class B (B1 Full Additive or B2 with interaction) most likely to balance fit and parsimony.

### Files Created:
- `experiments/designer_1/model_proposals.md`
- `experiments/designer_2/model_proposals.md`
- `experiments/designer_3/model_proposals.md`
- `experiments/experiment_plan.md` (synthesized)

---

## Phase 3: Model Development and Validation

**Status**: COMPLETE
**Approach**: Sequential validation of Class A models; Class B/C skipped based on evidence

### Class A Results:

| Experiment | Model | ELPD | sigma | Verdict |
|------------|-------|------|-------|---------|
| 1 | A1-Baseline (log_mpg ~ log_weight) | 147.6 | 0.166 | VIABLE but needs year |
| 2 | A2-Year (+ year) | 279.7 | 0.118 | **RECOMMENDED** |
| 3 | A3-Robust (Student-t) | 286.1 | 0.100 | nu~7 but gain not significant |

### Key Findings:

1. **Year effect is essential**: ELPD improved by +132 when adding year
2. **Origin NOT needed**: After conditioning on weight + year, all origin residual t-stats < 2
3. **Robust errors optional**: nu~7 confirms outliers but ELPD gain only +6.4 (z=0.26)
4. **Physical model validated**: beta_weight = -0.94 ≈ -1 (physics prediction)

### Decision Path:
- Model selector returned: **ADEQUATE**
- Decision auditor returned: **ACCEPT**
- Reason for skipping Class B/C: Origin effects vanish when conditioning on weight + year

### Final Model: A2-Year
```
log(mpg) ~ Normal(alpha + beta_weight * log_weight_c + beta_year * year_c, sigma)

Posteriors:
  alpha = 3.098 [3.087, 3.110]      # 22.2 mpg at mean weight in 1976
  beta_weight = -0.935 [-0.979, -0.893]  # ~1% weight increase = 0.94% mpg decrease
  beta_year = 0.033 [0.030, 0.036]  # 3.3% mpg improvement per year
  sigma = 0.118 [0.110, 0.127]      # ~12% multiplicative error
```

---

## Phase 6: Final Report

**Status**: COMPLETE
**Output**: `final_report.md`

The final report (404 lines) includes:
- Executive Summary with 4 key findings
- Data and Methods (full workflow description)
- Results (model comparison, parameter estimates, validation)
- Interpretation ("The Origin Myth" - key scientific insight)
- Limitations and Future Work
- Technical Appendix (priors, diagnostics, reproducibility)

---

## Project Summary

**Completed**: 2026-01-04

**Key Scientific Finding**: The apparent Japanese/European fuel efficiency advantage was a compositional effect (lighter cars, later market timing), not manufacturing superiority. Weight elasticity ≈ -1 matches physics; technology improved 3.3%/year across all origins.

**Workflow Statistics**:
- 3 experiments validated (Class A)
- 4 validation stages per experiment (prior predictive, recovery, fit, PPC)
- Class B/C skipped (empirically justified)
- Final model: A2-Year with ELPD = 279.7

