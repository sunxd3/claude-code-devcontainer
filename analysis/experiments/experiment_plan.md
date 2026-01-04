# Experiment Plan: Auto-MPG Bayesian Models

## Overview

This plan synthesizes proposals from three model designers, each exploring a different structural hypothesis. The models form a ladder from simple physical relationships to complex hierarchical structures.

## Model Classes

Based on the designer proposals, we organize models into three classes representing distinct structural hypotheses:

### Class A: Simple Physical Models
**Perspective**: Weight is the dominant mechanism; origin and year effects are secondary or absent.
**Source**: Designer 1

| Experiment | Model | Structure | Key Test |
|------------|-------|-----------|----------|
| experiment_1 | A1-Baseline | log_mpg ~ log_weight | Does physics alone suffice? |
| experiment_2 | A2-Year | log_mpg ~ log_weight + year | Does technology trend add value? |
| experiment_3 | A3-Robust | Same + Student-t errors | Are outliers problematic? |

### Class B: Combined Additive Models
**Perspective**: Weight, year, and origin all contribute additively.
**Source**: Designer 3

| Experiment | Model | Structure | Key Test |
|------------|-------|-----------|----------|
| experiment_4 | B1-Additive | log_mpg ~ log_weight + year + origin | Full additive baseline |
| experiment_5 | B2-Interaction | Same + year×origin | Do origins have different trends? |
| experiment_6 | B3-Robust | Same + Student-t | Robustness to outliers |

### Class C: Hierarchical by Origin
**Perspective**: Origin creates meaningful grouping structure requiring partial pooling.
**Source**: Designer 2

| Experiment | Model | Structure | Key Test |
|------------|-------|-----------|----------|
| experiment_7 | C1-Fixed | Origin as fixed effects | No-pooling baseline |
| experiment_8 | C2-Hierarchical | Origin with partial pooling | Does pooling help with 3 groups? |
| experiment_9 | C3-VaryingSlopes | Intercepts + slopes vary by origin | Do weight effects differ by market? |

## Execution Strategy

### Phase 1: Class A - Physical Models
Start with simplest models to establish baseline performance.

1. **Experiment 1 (A1-Baseline)**:
   - Prior predictive check → Recovery check → Fit → Posterior predictive check → Critique
   - Expected: Good baseline, but residuals may show origin patterns

2. **Experiment 2 (A2-Year)**:
   - Full validation pipeline
   - Compare to experiment_1 via LOO

3. **Experiment 3 (A3-Robust)**:
   - Full validation pipeline
   - Check posterior of nu to assess need for robustness

**Decision point**: If Class A residuals show clear origin structure, proceed to Class B/C. If Class A with robustness (A3) performs well with nu > 30, simpler models may suffice.

### Phase 2: Class B - Combined Models
Add origin effects to the best physical model.

4. **Experiment 4 (B1-Additive)**:
   - Full validation pipeline
   - This is the EDA-recommended model

5. **Experiment 5 (B2-Interaction)**:
   - Full validation pipeline
   - Compare to experiment_4

6. **Experiment 6 (B3-Robust)**:
   - Full validation pipeline
   - Compare to experiment_5

### Phase 3: Class C - Hierarchical
Test whether partial pooling is beneficial with only 3 groups.

7. **Experiment 7 (C1-Fixed)**:
   - Should be similar to B1 but with sum-to-zero constraint
   - Serves as non-pooling benchmark

8. **Experiment 8 (C2-Hierarchical)**:
   - Test partial pooling with non-centered parameterization
   - Examine shrinkage diagnostic

9. **Experiment 9 (C3-VaryingSlopes)**:
   - Most complex model
   - Test whether weight effects truly differ by origin

## Data Preparation

All models use the same prepared data:

```python
# Standard transforms
log_mpg = np.log(df['mpg'])
log_weight_c = np.log(df['weight']) - 7.96  # centered at mean log(weight)
year_c = df['model_year'] - 76  # centered at midpoint

# For hierarchical models
origin = df['origin'].values  # 1=USA, 2=Europe, 3=Japan
J = 3  # number of groups
```

## Prior Summary

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| alpha | Normal(3.1, 0.5) | Mean log(MPG) ~ 3.1 |
| beta_weight | Normal(-1, 0.5) | Physics predicts elasticity ~ -1 |
| beta_year | Normal(0.05, 0.03) | ~1.2 mpg/year = ~0.05 on log scale |
| alpha_origin | Normal(0, 0.3) | Origin differences ~0.3-0.4 on log scale |
| sigma | Exponential(3) | Expect residual SD < 0.33 |
| nu | Gamma(2, 0.1) | Mean ~20, allows 3-50+ |
| tau (hierarchical) | HalfNormal(0, 0.3) | Weakly informative for group SD |

## Success Criteria

A model is considered validated if:
1. **Prior predictive**: Generates plausible MPG distributions (8-50 mpg)
2. **Recovery**: Can recover true parameters from simulated data
3. **Convergence**: R-hat < 1.01, ESS > 400, no divergences
4. **Posterior predictive**: Captures observed MPG distribution and relationships

## Expected Outcome

Based on EDA findings, we expect:
- Class A will underperform due to missing origin effects
- Class B (B1 or B2) will likely be the best balance of fit and parsimony
- Class C may offer marginal improvements but partial pooling with 3 groups is limited
- Robust errors (nu parameter) will likely show nu > 20, indicating normal errors suffice

The final model will likely be **B1 (Full Additive)** or **B2 (with interaction)** depending on LOO comparison.

## File Organization

```
experiments/
  experiment_plan.md          # This file
  experiment_1/               # A1-Baseline
    model.stan
    prior_predictive/
    simulation/
    fit/
    posterior_predictive/
    critique/
  experiment_2/               # A2-Year
    ...
  (etc)
  model_assessment/
    assessment_report.md
```
