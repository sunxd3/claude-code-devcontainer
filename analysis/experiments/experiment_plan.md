# Consolidated Experiment Plan

**Date**: 2026-01-14
**Source**: Synthesized from designer_1 (hierarchical pooling) and designer_2 (treatment heterogeneity)

## Overview

Build a ladder of Bayesian models to estimate treatment effects on student test scores while accounting for school-level grouping. The key question: how much heterogeneity exists in treatment effects across schools, and what structure best captures it?

**Data**: 160 students across 8 schools, binary treatment, continuous test scores
**True DGP**: Hierarchical model with tau=8 (school intercepts), beta=5 (treatment), sigma=10 (residual)

## Model Ladder

### Experiment 1: Complete Pooling (Baseline)

**Purpose**: Performance floor. Ignores school structure entirely.

```stan
y_i ~ Normal(alpha + beta * treatment_i, sigma)

Priors:
  alpha ~ Normal(77, 15)
  beta ~ Normal(5, 5)
  sigma ~ Half-Normal(0, 15)
```

**Falsification**: Poor LOO vs hierarchical models; systematic school-level residual patterns.

---

### Experiment 2: Random Intercepts Only

**Purpose**: Test if treatment heterogeneity is real or just noise.

```stan
y_ij ~ Normal(alpha_j + beta * treatment_ij, sigma)
alpha_j = alpha_0 + tau_alpha * z_alpha_j
z_alpha_j ~ Normal(0, 1)

Priors:
  alpha_0 ~ Normal(77, 15)
  tau_alpha ~ Half-Normal(0, 10)
  beta ~ Normal(5, 5)
  sigma ~ Half-Normal(0, 15)
```

**Falsification**: Worse LOO than random slopes model; systematic treatment effect residuals by school.

---

### Experiment 3: Random Intercepts + Random Slopes (Scientific Target)

**Purpose**: Primary target model. Captures both school baseline differences and treatment effect heterogeneity.

```stan
y_ij ~ Normal(alpha_j + beta_j * treatment_ij, sigma)
alpha_j = alpha_0 + tau_alpha * z_alpha_j
beta_j = beta_0 + tau_beta * z_beta_j
z_alpha_j ~ Normal(0, 1)
z_beta_j ~ Normal(0, 1)

Priors:
  alpha_0 ~ Normal(77, 15)
  tau_alpha ~ Half-Normal(0, 10)
  beta_0 ~ Normal(5, 5)
  tau_beta ~ Half-Normal(0, 5)
  sigma ~ Half-Normal(0, 15)
```

**Justification**: EDA shows ICC ~ 0.14-0.39 and treatment effect SD ~ 5.16 across schools.

**Falsification**:
- tau_beta posterior near zero → revert to random intercepts only
- Poor posterior predictive fit for extreme schools (1, 3)

---

### Experiment 4: Correlated Random Effects

**Purpose**: Test if high-baseline schools respond differently to treatment.

```stan
y_ij ~ Normal(alpha_j + beta_j * treatment_ij, sigma)
[alpha_j, gamma_j]' ~ MVNormal([alpha_0, 0]', Sigma)
beta_j = beta_0 + gamma_j
Sigma = diag(tau) * Omega * diag(tau)
Omega ~ LKJCorr(2)

Priors:
  alpha_0 ~ Normal(77, 15)
  beta_0 ~ Normal(5, 5)
  tau ~ Half-Normal([0, 0], [10, 5])
  sigma ~ Half-Normal(0, 15)
```

**Falsification**: Correlation posterior includes zero → uncorrelated model sufficient.

---

### Experiment 5: Student-t Errors (Robust)

**Purpose**: Test if extreme school effects are outliers vs true heterogeneity.

```stan
y_ij ~ StudentT(nu, alpha_j + beta_j * treatment_ij, sigma)
// Same random effects structure as Experiment 3

Additional prior:
  nu ~ Gamma(2, 0.1)
```

**Falsification**: nu posterior > 30 → Normal likelihood sufficient.

---

## Validation Strategy

For each experiment:

1. **Prior predictive check**: Verify priors produce plausible data
2. **Parameter recovery**: Fit to simulated data with known truth
3. **Model fitting**: Fit to actual data with diagnostics
4. **Posterior predictive check**: Verify model can reproduce data patterns
5. **Model critique**: Assess strengths and weaknesses

## Comparison Strategy

- Compare all models via LOO-CV
- Check posterior predictive fit for school-specific effects
- Verify appropriate shrinkage for extreme schools (1 and 3)
- Report both population-average and school-specific treatment effects

## Expected Outcomes

- Baseline will underperform (floor)
- Random intercepts will improve substantially
- Random slopes (Exp 3) expected to be best based on EDA heterogeneity
- Correlated RE may not improve (weak correlation in EDA)
- Student-t unlikely to help (normal likelihood already appropriate)

## Experiment Directory Structure

```
experiments/
  experiment_1/   # Complete pooling
  experiment_2/   # Random intercepts only
  experiment_3/   # Random intercepts + slopes (target)
  experiment_4/   # Correlated RE
  experiment_5/   # Student-t robust
```

Each experiment folder will contain:
- `model.stan` - Stan model code
- `prior_predictive/` - Prior predictive check outputs
- `simulation/` - Parameter recovery outputs
- `fit/` - Model fitting outputs
- `posterior_predictive/` - PPC outputs
- `critique/` - Model assessment
