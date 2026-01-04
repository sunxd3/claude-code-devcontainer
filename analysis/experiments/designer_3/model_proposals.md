# Model Proposals: Combined Mechanisms for MPG Prediction

## Overview

This document proposes a ladder of models that synthesize three mechanisms identified in the EDA:

1. **Physical mechanism**: Vehicle weight determines energy requirements for propulsion. Weight has the strongest correlation with MPG (r = -0.83), and the relationship appears nonlinear on the original scale.

2. **Grouping mechanism**: Origin (USA, Europe, Japan) captures market segmentation effects. Japanese and European cars average 10 MPG higher than American cars (eta-squared = 0.33).

3. **Temporal mechanism**: Technological progress and regulatory pressure (CAFE standards, oil crises) drove efficiency improvements of +1.22 MPG per year.

All models use log(MPG) as the outcome, which improves normality and makes effects multiplicative. This means weight effects become elasticities (percent change in MPG per percent change in weight), providing natural physical interpretation.

## Data Structure

```
N = 392 observations (excluding 6 with missing horsepower, though we don't use horsepower)
Outcome: log_mpg = log(mpg)
Predictors:
  - log_weight_c = log(weight) - 7.96  (centered log weight)
  - year_c = model_year - 76           (centered at midpoint)
  - origin in {1, 2, 3}                (USA, Europe, Japan)
```

---

## Model 1: Baseline - Full Additive Model

### Generative Story

Each car's fuel efficiency is determined by three additive components on the log scale:
- A baseline efficiency depending on origin (market segment)
- An efficiency penalty proportional to log(weight)
- A technological improvement proportional to model year

```
log(mpg_i) ~ Normal(mu_i, sigma)
mu_i = alpha + alpha_origin[origin_i] + beta_weight * log_weight_c_i + beta_year * year_c_i
```

### Stan Implementation

```stan
data {
  int<lower=1> N;
  vector[N] log_mpg;
  vector[N] log_weight_c;     // log(weight) - 7.96
  vector[N] year_c;           // model_year - 76
  array[N] int<lower=1, upper=3> origin;
}

transformed data {
  // Contrast matrix for sum-to-zero origin effects
  // origin 3 (Japan) is reference: effect = -(origin1 + origin2)
}

parameters {
  real alpha;                 // grand mean intercept
  vector[2] alpha_origin_raw; // USA and Europe deviations
  real beta_weight;           // log-weight coefficient (elasticity)
  real beta_year;             // year coefficient
  real<lower=0> sigma;        // residual SD
}

transformed parameters {
  vector[3] alpha_origin;
  alpha_origin[1] = alpha_origin_raw[1];
  alpha_origin[2] = alpha_origin_raw[2];
  alpha_origin[3] = -(alpha_origin_raw[1] + alpha_origin_raw[2]);
}

model {
  // Priors
  alpha ~ normal(3.1, 0.5);
  alpha_origin_raw ~ normal(0, 0.3);
  beta_weight ~ normal(-1.0, 0.5);
  beta_year ~ normal(0.05, 0.03);
  sigma ~ exponential(3);

  // Likelihood
  vector[N] mu = alpha + alpha_origin[origin] +
                 beta_weight * log_weight_c +
                 beta_year * year_c;
  log_mpg ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  {
    vector[N] mu = alpha + alpha_origin[origin] +
                   beta_weight * log_weight_c +
                   beta_year * year_c;
    for (i in 1:N) {
      log_lik[i] = normal_lpdf(log_mpg[i] | mu[i], sigma);
      y_rep[i] = normal_rng(mu[i], sigma);
    }
  }
}
```

### Prior Justification

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| alpha | Normal(3.1, 0.5) | Mean log(MPG) is 3.09. Prior covers 10-45 MPG range at 95% level. |
| alpha_origin | Normal(0, 0.3) | EDA shows ~0.4 difference on log scale between origins. Prior allows this while regularizing. |
| beta_weight | Normal(-1, 0.5) | Physical reasoning: elasticity near -1 (doubling weight halves MPG). Allows -0.5 to -1.5. |
| beta_year | Normal(0.05, 0.03) | EDA shows +1.22 MPG/year; on log scale ~0.05. Prior centered there with reasonable uncertainty. |
| sigma | Exponential(3) | Mean 0.33 matches raw log(MPG) SD; expect smaller residuals after conditioning. |

### Mechanism Addressed

This model establishes a baseline that includes all three hypothesized mechanisms but assumes they act independently. The additive structure on the log scale implies multiplicative effects on MPG: a heavier car from a less efficient market segment in an earlier year has compounding disadvantages.

### Falsification Criteria

Abandon or modify if:
- Residual plots show systematic origin-by-year interaction patterns
- Posterior predictive checks show poor coverage of extreme MPG values
- PSIS-LOO shows excessive Pareto k values (>0.7) for specific observations

---

## Model 2: Scientific - Differential Technological Progress

### Generative Story

Building on Model 1, we hypothesize that technological progress differed by origin. The EDA shows:
- USA: +1.12 MPG/year
- Europe: +1.03 MPG/year
- Japan: +0.95 MPG/year

American manufacturers may have had more room to improve, or faced different regulatory pressures. This model adds year-by-origin interactions:

```
log(mpg_i) ~ Normal(mu_i, sigma)
mu_i = alpha + alpha_origin[origin_i] + beta_weight * log_weight_c_i
       + (beta_year + beta_year_origin[origin_i]) * year_c_i
```

### Stan Implementation

```stan
data {
  int<lower=1> N;
  vector[N] log_mpg;
  vector[N] log_weight_c;
  vector[N] year_c;
  array[N] int<lower=1, upper=3> origin;
}

parameters {
  real alpha;
  vector[2] alpha_origin_raw;
  real beta_weight;
  real beta_year;                // average year effect
  vector[2] beta_year_origin_raw; // origin-specific deviations
  real<lower=0> sigma;
}

transformed parameters {
  vector[3] alpha_origin;
  vector[3] beta_year_origin;

  alpha_origin[1] = alpha_origin_raw[1];
  alpha_origin[2] = alpha_origin_raw[2];
  alpha_origin[3] = -(alpha_origin_raw[1] + alpha_origin_raw[2]);

  beta_year_origin[1] = beta_year_origin_raw[1];
  beta_year_origin[2] = beta_year_origin_raw[2];
  beta_year_origin[3] = -(beta_year_origin_raw[1] + beta_year_origin_raw[2]);
}

model {
  // Priors
  alpha ~ normal(3.1, 0.5);
  alpha_origin_raw ~ normal(0, 0.3);
  beta_weight ~ normal(-1.0, 0.5);
  beta_year ~ normal(0.05, 0.03);
  beta_year_origin_raw ~ normal(0, 0.02);  // small deviations expected
  sigma ~ exponential(3);

  // Likelihood
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = alpha + alpha_origin[origin[i]] +
            beta_weight * log_weight_c[i] +
            (beta_year + beta_year_origin[origin[i]]) * year_c[i];
  }
  log_mpg ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  {
    vector[N] mu;
    for (i in 1:N) {
      mu[i] = alpha + alpha_origin[origin[i]] +
              beta_weight * log_weight_c[i] +
              (beta_year + beta_year_origin[origin[i]]) * year_c[i];
      log_lik[i] = normal_lpdf(log_mpg[i] | mu[i], sigma);
      y_rep[i] = normal_rng(mu[i], sigma);
    }
  }

  // Derived: total year effect by origin
  vector[3] total_year_effect;
  for (j in 1:3) {
    total_year_effect[j] = beta_year + beta_year_origin[j];
  }
}
```

### Prior Justification

New parameter:

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| beta_year_origin | Normal(0, 0.02) | Origin-specific deviations from mean trend. EDA suggests differences of ~0.01 on log scale. Tighter prior reflects skepticism about large differential effects. |

### Mechanism Addressed

This model tests whether the "catch-up" phenomenon observed in EDA (American cars improving faster than Japanese) represents a real differential or noise. The interaction structure allows different technological trajectories while sharing information about the overall trend.

### Falsification Criteria

Abandon or simplify back to Model 1 if:
- Posterior for beta_year_origin concentrates near zero (95% interval includes zero with most mass there)
- PSIS-LOO or WAIC shows no improvement over Model 1
- The added complexity does not improve posterior predictive performance

---

## Model 3: Extension - Robust Errors with Student-t

### Generative Story

The EDA identified influential observations:
- VW Rabbit Diesel at 46.6 MPG (statistical outlier)
- Heavy European diesels achieving unexpectedly high MPG
- Large American V8s at very low MPG

Rather than removing these observations (which represent real vehicles), we accommodate them with heavier tails. The Student-t distribution has a degrees-of-freedom parameter nu that controls tail weight: lower nu means heavier tails.

```
log(mpg_i) ~ Student-t(nu, mu_i, sigma)
mu_i = alpha + alpha_origin[origin_i] + beta_weight * log_weight_c_i
       + (beta_year + beta_year_origin[origin_i]) * year_c_i
```

### Stan Implementation

```stan
data {
  int<lower=1> N;
  vector[N] log_mpg;
  vector[N] log_weight_c;
  vector[N] year_c;
  array[N] int<lower=1, upper=3> origin;
}

parameters {
  real alpha;
  vector[2] alpha_origin_raw;
  real beta_weight;
  real beta_year;
  vector[2] beta_year_origin_raw;
  real<lower=0> sigma;
  real<lower=1> nu;             // degrees of freedom
}

transformed parameters {
  vector[3] alpha_origin;
  vector[3] beta_year_origin;

  alpha_origin[1] = alpha_origin_raw[1];
  alpha_origin[2] = alpha_origin_raw[2];
  alpha_origin[3] = -(alpha_origin_raw[1] + alpha_origin_raw[2]);

  beta_year_origin[1] = beta_year_origin_raw[1];
  beta_year_origin[2] = beta_year_origin_raw[2];
  beta_year_origin[3] = -(beta_year_origin_raw[1] + beta_year_origin_raw[2]);
}

model {
  // Priors
  alpha ~ normal(3.1, 0.5);
  alpha_origin_raw ~ normal(0, 0.3);
  beta_weight ~ normal(-1.0, 0.5);
  beta_year ~ normal(0.05, 0.03);
  beta_year_origin_raw ~ normal(0, 0.02);
  sigma ~ exponential(3);
  nu ~ gamma(2, 0.1);           // weakly informative on df

  // Likelihood with Student-t errors
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = alpha + alpha_origin[origin[i]] +
            beta_weight * log_weight_c[i] +
            (beta_year + beta_year_origin[origin[i]]) * year_c[i];
  }
  log_mpg ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  {
    vector[N] mu;
    for (i in 1:N) {
      mu[i] = alpha + alpha_origin[origin[i]] +
              beta_weight * log_weight_c[i] +
              (beta_year + beta_year_origin[origin[i]]) * year_c[i];
      log_lik[i] = student_t_lpdf(log_mpg[i] | nu, mu[i], sigma);
      y_rep[i] = student_t_rng(nu, mu[i], sigma);
    }
  }

  // Identify potential outliers via log-likelihood
  vector[N] outlier_score;
  {
    real threshold = student_t_lpdf(0.0 | nu, 0.0, sigma) - 4.0;
    for (i in 1:N) {
      outlier_score[i] = (log_lik[i] < threshold) ? 1.0 : 0.0;
    }
  }
}
```

### Prior Justification

New parameter:

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| nu | Gamma(2, 0.1) | Mean ~20, but allows values from 3-50. Low values (3-10) indicate heavy tails; values >30 approach normality. Prior is skeptical of extreme non-normality but allows data to inform. |

### Mechanism Addressed

This model addresses robustness to outliers without manual data cleaning. The Student-t likelihood downweights observations far from the predicted mean, making coefficient estimates less sensitive to extreme values. The posterior for nu tells us how heavy the tails need to be.

### Falsification Criteria

Simplify back to normal errors (Model 2) if:
- Posterior for nu concentrates at high values (median > 30), indicating normal is adequate
- No observations are flagged as outliers by the outlier_score metric
- Coefficient estimates are nearly identical to Model 2

---

## Model Comparison Strategy

### Within-Sample Diagnostics

For each fitted model, examine:
1. **Traceplots and R-hat**: All parameters should have R-hat < 1.01
2. **Effective sample size**: ESS > 400 for reliable posterior summaries
3. **Posterior predictive checks**: Compare y_rep distributions to observed data

### Cross-Validation

Compare models using PSIS-LOO:
1. Compute LOO-IC for each model
2. Compare using loo_compare() - prefer lower values
3. Examine Pareto k diagnostics - high k values indicate influential observations

### Interpretive Checks

1. **Do coefficients make physical sense?** Weight elasticity should be negative and near -1.
2. **Are origin effects consistent with EDA?** USA should be less efficient than Japan/Europe.
3. **Does the year effect match expectations?** Should be positive, around 0.05 on log scale.
4. **For Model 3: What does nu tell us?** If nu < 10, outliers are meaningful; if nu > 30, normal errors suffice.

### Decision Criteria

- If Model 2 shows no improvement over Model 1 via LOO: use Model 1 for parsimony
- If Model 3 shows nu > 30: use Model 2 (normal errors adequate)
- If Model 3 shows nu < 15 and improves LOO: prefer Model 3 for final inference

---

## Computational Notes

**Identifiability**: Sum-to-zero constraints on origin effects ensure the intercept alpha represents the grand mean. Without this, alpha and the origin effects would be unidentified up to a constant.

**Centering**: Predictors are centered to reduce posterior correlation between intercept and slopes. This improves sampling efficiency.

**No hierarchical structure**: With only 3 origin groups, we use fixed effects rather than partial pooling. Hierarchical models require more groups (typically 5+) to estimate group-level variance reliably.

**Student-t sampling**: The nu parameter can cause slow mixing if it explores very low values. The gamma(2, 0.1) prior constrains nu >= 1 and regularizes toward moderate values.

**Vectorization**: The likelihood is written with explicit loops for clarity but could be vectorized for efficiency. Stan's automatic differentiation handles either form.

---

## Summary Table

| Model | Mechanism | New Parameters | Expected Improvement |
|-------|-----------|----------------|---------------------|
| M1 Baseline | Additive synthesis | alpha, alpha_origin[3], beta_weight, beta_year, sigma | Establishes baseline with all mechanisms |
| M2 Scientific | Differential trends | + beta_year_origin[3] | Tests year-by-origin interaction |
| M3 Extension | Robust errors | + nu | Handles outliers without data exclusion |

The model ladder progresses from simple additive structure to richer interactions and finally to robust error modeling. Each step adds complexity only where EDA suggests it is warranted, and each has clear falsification criteria for when to step back.
