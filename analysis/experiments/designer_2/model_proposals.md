# Hierarchical Model Proposals: Market Segmentation Perspective

## Conceptual Framework

Vehicles are designed for market segments with different priorities. USA manufacturers historically emphasized size and power; Japanese manufacturers optimized for fuel efficiency; European manufacturers balanced performance with economy. This creates natural grouping where vehicles from the same origin share baseline fuel efficiency characteristics beyond what their physical attributes explain.

The hierarchical modeling question: how much should inference about one origin inform estimates for others? With 249 USA, 70 Europe, and 79 Japan observations, each group has sufficient data for independent estimation, but partial pooling can still improve estimates by regularizing toward the grand mean.

Key EDA findings supporting this perspective:
- Origin explains 33% of MPG variance (eta-squared = 0.33)
- Mean MPG differs substantially: USA=20.1, Europe=27.9, Japan=30.5
- On log scale, this is roughly 3.0, 3.3, 3.4 (differences of 0.3-0.4 log units)
- Weight effect may differ by origin due to different design philosophies

---

## Model 1: Fixed Effects Baseline

### Generative Story

Each origin has its own intercept estimated independently. There is no information sharing across origins. This represents the "no pooling" extreme and sets a performance baseline.

```
log(mpg_i) = alpha + alpha_origin[j] + beta_w * log_weight_c_i + beta_y * year_c_i + epsilon_i
epsilon_i ~ Normal(0, sigma)
```

where `j = origin[i]` and `alpha_origin` uses sum-to-zero constraint for identifiability.

### Stan Specification

```stan
data {
  int<lower=1> N;
  int<lower=1> J;                          // number of origins (3)
  vector[N] log_mpg;
  vector[N] log_weight_c;                  // centered: log(weight) - 7.96
  vector[N] year_c;                        // centered: year - 76
  array[N] int<lower=1,upper=J> origin;
}

transformed data {
  real log_mpg_mean = mean(log_mpg);       // approximately 3.09
  real log_mpg_sd = sd(log_mpg);           // approximately 0.33
}

parameters {
  real alpha;                              // grand intercept
  vector[J-1] alpha_origin_raw;            // origin effects (J-1 for identifiability)
  real beta_weight;                        // log(weight) coefficient
  real beta_year;                          // year coefficient
  real<lower=0> sigma;                     // residual SD
}

transformed parameters {
  vector[J] alpha_origin;
  alpha_origin[1:J-1] = alpha_origin_raw;
  alpha_origin[J] = -sum(alpha_origin_raw);  // sum-to-zero constraint
}

model {
  // Priors
  alpha ~ normal(3.1, 0.5);                // centered on observed mean
  alpha_origin_raw ~ normal(0, 0.5);       // weakly informative, allows large origin effects
  beta_weight ~ normal(-1, 0.5);           // physical prior: elasticity near -1
  beta_year ~ normal(0.02, 0.02);          // ~1.2 mpg/year implies ~0.02 on log scale
  sigma ~ exponential(3);                  // mode at 0.33, plausible residual SD

  // Likelihood
  vector[N] mu = alpha + alpha_origin[origin] + beta_weight * log_weight_c + beta_year * year_c;
  log_mpg ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    real mu_n = alpha + alpha_origin[origin[n]] + beta_weight * log_weight_c[n] + beta_year * year_c[n];
    log_lik[n] = normal_lpdf(log_mpg[n] | mu_n, sigma);
    y_rep[n] = normal_rng(mu_n, sigma);
  }
}
```

### Prior Justification

The prior on `alpha_origin_raw ~ Normal(0, 0.5)` is weakly informative. On the log scale, origin effects of 0.3-0.4 (observed in data) are within one standard deviation. This allows the data to determine origin effects while providing mild regularization.

The weight prior `Normal(-1, 0.5)` encodes physical reasoning: if mpg ~ 1/weight (constant fuel consumption per mile), then log(mpg) ~ -log(weight), implying beta = -1. The prior allows deviation from this ideal while keeping estimates in a physically plausible range.

### Expected Behavior

This model should fit well given the strong origin effects in the data. With 70+ observations per group, estimates will be data-dominated. The sum-to-zero parameterization centers origin effects at zero, making alpha interpretable as the average across origins.

### Falsification Criteria

- **Abandon if**: Posterior predictive checks show systematic bias within origin groups (e.g., underpredicting high-MPG Japanese cars), indicating the linear-in-log-weight assumption fails differently by origin.
- **Simplify if**: Origin effects are indistinguishable from zero (unlikely given EDA), suggesting weight alone suffices.

---

## Model 2: Hierarchical Partial Pooling (Scientific Model)

### Generative Story

Origin effects are drawn from a common distribution. This represents the belief that while origins differ, they are not completely unrelated; they share a common automotive context. The hyperparameter tau controls how much origins can deviate from the grand mean.

```
alpha_origin[j] ~ Normal(0, tau)       for j = 1, ..., J
tau ~ HalfNormal(0, 0.3)               hyperprior on group variation
log(mpg_i) = alpha + alpha_origin[j] + beta_w * log_weight_c_i + beta_y * year_c_i + epsilon_i
```

### Stan Specification

```stan
data {
  int<lower=1> N;
  int<lower=1> J;                          // number of origins (3)
  vector[N] log_mpg;
  vector[N] log_weight_c;
  vector[N] year_c;
  array[N] int<lower=1,upper=J> origin;
}

parameters {
  real alpha;
  vector[J] alpha_origin_raw;              // non-centered: unit normal
  real<lower=0> tau;                       // SD of origin effects
  real beta_weight;
  real beta_year;
  real<lower=0> sigma;
}

transformed parameters {
  vector[J] alpha_origin = tau * alpha_origin_raw;  // non-centered parameterization
}

model {
  // Hyperpriors
  tau ~ normal(0, 0.3);                    // half-normal (tau > 0)

  // Priors on group effects (standard normal due to non-centering)
  alpha_origin_raw ~ std_normal();

  // Fixed effect priors
  alpha ~ normal(3.1, 0.5);
  beta_weight ~ normal(-1, 0.5);
  beta_year ~ normal(0.02, 0.02);
  sigma ~ exponential(3);

  // Likelihood
  vector[N] mu = alpha + alpha_origin[origin] + beta_weight * log_weight_c + beta_year * year_c;
  log_mpg ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  real origin_shrinkage = tau^2 / (tau^2 + sigma^2 / 70);  // approximate shrinkage factor

  for (n in 1:N) {
    real mu_n = alpha + alpha_origin[origin[n]] + beta_weight * log_weight_c[n] + beta_year * year_c[n];
    log_lik[n] = normal_lpdf(log_mpg[n] | mu_n, sigma);
    y_rep[n] = normal_rng(mu_n, sigma);
  }
}
```

### Prior Justification

The key prior is `tau ~ HalfNormal(0, 0.3)`. This deserves careful thought:

**Why 0.3?** On the log scale, an origin effect of 0.3 corresponds to roughly 35% difference in MPG (exp(0.3) = 1.35). The observed differences are:
- Japan vs USA: log(30.5/20.1) = 0.42
- Europe vs USA: log(27.9/20.1) = 0.33

So tau ~ 0.3-0.4 would be consistent with the data. The HalfNormal(0, 0.3) prior places most mass below 0.6 while allowing larger values, thus being informative enough to identify tau with only 3 groups while not overwhelming the data.

**Non-centered parameterization**: With only 3 groups, the centered parameterization can create a "funnel" geometry where tau and alpha_origin become correlated in ways that challenge sampling. The non-centered form (alpha_origin = tau * alpha_origin_raw) breaks this correlation.

### Expected Behavior

Given the sample sizes (70-249 per origin) and strong origin effects, we expect modest shrinkage. The posterior for tau should concentrate around 0.2-0.4, reflecting genuine origin variation. The hierarchical model should produce similar point estimates to the fixed effects model but with better-calibrated uncertainty.

The `origin_shrinkage` generated quantity provides a diagnostic: values near 1 indicate minimal pooling (like fixed effects), values near 0 indicate heavy pooling toward grand mean.

### Falsification Criteria

- **Abandon if**: tau posterior concentrates at 0, suggesting no meaningful origin variation after controlling for weight (would contradict strong EDA evidence).
- **Abandon if**: tau posterior is extremely diffuse with no concentration, suggesting the prior dominates (unlikely with this sample size).
- **Simplify to baseline if**: Fixed effects and hierarchical estimates diverge substantially, indicating the pooling assumption is inappropriate. This would suggest origins are so different that borrowing strength is harmful.
- **Extend if**: Residuals within origin groups show different patterns, suggesting origin-specific slopes are needed.

---

## Model 3: Varying Slopes Extension

### Generative Story

Both intercepts and weight slopes vary by origin. This encodes the hypothesis that different automotive design philosophies create different weight-to-efficiency relationships:
- Japanese manufacturers may achieve consistent efficiency across weight ranges through design optimization
- American manufacturers may show steeper MPG degradation with weight due to less aerodynamic designs

```
alpha_origin[j] ~ Normal(0, tau_alpha)
beta_origin[j] ~ Normal(0, tau_beta)
log(mpg_i) = alpha + alpha_origin[j] + (beta_w + beta_origin[j]) * log_weight_c_i + beta_y * year_c_i + epsilon_i
```

### Stan Specification

```stan
data {
  int<lower=1> N;
  int<lower=1> J;
  vector[N] log_mpg;
  vector[N] log_weight_c;
  vector[N] year_c;
  array[N] int<lower=1,upper=J> origin;
}

parameters {
  real alpha;
  real beta_weight;                        // population-level weight effect
  real beta_year;

  // Hierarchical parameters (non-centered)
  vector[J] z_alpha;                       // standardized origin intercepts
  vector[J] z_beta;                        // standardized origin slopes
  real<lower=0> tau_alpha;                 // SD of origin intercepts
  real<lower=0> tau_beta;                  // SD of origin slope deviations

  real<lower=0> sigma;
}

transformed parameters {
  vector[J] alpha_origin = tau_alpha * z_alpha;
  vector[J] beta_origin = tau_beta * z_beta;
}

model {
  // Hyperpriors
  tau_alpha ~ normal(0, 0.3);              // intercept variation
  tau_beta ~ normal(0, 0.3);               // slope variation (same scale, different meaning)

  // Standardized group effects
  z_alpha ~ std_normal();
  z_beta ~ std_normal();

  // Fixed effect priors
  alpha ~ normal(3.1, 0.5);
  beta_weight ~ normal(-1, 0.5);
  beta_year ~ normal(0.02, 0.02);
  sigma ~ exponential(3);

  // Likelihood
  vector[N] mu;
  for (n in 1:N) {
    int j = origin[n];
    mu[n] = alpha + alpha_origin[j] + (beta_weight + beta_origin[j]) * log_weight_c[n] + beta_year * year_c[n];
  }
  log_mpg ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  vector[J] total_slope = beta_weight + beta_origin;  // origin-specific weight effects

  for (n in 1:N) {
    int j = origin[n];
    real mu_n = alpha + alpha_origin[j] + (beta_weight + beta_origin[j]) * log_weight_c[n] + beta_year * year_c[n];
    log_lik[n] = normal_lpdf(log_mpg[n] | mu_n, sigma);
    y_rep[n] = normal_rng(mu_n, sigma);
  }
}
```

### Prior Justification

The prior `tau_beta ~ HalfNormal(0, 0.3)` reflects uncertainty about whether slope variation exists. On the log-log scale, a slope difference of 0.3 means that for a 10% increase in weight, MPG changes by an additional ~3% in one origin versus another. This is plausible but not certain.

The same scale (0.3) for both tau_alpha and tau_beta is a modeling choice. If prior knowledge suggested slopes should vary less than intercepts, tau_beta could have a tighter prior (e.g., HalfNormal(0, 0.15)).

**Potential identifiability concern**: With only 3 groups and varying slopes, the model has 6 group-level parameters (3 intercepts + 3 slopes) plus 2 hypervariances. This is still identifiable but the slope variation may be weakly estimated. The non-centered parameterization and informative hyperpriors help.

### Expected Behavior

If design philosophies truly differ, we expect:
- USA: steeper negative slope (heavier cars suffer more)
- Japan: flatter slope (efficient across weight range)
- Europe: intermediate

The `total_slope` generated quantity directly shows origin-specific elasticities for interpretation.

If slope variation is minimal, tau_beta should concentrate near zero, and the model should give similar predictive performance to Model 2.

### Falsification Criteria

- **Abandon if**: tau_beta posterior concentrates strongly at 0, indicating no evidence for varying slopes. In this case, simplify to Model 2.
- **Abandon if**: Sampling diagnostics (divergences, low ESS for tau_beta) suggest the model is over-parameterized for these data.
- **Abandon if**: LOO-CV shows worse expected predictive performance than Model 2, indicating overfitting.
- **Simplify if**: Origin-specific slopes are indistinguishable (overlapping posteriors), suggesting the complexity is not warranted.

---

## Model Comparison Strategy

1. **Fit all three models** using identical data preparation
2. **Check diagnostics**: R-hat < 1.01, ESS > 400, no divergences
3. **Compare via LOO-CV**: Models should be ranked by expected log predictive density (elpd)
4. **Posterior predictive checks**: Examine residuals by origin to detect misspecification
5. **Examine shrinkage**: Compare hierarchical estimates to fixed effects; verify shrinkage is sensible

Expected ranking: Model 2 (Scientific) should provide the best bias-variance tradeoff. Model 1 may overfit slightly with independent origin estimates. Model 3 adds complexity that may not be warranted by data.

## Computational Notes

**Sampling recommendations**:
- 4 chains, 1000 warmup, 1000 sampling iterations
- Increase `adapt_delta` to 0.95 for hierarchical models to reduce divergences
- Check ESS specifically for tau parameters (often lowest)

**Potential issues**:
- Model 2: Non-centered parameterization should prevent funnel; if divergences persist, try centered parameterization and compare
- Model 3: May show lower ESS for tau_beta; this is expected with weak identification
- All models: sigma should be well-identified given the data volume

---

## Appendix: Data Preparation

All models expect the following transformations:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("auto_mpg_cleaned.csv")
df = df.dropna(subset=['horsepower'])  # remove 6 missing

# Transformations
df['log_mpg'] = np.log(df['mpg'])
df['log_weight'] = np.log(df['weight'])
df['log_weight_c'] = df['log_weight'] - df['log_weight'].mean()  # center at ~7.96
df['year_c'] = df['model_year'] - 76  # center at midpoint

# Stan data dict
stan_data = {
    'N': len(df),
    'J': 3,
    'log_mpg': df['log_mpg'].values,
    'log_weight_c': df['log_weight_c'].values,
    'year_c': df['year_c'].values,
    'origin': df['origin'].values
}
```

Note: Origin is already coded 1, 2, 3 in the data (USA, Europe, Japan).
