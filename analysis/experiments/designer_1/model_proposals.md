# Physical Efficiency Models for Auto-MPG Prediction

## Physical Foundation

Fuel efficiency is governed by physics: the energy required to move a vehicle depends on its mass, aerodynamic drag, and rolling resistance. The fundamental relationship is:

```
Energy/distance ~ mass + drag + friction
```

Since gallons/mile is proportional to energy expended, and MPG = 1/(gallons/mile), we expect:

```
MPG ~ 1 / (a*weight + b)
```

For heavy vehicles where weight dominates drag, this simplifies to MPG ~ 1/weight, or equivalently log(MPG) ~ -log(weight). The EDA supports this: weight has the strongest correlation with MPG (r = -0.83), and residual plots show curvature consistent with an inverse or log-log relationship.

## Model Ladder

### Model 1: Log-Log Baseline

**Purpose**: Establish a baseline with the simplest physically-motivated specification. The log-log transformation captures the inverse relationship between MPG and weight while ensuring positivity and handling the right-skewed MPG distribution.

**Generative Story**:
1. Each vehicle has a weight w_i
2. Log(MPG) depends linearly on log(weight) with slope beta (the elasticity)
3. Residual variation is normally distributed on the log scale

**Physical Interpretation**: The elasticity beta represents the percent change in MPG per percent change in weight. Physics predicts beta ~ -1 (doubling weight roughly halves MPG), though aerodynamic and friction losses may cause deviations.

**Stan Specification**:

```stan
data {
  int<lower=1> N;
  vector[N] log_mpg;        // log(mpg)
  vector[N] log_weight_c;   // log(weight) - mean(log(weight)), centered
}

parameters {
  real alpha;               // intercept (log-MPG at mean log-weight)
  real beta;                // elasticity of MPG w.r.t. weight
  real<lower=0> sigma;      // residual SD on log scale
}

model {
  // Priors
  alpha ~ normal(3.1, 0.3);       // log(22) ~ 3.1, most cars 15-35 mpg
  beta ~ normal(-1, 0.3);         // physics suggests ~ -1
  sigma ~ exponential(5);         // expect small residual SD (~0.2)

  // Likelihood
  log_mpg ~ normal(alpha + beta * log_weight_c, sigma);
}

generated quantities {
  vector[N] log_mpg_rep;
  for (n in 1:N)
    log_mpg_rep[n] = normal_rng(alpha + beta * log_weight_c[n], sigma);
}
```

**Data Requirements**:
- `log_mpg`: log-transformed MPG values
- `log_weight_c`: log(weight) centered at its mean (~7.96)

**Prior Justification**:
- `alpha ~ normal(3.1, 0.3)`: At mean weight (~2970 lbs), we expect MPG around 20-25, so log(MPG) ~ 3.0-3.2. The prior is weakly informative, allowing the data to dominate.
- `beta ~ normal(-1, 0.3)`: Physics predicts elasticity near -1. The prior SD of 0.3 allows values from roughly -1.6 to -0.4, accommodating uncertainty about how much drag/friction contribute beyond mass effects.
- `sigma ~ exponential(5)`: Implies prior mean 0.2 on log scale (about 20% multiplicative error). This reflects that physical models leave substantial unexplained variation from engine efficiency, driving conditions, etc.

**Expected Behavior**:
- beta should be strongly negative, likely between -0.8 and -1.2
- R-squared (on log scale) around 0.65-0.70 based on correlation structure
- Residuals should be roughly homoscedastic with mild remaining curvature

**Falsification Criteria**:
- If beta is significantly different from -1 (outside -0.5 to -1.5), the pure inverse relationship may be wrong
- If residuals show strong heteroscedasticity by year or origin, the model misses important structure
- If posterior predictive checks show systematic under/over-prediction at weight extremes, nonlinear extensions are needed


### Model 2: Physical Power Law with Technological Drift

**Purpose**: Extend the baseline to capture technological improvement over time. The 1970-1982 period saw major efficiency gains from fuel crises and CAFE regulations. Including year allows us to estimate the "pure" weight effect controlling for technological era.

**Generative Story**:
1. Baseline efficiency depends on weight via power law: MPG ~ weight^beta
2. Technology improved roughly linearly over time, adding a multiplicative factor
3. On log scale: log(MPG) = alpha + beta*log(weight) + gamma*year

**Physical Interpretation**:
- beta captures the fundamental physics (energy ~ mass)
- gamma captures technology drift: better engines, lighter materials, improved aerodynamics
- Separating these lets us estimate what MPG improvement came from making smaller cars vs. making better cars

**Stan Specification**:

```stan
data {
  int<lower=1> N;
  vector[N] log_mpg;
  vector[N] log_weight_c;      // log(weight) - 7.96
  vector[N] year_c;            // model_year - 76 (centered at midpoint)
}

parameters {
  real alpha;                   // intercept
  real beta;                    // weight elasticity
  real gamma;                   // year effect (technology drift)
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(3.1, 0.3);
  beta ~ normal(-1, 0.3);
  gamma ~ normal(0.03, 0.02);   // ~3% improvement per year on MPG
  sigma ~ exponential(5);

  // Likelihood
  log_mpg ~ normal(alpha + beta * log_weight_c + gamma * year_c, sigma);
}

generated quantities {
  vector[N] log_mpg_rep;
  vector[N] mu;
  for (n in 1:N) {
    mu[n] = alpha + beta * log_weight_c[n] + gamma * year_c[n];
    log_mpg_rep[n] = normal_rng(mu[n], sigma);
  }
}
```

**Prior Justification**:
- `gamma ~ normal(0.03, 0.02)`: EDA shows +1.2 mpg/year trend. At mean MPG ~23, this is about 5%/year. On log scale, 5% ~ 0.05 in log units. However, some of this comes from weight changes, so we expect the partial effect to be smaller, hence centering at 0.03. The prior allows gamma from roughly -0.01 to 0.07, covering zero effect and up to 7%/year.

**Expected Behavior**:
- beta should remain near -1, possibly slightly less negative once year is controlled
- gamma should be positive (0.02-0.05), indicating real technological progress
- Residual SD should decrease relative to Model 1
- Model should better capture late-period high-MPG vehicles

**Falsification Criteria**:
- If gamma is near zero or negative, technological progress story is wrong
- If beta changes dramatically when adding year, weight and year are confounded in problematic ways
- If residuals still show origin-based patterns, market segmentation effects are needed


### Model 3: Power Law with Robust Errors

**Purpose**: Test whether outliers or heavy-tailed variation affect inference. The EDA notes a few unusual observations (high-MPG V8s, heavy 4-cylinders). A Student-t likelihood provides robustness while remaining interpretable.

**Generative Story**: Same as Model 2, but residuals follow a t-distribution rather than normal. This accommodates occasional large deviations from the physical model (unusual engines, measurement error, extreme driving conditions).

**Physical Interpretation**: The core physics remains unchanged; we simply acknowledge that some vehicles deviate more than a normal model predicts. The degrees-of-freedom parameter nu tells us how heavy the tails are:
- nu > 30: essentially normal
- nu ~ 4-10: moderate outlier robustness
- nu < 4: heavy tails, substantial outlier accommodation

**Stan Specification**:

```stan
data {
  int<lower=1> N;
  vector[N] log_mpg;
  vector[N] log_weight_c;
  vector[N] year_c;
}

parameters {
  real alpha;
  real beta;
  real gamma;
  real<lower=0> sigma;
  real<lower=2> nu;            // degrees of freedom for t-distribution
}

model {
  // Priors
  alpha ~ normal(3.1, 0.3);
  beta ~ normal(-1, 0.3);
  gamma ~ normal(0.03, 0.02);
  sigma ~ exponential(5);
  nu ~ gamma(2, 0.1);          // prior centered around 20, allows heavy tails

  // Likelihood
  log_mpg ~ student_t(nu, alpha + beta * log_weight_c + gamma * year_c, sigma);
}

generated quantities {
  vector[N] log_mpg_rep;
  vector[N] mu;
  for (n in 1:N) {
    mu[n] = alpha + beta * log_weight_c[n] + gamma * year_c[n];
    log_mpg_rep[n] = student_t_rng(nu, mu[n], sigma);
  }
}
```

**Prior Justification**:
- `nu ~ gamma(2, 0.1)`: This gamma prior has mean 20 and allows values from ~4 to ~50+. It gently regularizes toward normality while permitting heavy tails if the data warrant. The lower bound of 2 ensures finite variance.

**Expected Behavior**:
- If data are well-behaved: nu should be large (>20), and results should match Model 2
- If outliers matter: nu should be 4-15, and beta/gamma estimates may shift slightly
- sigma may be smaller than Model 2 since extreme observations are accommodated by heavy tails

**Falsification Criteria**:
- If nu < 4 with high posterior certainty, the residual structure is fundamentally non-normal and may indicate model misspecification beyond just outliers
- If coefficient estimates change substantially from Model 2, the normal model was giving biased estimates due to outlier influence

**Computational Notes**: Student-t likelihoods are well-behaved in Stan. The main concern is that nu near 2 can cause occasional divergences; the lower bound at 2 and regularizing prior mitigate this.


## Model Comparison Strategy

**Sequence**: Fit models in order (1, 2, 3) since each adds complexity to the previous.

**Diagnostics to examine**:
1. MCMC convergence: Rhat < 1.01, ESS > 400 for all parameters
2. Posterior predictive checks: distributions of replicated data vs observed
3. Residual plots: vs fitted values, vs weight, vs year
4. Leave-one-out cross-validation (LOO-CV) for predictive comparison

**Key Comparisons**:
- Model 1 vs 2: Does year improve prediction? Compare LOO-ELPD. If the difference is small (<4 SE), the simpler model may suffice.
- Model 2 vs 3: Does robustness matter? Compare posterior of nu. If nu > 30 with certainty, Model 2 is adequate.

**Decision Rules**:
- Prefer simpler model unless complex model shows clear improvement (LOO difference > 4 SE)
- If Model 3 shows nu < 10, investigate the outlying observations more carefully
- If residuals show origin patterns in all models, a hierarchical extension by origin is warranted (but that is for a different modeling perspective)


## Data Preparation Notes

The following transformations are needed before fitting:

```python
import numpy as np
import pandas as pd

df = pd.read_csv('auto_mpg_cleaned.csv')

# Remove rows with missing values (only 6 missing horsepower)
df_complete = df.dropna()

# Compute transforms
log_mpg = np.log(df_complete['mpg'])
log_weight = np.log(df_complete['weight'])
log_weight_c = log_weight - log_weight.mean()  # center at ~7.96
year_c = df_complete['model_year'] - 76        # center at midpoint

# Stan data dictionary
stan_data = {
    'N': len(df_complete),
    'log_mpg': log_mpg.values,
    'log_weight_c': log_weight_c.values,
    'year_c': year_c.values
}
```

**Centering Rationale**: Centering log_weight and year improves interpretability (alpha is predicted log-MPG for an average-weight car in 1976) and can aid sampling by reducing correlation between intercept and slopes.


## Summary Table

| Model | Parameters | Key Test | Expected Outcome |
|-------|-----------|----------|------------------|
| 1. Log-Log Baseline | alpha, beta, sigma | Pure weight-MPG relationship | beta ~ -1, establishes floor |
| 2. Power Law + Year | + gamma | Technology drift | gamma > 0, better late-period fit |
| 3. Robust Errors | + nu | Outlier sensitivity | nu informs whether t vs normal matters |

All three models share the physical core (log-log relationship between MPG and weight). The ladder tests progressively: (1) whether basic physics suffices, (2) whether technology matters, and (3) whether distributional assumptions are adequate.
