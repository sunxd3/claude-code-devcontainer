# Experiment Plan: Treatment Effect Heterogeneity

**Designer**: designer_2
**Date**: 2026-01-14
**Perspective**: Treatment effect heterogeneity across schools

## Overview

The EDA reveals substantial variation in school-specific treatment effects (range: -0.5 to 15.0 points, SD = 5.16) despite an overall strong effect (ATE = 7.03, Cohen's d = 0.71). This ladder explores whether treatment effects are fixed, vary randomly across schools, or require robust modeling due to outliers or heavy tails.

Core question: Do we need random slopes, correlated random effects, or alternative error distributions to capture treatment heterogeneity?

---

## Baseline: Complete Pooling

**Generative Story**

All students share a common baseline and treatment effect, ignoring school structure entirely:

```
y_i ~ Normal(mu_i, sigma)
mu_i = alpha + beta * treatment_i

Parameters:
- alpha: population mean for control group
- beta: treatment effect (same for all schools)
- sigma: residual standard deviation

Priors:
alpha ~ Normal(77, 15)    # centered on observed grand mean
beta ~ Normal(5, 5)        # weakly informative, centered on observed ATE range
sigma ~ Half-Normal(0, 15) # allows residual SD up to full score range
```

**Justification**

This model will fail to capture observed heterogeneity but provides a performance floor. It tests the null hypothesis that all variation in treatment effects is sampling noise.

**Falsification Criteria**

- Posterior predictive checks will show poor coverage of school-specific effects
- LOO will be substantially worse than models with school structure
- If this model is competitive on LOO, treatment heterogeneity is not meaningful

**Computational Notes**

No convergence issues expected. This is the simplest possible model and will converge easily.

**Prior Considerations**

Priors are weakly informative to allow the data to dominate. The prior on beta is wide enough to include the full range of observed school-specific effects (though the model cannot represent them). The prior SD of 15 for sigma allows for substantial unexplained variance.

---

## Scientific: Random Intercepts + Random Slopes (Uncorrelated)

**Generative Story**

Each school has its own baseline level and its own treatment effect. These are drawn independently from population distributions:

```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + beta_j * treatment_ij
alpha_j ~ Normal(alpha_0, tau_alpha)
beta_j ~ Normal(beta_0, tau_beta)

Parameters:
- alpha_0: population mean for control group
- beta_0: population-average treatment effect
- tau_alpha: between-school SD of baselines
- tau_beta: between-school SD of treatment effects
- sigma: residual SD (within-school variation)

Priors:
alpha_0 ~ Normal(77, 15)
beta_0 ~ Normal(5, 5)
tau_alpha ~ Half-Normal(0, 10)  # between-school baseline variation
tau_beta ~ Half-Normal(0, 5)    # treatment effect heterogeneity
sigma ~ Half-Normal(0, 15)
```

**Justification**

The EDA shows school-specific ATEs ranging from -0.5 to 15.0 with SD = 5.16. Random slopes allow each school to have its own treatment effect, capturing this observed heterogeneity. The uncorrelated structure assumes no systematic relationship between school baseline and treatment response.

**Falsification Criteria**

- If posterior for tau_beta has substantial mass near zero (e.g., 95% CI includes zero or lower bound < 1), heterogeneity is not meaningful and we should revert to fixed slopes
- If posterior predictive checks show systematic misfit for specific schools, model structure may be wrong
- If LOO is not substantially better than random intercepts alone, random slopes are not justified

**Computational Notes**

With only J=8 schools, tau_beta may be difficult to estimate precisely. Use non-centered parameterization if there are divergences:

```
beta_j = beta_0 + tau_beta * z_beta_j
z_beta_j ~ Normal(0, 1)
```

This separates the scale (tau_beta) from the group-level deviations (z_beta_j), avoiding funnel geometry when tau_beta is small.

**Prior Considerations**

The prior Half-Normal(0, 5) on tau_beta reflects the observed SD of 5.16 in school-specific effects. This provides mild regularization to prevent overfitting with few schools while allowing the data to inform the degree of heterogeneity. The prior implies that 95% of schools should have treatment effects within roughly ±10 points of the average (beta_0 ± 2*tau_beta), which is reasonable given the observed range.

---

## Extension 1: Correlated Random Effects

**Generative Story**

Schools with higher baselines may respond differently to treatment. We allow correlation between intercepts and slopes:

```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + beta_j * treatment_ij

[alpha_j]   ~ MVNormal([alpha_0], Sigma)
[beta_j ]               [beta_0]

Sigma = [tau_alpha^2,          rho*tau_alpha*tau_beta]
        [rho*tau_alpha*tau_beta,  tau_beta^2          ]

Parameters:
- rho: correlation between school intercepts and slopes (-1 to 1)
- other parameters as in Scientific model

Priors:
alpha_0 ~ Normal(77, 15)
beta_0 ~ Normal(5, 5)
tau_alpha ~ Half-Normal(0, 10)
tau_beta ~ Half-Normal(0, 5)
rho ~ LKJ(2)  # weakly informative, favors small correlations
sigma ~ Half-Normal(0, 15)
```

**Justification**

The EDA hints at weak negative correlation between school baselines and treatment effects (schools 1 and 3 have low baselines, schools 7 has high baseline and small effect). This model tests whether such patterns are systematic. The LKJ(2) prior provides mild regularization toward independence while allowing the data to identify strong correlations if present.

**Falsification Criteria**

- If posterior for rho has 95% CI including zero, correlation is not meaningful and we revert to uncorrelated model
- If model fit (LOO) is not substantially better than uncorrelated model, added complexity is not justified
- If rho posterior is near ±1, model may be overparameterized or have identification issues

**Computational Notes**

Correlated random effects can create funnel geometry. Use Cholesky decomposition and non-centered parameterization:

```
L ~ LKJCholesky(2)  # Cholesky factor of correlation matrix
z_j ~ Normal(0, 1)  # standard normal draws
[alpha_j] = [alpha_0] + diag([tau_alpha]) * L * z_j
[beta_j ]   [beta_0]      [tau_beta  ]
```

This is more efficient than building the covariance matrix directly and avoids funnel issues.

**Prior Considerations**

The LKJ(2) prior on the correlation matrix concentrates mass near zero correlation but allows for moderate to strong correlations if the data support them. With only 8 schools, estimating correlation is challenging. The prior provides regularization to prevent spurious correlations from small sample variation.

---

## Extension 2: Student-t Errors

**Generative Story**

Treatment heterogeneity may arise from heavy-tailed errors rather than true variation in treatment effects. A Student-t likelihood allows for outliers while maintaining random slopes:

```
y_ij ~ StudentT(nu, mu_ij, sigma)
mu_ij = alpha_j + beta_j * treatment_ij
alpha_j ~ Normal(alpha_0, tau_alpha)
beta_j ~ Normal(beta_0, tau_beta)

Parameters:
- nu: degrees of freedom (controls tail heaviness)
- other parameters as in Scientific model

Priors:
alpha_0 ~ Normal(77, 15)
beta_0 ~ Normal(5, 5)
tau_alpha ~ Half-Normal(0, 10)
tau_beta ~ Half-Normal(0, 5)
nu ~ Gamma(2, 0.1)  # mode around 20, allows heavier tails
sigma ~ Half-Normal(0, 15)
```

**Justification**

Schools 1 and 3 show extreme deviations in treatment effects (15.0 and -0.5 respectively). This model tests whether such extremes reflect heavy-tailed noise rather than true treatment variation. The EDA found no evidence against normality (Shapiro-Wilk p = 0.18), but this tests robustness.

**Falsification Criteria**

- If posterior for nu has substantial mass above 30, Student-t is unnecessary (approximates Normal) and we revert to Gaussian errors
- If tau_beta does not shrink compared to Scientific model, heavy tails are not explaining heterogeneity
- If LOO is not better than Normal likelihood, added flexibility is not justified

**Computational Notes**

Student-t likelihood can introduce multimodality if nu is estimated too freely. The Gamma(2, 0.1) prior provides regularization while allowing for moderate tail heaviness (mode ~20, mean ~20). If convergence issues arise, fix nu at specific values (e.g., nu=4 for heavy tails, nu=10 for mild robustness) and compare models.

**Prior Considerations**

The Gamma(2, 0.1) prior on nu balances robustness against overfitting. Very low nu (e.g., <4) creates extremely heavy tails that may overfit outliers. Very high nu (>30) approximates the Normal. The prior mode around 20 provides mild robustness without dramatically changing the likelihood shape. This prior keeps the sampler in reasonable regions while allowing the data to identify heavy tails if they exist.

---

## Extension 3: Fixed School Effects + Random Slopes

**Generative Story**

If schools are not exchangeable (e.g., different curricula, neighborhoods), we use fixed effects for baselines but maintain random slopes for treatment heterogeneity:

```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + beta_j * treatment_ij
alpha_j: separate parameter for each school (no hierarchical structure)
beta_j ~ Normal(beta_0, tau_beta)

Parameters:
- alpha_j: separate intercept for each j=1..8 (8 parameters)
- beta_0: population-average treatment effect
- tau_beta: between-school SD of treatment effects
- sigma: residual SD

Priors:
alpha_j ~ Normal(77, 20)  # independent priors for each school
beta_0 ~ Normal(5, 5)
tau_beta ~ Half-Normal(0, 5)
sigma ~ Half-Normal(0, 15)
```

**Justification**

If schools represent distinct contexts (urban/rural, different programs), exchangeability may be inappropriate. Fixed effects allow each school its own baseline without shrinkage, while random slopes still pool information about treatment variation. This tests whether the hierarchical structure on intercepts is helping or hurting.

**Falsification Criteria**

- If LOO is worse than random intercepts model (Scientific), hierarchical structure on intercepts is valuable and should be retained
- If fixed effects estimates match closely with random effects posteriors from Scientific model, exchangeability is fine
- If model has convergence issues or extreme parameter estimates, exchangeability assumption is appropriate

**Computational Notes**

No special parameterization needed for fixed effects. With 8 schools and balanced sample sizes, estimation should be straightforward. Watch for schools with small sample sizes (school 1 has only n=15) where fixed effects may be poorly identified.

**Prior Considerations**

Each alpha_j gets an independent prior centered at the grand mean with wider SD (20 vs 15) to allow more flexibility. This avoids imposing hierarchical shrinkage on baselines while maintaining pooling for treatment effects. The weaker prior reflects that we're not borrowing strength across schools for baselines.

---

## Implementation Notes

**Stan Translation**

All models are specified concretely enough for direct Stan implementation:
- Data block: `int N, J; vector[N] y, treatment; array[N] int school;`
- Parameters: as listed in each model
- Model block: likelihood + priors as specified
- Generated quantities: LOO pointwise log-likelihood, posterior predictive draws

**Comparison Strategy**

1. Fit all models and check convergence (R-hat < 1.01, ESS > 400)
2. Compare via LOO-CV (prefer lower ELPD_LOO)
3. Check posterior predictive fits for school-specific effects
4. Assess parameter recovery if simulation study is performed

**Expected Outcomes**

- Baseline will underperform but set the floor
- Scientific model should capture heterogeneity well
- Correlated RE may not improve fit (weak correlation in EDA)
- Student-t may not help (normal likelihood already appropriate)
- Fixed effects will likely underperform due to loss of shrinkage with only 8 schools

The scientific model (uncorrelated random slopes) is the primary target. Extensions probe whether additional structure is needed.
