# Experiment Plan: Hierarchical Pooling Strategy

**Designer**: designer_1
**Perspective**: Hierarchical pooling and school-level structure
**Date**: 2026-01-14

## The Structural Question

With 8 schools and 160 students, how should we share information across schools when estimating treatment effects? The EDA reveals substantial between-school variation (ICC = 0.14) and heterogeneous treatment effects (school-level ATEs range from -0.5 to 15.0 points, SD = 5.16). This calls for partial pooling via random effects, but the small number of schools (J=8) makes prior choices and parameterization critical.

The core trade-off: complete pooling ignores meaningful school differences and biases estimates; no pooling overfits with limited data per school (15-25 students); partial pooling borrows strength while respecting heterogeneity.

## Model Ladder

### Baseline: Complete Pooling

**Generative story**: All students are exchangeable. Schools are irrelevant labels. Treatment effect is constant across the population.

```
y_i ~ Normal(mu_i, sigma)
mu_i = alpha + beta * treatment_i

Priors:
alpha ~ Normal(77, 15)      # grand mean at observed scale
beta ~ Normal(5, 5)          # weakly informative ATE
sigma ~ Half-Normal(0, 15)   # residual SD
```

**Justification**: Guarantees a fit. Establishes performance floor. Maximum statistical power if schools truly don't matter.

**Falsification criteria**:
- Posterior predictive checks show systematic school-level residuals (not random noise)
- School-specific mean residuals deviate from zero in predictable patterns
- LOO predictive density much worse than partial pooling

**Computational notes**: Simple model, fast sampling, no identifiability issues.

**Prior reasoning**: Population-level priors are weakly informative at the observed data scale. Alpha centered at 77 (observed mean), beta at 5 (plausible ATE between observed 7 and zero), sigma allows wide range of residual variation.

---

### Scientific: Partial Pooling with Random Intercepts and Slopes

**Generative story**: Schools have baseline differences (random intercepts) and treatment effects vary by school (random slopes). School-level effects are drawn from population distributions, enabling shrinkage toward the population mean. This respects both the ICC (14% variance between schools) and the observed treatment heterogeneity (SD = 5.16).

```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + (beta + gamma_j) * treatment_ij

Non-centered parameterization:
alpha_j = alpha_0 + tau_alpha * z_alpha_j
gamma_j = tau_gamma * z_gamma_j
z_alpha_j ~ Normal(0, 1)
z_gamma_j ~ Normal(0, 1)

Priors:
alpha_0 ~ Normal(77, 15)
tau_alpha ~ Half-Normal(0, 10)
beta ~ Normal(5, 5)
tau_gamma ~ Half-Normal(0, 5)
sigma ~ Half-Normal(0, 15)
```

**Justification**: EDA shows clear evidence for both school-level baseline differences and treatment effect heterogeneity. Random intercepts capture the ICC. Random slopes accommodate the wide range of school-level ATEs. Non-centered parameterization avoids funnel geometry when tau is small. Uncorrelated random effects (simpler assumption to start).

**Falsification criteria**:
- If tau_alpha or tau_gamma posterior mass concentrates near zero, hierarchy unnecessary (revert to simpler model)
- If divergences persist despite non-centered parameterization, model misspecified
- If school-level effects don't show appropriate shrinkage (extreme schools like School 1 with ATE=15.0 should shrink toward population mean)
- Poor posterior predictive fit for extreme schools

**Computational notes**: Non-centered parameterization essential. With J=8, tau parameters may have low effective sample size (monitor ESS_bulk > 400). Regularizing priors on tau prevent divergences. Expect ~1000 iterations warmup, 1000 sampling sufficient.

**Prior reasoning**:
- Half-Normal(0, 10) on tau_alpha: regularizing but allows ICC up to ~0.5, appropriate for observed ICC=0.14
- Half-Normal(0, 5) on tau_gamma: stronger regularization because treatment effect variation is harder to identify with J=8 schools; keeps sampling in plausible region
- These priors encode: "school differences exist but aren't extreme" which matches data scale and prevents the sampler from wandering into pathological regions

---

### Extension 1: No Pooling

**Generative story**: Each school is a distinct population. No information sharing. Estimate separate (alpha_j, beta_j) for each school.

```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + beta_j * treatment_ij

Priors (per school):
alpha_j ~ Normal(77, 15)
beta_j ~ Normal(5, 5)
sigma ~ Half-Normal(0, 15)
```

**Justification**: Demonstrates the cost of ignoring pooling. With 15-25 observations per school, estimates will be noisy. This model should underperform partial pooling on LOO.

**Falsification criteria**:
- If this model has better LOO than partial pooling, the hierarchical structure is inappropriate (schools are genuinely distinct populations)
- Wide credible intervals for school-specific parameters
- Poor out-of-sample prediction

**Computational notes**: No hierarchical structure, straightforward sampling. May see high posterior uncertainty for small schools.

**Prior reasoning**: Same population-level priors as baseline, applied independently to each school. No regularization across schools.

---

### Extension 2: Random Intercepts Only

**Generative story**: Schools differ in baseline performance but treatment effect is constant. Simpler hierarchy than the scientific model.

```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + beta * treatment_ij

Non-centered:
alpha_j = alpha_0 + tau_alpha * z_alpha_j
z_alpha_j ~ Normal(0, 1)

Priors:
alpha_0 ~ Normal(77, 15)
tau_alpha ~ Half-Normal(0, 10)
beta ~ Normal(5, 5)
sigma ~ Half-Normal(0, 15)
```

**Justification**: Tests whether treatment effect heterogeneity is real or noise. If tau_gamma in the scientific model is small and this model has similar LOO, random slopes aren't needed.

**Falsification criteria**:
- LOO substantially worse than scientific model → random slopes necessary
- Posterior predictive checks show systematic treatment effect residuals by school
- School-level treatment effects from data don't shrink to common beta

**Computational notes**: Simpler than scientific model, faster sampling. Non-centered parameterization still recommended.

**Prior reasoning**: Identical to scientific model but removes tau_gamma. Tests parsimony.

---

### Extension 3: Correlated Random Effects

**Generative story**: Schools with lower baseline scores benefit more from treatment (or vice versa). Intercepts and slopes trade off.

```
y_ij ~ Normal(mu_ij, sigma)
mu_ij = alpha_j + (beta + gamma_j) * treatment_ij

(alpha_j, gamma_j) ~ Multivariate_Normal([alpha_0, 0], Sigma)
Sigma = diag(tau) * Omega * diag(tau)
where tau = [tau_alpha, tau_gamma]
Omega ~ LKJ(2)

Priors:
alpha_0 ~ Normal(77, 15)
beta ~ Normal(5, 5)
tau_alpha ~ Half-Normal(0, 10)
tau_gamma ~ Half-Normal(0, 5)
sigma ~ Half-Normal(0, 15)
LKJ(2) prior on correlation matrix
```

**Justification**: If School 1 (low baseline, large treatment effect) represents a pattern, correlation captures it. EDA analysts noted "weak negative correlation" possibility.

**Falsification criteria**:
- Posterior correlation includes zero and is weak → uncorrelated model sufficient
- LOO similar to uncorrelated scientific model → added complexity unjustified
- Correlation doesn't improve posterior predictive fit for extreme schools

**Computational notes**: LKJ(2) prior weakly favors no correlation. More parameters to estimate with J=8 schools; may have convergence issues. Monitor correlation parameter ESS carefully.

**Prior reasoning**: LKJ(2) is mildly regularizing toward independence. If data supports correlation, it will emerge; if not, posterior stays near zero. This prevents overfitting while allowing discovery of genuine patterns.

---

### Extension 4: Heterogeneous Variance

**Generative story**: Residual variation differs by treatment status or by school. Treatment might reduce or increase individual-level variance. Or schools differ in student homogeneity.

**Option A - By treatment**:
```
y_ij ~ Normal(mu_ij, sigma_treatment[treatment_ij])
mu_ij = alpha_j + (beta + gamma_j) * treatment_ij

sigma_control ~ Half-Normal(0, 15)
sigma_treated ~ Half-Normal(0, 15)
```

**Option B - By school**:
```
y_ij ~ Normal(mu_ij, sigma_j)
mu_ij = alpha_j + (beta + gamma_j) * treatment_ij

sigma_j ~ Half-Normal(0, 15)
```

**Justification**: EDA reports Levene test p=0.55 (equal variance), but this is worth checking. Some treatments homogenize outcomes, others increase variance. Option B tests if schools differ in measurement precision or student heterogeneity.

**Falsification criteria**:
- Posterior ratio sigma_treated/sigma_control includes 1.0 with high probability → homogeneous variance
- LOO no better than constant variance model
- Posterior predictive variance checks don't show patterns

**Computational notes**: Option A adds one parameter (manageable). Option B adds 8 parameters with only ~20 obs per school (poorly identified, expect wide posteriors). More careful prior needed for Option B.

**Prior reasoning**: Same Half-Normal(0, 15) for all sigma parameters maintains exchangeability before seeing data. With limited data per group/school, priors provide essential regularization.

---

## Summary

The ladder progresses from complete pooling (performance floor) through partial pooling strategies of increasing complexity. The scientific model (random intercepts + uncorrelated random slopes) is the primary target, justified by EDA evidence of both baseline school differences and treatment heterogeneity. Extensions test:

1. **No pooling**: What we lose without regularization
2. **Random intercepts only**: Whether random slopes are necessary
3. **Correlated effects**: Whether intercept/slope trade-offs exist
4. **Heterogeneous variance**: Whether residual variation differs systematically

With only 8 schools, prior choices are load-bearing. Regularizing priors on tau parameters prevent divergences and keep sampling in plausible regions. Non-centered parameterizations avoid funnel geometry. All models maintain falsification criteria tied to LOO, posterior predictive checks, and parameter interpretability.
