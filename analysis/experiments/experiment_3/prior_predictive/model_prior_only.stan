// Prior-only model for A3-Robust: Student-t errors
// No likelihood - samples from priors only to generate prior predictive

data {
  int<lower=1> N;
  vector[N] log_mpg;         // not used in sampling, only for y_rep shape
  vector[N] log_weight_c;
  vector[N] year_c;
}

parameters {
  real alpha;
  real beta_weight;
  real beta_year;
  real<lower=0> sigma;
  real<lower=2> nu;         // degrees of freedom for t-distribution
}

model {
  // Priors only (same as main model)
  alpha ~ normal(3.1, 0.3);
  beta_weight ~ normal(-1, 0.3);
  beta_year ~ normal(0.03, 0.02);
  sigma ~ exponential(5);
  nu ~ gamma(2, 0.1);       // prior centered around 20

  // No likelihood term - MCMC samples from priors
}

generated quantities {
  vector[N] y_rep;

  for (n in 1:N) {
    real mu_n = alpha + beta_weight * log_weight_c[n] + beta_year * year_c[n];
    y_rep[n] = student_t_rng(nu, mu_n, sigma);
  }
}
