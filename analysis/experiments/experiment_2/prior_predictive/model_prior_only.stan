// Prior-only model for A2-Year prior predictive checks
// Samples from priors only (no likelihood) to generate prior predictive distribution

data {
  int<lower=1> N;
  vector[N] log_mpg;           // Not used for sampling, but needed for structure
  vector[N] log_weight_c;      // log(weight) - 7.96
  vector[N] year_c;            // model_year - 76
}

parameters {
  real alpha;
  real beta_weight;
  real beta_year;
  real<lower=0> sigma;
}

model {
  // Priors only - no likelihood term
  alpha ~ normal(3.1, 0.3);
  beta_weight ~ normal(-1, 0.3);
  beta_year ~ normal(0.03, 0.02);
  sigma ~ exponential(5);
}

generated quantities {
  vector[N] y_rep;

  for (n in 1:N) {
    real mu_n = alpha + beta_weight * log_weight_c[n] + beta_year * year_c[n];
    y_rep[n] = normal_rng(mu_n, sigma);
  }
}
