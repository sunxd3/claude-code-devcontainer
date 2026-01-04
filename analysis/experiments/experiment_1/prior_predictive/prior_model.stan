// Prior predictive model for A1-Baseline
// Samples from priors and generates predictive observations

data {
  int<lower=1> N;
  vector[N] log_weight_c;  // centered log(weight)
}

generated quantities {
  // Sample from priors
  real alpha = normal_rng(3.1, 0.3);
  real beta_weight = normal_rng(-1, 0.3);
  real<lower=0> sigma = exponential_rng(5);

  // Generate prior predictive observations
  vector[N] y_rep;
  for (n in 1:N) {
    real mu_n = alpha + beta_weight * log_weight_c[n];
    y_rep[n] = normal_rng(mu_n, sigma);
  }
}
