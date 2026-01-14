data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> treatment;
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  // Priors only - no likelihood for prior predictive
  alpha ~ normal(77, 15);
  beta ~ normal(5, 5);
  sigma ~ normal(0, 15);  // Half-Normal due to <lower=0> constraint
}

generated quantities {
  vector[N] mu = alpha + beta * to_vector(treatment);
  vector[N] y_prior_pred;

  for (n in 1:N) {
    y_prior_pred[n] = normal_rng(mu[n], sigma);
  }
}
