data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> treatment;
  vector[N] y;
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(77, 15);
  beta ~ normal(5, 5);
  sigma ~ normal(0, 15);  // Half-Normal due to <lower=0> constraint

  // Likelihood
  vector[N] mu = alpha + beta * to_vector(treatment);
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] mu = alpha + beta * to_vector(treatment);
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    y_rep[n] = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
