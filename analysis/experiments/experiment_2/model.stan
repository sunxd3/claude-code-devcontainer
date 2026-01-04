// Model A2-Year: Log-Log Physical Model with Technological Drift
// log(mpg) ~ log(weight) + year
// Tests whether technology trend adds predictive value beyond weight

data {
  int<lower=1> N;
  vector[N] log_mpg;
  vector[N] log_weight_c;   // log(weight) - 7.96
  vector[N] year_c;         // model_year - 76 (centered at midpoint)
}

parameters {
  real alpha;               // intercept
  real beta_weight;         // weight elasticity
  real beta_year;           // year effect (technology drift)
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(3.1, 0.3);
  beta_weight ~ normal(-1, 0.3);
  beta_year ~ normal(0.03, 0.02);   // ~3% improvement per year on MPG
  sigma ~ exponential(5);

  // Likelihood
  log_mpg ~ normal(alpha + beta_weight * log_weight_c + beta_year * year_c, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = alpha + beta_weight * log_weight_c[n] + beta_year * year_c[n];
    y_rep[n] = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(log_mpg[n] | mu_n, sigma);
  }
}
