// Model A3-Robust: Log-Log Physical Model with Student-t Errors
// log(mpg) ~ log(weight) + year, with Student-t likelihood
// Tests robustness to outliers

data {
  int<lower=1> N;
  vector[N] log_mpg;
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
  // Priors
  alpha ~ normal(3.1, 0.3);
  beta_weight ~ normal(-1, 0.3);
  beta_year ~ normal(0.03, 0.02);
  sigma ~ exponential(5);
  nu ~ gamma(2, 0.1);       // prior centered around 20, allows heavy tails

  // Likelihood with Student-t errors
  log_mpg ~ student_t(nu, alpha + beta_weight * log_weight_c + beta_year * year_c, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = alpha + beta_weight * log_weight_c[n] + beta_year * year_c[n];
    y_rep[n] = student_t_rng(nu, mu_n, sigma);
    log_lik[n] = student_t_lpdf(log_mpg[n] | nu, mu_n, sigma);
  }
}
