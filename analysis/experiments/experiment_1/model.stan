// Model A1-Baseline: Log-Log Physical Model
// log(mpg) ~ log(weight)
// Tests whether weight alone explains fuel efficiency via physics

data {
  int<lower=1> N;
  vector[N] log_mpg;        // log(mpg)
  vector[N] log_weight_c;   // log(weight) - mean(log(weight)), centered at ~7.96
}

parameters {
  real alpha;               // intercept (log-MPG at mean log-weight)
  real beta_weight;         // elasticity of MPG w.r.t. weight
  real<lower=0> sigma;      // residual SD on log scale
}

model {
  // Priors
  alpha ~ normal(3.1, 0.3);       // log(22) ~ 3.1, most cars 15-35 mpg
  beta_weight ~ normal(-1, 0.3);  // physics suggests elasticity ~ -1
  sigma ~ exponential(5);         // expect small residual SD (~0.2)

  // Likelihood
  log_mpg ~ normal(alpha + beta_weight * log_weight_c, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    real mu_n = alpha + beta_weight * log_weight_c[n];
    y_rep[n] = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(log_mpg[n] | mu_n, sigma);
  }
}
