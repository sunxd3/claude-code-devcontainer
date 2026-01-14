data {
  int<lower=1> N;                       // number of students
  int<lower=1> J;                       // number of schools
  array[N] int<lower=1, upper=J> school;  // school ID for each student
  array[N] int<lower=0, upper=1> treatment;  // treatment indicator
  array[N] real y;                      // test scores (observed)
}

parameters {
  real alpha_0;                         // population mean intercept
  real<lower=0> tau_alpha;              // SD of school intercepts
  vector[J] z_alpha;                    // standardized school intercepts

  real beta_0;                          // population mean treatment effect
  real<lower=0> tau_beta;               // SD of school treatment effects
  vector[J] z_beta;                     // standardized school slopes

  real<lower=0> sigma;                  // residual SD
}

transformed parameters {
  vector[J] alpha;                      // school intercepts
  vector[J] beta;                       // school treatment effects
  vector[N] mu;                         // expected outcome for each student

  // Non-centered parameterization
  alpha = alpha_0 + tau_alpha * z_alpha;
  beta = beta_0 + tau_beta * z_beta;

  // Expected value for each student
  for (n in 1:N) {
    mu[n] = alpha[school[n]] + beta[school[n]] * treatment[n];
  }
}

model {
  // Priors
  alpha_0 ~ normal(77, 15);
  tau_alpha ~ normal(0, 10);
  z_alpha ~ std_normal();

  beta_0 ~ normal(5, 5);
  tau_beta ~ normal(0, 5);
  z_beta ~ std_normal();

  sigma ~ normal(0, 15);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;                    // pointwise log-likelihood
  array[N] real y_rep;                  // posterior/prior predictive draws

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
    y_rep[n] = normal_rng(mu[n], sigma);
  }
}
