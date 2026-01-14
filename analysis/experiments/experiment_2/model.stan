// Experiment 2: Random Intercepts Only
// Non-centered parameterization for school-level random intercepts
// Fixed treatment effect across all schools

data {
  int<lower=1> N;                           // number of students
  int<lower=1> J;                           // number of schools
  array[N] int<lower=1, upper=J> school;    // school indicator for each student
  array[N] real treatment;                  // treatment indicator (0/1)
  array[N] real y;                          // test scores (observed)
}

parameters {
  real alpha_0;                             // population mean intercept
  real<lower=0> tau_alpha;                  // SD of school random intercepts
  real beta;                                // treatment effect (fixed across schools)
  real<lower=0> sigma;                      // residual SD
  vector[J] z_alpha;                        // standardized school deviations
}

transformed parameters {
  vector[J] alpha;                          // school-specific intercepts
  vector[N] mu;                             // expected value for each student

  // Non-centered parameterization
  alpha = alpha_0 + tau_alpha * z_alpha;

  // Linear predictor
  for (n in 1:N) {
    mu[n] = alpha[school[n]] + beta * treatment[n];
  }
}

model {
  // Priors
  alpha_0 ~ normal(77, 15);
  tau_alpha ~ normal(0, 10);
  beta ~ normal(5, 5);
  sigma ~ normal(0, 15);
  z_alpha ~ std_normal();

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;                        // pointwise log-likelihood
  vector[N] y_rep;                          // posterior predictive draws

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
    y_rep[n] = normal_rng(mu[n], sigma);
  }
}
