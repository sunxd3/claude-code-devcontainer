// Prior Predictive Model for Experiment 2: Random Intercepts Only
// Samples from priors and generates synthetic data without conditioning on observed y

data {
  int<lower=1> N;                           // number of students
  int<lower=1> J;                           // number of schools
  array[N] int<lower=1, upper=J> school;    // school indicator for each student
  array[N] real treatment;                  // treatment indicator (0/1)
}

generated quantities {
  // Sample from priors
  real alpha_0 = normal_rng(77, 15);
  real<lower=0> tau_alpha = abs(normal_rng(0, 10));
  real beta = normal_rng(5, 5);
  real<lower=0> sigma = abs(normal_rng(0, 15));

  // Generate school-level random effects
  vector[J] z_alpha;
  vector[J] alpha;
  for (j in 1:J) {
    z_alpha[j] = normal_rng(0, 1);
  }
  alpha = alpha_0 + tau_alpha * z_alpha;

  // Generate synthetic observations
  vector[N] mu;
  array[N] real y_prior_pred;
  for (n in 1:N) {
    mu[n] = alpha[school[n]] + beta * treatment[n];
    y_prior_pred[n] = normal_rng(mu[n], sigma);
  }
}
