
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> school;
  array[N] int<lower=0, upper=1> treatment;
}

generated quantities {
  // Sample from priors
  real alpha_0 = normal_rng(77, 15);
  real<lower=0> tau_alpha = abs(normal_rng(0, 10));
  vector[J] z_alpha;

  real beta_0 = normal_rng(5, 5);
  real<lower=0> tau_beta = abs(normal_rng(0, 5));
  vector[J] z_beta;

  real<lower=0> sigma = abs(normal_rng(0, 15));

  // Transformed parameters
  vector[J] alpha;
  vector[J] beta;

  // Generate replicated data
  array[N] real y_rep;

  // Sample standardized random effects
  for (j in 1:J) {
    z_alpha[j] = normal_rng(0, 1);
    z_beta[j] = normal_rng(0, 1);
  }

  // Non-centered parameterization
  alpha = alpha_0 + tau_alpha * z_alpha;
  beta = beta_0 + tau_beta * z_beta;

  // Generate replicated observations
  for (n in 1:N) {
    real mu = alpha[school[n]] + beta[school[n]] * treatment[n];
    y_rep[n] = normal_rng(mu, sigma);
  }
}
