data {
  int<lower=1> N;              // number of grid points (n_grid^2)
  int<lower=1> T;              // number of observed time points per grid point (e.g., 10)
  matrix[N, T] y;             // observed trajectories (positions)
  real<lower=0> sigma;
}

parameters {
  real alpha;
  real mu_beta;
  real log_std_beta;
  vector[N] beta_raw;
}

transformed parameters {
  // Non-centered parameterization
  real<lower=0> std_beta = exp(log_std_beta);
  vector[N] beta = mu_beta + std_beta * beta_raw;
}

model {
  // Priors for the global parameters:
  alpha ~ normal(0, 1);
  mu_beta ~ normal(0, 1);
  log_std_beta ~ normal(-1, 1);

  // Non-centered prior
  beta_raw ~ normal(0, 1);

  // Likelihood
  for (n in 1:N) {
      y[n, 1] ~ normal(0, sigma);
      for (t in 2:T) {
        y[n, t] ~ normal(alpha + beta[n] * y[n, t-1], sigma);
      }
  }
}
