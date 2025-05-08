data {
  int<lower=1> N;              // number of grid points (n_grid^2)
  int<lower=1> T;              // number of observed time points per grid point (e.g., 5)
  matrix[N, T] y;             // observed trajectories (positions)
  real<lower=0> sigma;
}

parameters {
  real alpha;
  real beta;
  real log_std_eta;
  vector[N] eta_raw;
}

transformed parameters {
  // Non-centered parameterization
  real<lower=0> std_eta = exp(log_std_eta);
  vector[N] eta = 2*inv_logit(beta + eta_raw) - 1;
}

model {
  // Priors for the global parameters:
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  log_std_eta ~ normal(0, 1);

  // Local priors
  eta_raw ~ normal(0, std_eta);

  // Likelihood
  for (n in 1:N) {
      y[n, 1] ~ normal(0, sigma);
      for (t in 2:T) {
        y[n, t] ~ normal(alpha + eta[n] * y[n, t-1], sigma);
      }
  }
}
