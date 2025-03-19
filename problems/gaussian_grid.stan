data {
  int<lower=1> N;              // number of grid points (n_grid^2)
  int<lower=1> T;              // number of observed time points per grid point (e.g., 10)
  matrix[N, T] dx;             // observed trajectories (positions)
  real dt_obs;                 // time step between observations
  real<lower=0> sigma_noise;   // noise level
}

parameters {
  real mu;                     // global drift mean
  real log_sigma;              // log-scale for the local drift parameters
  vector[N] theta_raw;         // standardized local drift (non-centered)
}

transformed parameters {
  // Non-centered parameterization for drift parameters:
  // theta[i] = mu + exp(log_sigma) * theta_raw[i]
  real<lower=0> sigma = exp(log_sigma);
  vector[N] theta = mu + sigma * theta_raw;
}

model {
  // Priors for the global parameters:
  mu ~ normal(0, 3);
  log_sigma ~ normal(0, 1);

  // Non-centered prior for the latent drift components:
  theta_raw ~ normal(0, 1);

  // Likelihood: vectorize over time points for each grid point.
  // Each increment: dx[i,t] ~ Normal(theta[i] * dt_obs, sigma_noise * sqrt(dt_obs))
  for (i in 1:N)
    for (t in 1:T)
      dx[i, t] ~ normal(theta[i] * dt_obs, sigma_noise * sqrt(dt_obs));
}
