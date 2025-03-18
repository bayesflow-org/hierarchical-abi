
data {
          int<lower=1> N;         // number of grid points (n_grid^2)
          int<lower=1> T;         // number of observed time points per grid point (10)
          matrix[N, T] dx;         // observed trajectories (positions)
          real dt_obs;            // time step between observations (derived from simulator settings)
        }

parameters {
          real mu;              // global drift mean
          real log_sigma;         // log-scale for the local drift parameters
          vector[N] theta_raw;      // standardized local drift for each grid point
        }

transformed parameters {
  real<lower=0> sigma = exp(log_sigma);  // actual standard deviation
  vector[N] theta = mu + sigma * theta_raw;  // actual drift parameters
}

model {
          // Priors
          mu ~ normal(0, 3);
          log_sigma ~ normal(0, 1);
          theta_raw ~ multi_normal(rep_vector(0.0, N), diag_matrix(rep_vector(1.0, N)));

          // Likelihood: Each increment is an independent observation
          // Increments between observed points are distributed as:
          //    Δx ~ Normal(theta * dt_obs, dt_obs)
          for (i in 1:N)
            for (t in 1:T)
              dx[i, t] ~ normal(theta[i] * dt_obs, sqrt(dt_obs));
        }
