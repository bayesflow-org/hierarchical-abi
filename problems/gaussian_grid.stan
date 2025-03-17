
data {
          int<lower=1> N;         // number of grid points (n_grid^2)
          int<lower=1> T;         // number of observed time points per grid point (10)
          matrix[N, T] x;         // observed trajectories (positions)
          real dt_obs;            // time step between observations (derived from simulator settings)
        }

transformed data {
          // Compute increments between observed points
          matrix[N, T] dx;
          for (i in 1:N) {
            dx[i, 1] = x[i, 1];  // First increment (assuming x0 = 0)
            for (t in 2:T)
              dx[i, t] = x[i, t] - x[i, t-1];
          }
        }

parameters {
          real mu;              // global drift mean
          real log_tau;         // log-scale for the local drift parameters
          vector[N] theta;      // local drift for each grid point
        }

transformed parameters {
          real tau;
          tau = exp(log_tau);
        }

model {
          // Priors
          mu ~ normal(0, 3);
          log_tau ~ normal(0, 1);
          theta ~ normal(mu, tau);

          // Likelihood: Each increment is an independent observation
          // Increments between observed points are distributed as:
          //    Δx ~ Normal(theta * dt_obs, dt_obs)
          for (i in 1:N)
            for (t in 1:T)
              dx[i, t] ~ normal(theta[i] * dt_obs, sqrt(dt_obs));
        }
