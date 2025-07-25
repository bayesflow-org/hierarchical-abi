import os

import numpy as np
from cmdstanpy import CmdStanModel

stan_file = os.path.join('experiments', 'problems', 'ar1_grid.stan')
stan_model = CmdStanModel(stan_file=stan_file)
def get_stan_posterior(sim_test, sigma_noise, chains=4):
    N, T = sim_test.shape

    # Suppose data is a numpy array of shape (N, T)
    # Prepare data for Stan
    stan_data = {
        'N': N,
        'T': T,
        'y': sim_test,
        'sigma': sigma_noise
    }

    # Fit the model to the data
    fit = stan_model.sample(data=stan_data, show_progress=False, chains=chains)

    global_posterior = np.concatenate([fit.draws_pd("alpha"),
                                       fit.draws_pd("beta"), fit.draws_pd("log_std_eta")], axis=-1)
    local_posterior = fit.draws_pd("eta").T
    return global_posterior, local_posterior
