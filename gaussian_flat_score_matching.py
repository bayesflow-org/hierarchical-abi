#%% md
# # Flat Gaussian with compositional score matching

import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

os.environ['KERAS_BACKEND'] = 'torch'
from bayesflow import diagnostics
from torch.utils.data import DataLoader
import itertools
import sys

from diffusion_model import CompositionalScoreModel, SDE, train_score_model, adaptive_sampling
from problems.gaussian_flat import GaussianProblem, Prior, generate_synthetic_data, \
    sample_posterior, analytical_posterior_mean_std, posterior_contraction
#%%
torch_device = torch.device("cuda")


# get arguments
max_number_of_obs = int(sys.argv[1])
experiment_id = int(sys.argv[2])

variables_of_interest = ['mini_batch', 'cosine_shift', 'damping_factor_t'] # 'damping_factor', 'damping_factor_prior'
if max_number_of_obs > 1:
    variables_of_interest = ['n_conditions']
model_ids = np.arange(10)  # train 10 models
model_id, variable_of_interest = list(itertools.product(model_ids, variables_of_interest))[experiment_id]

print('Exp:', experiment_id, 'Model:', model_id, variable_of_interest)

#%%
prior = Prior()
#simulator_test = Simulator()
np.random.seed(experiment_id)

# test the simulator
#prior_test = prior.sample(2)
#sim_test = simulator_test(prior_test, n_obs=1000)
#visualize_simulation_output(sim_test['observable'])
#%%
batch_size = 128
#max_number_of_obs = 1  # larger than one means we condition the score on multiple observations

dataset = GaussianProblem(
    n_data=10000,
    prior=prior,
    online_learning=True,
    max_number_of_obs=max_number_of_obs
)
dataset_valid = GaussianProblem(
    n_data=batch_size*2,
    prior=prior,
    max_number_of_obs=max_number_of_obs
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
#%%
# Define diffusion model
current_sde = SDE(
    kernel_type=['variance_preserving', 'sub_variance_preserving'][0],
    noise_schedule=['linear', 'cosine', 'flow_matching'][1]
)

score_model = CompositionalScoreModel(
    input_dim_theta_global=prior.n_params_global,
    input_dim_x=prior.D,
    hidden_dim=64,
    n_blocks=3,
    max_number_of_obs=max_number_of_obs,
    prediction_type=['score', 'e', 'x', 'v'][3],
    sde=current_sde,
    time_embed_dim=16,
    use_film=False,
    weighting_type=[None, 'likelihood_weighting', 'flow_matching', 'sigmoid'][1],
    prior=prior,
    name_prefix=f'model{model_id}_'
)
#count_parameters(score_model)
print(score_model.name)

# make dir for plots
if not os.path.exists(f"plots/{score_model.name}"):
    os.makedirs(f"plots/{score_model.name}")
    #%%
    # train model
    loss_history = train_score_model(score_model, dataloader, dataloader_valid=dataloader_valid,
                                     epochs=500, device=torch_device)
    torch.save(score_model.state_dict(), f"models/{score_model.name}.pt")

    # plot loss history
    plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(loss_history[:, 0], label='Mean Train')
    plt.plot(loss_history[:, 1], label='Mean Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{score_model.name}/loss_training.png')
    plt.show()
    # %%
else:
    score_model.load_state_dict(
        torch.load(f"models/{score_model.name}.pt", weights_only=True, map_location=torch.device(torch_device)))

score_model.eval()

#%% md
# # Step Size for different Grid Sizes
# 
# - we compare score model with only one condition, and with $k$-conditions
# - we show that the scaling in the number of needed sampling steps only depends on the Bayesian Units used
# - error reduces when using more conditions, but since network size stays the same, increases at some point again
# - we show how mini batching effects the posterior
# 
# Metrics:
# - MMD between true and estimated posterior samples
# - RMSE between the medians of true and estimated posterior samples
# - Posterior contraction: (1 - var_empirical_posterior / var_prior) / (1 - var_true_posterior / var_prior), and using the mean variances over all parameters
#%%
def gaussian_kernel(x, y, sigma):
    """Compute Gaussian kernel between two sets of samples."""
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    sq_dists = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    return np.exp(-sq_dists / (2 * sigma ** 2))

def compute_mmd(x, y, sigma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        x (np.ndarray): Samples from distribution P, shape (n, d).
        y (np.ndarray): Samples from distribution Q, shape (m, d).
        sigma (float): Bandwidth for the Gaussian kernel.

    Returns:
        float: Estimated MMD^2 value.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Compute kernel matrices
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)

    # Compute MMD^2
    mmd_squared = (np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy))
    return mmd_squared


def kl_divergence(x, samples_q):
    try:
        mu_p, std_p = analytical_posterior_mean_std(x, prior_std=prior.scale, likelihood_std=prior.simulator.scale)
        cov_p = np.diag(std_p**2)

        mu_q = np.mean(samples_q, axis=0)
        cov_q = np.cov(samples_q, rowvar=False)

        d = mu_p.shape[0]
        cov_q_inv = np.linalg.inv(cov_q)

        term1 = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
        term2 = np.trace(cov_q_inv @ cov_p)
        term3 = (mu_q - mu_p).T @ cov_q_inv @ (mu_q - mu_p)

        return 0.5 * (term1 - d + term2 + term3)
    except Exception as e:  # sometimes a linalg error occurs
        print(e)
        return np.nan
#%%
# Ensure we generate enough synthetic data samples.
n_samples_data = 100
n_post_samples = 100
score_model.current_number_of_obs = 1
max_steps = 10000
#variables_of_interest = ['mini_batch', 'cosine_shift', 'damping_factor_t'] # 'damping_factor', 'damping_factor_prior'
#if score_model.max_number_of_obs > 1:
#    variables_of_interest.append('n_conditions')

#for variable_of_interest in variables_of_interest:
#variable_of_interest = variables_of_interest[0]

mini_batch = [100]
n_conditions = [1]
cosine_shifts = [0]
d_factors = [1]  # using the d factor depending on the mini batch size
data_sizes = np.array([1, 10, 100, 1000, 10000, 100000])

if variable_of_interest == 'mini_batch':
    # Set up your data sizes and mini-batch parameters.
    mini_batch = [1, 10, 100, 1000, 10000, None]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'n_conditions':
    n_conditions = [1, 5, 10, 20, 50, 100]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'cosine_shift':
    cosine_shifts = [0, -1, 1, 2, 5, 10]
    second_variable_of_interest = 'data_size'

elif variable_of_interest in ['damping_factor', 'damping_factor_prior', 'damping_factor_t']:
    d_factors = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.75, 0.9, 1]
    second_variable_of_interest = 'data_size'
else:
    raise ValueError('Unknown variable_of_interest')

df_path = f'plots/{score_model.name}/df_results_{variable_of_interest}.csv'
if os.path.exists(df_path):
    # Load CSV
    df_results = pd.read_csv(df_path, index_col=0)
    # Convert string representations back to lists
    df_results['list_steps'] = df_results['list_steps'].apply(lambda x: ast.literal_eval(x))

    if variable_of_interest == 'damping_factor_prior':
        df_results['damping_factor_prior'] = df_results['damping_factor']
    elif variable_of_interest == 'damping_factor_t':
        df_results['damping_factor_t'] = df_results['damping_factor']
else:
    df_results = None
#%%
# List to store results.
results = []
reached_max_evals = []

# Iterate over data sizes.
for n in data_sizes:
    # Generate synthetic data with enough samples
    true_params, test_data = generate_synthetic_data(prior, n_samples=n_samples_data, data_size=n,
                                                     normalize=False, random_seed=experiment_id)
    true_params = true_params.numpy()
    # Iterate over experimental setting
    for mb, nc, cs, d_factor in itertools.product(mini_batch, n_conditions, cosine_shifts, d_factors):
        # Skip mini-batch settings that are larger than or equal to the data size.
        if mb is not None and mb > n:
            continue
        if mb == n:
            mb = None
        if nc > n:
            continue

        for max_reached in reached_max_evals:
            if max_reached[1] == nc and max_reached[2] == cs and max_reached[3] == d_factor:
                # for this condition, if a lower mini batch size already failed we can skip that as well
                if max_reached[0] is None or mb is None:
                    pass
                elif max_reached[0] <= mb:
                    print(f'smaller mini batch size already failed, skipping {nc}, {cs}')
                    continue

        print(f"Data Size: {n}, Mini Batch: {mb}, Conditions: {nc}, Cosine shift: {cs}, Damping Factor: {d_factor}")
        # Set current number of conditions
        score_model.current_number_of_obs = nc

        # Set cosine shit
        score_model.sde.s_shift_cosine = cs

        # Damping factor
        if variable_of_interest == 'damping_factor_t':
            t0_value = 1
            t1_value = d_factor
            damping_factor = lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2*t)
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor, 'damping_factor_prior': damping_factor}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor,
                                  'damping_factor_prior': damping_factor}
        elif variable_of_interest == 'damping_factor_prior':
            damping_factor = lambda t: torch.ones_like(t) * d_factor
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor, 'damping_factor_prior': damping_factor}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor, 'damping_factor_prior': damping_factor}
        else:
            damping_factor = lambda t: torch.ones_like(t) * d_factor
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor}

        # Run adaptive sampling.
        test_samples, list_steps = adaptive_sampling(score_model, test_data, conditions=None,
                                                     n_post_samples=n_post_samples,
                                                     mini_batch_arg=mini_batch_arg,
                                                     max_evals=max_steps*2,
                                                     t_end=0, random_seed=0, device=torch_device,
                                                     run_sampling_in_parallel=False,  # can actually be faster
                                                     return_steps=True)
        # Sample the true posterior.
        true_samples = np.stack([sample_posterior(x, prior_sigma=prior.scale,
                                                  sigma=prior.simulator.scale, n_samples=n_post_samples) for x in test_data], axis=0)

        # Compute metrics.
        mmd = [compute_mmd(test_samples[i], true_samples[i]) for i in range(n_samples_data)]
        if test_samples.shape[1] > 1:
            kl = [kl_divergence(test_data[i], test_samples[i]) for i in range(n_samples_data)]
        else:
            kl = [np.nan for i in range(n_samples_data)]

        rmse = diagnostics.root_mean_squared_error(test_samples, true_params)['values'].mean()
        c_error = diagnostics.calibration_error(test_samples, true_params)['values'].mean()

        contractions = diagnostics.posterior_contraction(test_samples, true_params)['values'].mean()
        true_contraction = posterior_contraction(prior_std=prior.scale, likelihood_std=prior.simulator.scale, n_obs=n).mean()
        rel_contraction = (contractions / true_contraction)

        # Number of steps
        if np.isnan(test_samples).any():
            n_steps = np.inf
            reached_max_evals.append((mb, nc, cs, d_factor))
        else:
            n_steps = np.mean([len(ls) for ls in list_steps])
            if n_steps >= max_steps:
                # no need to check larger mini batches, will also fail to converge
                reached_max_evals.append((mb, nc, cs, d_factor))

        # Print current metrics.
        print(f"KL: {np.mean(kl)}, #Steps: {n_steps}")

        # Save results into a dictionary.
        for i in range(n_samples_data):  # might be less than the actual data points because inference failed
            results.append({
                "data_size": n,
                "data_id": i,
                "mini_batch": mb if mb is not None else n,
                "damping_factor": d_factor,
                'n_conditions': nc,
                'cosine_shift': cs,
                "n_steps": n_steps,
                "list_steps": np.where(np.isnan(list_steps[0]), None, list_steps[0]).tolist(),  # only for the first sample
                "mmd": mmd[i],
                "kl": kl[i],
                "median": np.median(test_samples, axis=1)[i],
                "median_rmse": rmse,
                "c_error": c_error,
                "contractions": contractions,
                "rel_contraction": rel_contraction
            })

        # Create a DataFrame from the results list. Save intermediate results
        df_results = pd.DataFrame(results)
        # Convert lists to strings for CSV storage
        df_results['list_steps'] = df_results['list_steps'].apply(lambda x: str(x))
        df_results.to_csv(df_path)
