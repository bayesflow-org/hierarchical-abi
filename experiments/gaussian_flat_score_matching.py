#%% md
# # Flat Gaussian with compositional score matching

import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ['KERAS_BACKEND'] = 'torch'
from bayesflow import diagnostics
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffusion_model import ScoreModel, SDE, GaussianFourierProjection, ShallowSet, train_score_model, adaptive_sampling
from experiments.problems.gaussian_flat import GaussianProblem, Prior, generate_synthetic_data, sample_posterior, kl_divergence, posterior_contraction
#%%
torch_device = torch.device("cuda")


# get arguments
max_number_of_obs = int(sys.argv[1])
experiment_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
noise_schedule = ['cosine', 'linear', 'edm-training', 'edm-sampling'][0]

variables_of_interest = [#'mini_batch', 'cosine_shift', 'damping_factor_t',
                         'damping_factor_t_linear', 'damping_factor_t_cosine']
if max_number_of_obs > 1:
    variables_of_interest = ['n_conditions']
model_ids = np.arange(10)  # train 10 models
variable_of_interest, model_id = list(itertools.product(variables_of_interest, model_ids))[experiment_id]

print('Exp:', experiment_id, 'Model:', model_id, variable_of_interest)

#%%
prior = Prior()
np.random.seed(experiment_id)
batch_size = 128
if max_number_of_obs == 1:
    number_of_obs = 1
else:
    number_of_obs = [1, 5, 10, 20, 50, 100]

#%%
current_sde = SDE(
    kernel_type=['variance_preserving', 'sub_variance_preserving'][0],
    noise_schedule=noise_schedule
)

dataset = GaussianProblem(
    n_data=10000,
    prior=prior,
    sde=current_sde,
    online_learning=True,
    number_of_obs=number_of_obs
)
dataset_valid = GaussianProblem(
    n_data=1000,
    prior=prior,
    sde=current_sde,
    number_of_obs=number_of_obs
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

time_embedding = nn.Sequential(
    GaussianFourierProjection(8),
    nn.Linear(8,8),
    nn.Mish()
)
summary_dim = 10
summary_net = ShallowSet(dim_input=10, dim_output=summary_dim, dim_hidden=8) if isinstance(number_of_obs, list) else None

max_number_of_obs = max(number_of_obs) if isinstance(number_of_obs, list) else number_of_obs
score_model = ScoreModel(
    input_dim_theta=prior.n_params_global,
    input_dim_x=summary_dim,
    summary_net=summary_net,
    time_embedding=time_embedding,
    hidden_dim=256,
    n_blocks=5,
    max_number_of_obs=max_number_of_obs,
    prediction_type='v' if not 'edm' in noise_schedule else 'F',
    sde=current_sde,
    weighting_type='likelihood_weighting' if not 'edm' in noise_schedule else 'edm',
    prior=prior,
    name_prefix=f'gaussian_flat{model_id}_{max_number_of_obs}'
)

# make dir for plots
if not os.path.exists(f"experiments/models/{score_model.name}.pt"):
    os.makedirs(f"experiments/plots/{score_model.name}", exist_ok=True)
    #%%
    # train model
    loss_history = train_score_model(score_model, dataloader, dataloader_valid=dataloader_valid,
                                     epochs=1000, device=torch_device)
    torch.save(score_model.state_dict(), f"experiments/models/{score_model.name}.pt")

    # plot loss history
    plt.figure(figsize=(16, 4), tight_layout=True)
    plt.plot(loss_history[:, 0], label='Training', color="#132a70", lw=2.0, alpha=0.9)
    plt.plot(loss_history[:, 1], label='Validation', linestyle="--", marker="o", color='black')
    plt.grid(alpha=0.5)
    plt.xlabel('Training epoch #')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'experiments/plots/{score_model.name}/loss_training.png')

    # %%
else:
    score_model.load_state_dict(
        torch.load(f"experiments/models/{score_model.name}.pt", weights_only=True, map_location=torch.device(torch_device)))

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
# - KL Divergence between true and estimated posterior samples
# - RMSE between the medians of true and estimated posterior samples
# - Posterior contraction: (1 - var_empirical_posterior / var_prior) / (1 - var_true_posterior / var_prior), and using the mean variances over all parameters
#%%
#%%
# Ensure we generate enough synthetic data samples.
n_samples_data = 100
n_post_samples = 100
score_model.current_number_of_obs = 1
obs_n_time_steps = 0
max_steps = 10000

mini_batch = ['10%']
n_conditions = [1]
cosine_shifts = [0]
d_factors = [1]  # using the d factor depending on the mini batch size
data_sizes = np.array([1, 10, 100, 1000, 10000])

if variable_of_interest == 'mini_batch':
    # Set up your data sizes and mini-batch parameters.
    #mini_batch = [1, 10, 100, 1000, 10000, None]
    mini_batch = [1, 10, 100, None]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'n_conditions':
    n_conditions = [1, 5, 10, 20, 50, 100]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'cosine_shift':
    #cosine_shifts = [0, -1, 1, 2, 5, 10]
    cosine_shifts = [0, 2, 5, 10]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'damping_factor_t':
    d_factors = np.square([1e-05, 0.01, 1.0])  # we used a factor of 2 before
    second_variable_of_interest = 'data_size'
elif variable_of_interest == 'damping_factor_t_cosine':
    d_factors = np.square([1e-05, 0.01, 1.0])  # we used a factor of 2 before
    second_variable_of_interest = 'data_size'
elif variable_of_interest == 'damping_factor_t_linear':
    d_factors = np.square([1e-05, 0.01, 1.0])  # we used a factor of 2 before
    second_variable_of_interest = 'data_size'
else:
    raise ValueError('Unknown variable_of_interest')

df_path = f'experiments/plots/{score_model.name}/df_results_{variable_of_interest}.csv'

def exponential_decay(t, d0, d1):
    return d0 * torch.exp(-np.log(d0 / d1) * t)

def linear_decay(t, d0, d1):
    start = torch.as_tensor(d0, dtype=t.dtype, device=t.device)
    end = torch.as_tensor(d1, dtype=t.dtype, device=t.device)
    return torch.lerp(input=start, end=end, weight=t)

def cosine_decay(t, d0, d1):
    return d1 + 0.5 * (d0 - d1) * (1 + torch.cos(torch.pi * t))

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
        if mb == '10%':
            mb = max(int(n * 0.1), 1)
        if mb is not None and mb >= n:
            continue
        if nc > n:
            continue

        skip = False
        for max_reached in reached_max_evals:
            if max_reached[0] <= n:  # check if for a smaller data size we already failed
                if max_reached[2] == nc and max_reached[3] == cs and max_reached[4] == d_factor:
                    # all conditions are the same, only mini batch size is different
                    if max_reached[1] is None:
                        pass
                    elif mb is None or max_reached[1] < mb:
                        print(f'smaller mini batch size already failed, skipping {nc}, {cs}')
                        skip = True
                        break
        if skip:
            results.append({
                "data_size": n,
                "data_id": -1,
                "mini_batch": mb if mb is not None else n,
                "damping_factor": d_factor,
                'n_conditions': nc,
                'cosine_shift': cs,
                "n_steps": max_steps,
                "kl": np.nan,
                "median": np.nan,
                "median_rmse": np.nan,
                "c_error": np.nan,
                "contractions": np.nan,
                "rel_contraction": np.nan,
            })
            df_results = pd.DataFrame(results)
            df_results.to_csv(df_path)
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
            damping_factor = lambda t: exponential_decay(t=t, d0=t0_value, d1=t1_value)
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor}
        elif variable_of_interest == 'damping_factor_t_cosine':
            t0_value = 1
            t1_value = d_factor
            damping_factor = lambda t: cosine_decay(t=t, d0=t0_value, d1=t1_value)
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor}
        elif variable_of_interest == 'damping_factor_t_linear':
            t0_value = 1
            t1_value = d_factor
            damping_factor = lambda t: linear_decay(t=t, d0=t0_value, d1=t1_value)
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor}
        else:
            damping_factor = lambda t: torch.ones_like(t) * d_factor
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor, 'damping_factor_prior': 1}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor, 'damping_factor_prior': 1}

        # Run adaptive sampling.
        try:
            test_samples, list_steps = adaptive_sampling(score_model, test_data,
                                                         conditions=None,
                                                         n_post_samples=n_post_samples,
                                                         sampling_arg=mini_batch_arg,
                                                         max_evals=max_steps*2,
                                                         t_end=0, random_seed=experiment_id, device=torch_device,
                                                         run_sampling_in_parallel=False,  # can actually be faster
                                                         return_steps=True)
        except torch.OutOfMemoryError as e:
            print(e)
            results.append({
                "data_size": n,
                "data_id": -1,
                "mini_batch": mb if mb is not None else n,
                "damping_factor": d_factor,
                'n_conditions': nc,
                'cosine_shift': cs,
                "n_steps": max_steps,
                "kl": np.nan,
                "median": np.nan,
                "median_rmse": np.nan,
                "c_error": np.nan,
                "contractions": np.nan,
                "rel_contraction": np.nan,
            })
            df_results = pd.DataFrame(results)
            df_results.to_csv(df_path)
            continue

        # Sample the true posterior.
        true_samples = np.stack([sample_posterior(x, prior_sigma=prior.scale,
                                                  sigma=prior.simulator.scale, n_samples=n_post_samples) for x in test_data], axis=0)

        # Compute metrics.
        if test_samples.shape[1] > 1:
            kl = [kl_divergence(test_data[i], test_samples[i], prior=prior) for i in range(n_samples_data)]
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
            reached_max_evals.append((n, mb, nc, cs, d_factor))
        else:
            n_steps = np.mean([len(ls) for ls in list_steps])
            if n_steps >= max_steps:
                # others will also fail to converge
                reached_max_evals.append((n, mb, nc, cs, d_factor))

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
                "kl": kl[i],
                "median": np.median(test_samples, axis=1)[i],
                "median_rmse": rmse,
                "c_error": c_error,
                "contractions": contractions,
                "rel_contraction": rel_contraction
            })

        # Create a DataFrame from the results list. Save intermediate results
        df_results = pd.DataFrame(results)
        df_results.to_csv(df_path)
