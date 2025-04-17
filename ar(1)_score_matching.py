#%% md
# # Hierarchical Ar(1) on a Grid Test with compositional score matching
# 
# In this notebook, we will test the compositional score matching on a hierarchical problem defined on a grid.
# - The observations are on grid with `n_grid` x `n_grid` points.
# - The global parameters are the same for all grid points with hyper-priors:
# $$ \alpha \sim \mathcal{N}(0, 1) \quad
#   \mu_\beta \sim \mathcal{N}(0, 1) \quad
#   \log\text{std}_\beta \sim \mathcal{N}(-1, 1);$$
# 
# - The local parameters are different for each grid point
# $$ \beta_{i,j}^\text{raw} \sim \mathcal{N}(\mu_\beta, \text{std}_\beta^2), \qquad \beta_{i,j} = 2\operatorname{sigmoid}(\beta_{i,j}^\text{raw})-1$$
# 
# -  In each grid point, we have a time series of `T` observations. For the time beeing, we fix $\sigma=1$.
# $$ y_{i,j} \sim \mathcal{N}(\alpha + \beta_{i,j}y_{i,j-1}, \sigma^2), y_{i,0} \sim \mathcal{N}(0, \sigma^2)$$
# - We observe $T=5$ time points for each grid point. We can also amortize over the time dimension.
#%%
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

from diffusion_model import HierarchicalScoreModel, SDE, euler_maruyama_sampling, adaptive_sampling, train_score_model
from diffusion_model.helper_networks import GaussianFourierProjection, ShallowSet
from problems.ar1_grid import AR1GridProblem, Prior

#%%
torch_device = torch.device("cuda")

# get arguments
max_number_of_obs = int(sys.argv[1])
experiment_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

variables_of_interest = ['mini_batch', 'cosine_shift', 'damping_factor_t', 'compare_stan', 'max_results']
if max_number_of_obs > 1:
    variables_of_interest = ['n_conditions', 'max_results']
model_ids = np.arange(10)  # train 10 models
variable_of_interest, model_id = list(itertools.product(variables_of_interest, model_ids))[experiment_id]

print('Exp:', experiment_id, 'Model:', model_id, variable_of_interest)


#%%
prior = Prior()
np.random.seed(experiment_id)

#%%
batch_size = 128
if max_number_of_obs == 1:
    number_of_obs = 1
else:
    number_of_obs = [1, 4, 8, 16, 64, 128]
current_sde = SDE(
    kernel_type=['variance_preserving', 'sub_variance_preserving'][0],
    noise_schedule=['linear', 'cosine', 'flow_matching'][1]
)

dataset = AR1GridProblem(
    n_data=10000,
    prior=prior,
    sde=current_sde,
    online_learning=True,
    number_of_obs=number_of_obs,
    amortize_time=False,
    as_set=True
)

dataset_valid = AR1GridProblem(
    n_data=1000,
    prior=prior,
    sde=current_sde,
    number_of_obs=number_of_obs,
    as_set=True,
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

#%%
# Define diffusion model
global_summary_dim = 5
obs_n_time_steps = 0
global_summary_net = ShallowSet(dim_input=5, dim_output=global_summary_dim, dim_hidden=16)

time_dim = 8
time_embedding_local = nn.Sequential(
    GaussianFourierProjection(time_dim),
    nn.Linear(time_dim, time_dim),
    nn.Mish()
)
time_embedding_global = nn.Sequential(
    GaussianFourierProjection(time_dim),
    nn.Linear(time_dim, time_dim),
    nn.Mish()
)

max_number_of_obs = max(number_of_obs) if isinstance(number_of_obs, list) else number_of_obs
score_model = HierarchicalScoreModel(
    input_dim_theta_global=prior.n_params_global,
    input_dim_theta_local=prior.n_params_local,
    input_dim_x_global=global_summary_dim,
    global_summary_net=global_summary_net if isinstance(number_of_obs, list) else None,
    input_dim_x_local=global_summary_dim,
    time_embedding_local=time_embedding_local,
    time_embedding_global=time_embedding_global,
    hidden_dim=256,
    n_blocks=5,
    max_number_of_obs=max_number_of_obs,
    prediction_type=['score', 'e', 'x', 'v'][3],
    sde=current_sde,
    weighting_type=[None, 'likelihood_weighting', 'flow_matching', 'sigmoid'][1],
    prior=prior,
    name_prefix=f'ar1_{model_id}_{max_number_of_obs}'
)

# make dir for plots
if not os.path.exists(f"models/{score_model.name}.pt"):
    os.makedirs(f"plots/{score_model.name}", exist_ok=True)
    #%%
    # train model
    loss_history = train_score_model(score_model, dataloader, dataloader_valid=dataloader_valid, hierarchical=True,
                                     epochs=2000, device=torch_device)
    torch.save(score_model.state_dict(), f"models/{score_model.name}.pt")

    # plot loss history
    plt.figure(figsize=(16, 4), tight_layout=True)
    plt.plot(loss_history[:, 0], label='Training', color="#132a70", lw=2.0, alpha=0.9)
    plt.plot(loss_history[:, 1], label='Validation', linestyle="--", marker="o", color='black')
    plt.grid(alpha=0.5)
    plt.xlabel('Training epoch #')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'plots/{score_model.name}/loss_training.png')
    # %%
else:
    score_model.load_state_dict(
        torch.load(f"models/{score_model.name}.pt", weights_only=True, map_location=torch.device(torch_device))
    )
    print("Model loaded")

score_model.eval()

#%% md
# # Check different sampling schemes
#%%
# Ensure we generate enough synthetic data samples.
n_samples_data = 100
n_post_samples = 100
score_model.current_number_of_obs = 1
max_steps = 10000
print(variable_of_interest)

mini_batch = ['10%']
n_conditions = [1]
cosine_shifts = [0]
d_factors = [1]  # using the d factor depending on the mini batch size
data_sizes = np.array([4*4, 16*16, 64*64, 512*512])

if variable_of_interest == 'mini_batch':
    # Set up your data sizes and mini-batch parameters.
    mini_batch = [1, 10, 100, 1000, 10000, None]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'n_conditions':
    n_conditions = [1, 4, 8, 16, 64, 128]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'cosine_shift':
    cosine_shifts = [0, -1, 1, 2, 5, 10]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'damping_factor_t':
    d_factors = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.75, 0.9, 1]
    second_variable_of_interest = 'data_size'

elif variable_of_interest == 'compare_stan':
    N = 32 * 32
    global_posterior_stan = np.load(f'problems/ar1/global_posterior_{N}.npy')
    local_posterior_stan = np.load(f'problems/ar1/local_posterior_{N}.npy')
    true_global = np.load(f'problems/ar1/true_global_{N}.npy')
    true_local = np.load(f'problems/ar1/true_local_{N}.npy')

    n_grid_stan = int(np.sqrt(true_local.shape[1]))

    test_data = []
    for g, l in zip(true_global, true_local):
        sim_dict = {'alpha': g[0],
                    'beta': l}
        td = prior.simulator(sim_dict)['observable']
        test_data.append(td.reshape(1, n_grid_stan * n_grid_stan, 5))
    test_data = np.concatenate(test_data)

    n_obs = n_grid_stan * n_grid_stan
    batch_size = test_data.shape[0]
    n_post_samples = 100
    score_model.current_number_of_obs = 1

    global_param_names = prior.global_param_names
    local_param_names = prior.get_local_param_names(n_grid_stan * n_grid_stan)
    param_names_stan = ['STAN ' + p for p in global_param_names]

    print(n_grid_stan * n_grid_stan, test_data.shape)

    import optuna
    def objective(trial):
        t1_value = trial.suggest_float('t1_value', 1e-5, 1)
        s_shift_cosine = trial.suggest_float('s_shift_cosine', 0, 5)

        t0_value = 1
        mini_batch_arg = {
            'size': 16,
            'damping_factor': lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2 * t),
        }
        score_model.sde.s_shift_cosine = s_shift_cosine

        test_global_samples = adaptive_sampling(score_model, test_data, obs_n_time_steps=obs_n_time_steps,
                                                n_post_samples=100,
                                                # max_evals=1000,
                                                mini_batch_arg=mini_batch_arg,
                                                run_sampling_in_parallel=False,
                                                device=torch_device, verbose=False)

        cerror = diagnostics.calibration_error(test_global_samples, true_global)['values'].mean()
        return cerror.cpu().numpy()


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print(study.best_params)

    t1_value = study.best_params['t1_value']
    t0_value = 1
    mini_batch_arg = {
        'size': 16,
        'damping_factor': lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2 * t),
    }
    score_model.sde.s_shift_cosine = study.best_params['s_shift_cosine']

    posterior_global_samples_test = adaptive_sampling(score_model, test_data, obs_n_time_steps=obs_n_time_steps,
                                                      n_post_samples=100,
                                                      mini_batch_arg=mini_batch_arg,
                                                      run_sampling_in_parallel=False,
                                                      device=torch_device, verbose=False)

    score_model.sde.s_shift_cosine = 0
    posterior_local_samples_test = euler_maruyama_sampling(score_model, test_data, obs_n_time_steps=obs_n_time_steps,
                                                           n_post_samples=n_post_samples,
                                                           conditions=posterior_global_samples_test,
                                                           diffusion_steps=100,
                                                           device=torch_device, verbose=False)

    posterior_local_samples_test = score_model.prior.transform_local_params(posterior_local_samples_test)

    np.save(f'problems/ar1/posterior_global_samples_test.npy', posterior_global_samples_test)
    np.save(f'problems/ar1/posterior_local_samples_test.npy', posterior_local_samples_test)

    fig = diagnostics.recovery(posterior_global_samples_test, true_global, variable_names=global_param_names)
    fig.savefig(f'plots/{score_model.name}/recovery_global_ours.png')

    fig = diagnostics.recovery(posterior_global_samples_test, np.median(global_posterior_stan, axis=1),
                               variable_names=global_param_names, xlabel='STAN Median Estimate')
    fig.savefig(f'plots/{score_model.name}/recovery_global_ours_vs_STAN.png')

    fig = diagnostics.calibration_ecdf(posterior_global_samples_test, true_global, difference=True,
                                       variable_names=global_param_names)
    fig.savefig(f'plots/{score_model.name}/ecdf_global_ours.png')

    fig = diagnostics.recovery(posterior_local_samples_test.reshape(test_data.shape[0], n_post_samples, -1)[:, :, :12],
                               np.median(local_posterior_stan[:, :, :12], axis=1), ylabel='Score Based Estimates',
                               xlabel='STAN Median Estimate')
    fig.savefig(f'plots/{score_model.name}/recovery_local_ours_vs_STAN.png')
    exit()

elif variable_of_interest == 'max_results':
    n_grid = 512
    # np.random.seed(0)
    prior_dict = prior.sample(batch_size=100, n_local_samples=n_grid * n_grid)
    true_global, true_local, test_data = prior_dict['global_params'], prior_dict['local_params'], \
    prior_dict['data']

    n_obs = n_grid * n_grid
    batch_size = test_data.shape[0]
    n_post_samples = 100
    score_model.current_number_of_obs = max_number_of_obs

    global_param_names = prior.global_param_names
    local_param_names = prior.get_local_param_names(n_grid * n_grid)
    param_names_stan = ['STAN ' + p for p in global_param_names]

    print(n_grid * n_grid, test_data.shape)

    import optuna
    def objective(trial):
        t1_value = trial.suggest_float('t1_value', 1e-7, 1)
        s_shift_cosine = trial.suggest_float('s_shift_cosine', 0, 10)

        t0_value = 1
        mini_batch_arg = {
            'size': 2,
            'damping_factor': lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2 * t),
        }
        score_model.sde.s_shift_cosine = s_shift_cosine

        test_global_samples = adaptive_sampling(score_model, test_data, obs_n_time_steps=obs_n_time_steps,
                                                n_post_samples=100,
                                                # max_evals=1000,
                                                mini_batch_arg=mini_batch_arg,
                                                run_sampling_in_parallel=False,
                                                device=torch_device, verbose=False)

        cerror = diagnostics.calibration_error(test_global_samples, true_global)['values'].mean()
        return cerror.cpu().numpy()


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print(study.best_params)

    t1_value = study.best_params['t1_value']
    t0_value = 1
    mini_batch_arg = {
        'size': 16,
        'damping_factor': lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2 * t),
    }
    score_model.sde.s_shift_cosine = study.best_params['s_shift_cosine']

    posterior_global_samples_test = adaptive_sampling(score_model, test_data, obs_n_time_steps=obs_n_time_steps,
                                                      n_post_samples=100,
                                                      mini_batch_arg=mini_batch_arg,
                                                      run_sampling_in_parallel=False,
                                                      device=torch_device, verbose=False)

    score_model.sde.s_shift_cosine = 0
    posterior_local_samples_test = euler_maruyama_sampling(score_model, test_data, obs_n_time_steps=obs_n_time_steps,
                                                           n_post_samples=n_post_samples,
                                                           conditions=posterior_global_samples_test,
                                                           diffusion_steps=100,
                                                           device=torch_device, verbose=False)

    posterior_local_samples_test = score_model.prior.transform_local_params(posterior_local_samples_test)

    np.save(f'problems/ar1/posterior_global_samples_test512grid.npy', posterior_global_samples_test)
    np.save(f'problems/ar1/posterior_local_samples_test512grid.npy', posterior_local_samples_test)

    fig = diagnostics.recovery(posterior_global_samples_test, true_global, variable_names=global_param_names)
    fig.savefig(f'plots/{score_model.name}/recovery_global_ours.png')

    fig = diagnostics.calibration_ecdf(posterior_global_samples_test, true_global, difference=True,
                                       variable_names=global_param_names)
    fig.savefig(f'plots/{score_model.name}/ecdf_global_ours.png')

    local_rmse = diagnostics.root_mean_squared_error(posterior_local_samples_test, true_local)['values'].mean()
    print(local_rmse)

    fig = diagnostics.recovery(posterior_local_samples_test.reshape(test_data.shape[0], n_post_samples, -1)[:, :, :12],
                               true_local.reshape(test_data.shape[0], -1)[:, :12],
                               variable_names=local_param_names)
    fig.savefig(f'plots/{score_model.name}/recovery_global_ours.png')

    exit()

else:
    raise ValueError('Unknown variable_of_interest')

df_path = f'plots/{score_model.name}/df_results_{variable_of_interest}.csv'

#%%
# List to store results.
results = []
reached_max_evals = []

# Iterate over data sizes.
for n in data_sizes:
    # Generate synthetic data with enough samples
    prior_dict = prior.sample(batch_size=n_samples_data, n_local_samples=n)
    true_params_global, true_params_local, test_data = prior_dict['global_params'], prior_dict['local_params'], prior_dict['data']

    true_params_global = true_params_global.numpy()
    true_params_local = true_params_local.numpy()
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
                #elif max_reached[2] == nc and max_reached[3] == cs and max_reached[4] < d_factor:
                #    # all conditions are the same (assuming mini-batching does not change)
                #    # check if smaller damping factor already failed
                #    print(f'smaller damping factor already failed, skipping {nc}, {cs}')
                #    skip = True
                #    break
        if skip:
            results.append({
                "data_size": n,
                "data_id": -1,
                "mini_batch": mb if mb is not None else n,
                "damping_factor": d_factor,
                'n_conditions': nc,
                'cosine_shift': cs,
                "n_steps": max_steps,
                "median": np.nan,
                "rmse_global":  np.nan,
                "c_error_global":  np.nan,
                "contractions_global":  np.nan,
                "rmse_local":  np.nan,
                "c_error_local":  np.nan,
                "contractions_local":  np.nan,
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
            damping_factor = lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2*t)
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor}
        else:
            damping_factor = lambda t: torch.ones_like(t) * d_factor
            if mb is None:
                mini_batch_arg = {'damping_factor': damping_factor}
            else:
                mini_batch_arg = {'size': mb, 'damping_factor': damping_factor}

        # Run adaptive sampling.
        try:
            print('global sampling')
            test_global_samples, list_steps = adaptive_sampling(score_model, test_data, conditions=None,
                                                         obs_n_time_steps=obs_n_time_steps,
                                                         n_post_samples=n_post_samples,
                                                         mini_batch_arg=mini_batch_arg,
                                                         max_evals=max_steps*2,
                                                         t_end=0, random_seed=experiment_id, device=torch_device,
                                                         run_sampling_in_parallel=False,  # can actually be faster
                                                         return_steps=True)

            score_model.current_number_of_obs = 1
            score_model.sde.s_shift_cosine = 0
            print('local sampling')
            test_local_samples = euler_maruyama_sampling(score_model, test_data, obs_n_time_steps=obs_n_time_steps,
                                                      n_post_samples=test_global_samples.shape[1],
                                                      conditions=test_global_samples,
                                                      diffusion_steps=200, random_seed=experiment_id,
                                                      device=torch_device, verbose=False)

            test_local_samples = score_model.prior.transform_local_params(test_local_samples)[..., 0]

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
                "median": np.nan,
                "rmse_global":  np.nan,
                "c_error_global":  np.nan,
                "contractions_global":  np.nan,
                "rmse_local":  np.nan,
                "c_error_local":  np.nan,
                "contractions_local":  np.nan,
            })
            df_results = pd.DataFrame(results)
            df_results.to_csv(df_path)
            continue

        # Number of steps
        if np.isnan(test_global_samples).any():
            n_steps = np.inf
            reached_max_evals.append((n, mb, nc, cs, d_factor))
            print('nan in global samples')
        else:
            n_steps = np.mean([len(ls) for ls in list_steps])
            if n_steps >= max_steps:
                # others will also fail to converge
                reached_max_evals.append((n, mb, nc, cs, d_factor))
                print('max steps reached')

        print('global shapes', test_global_samples.shape, true_params_global.shape)
        rmse_global = diagnostics.root_mean_squared_error(test_global_samples, true_params_global)['values'].mean()
        c_error_global = diagnostics.calibration_error(test_global_samples, true_params_global)['values'].mean()
        contractions_global = diagnostics.posterior_contraction(test_global_samples, true_params_global)['values'].mean()

        print('local shapes', test_local_samples.shape, true_params_local.shape)
        rmse_local = diagnostics.root_mean_squared_error(test_local_samples, true_params_local)['values'].mean()
        c_error_local = diagnostics.calibration_error(test_local_samples, true_params_local)['values'].mean()
        contractions_local = diagnostics.posterior_contraction(test_local_samples, true_params_local)['values'].mean()

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
                "median_global": np.median(test_global_samples, axis=1)[i],
                "median_local": np.median(test_local_samples, axis=1)[i],
                "rmse_global": rmse_global,
                "c_error_global": c_error_global,
                "contractions_global": contractions_global,
                "rmse_local": rmse_local,
                "c_error_local": c_error_local,
                "contractions_local": contractions_local,
            })

        # Create a DataFrame from the results list. Save intermediate results
        df_results = pd.DataFrame(results)
        df_results.to_csv(df_path)
