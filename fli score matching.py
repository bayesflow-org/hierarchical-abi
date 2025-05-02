#%% md
# # FLI with compositional score matching
# 
#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import itertools

os.environ['KERAS_BACKEND'] = 'torch'
from bayesflow import diagnostics

from torch.utils.data import DataLoader

from diffusion_model import HierarchicalScoreModel, SDE, euler_maruyama_sampling, train_score_model
from diffusion_model.helper_networks import GaussianFourierProjection, ShallowSet
from diffusion_model.bayesflow_summary_nets import TimeSeriesNetwork
from problems.fli import FLIProblem, FLI_Prior, generate_synthetic_data
from problems import visualize_simulation_output
#%%
torch_device = torch.device("cuda")
#%%
prior = FLI_Prior()
batch_size = 64
number_of_obs = 1 #[4] # 1
max_number_of_obs = number_of_obs if isinstance(number_of_obs, int) else max(number_of_obs)
experiment_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

current_sde = SDE(
    kernel_type=['variance_preserving', 'sub_variance_preserving'][0],
    noise_schedule=['linear', 'cosine', 'flow_matching'][1]
)

dataset = FLIProblem(
    n_data=20000,
    prior=prior,
    sde=current_sde,
    online_learning=True,
    number_of_obs=number_of_obs,
)

dataset_valid = FLIProblem(
    n_data=1000,
    prior=prior,
    sde=current_sde,
    number_of_obs=number_of_obs
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

#%%
# Define diffusion model
n_blocks = [5, 6]
hidden_dim = [256, 512]
hidden_dim_summary = [10, 14, 18, 22, 32]
split_summary_vector = [True, False]
n_blocks, hidden_dim, hidden_dim_summary, split_summary_vector = list(itertools.product(n_blocks, hidden_dim, hidden_dim_summary, split_summary_vector))[experiment_id]
summary_net = TimeSeriesNetwork(input_dim=1, recurrent_dim=256, summary_dim=hidden_dim_summary)

global_summary_dim = hidden_dim_summary
global_summary_net = ShallowSet(dim_input=hidden_dim_summary, dim_output=global_summary_dim, dim_hidden=128)

time_embedding_local = nn.Sequential(
    GaussianFourierProjection(8),
    nn.Linear(8, 8),
    nn.Mish()
)
time_embedding_global = nn.Sequential(
    GaussianFourierProjection(8),
    nn.Linear(8, 8),
    nn.Mish()
)

score_model = HierarchicalScoreModel(
    input_dim_theta_global=prior.n_params_global,
    input_dim_theta_local=prior.n_params_local,
    input_dim_x_global=global_summary_dim,
    input_dim_x_local=hidden_dim_summary,
    summary_net=summary_net,
    global_summary_net=global_summary_net if isinstance(number_of_obs, list) else None,
    #time_embedding_local=time_embedding_local,
    #time_embedding_global=time_embedding_global,
    hidden_dim=hidden_dim,
    n_blocks=n_blocks,
    max_number_of_obs=max_number_of_obs,
    prediction_type=['score', 'e', 'x', 'v'][3],
    sde=current_sde,
    weighting_type=[None, 'likelihood_weighting', 'flow_matching', 'sigmoid'][1],
    prior=prior,
    name_prefix=f'FLI_{max_number_of_obs}_{hidden_dim_summary}_{hidden_dim}_{n_blocks}{"_split" if split_summary_vector else ""}_{summary_net.name}_',
    split_summary_vector=split_summary_vector
)

# make dir for plots
if not os.path.exists(f"plots/{score_model.name}"):
    os.makedirs(f"plots/{score_model.name}")
#%%
if not os.path.exists(f"models/{score_model.name}.pt"):
    # train model
    loss_history = train_score_model(score_model, dataloader, dataloader_valid=dataloader_valid, hierarchical=True,
                                                  epochs=3000, device=torch_device)
    score_model.eval()
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

#%%
score_model.load_state_dict(
    torch.load(f"models/{score_model.name}.pt", weights_only=True, map_location=torch.device(torch_device))
)
score_model.eval()
#%% md
# # Validation
#%%
n_local_samples = 1024*max_number_of_obs
valid_prior_global, valid_prior_local, valid_data = generate_synthetic_data(prior=prior, n_data=100,
                                                                            n_local_samples=n_local_samples,
                                                                            random_seed=0)
n_post_samples = 100
global_param_names = prior.global_param_names
local_param_names = prior.get_local_param_names(n_local_samples)
score_model.current_number_of_obs = max_number_of_obs
print(valid_data.shape, score_model.current_number_of_obs)
#%%
#t1_value = 0.0009
#t0_value = 1
t1_value = 0.01
t0_value = 0.05
sampling_arg = {
    'size': 10,
    #'damping_factor': lambda t: (1-torch.ones_like(t)) * 1/n_local_samples + 0.01,
    'damping_factor': lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2*t),
}

#%%
score_model.sde.s_shift_cosine = 0
posterior_global_samples_valid = euler_maruyama_sampling(score_model, valid_data,
                                                   n_post_samples=n_post_samples,
                                                   sampling_arg=sampling_arg,
                                                   diffusion_steps=1000,
                                                   device=torch_device, verbose=False)
#%%
fig = diagnostics.recovery(posterior_global_samples_valid, np.array(valid_prior_global), variable_names=global_param_names)
fig.savefig(f'plots/{score_model.name}/recovery_global.png')

fig = diagnostics.calibration_ecdf(posterior_global_samples_valid, np.array(valid_prior_global),
                          difference=True, variable_names=global_param_names)
fig.savefig(f'plots/{score_model.name}/ecdf_global.png')
#%%
conditions_global = (np.median(posterior_global_samples_valid, axis=0), posterior_global_samples_valid)[1]
score_model.sde.s_shift_cosine = 0
score_model.current_number_of_obs = 1
posterior_local_samples_valid = euler_maruyama_sampling(score_model, valid_data[:, :12],
                                                        n_post_samples=n_post_samples, conditions=conditions_global,
                                                        diffusion_steps=300, device=torch_device, verbose=False)
#%%
fig = diagnostics.recovery(posterior_local_samples_valid.reshape(valid_data.shape[0], n_post_samples, -1)[:, :, :12],
                          np.array(valid_prior_local).reshape(valid_data.shape[0], -1)[:, :12],
                          variable_names=local_param_names[:12])
fig.savefig(f'plots/{score_model.name}/recovery_local.png')

#%% Real data

grid_data = 512
global_param_names = prior.global_param_names
local_param_names = prior.get_local_param_names(grid_data * grid_data)

binned_data = np.load('problems/FLI/exp_binned_data.npy')[:grid_data, :grid_data]
binned_data = binned_data.reshape(1, grid_data * grid_data, 256, 1) / np.max(binned_data)

data = np.load('problems/FLI/final_Data.npy')[:, :grid_data, :grid_data]
data = data.reshape(1, grid_data * grid_data, 256, 1) / np.max(data)
binary_mask = (np.sum(data, axis=2) != 0) * 1

for i, real_data in enumerate([binned_data, data]):

    t1_value = 0.0009
    t0_value = 1
    n_post_samples = 100
    sampling_arg = {
        'size': 10,
        #'damping_factor': lambda t: (torch.ones_like(t) / real_data.shape[1] * 100) * (t0_value * torch.exp(-np.log(t0_value / t1_value) * 2*t)),
        #'damping_factor': lambda t: (1-torch.ones_like(t)) / real_data.shape[1] + 0.1,
        #'damping_factor': lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2*t),
        #'damping_factor': lambda t: (1-torch.ones_like(t)) * 1/(grid_data**2) * 0.0001 + 0.001,
        #'damping_factor': lambda t: torch.ones_like(t) * 1/(grid_data**2) + 0.1,
        'damping_factor': lambda t: torch.ones_like(t) * 1e-10 + 0.0001,
        #'damping_factor': lambda t: (1-torch.ones_like(t)) * 1e-7 + 2e-4,
        #'sampling_chunk_size': 512,
        "sampling_weights": binary_mask.reshape(-1),
    }
    score_model.sde.s_shift_cosine = 0

    posterior_global_samples_real = euler_maruyama_sampling(score_model, real_data,
                                                             n_post_samples=n_post_samples,
                                                             sampling_arg=sampling_arg,
                                                             diffusion_steps=1000, device=torch_device, verbose=False)

    prior_dict = {}
    posterior_dict = {}
    prior_tranf_dict = {}
    posterior_tranf_dict = {}
    for i in range(len(global_param_names)):
        prior_dict[global_param_names[i]] = valid_prior_global[:, i]
        posterior_dict[global_param_names[i]] = posterior_global_samples_real[0, :, i]

    tau, tau_2, A = prior.transform_raw_params(
            log_tau=prior_dict[global_param_names[0]],
            log_delta_tau=prior_dict[global_param_names[2]],
            a=prior_dict[global_param_names[4]]
        )
    prior_tranf_dict = {
        r'$\tau$': tau,
        r'$\tau_2$': tau_2,
        r'$A$': A
    }

    tau, tau_2, A = prior.transform_raw_params(
            log_tau=posterior_dict[global_param_names[0]],
            log_delta_tau=posterior_dict[global_param_names[2]],
            a=posterior_dict[global_param_names[4]]
        )
    posterior_tranf_dict = {
        r'$\tau$': tau,
        r'$\tau_2$': tau_2,
        r'$A$': A
    }

    fig = diagnostics.pairs_posterior(
        posterior_dict,
        priors=prior_dict,
    )
    fig.savefig(f'plots/{score_model.name}/real_data_global_posterior_{i}.png')

    fig = diagnostics.pairs_posterior(
        posterior_tranf_dict,
        priors=prior_tranf_dict,
    )
    fig.savefig(f'plots/{score_model.name}/real_data_global_posterior_transf_{i}.png')

    score_model.sde.s_shift_cosine = 0
    posterior_local_samples_real = euler_maruyama_sampling(score_model, real_data,
                                                            conditions=posterior_global_samples_real,
                                                            n_post_samples=n_post_samples,
                                                            diffusion_steps=300, device=torch_device, verbose=False)

    tau, tau_2, A = prior.transform_raw_params(
        log_tau=posterior_local_samples_real[0, :, :, 0].reshape(n_post_samples, grid_data, grid_data),
        log_delta_tau=posterior_local_samples_real[0, :, :, 1].reshape(n_post_samples, grid_data, grid_data),
        a=posterior_local_samples_real[0, :, :, 2].reshape(n_post_samples, grid_data, grid_data),
    )
    ps = np.concatenate([tau[:, :, :, np.newaxis], tau_2[:, :, :, np.newaxis], A[:, :, :, np.newaxis]], axis=-1)
    transf_local_param_names = [r'$\tau_1^L$', r'$\tau_2^L$', r'$A^L$']

    med = np.median(ps, axis=0)
    std = np.std(ps, axis=0)
    visualize_simulation_output(med, title_prefix=['Posterior Median ' + p for p in transf_local_param_names],
                                cmap='turbo', save_path=f"plots/{score_model.name}/real_data_median_same_{i}.png")
    visualize_simulation_output(std, title_prefix=['Posterior Std ' + p for p in transf_local_param_names],
                                cmap='turbo', save_path=f"plots/{score_model.name}/real_data_std_same_{i}.png")
    visualize_simulation_output(med, title_prefix=['Posterior Median ' + p for p in transf_local_param_names], same_scale=False,
                                cmap='turbo', save_path=f"plots/{score_model.name}/real_data_median_{i}.png")
    visualize_simulation_output(std, title_prefix=['Posterior Std ' + p for p in transf_local_param_names], same_scale=False,
                                cmap='turbo', save_path=f"plots/{score_model.name}/real_data_std_{i}.png")
    np.save(f'plots/{score_model.name}/fli_local_median_{i}', med)
    np.save(f'plots/{score_model.name}/fli_local_std_{i}', std)

    fig, axis = plt.subplots(1, 5, figsize=(10, 3), tight_layout=True, sharex=True, sharey=True)
    axis = axis.flatten()
    pixel_ids = [0, 0]
    for ax in axis:
        plot_index = np.random.randint(0, tau.shape[0])

        simulations = np.array([
            prior.simulator.decay_gen_single(
                tau_L=tau[post_index, pixel_ids[0], pixel_ids[1]],
                tau_L_2=tau_2[post_index, pixel_ids[0], pixel_ids[1]],
                A_L=A[post_index, pixel_ids[0], pixel_ids[1]]
            ) for post_index in range(tau.shape[0])
        ])

        ax.plot(real_data.reshape(grid_data, grid_data, 256)[pixel_ids[0], pixel_ids[1]], label='data')
        ax.plot(np.median(simulations, axis=0), label='posterior median', alpha=0.8, color='orange')
        ax.fill_between(
            np.arange(simulations.shape[1]),
            np.quantile(simulations, 0.025, axis=0),
            np.quantile(simulations, 0.975, axis=0),
            alpha=0.4,
            color='orange',
            label='posterior 95% CI'
        )
        ax.set_xlabel('Time')
    axis[0].set_ylabel('Normalized Photon Count')
    fig.legend(labels=['data', 'posterior median', 'posterior 95% CI'], bbox_to_anchor=(0.5, -0.07),
               ncol=3, loc='lower center')
    plt.savefig(f'plots/{score_model.name}/real_data_fit_{i}.png')
    plt.close()
