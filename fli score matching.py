#%% md
# # FLI with compositional score matching
# 
#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
import itertools

os.environ['KERAS_BACKEND'] = 'torch'
from bayesflow import diagnostics

from torch.utils.data import DataLoader

from diffusion_model import HierarchicalScoreModel, SDE, euler_maruyama_sampling, train_score_model
from diffusion_model.helper_networks import GaussianFourierProjection
from diffusion_model.bayesflow_summary_nets import TimeSeriesNetwork
from problems.fli import FLIProblem, FLI_Prior, generate_synthetic_data
from problems import plot_shrinkage, visualize_simulation_output
#%%
torch_device = torch.device("cuda")
#%%
prior = FLI_Prior()
batch_size = 64
number_of_obs = 1 #[16]
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
print('data generated')
# Create dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

#%%
# Define diffusion model
n_blocks = [5, 6]
hidden_dim = [256, 512]
hidden_dim_summary = [10, 14, 18, 22, 32]
n_blocks, hidden_dim, hidden_dim_summary = list(itertools.product(n_blocks, hidden_dim, hidden_dim_summary))[experiment_id]
summary_net = TimeSeriesNetwork(input_dim=1, recurrent_dim=256, summary_dim=hidden_dim_summary)

global_summary_dim = hidden_dim_summary
#global_summary_net = ShallowSet(dim_input=hidden_dim_summary, dim_output=global_summary_dim, dim_hidden=16)

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
    #global_summary_net=global_summary_net,
    time_embedding_local=time_embedding_local,
    time_embedding_global=time_embedding_global,
    hidden_dim=hidden_dim,
    n_blocks=n_blocks,
    max_number_of_obs=number_of_obs if isinstance(number_of_obs, int) else max(number_of_obs),
    prediction_type=['score', 'e', 'x', 'v'][3],
    sde=current_sde,
    weighting_type=[None, 'likelihood_weighting', 'flow_matching', 'sigmoid'][1],
    prior=prior,
    name_prefix=f'FLI_{hidden_dim_summary}_{hidden_dim}_{n_blocks}_{summary_net.name}_'
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
n_local_samples = 10
valid_prior_global, valid_prior_local, valid_data = generate_synthetic_data(prior=prior, n_data=100,
                                                                            n_local_samples=n_local_samples,
                                                                            random_seed=0)
n_post_samples = 100
global_param_names = prior.global_param_names
local_param_names = prior.get_local_param_names(n_local_samples)
#score_model.current_number_of_obs = 4  # we can choose here, how many observations are passed together through the score
#score_model.current_number_of_obs = 4
print(valid_data.shape, score_model.current_number_of_obs)
#%%
mini_batch_size = 10
t1_value = 0.01
t0_value = 1
sampling_arg = {
    'size': mini_batch_size,
    #'damping_factor': lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * 2*t),
}
#plt.plot(torch.linspace(0, 1, 100), mini_batch_arg['damping_factor'](torch.linspace(0, 1, 100)))
#plt.show()

t0_value, t1_value
#%%
#score_model.sde.s_shift_cosine = 4
posterior_global_samples_valid = euler_maruyama_sampling(score_model, valid_data,
                                                   n_post_samples=n_post_samples,
                                                   sampling_arg=sampling_arg,
                                                   diffusion_steps=500,
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
posterior_local_samples_valid = euler_maruyama_sampling(score_model, valid_data,
                                                        n_post_samples=n_post_samples, conditions=conditions_global,
                                                        diffusion_steps=200, device=torch_device, verbose=False)
#%%
fig = diagnostics.recovery(posterior_local_samples_valid.reshape(valid_data.shape[0], n_post_samples, -1),
                          np.array(valid_prior_local).reshape(valid_data.shape[0], -1),
                          variable_names=local_param_names)
fig.savefig(f'plots/{score_model.name}/recovery_local.png')
#%%
plot_shrinkage(posterior_global_samples_valid[:12], posterior_local_samples_valid[:12], min_max=(-10, 10))
#%%
valid_id = 0
print('Global Estimates')
print('mu:', np.median(posterior_global_samples_valid[valid_id, :, 0]), np.std(posterior_global_samples_valid[valid_id, :, 0]))
print('log sigma:', np.median(posterior_global_samples_valid[valid_id, :, 1]), np.std(posterior_global_samples_valid[valid_id, :, 1]))
print('True')
print('mu:', valid_prior_global[valid_id][0].item())
print('log sigma:', valid_prior_global[valid_id][1].item())
#%%
n_grid = int(np.sqrt(n_local_samples))
ps = posterior_local_samples_valid[valid_id, :, :n_grid*n_grid].reshape(n_post_samples, n_grid, n_grid, 3).copy()
true = valid_prior_local[valid_id, :n_grid*n_grid].numpy().copy().reshape(n_grid, n_grid, 3)
ps[:, :, :, 0] = np.exp(ps[:, :, :, 0])
true[:, :, 0] = np.exp(true[:, :, 0])
ps[:, :, :, 1] = ps[:, :, :, 0] + np.exp(ps[:, :, :, 1])
true[:, :, 1] = true[:, :, 0] + np.exp(true[:, :, 1])
ps[:, :, :, 2] = expit(ps[:, :, :, 2])
true[:, :, 2] = expit(true[:, :, 2])
transf_local_param_names = [r'$\tau_1^L$', r'$\tau_2^L$', r'$A^L$']

med = np.median(ps, axis=0)
std = np.std(ps, axis=0)
error = (med-true)**2
visualize_simulation_output(med, title_prefix=['Posterior Median ' + p for p in transf_local_param_names],
                            cmap='turbo', save_path=f"plots/{score_model.name}/simulation_median_{valid_id}.png")
visualize_simulation_output(true, title_prefix=['True ' + p for p in transf_local_param_names],
                            cmap='turbo', save_path=f"plots/{score_model.name}/simulation_true_{valid_id}.png")

visualize_simulation_output(std, title_prefix=['Posterior Std ' + p for p in transf_local_param_names],
                            cmap='turbo', save_path=f"plots/{score_model.name}/simulation_std_{valid_id}.png")
visualize_simulation_output(error, title_prefix=['Error ' + p for p in transf_local_param_names],
                            cmap='turbo', save_path=f"plots/{score_model.name}/simulation_error_{valid_id}.png")
#%%
fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
for i in range(3):
    ax[i].errorbar(x=true[:, :, i].flatten(), y=med[:, :, i].flatten(), yerr=1.96*std[:, :, i].flatten(), fmt='o')
    #ax[i].plot([np.min(true[:, :, i]), np.max(true[:, :, i])], [np.min(true[:, :, i]), np.max(true[:, :, i])], 'k--')
    ax[i].axhline(np.median(posterior_global_samples_valid[valid_id, :, i], axis=0), color='red', linestyle='--',
                label='Global posterior mean', alpha=0.75)
    ax[i].set_ylabel('Prediction')
    ax[i].set_xlabel('True')
    ax[i].legend()
plt.show()
#%%
