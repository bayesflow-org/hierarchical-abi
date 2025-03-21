#%% md
# # FLI with compositional score matching
# 
#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

os.environ['KERAS_BACKEND'] = 'torch'
from bayesflow import diagnostics

from torch.utils.data import DataLoader

from diffusion_model import HierarchicalScoreModel, SDE, weighting_function, euler_maruyama_sampling, adaptive_sampling, \
    generate_diffusion_time, count_parameters, train_hierarchical_score_model
from problems.fli import FLIProblem, FLI_Prior, generate_synthetic_data
from problems import plot_shrinkage, visualize_simulation_output
#%%
torch_device = torch.device("cuda")
#%%
prior = FLI_Prior()
#%%
batch_size = 128
number_of_obs = [1, 2, 4, 5, 10]  # or a list

dataset = FLIProblem(
    n_data=20000,
    prior=prior,
    online_learning=True,
    number_of_obs=number_of_obs,
)

dataset_valid = FLIProblem(
    n_data=batch_size*10,
    prior=prior,
    number_of_obs=number_of_obs,
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

score_model = HierarchicalScoreModel(
    input_dim_theta_global=prior.n_params_global,
    input_dim_theta_local=prior.n_params_local,
    input_dim_x=1,
    hidden_dim=512,
    n_blocks=5,
    max_number_of_obs=number_of_obs if isinstance(number_of_obs, int) else max(number_of_obs),
    prediction_type=['score', 'e', 'x', 'v'][3],
    sde=current_sde,
    time_embed_dim=32,
    weighting_type=[None, 'likelihood_weighting', 'flow_matching', 'sigmoid'][1],
    prior=prior,
    name_prefix='FLI_'
)
print(score_model.name)
count_parameters(score_model)

# make dir for plots
if not os.path.exists(f"plots/{score_model.name}"):
    os.makedirs(f"plots/{score_model.name}")
#%%
# train model
loss_history = train_hierarchical_score_model(score_model, dataloader, dataloader_valid=dataloader_valid,
                                              epochs=2000, device=torch_device)
score_model.eval()
torch.save(score_model.state_dict(), f"models/{score_model.name}.pt")

# plot loss history
plt.figure(figsize=(6, 3), tight_layout=True)
plt.plot(loss_history[:, 0], label='Train')
plt.plot(loss_history[:, 1], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'plots/{score_model.name}/loss_training.png')
plt.show()
