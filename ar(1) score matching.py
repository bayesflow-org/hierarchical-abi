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
# $$ \beta_{i,j} \sim \mathcal{N}(\mu_\beta, \text{std}_\beta^2)$$
# 
# -  In each grid point, we have a time series of `T` observations. For the time beeing, we fix $\sigma=1$.
# $$ y_{i,j} \sim \mathcal{N}(\alpha + \beta_{i,j}y_{i,j-1}, \sigma^2), y_{i,0} \sim \mathcal{N}(0, \sigma^2)$$
# - We observe $T=10$ time points for each grid point. We can also amortize over the time dimension.
#%%
import os

import matplotlib.pyplot as plt
import torch

os.environ['KERAS_BACKEND'] = 'torch'

from torch.utils.data import DataLoader

from diffusion_model import HierarchicalScoreModel, SDE, count_parameters, train_hierarchical_score_model
from problems.ar1_grid import AR1GridProblem, Prior

#%%
torch_device = torch.device("cuda")
#%%
prior = Prior()

#%%
batch_size = 128
number_of_obs = [1, 2, 4, 5, 10]  # or a list

dataset = AR1GridProblem(
    n_data=20000,
    prior=prior,
    online_learning=True,
    number_of_obs=number_of_obs,
    amortize_time=False
)

dataset_valid = AR1GridProblem(
    n_data=batch_size*10,
    prior=prior,
    number_of_obs=number_of_obs
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
    name_prefix='AR1_'
)
count_parameters(score_model)
print(score_model.name)

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
