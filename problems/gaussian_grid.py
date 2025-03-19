import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from cmdstanpy import CmdStanModel
from torch.utils.data import Dataset


class Simulator:
    def __init__(self, n_time_points, max_time=1, sigma_noise=1):
        self.max_time = max_time
        self.dt = self.max_time / n_time_points
        self.n_time_points = n_time_points
        self.sigma_noise = sigma_noise

    def __call__(self, params):
        """
        Simulate Brownian motion with drift.

        The SDE is:
            dx(t) = theta * dt + sigma_noise * sqrt(dt) * dW(t)
        starting from 0.

        The simulation runs for self.n_time_steps steps (with step dt)
        and then returns self.n_time_points evenly spaced observations
        between 0 and self.max_time.

        The parameter dict 'params' must contain:
            - 'theta': drift coefficient

        These parameters can be provided as:
            - A scalar (for a single grid element),
            - A 2D array of shape (batch_size, 1) or (batch_size, n_grid)
              for batch simulations over a one-dimensional grid,
            - A 3D array of shape (batch_size, n_grid, n_grid)
              for batch simulations over a two-dimensional grid.
        """
        # Convert parameters to numpy arrays.
        theta = np.array(params['theta'])
        n_grid = 1 if theta.ndim in (0, 1) else theta.shape[-1]

        # Determine simulation mode and grid shape.
        if theta.ndim in (0,1):
            # Scalar: simulate a single grid element.
            grid_shape = (1,1)
            theta = np.full(grid_shape, theta)
        elif theta.ndim == 2:
            # 2D array: shape (batch_size, d) where d==1.
            if theta.shape[1] == 1:
                grid_shape = (1,1)
            else:
                raise ValueError("For 2D 'theta', the second dimension must be 1.")
        elif theta.ndim == 3:
            # 3D array: shape (batch_size, n_grid, n_grid)
            if theta.shape[1] != n_grid or theta.shape[2] != n_grid:
                raise ValueError("For 3D 'theta', the second and third dimensions must equal n_grid.")
            grid_shape = (n_grid, n_grid)
        else:
            raise ValueError("Parameter 'theta' must be provided as a scalar, 2D array, or 3D array.")
        batch_size = theta.shape[0]

        # Simulate the full trajectory.
        # The noise will have shape: (batch_size, n_time_steps, *grid_shape)
        noise_shape = (batch_size, self.n_time_points) + grid_shape
        noise = np.random.normal(loc=0, scale=1, size=noise_shape)

        # Expand mu and sigma to include a time axis.
        if theta.ndim in (1, 2):
            # mu and sigma have shape (batch_size, grid) in the 2D case
            # For a scalar, we already set them to shape (1,)
            # Expand to (batch_size, 1, grid)
            if batch_size == 1:
                # Ensure shape is (1, 1, grid)
                theta_expanded = theta[np.newaxis, np.newaxis, :]
            else:
                theta_expanded = theta[:, np.newaxis, np.newaxis, :]
        else:
            # For 3D parameters, mu and sigma have shape (batch_size, n_grid, n_grid)
            # Expand to (batch_size, 1, n_grid, n_grid)
            theta_expanded = theta[:, np.newaxis, :, :]

        # Compute increments:
        #   increment = theta * dt + sqrt(dt) * noise
        increments = theta_expanded * self.dt + self.sigma_noise * np.sqrt(self.dt) * noise

        # Full trajectory: shape (batch_size, n_time_steps+1, *grid_shape)
        traj_full = np.cumsum(increments, axis=1)

        if theta.ndim == 2:  # just one grid element
            traj_full = traj_full.reshape(batch_size, self.n_time_points, 1)
            increments = increments.reshape(batch_size, self.n_time_points, 1)
        return dict(observable=traj_full, increments=increments)

class Prior:
    def __init__(self, n_time_points):
        self.mu_mean = 0
        self.mu_std = 3
        self.log_sigma_mean = 0
        self.log_sigma_std = 1
        self.n_params_global = 2
        self.n_params_local = 1

        np.random.seed(0)
        test_prior = self.sample_single(1000)
        self.simulator = Simulator(n_time_points=n_time_points)
        test = self.simulator(test_prior)
        self.x_mean = torch.tensor([np.mean(test['observable'])], dtype=torch.float32)
        self.x_std = torch.tensor([np.std(test['observable'])], dtype=torch.float32)
        self.prior_global_mean = torch.tensor(np.array([np.mean(test_prior['mu']), np.mean(test_prior['log_sigma'])]),
                                              dtype=torch.float32)
        self.prior_global_std = torch.tensor(np.array([np.std(test_prior['mu']), np.std(test_prior['log_sigma'])]),
                                             dtype=torch.float32)
        self.prior_local_mean = torch.tensor(np.array([np.mean(test_prior['theta'])]),
                                             dtype=torch.float32)
        self.prior_local_std = torch.tensor(np.array([np.std(test_prior['theta'])]),
                                            dtype=torch.float32)
        self.device = 'cpu'

    def __call__(self, batch_size):
        return self.sample_single(batch_size)

    def sample_single(self, batch_size):
        mu = np.random.normal(loc=self.mu_mean, scale=self.mu_std, size=(batch_size,1))
        log_sigma = np.random.normal(loc=self.log_sigma_mean, scale=self.log_sigma_std, size=(batch_size,1))
        theta = np.random.normal(loc=mu, scale=np.exp(log_sigma), size=(batch_size, 1))
        return dict(mu=mu, log_sigma=log_sigma, theta=theta)

    def sample_full(self, batch_size, n_grid):
        mu = np.random.normal(loc=self.mu_mean, scale=self.mu_std, size=(batch_size, 1))
        log_sigma = np.random.normal(loc=self.log_sigma_mean, scale=self.log_sigma_std, size=(batch_size, 1))
        theta = np.random.normal(loc=mu[:, np.newaxis], scale=np.exp(log_sigma)[:, np.newaxis],
                                 size=(batch_size, n_grid, n_grid))
        return dict(mu=mu, log_sigma=log_sigma, theta=theta)

    def score_global_batch(self, theta_batch_norm, condition_norm=None):
        """ Computes the global score for a batch of parameters."""
        theta_batch = self.denormalize_theta(theta_batch_norm, global_params=True)
        mu, log_sigma = theta_batch[..., 0], theta_batch[..., 1]
        grad_logp_mu = -(mu - self.mu_mean) / (self.mu_std**2)
        grad_logp_sigma = -(log_sigma - self.log_sigma_mean) / (self.log_sigma_std**2)
        # correct the score for the normalization
        score = torch.stack([grad_logp_mu, grad_logp_sigma], dim=-1)
        return score * self.prior_global_std

    def score_local_batch(self, theta_batch_norm, condition_norm):
        """ Computes the local score for a batch of samples. """
        theta = self.denormalize_theta(theta_batch_norm, global_params=False)
        condition = self.denormalize_theta(condition_norm, global_params=True)
        mu, log_sigma = condition[..., 0], condition[..., 1]
        # Gradient w.r.t theta conditioned on mu and log_sigma
        grad_logp_theta = -(theta - mu) / torch.exp(log_sigma*2)
        # correct the score for the normalization
        score = grad_logp_theta * self.prior_local_std
        return score

    def normalize_theta(self, theta, global_params):
        if self.device != theta.device:
            self.prior_global_mean = self.prior_global_mean.to(theta.device)
            self.prior_global_std = self.prior_global_std.to(theta.device)
            self.prior_local_mean = self.prior_local_mean.to(theta.device)
            self.prior_local_std = self.prior_local_std.to(theta.device)
            self.device = theta.device
            #print(f"Moving prior to device {theta.device}")
        if global_params:
            return (theta - self.prior_global_mean) / self.prior_global_std
        return (theta - self.prior_local_mean) / self.prior_local_std

    def denormalize_theta(self, theta, global_params):
        if self.device != theta.device:
            self.prior_global_mean = self.prior_global_mean.to(theta.device)
            self.prior_global_std = self.prior_global_std.to(theta.device)
            self.prior_local_mean = self.prior_local_mean.to(theta.device)
            self.prior_local_std = self.prior_local_std.to(theta.device)
            self.x_mean = self.x_mean.to(theta.device)
            self.x_std = self.x_std.to(theta.device)
            self.device = theta.device
            #print(f"Moving prior to device {theta.device}")
        if global_params:
            return theta * self.prior_global_std + self.prior_global_mean
        return theta * self.prior_local_std + self.prior_local_mean

    def normalize_data(self, x):
        if self.device != x.device:
            self.prior_global_mean = self.prior_global_mean.to(x.device)
            self.prior_global_std = self.prior_global_std.to(x.device)
            self.prior_local_mean = self.prior_local_mean.to(x.device)
            self.prior_local_std = self.prior_local_std.to(x.device)
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x - self.x_mean) / self.x_std


def generate_synthetic_data(prior, n_samples, n_time_points, grid_size=None, data_size=None, normalize=False, random_seed=None):
    """Generate synthetic data for the hierarchical model.

    Parameters:
        prior (Prior): Prior distribution for the model.
        n_samples (int): Number of samples to generate.
        n_time_points (int): Number of time points for the simulator.
        grid_size (int): Size of the grid for the simulator.
        data_size (int): Size of the grid=data_size**2 for the simulator, but then only return data_size (so a batch
            and not the full grid). Grid size must be proved as well for data_size to work.
        normalize (bool): Whether to normalize the data.
        random_seed (int): Random seed for reproducibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if grid_size is not None:
        batch_params = prior.sample_full(n_samples, n_grid=grid_size)
    else:
        batch_params = prior.sample_single(n_samples)
    if data_size is not None:
        if grid_size is None:
            raise ValueError("Grid size must be provided.")
    simulator = Simulator(n_time_points=n_time_points)
    sim_batch = simulator(batch_params)

    param_global = torch.tensor(np.concatenate((batch_params['mu'], batch_params['log_sigma']), axis=1),
                                dtype=torch.float32)
    param_local = torch.tensor(batch_params['theta'], dtype=torch.float32)
    data = torch.tensor(sim_batch['observable'], dtype=torch.float32)
    if normalize:
        param_global = prior.normalize_theta(param_global, global_params=True)
        param_local = prior.normalize_theta(param_local, global_params=False)
        data = prior.normalize_data(data)
    if grid_size is not None and data_size is not None:
        # create local params and data in shape (n_batch, n_samples, n_time_points, n_features), where n_samples=grid_size^2
        param_local = param_local.unsqueeze(1).reshape(n_samples, grid_size**2, -1)
        data = data.reshape(n_samples, data.shape[1], grid_size ** 2, -1)
        # Now, permute dimensions to have (n_samples, grid_size**2, n_time_points, n_features)
        data = data.permute(0, 2, 1, 3)
        param_local = param_local[:, :data_size]
        data = data[:, :data_size]
    return param_global, param_local, data


class GaussianGridProblem(Dataset):
    def __init__(self, n_data, prior, online_learning=False, amortize_time=False, max_number_of_obs=1):
        # Create model and dataset
        self.prior = prior
        self.n_data = n_data
        self.max_number_of_obs = max_number_of_obs
        self.n_obs = self.max_number_of_obs  # this can change for each batch
        self.online_learning = online_learning
        self.amortize_time = amortize_time
        self.max_number_of_time_points = 100
        self.n_time_points = prior.simulator.n_time_points
        if self.amortize_time and not self.online_learning:
            raise NotImplementedError
        self.generate_data()

    def generate_data(self):
        # Create model and dataset
        self.thetas_global, self.thetas_local, self.xs = generate_synthetic_data(
            self.prior,
            n_time_points=self.n_time_points,
            grid_size=int(np.ceil(np.sqrt(self.max_number_of_obs))) if self.max_number_of_obs > 1 else None,  # no need to simulate the full grid
            data_size=self.max_number_of_obs if self.max_number_of_obs > 1 else None,
            n_samples=self.n_data, normalize=True
        )
        self.epsilon_global = torch.randn_like(self.thetas_global, dtype=torch.float32)
        self.epsilon_local = torch.randn_like(self.thetas_local, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.thetas_global)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features_global = self.thetas_global[idx]
        if self.max_number_of_obs > 1:
            features_local = self.thetas_local[idx, :self.n_obs]
            target = self.xs[idx, :self.n_obs]
        else:
            features_local = self.thetas_local[idx]
            target = self.xs[idx]
        noise_global = self.epsilon_global[idx]
        noise_local = self.epsilon_local[idx]
        return features_global, noise_global, features_local, noise_local, target

    def on_epoch_end(self):  # for online learning
        # Regenerate data at the end of each epoch
        if self.online_learning:
            self.generate_data()

    def on_batch_end(self):
        # Called at the end of each batch
        if self.max_number_of_obs > 1:
            # sample number of observations
            self.n_obs = np.random.choice([1, 5, 10, 20, 50, 100])
        if self.amortize_time:
            self.n_time_points = np.random.randint(2, self.max_number_of_time_points + 1)


def visualize_simulation_output(sim_output, title_prefix="Time", cmap="viridis", same_scale=True):
    """
    Visualize the full simulation trajectory on a grid of subplots.

    Parameters:
        sim_output (np.ndarray): Simulation trajectory output.
            For a single simulation, it can be either:
              - 2D: shape (n_time_points, grid_size) for a 1D grid, or
              - 3D: shape (n_time_points, n_grid, n_grid) for a 2D grid.
            For batched simulations, the shape is:
              - 3D: (batch_size, n_time_points, grid_size) or
              - 4D: (batch_size, n_time_points, n_grid, n_grid).
            In such cases, only the first simulation (i.e. first batch element) is visualized.
        title_prefix (str, list): Prefix for subplot titles.
        cmap (str): Colormap for imshow when visualizing 2D grid outputs.
        same_scale (bool): Whether to use the same color scale for all subplots.
    """
    # If a batch dimension is present, select the first simulation.
    if sim_output.ndim == 4:
        # (batch_size, n_time_points, n_grid, n_grid)
        sim_output = sim_output[0]

    # Determine number of time points.
    n_time_points = sim_output.shape[0]

    # Automatically choose grid layout (approximate square).
    n_cols = n_time_points
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    # Flatten axes array in case it's 2D.
    axes = axes.flatten()

    for i in range(n_time_points):
        ax = axes[i]
        # Check if the grid is 1D or 2D.
        # 2D grid: shape (n_time_points, n_grid, n_grid)
        if same_scale:
            im = ax.imshow(sim_output[i], cmap=cmap, vmin=sim_output.min(), vmax=sim_output.max())
        else:
            im = ax.imshow(sim_output[i], cmap=cmap)
        if isinstance(title_prefix, list):
            ax.set_title(title_prefix[i])
        else:
            ax.set_title(f"{title_prefix} {i}")
        fig.colorbar(im, ax=ax)

    # Hide any unused subplots.
    for j in range(n_time_points, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    return


def plot_shrinkage(global_samples, local_samples, ci=95, min_max=None):
    """
    Plots the shrinkage of local estimates toward the global mean for each n_data.

    Parameters:
      global_samples: np.ndarray of shape (n_data, n_samples, 2)
                      The last dimension holds [global_mean, log_std].
      local_samples:  np.ndarray of shape (n_data, n_samples, n_individuals, 1)
                      The last dimension holds the local parameter.
      ci:             Confidence interval percentage (default 95).
    """
    n_data, n_samples, _ = global_samples.shape
    n_individuals = local_samples.shape[2]

    # Create a subplot for each n_data
    nrows, ncols = int(np.ceil(n_data / 4)), 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, int(np.ceil(n_data / 4))*2),
                             sharex=True, sharey=True, tight_layout=True)
    axes = axes.flatten()

    # If there is only one subplot, wrap it in a list for consistent indexing.
    if n_data == 1:
        axes = [axes]

    for i in range(n_data):
        ax = axes[i]

        # Process global posterior for this n_data:
        global_mean_samples = global_samples[i, :, 0]
        global_mean_est = np.mean(global_mean_samples)
        global_ci = [global_mean_est-1.96*np.mean(np.exp(global_samples[i, :, 1])),
                     global_mean_est+1.96*np.mean(np.exp(global_samples[i, :, 1]))]

        # Process local posterior for each individual at data index i:
        local_means = np.zeros(n_individuals)
        local_cis = np.zeros((n_individuals, 2))

        for j in range(n_individuals):
            samples_j = local_samples[i, :, j, 0]
            local_means[j] = np.mean(samples_j)
            local_cis[j, :] = np.percentile(samples_j, [50 - ci/2, 50 + ci/2])

        indices = np.arange(n_individuals)
        # Plot local estimates with error bars
        h1 = ax.errorbar(indices, local_means,
                    yerr=[local_means - local_cis[:, 0], local_cis[:, 1] - local_means],
                    fmt='o', capsize=5, label='Local posterior mean')

        # Plot the global estimate as a horizontal dashed line
        h2 = ax.axhline(global_mean_est, color='red', linestyle='--', label='Global posterior mean')
        # Shade the global CI
        h3 = ax.fill_between(indices, global_ci[0], global_ci[1],
                        color='red', alpha=0.2, label='Global 95% CI')

        ax.set_ylabel("Parameter Value")
        ax.set_title(f"Data {i}")
        if min_max is not None:
            ax.set_ylim(min_max)
    fig.legend(handles=[h1, h2, h3], loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.05))
    axes[-1].set_xlabel("Individual Index")
    for i in range(n_data, len(axes)):
        # disable axis
        axes[i].set_visible(False)
    plt.show()



stan_file = os.path.join('problems', 'gaussian_grid.stan')
stan_model = CmdStanModel(stan_file=stan_file)


def get_stan_posterior(sim_test, dt_obs, sigma_noise, chains=4):
    time_steps, n_grid, _ = sim_test.shape
    sim_test = sim_test.reshape(-1, time_steps)
    n_obs = sim_test.shape[0]

    # Suppose data is a numpy array of shape (N, T)
    # Prepare data for Stan
    stan_data = {
        'N': n_obs,
        'T': time_steps,
        'dx': sim_test,
        'dt_obs': dt_obs,
        'sigma_noise': sigma_noise
    }

    # Fit the model to the data
    fit = stan_model.sample(data=stan_data, show_progress=False, chains=chains)

    global_posterior = np.concatenate([fit.draws_pd("mu"), fit.draws_pd("log_sigma")], axis=-1)
    local_posterior = fit.draws_pd("theta").T
    return global_posterior, local_posterior
