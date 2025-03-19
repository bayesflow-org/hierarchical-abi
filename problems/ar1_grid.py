import os

import numpy as np
import torch
from cmdstanpy import CmdStanModel
from torch.utils.data import Dataset


class Simulator:
    def __init__(self, sigma_noise=1):
        """
        Simulator for the hierarchical AR(1) model:
            y[t] = y[t-1] + theta + sigma_noise * epsilon[t]
        starting from an initial value (default: 0).

        Parameters:
            sigma_noise (float): noise standard deviation.
        """
        self.sigma_noise = sigma_noise

    def __call__(self, params, n_time_points=10):
        beta = np.array(params['beta'])
        alpha = np.array(params['alpha'])
        N = beta.size
        if beta.ndim > 1:
            raise ValueError("beta must be a 1D array.")

        # Generate noise for the increments: shape (N, n_time_points)
        noise = np.random.normal(
            loc=0,
            scale=self.sigma_noise,
            size=(N, n_time_points)
        )

        # Initialize trajectories with the initial condition
        traj = np.zeros((N, n_time_points))

        # Simulate the AR(1) process for each trajectory and each batch
        traj[:, 0] =  noise[:, 0]
        for t in range(1, n_time_points):
            traj[:, t] = alpha + traj[:, t - 1] * beta + noise[:, t]

        return dict(observable=traj)


class Prior:
    def __init__(self):
        """
        Hierarchical prior for the AR(1) model.

        """
        self.alpha_mean = 0
        self.alpha_std = 1
        self.beta_mu_mean = 0
        self.beta_mu_std = 1
        self.log_beta_std_mean = -1
        self.log_beta_std_std = 1
        self.n_params_global = 3
        self.n_params_local = 1
        self.global_param_names = [r'$\alpha$', r'$\beta_{\mu}$', r'$\log \sigma_{\beta}$']

        # Build prior parameters as tensors.
        self.hyper_prior_means = torch.tensor(
            [self.alpha_mean,
             self.beta_mu_mean,
             self.log_beta_std_mean],
            dtype=torch.float32
        )
        self.hyper_prior_stds = torch.tensor(
            [self.alpha_std,
             self.beta_mu_std,
             self.log_beta_std_std],
            dtype=torch.float32
        )

        np.random.seed(0)
        self.simulator = Simulator()

        # Compute normalization constants
        test = self.sample(1000)
        self.norm_x_mean = torch.mean(test['data'])
        self.norm_x_std = torch.std(test['data'])
        self.norm_prior_global_mean = torch.mean(test['global_params'], dim=0)
        self.norm_prior_global_std = torch.std(test['global_params'], dim=0)
        self.norm_prior_local_mean = torch.mean(test['local_params'], dim=0)
        self.norm_prior_local_std = torch.std(test['local_params'], dim=0)

        self.current_device = 'cpu'


    def __call__(self, batch_size):
        return self.sample(batch_size=batch_size)

    @staticmethod
    def get_local_param_names(n_local_samples):
        return [r'$\beta_{' + str(i) + '}$' for i in range(n_local_samples)]

    def _sample_global(self):
        # Sample global parameters
        self.alpha = np.random.normal(loc=self.alpha_mean, scale=self.alpha_std)
        self.beta_mu = np.random.normal(loc=self.beta_mu_mean, scale=self.beta_mu_std)
        self.log_beta_std = np.random.normal(loc=self.log_beta_std_mean, scale=self.log_beta_std_std)
        return dict(alpha=self.alpha, beta_mu=self.beta_mu, log_beta_std=self.log_beta_std)

    def _sample_local(self, n_local_samples=1):
        # Sample local parameters
        self.beta = np.random.normal(loc=self.beta_mu, scale=np.exp(self.log_beta_std), size=n_local_samples)
        return dict(beta=self.beta)

    def sample(self, batch_size, n_local_samples=1, n_time_points=10):
        # Sample global and local parameters and simulate data
        global_params = np.zeros((batch_size, self.n_params_global))
        local_params = np.zeros((batch_size, n_local_samples))
        data = np.zeros((batch_size, n_local_samples, n_time_points))

        for i in range(batch_size):
            global_sample = self._sample_global()
            local_sample = self._sample_local(n_local_samples=n_local_samples)
            sim_dict = {'alpha': global_sample['alpha'], 'beta': local_sample['beta']}
            sim = self.simulator(sim_dict, n_time_points=n_time_points)

            global_params[i] = [global_sample['alpha'], global_sample['beta_mu'], global_sample['log_beta_std']]
            local_params[i] = local_sample['beta']
            data[i] = sim['observable']

        # add axis for parameter
        local_params = local_params[:, :, np.newaxis]

        # Convert to tensors
        global_params = torch.tensor(global_params, dtype=torch.float32)
        local_params = torch.tensor(local_params, dtype=torch.float32)
        data = torch.tensor(data, dtype=torch.float32)
        return dict(global_params=global_params, local_params=local_params, data=data)


    def score_global_batch(self, theta_batch_norm, condition_norm=None):
        """ Computes the global score for a batch of parameters."""
        theta_batch = self.denormalize_theta(theta_batch_norm, global_params=True)
        # Compute the score (gradient of log prior) elementwise.
        score = -(theta_batch - self.hyper_prior_means) / (self.hyper_prior_stds ** 2)
        # correct the score for the normalization
        return score * self.norm_prior_global_std

    def score_local_batch(self, theta_batch_norm, condition_norm):
        """ Computes the local score for a batch of samples. """
        theta = self.denormalize_theta(theta_batch_norm, global_params=False)
        condition = self.denormalize_theta(condition_norm, global_params=True)

        # Extract the relevant entries
        # condition = [alpha, beta_mu, log_beta_std]
        local_means = condition[..., 1]
        local_stds = torch.exp(condition[..., 2])

        # Compute the local score.
        score = -(theta - local_means) / (local_stds ** 2)
        # correct the score for the normalization
        return score * self.norm_prior_local_std

    def normalize_theta(self, theta, global_params):
        self._move_to_device(theta.device)
        if global_params:
            return (theta - self.norm_prior_global_mean) / self.norm_prior_global_std
        return (theta - self.norm_prior_local_mean) / self.norm_prior_local_std

    def denormalize_theta(self, theta, global_params):
        self._move_to_device(theta.device)
        if global_params:
            return theta * self.norm_prior_global_std + self.norm_prior_global_mean
        return theta * self.norm_prior_local_std + self.norm_prior_local_mean

    def normalize_data(self, x):
        self._move_to_device(x.device)
        return (x - self.norm_x_mean) / self.norm_x_std

    def _move_to_device(self, device):
        if self.current_device != device:
            print(f"Moving prior to device: {device}")
            self.norm_prior_global_mean = self.norm_prior_global_mean.to(device)
            self.norm_prior_global_std = self.norm_prior_global_std.to(device)
            self.norm_prior_local_mean = self.norm_prior_local_mean.to(device)
            self.norm_prior_local_std = self.norm_prior_local_std.to(device)
            self.hyper_prior_means = self.hyper_prior_means.to(device)
            self.hyper_prior_stds = self.hyper_prior_stds.to(device)
            self.norm_x_mean = self.norm_x_mean.to(device)
            self.norm_x_std = self.norm_x_std.to(device)
            self.current_device = device
        return



class AR1GridProblem(Dataset):
    def __init__(self, n_data, prior, online_learning=False, amortize_time=False, max_number_of_obs=1):
        # Create model and dataset
        self.prior = prior
        self.n_data = n_data

        self.max_number_of_obs = max_number_of_obs
        self.n_obs = self.max_number_of_obs  # this can change for each batch
        self.online_learning = online_learning

        self.amortize_time = amortize_time
        self.max_number_of_time_points = 100
        self.n_time_points = 10
        if self.amortize_time and not self.online_learning:
            raise NotImplementedError
        self.generate_data()

    def generate_data(self):
        # Create model and dataset
        sim_dict = self.prior.sample(
            batch_size=self.n_data,
            n_local_samples=self.n_obs,
            n_time_points=self.n_time_points
        )
        self.thetas_global = self.prior.normalize_theta(sim_dict['global_params'], global_params=True)
        self.thetas_local = self.prior.normalize_theta(sim_dict['local_params'], global_params=False)
        self.xs = self.prior.normalize_data(sim_dict['data'])

        if self.max_number_of_obs == 1:
            # squeeze obs-dimension for RNN
            self.xs = self.xs.squeeze(1)
        # add feature dimension for RNN
        self.xs = self.xs.unsqueeze(-1)

        self.epsilon_global = torch.randn_like(self.thetas_global, dtype=torch.float32)
        self.epsilon_local = torch.randn_like(self.thetas_local, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.thetas_global)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features_global = self.thetas_global[idx]
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


stan_file = os.path.join('problems', 'ar1_grid.stan')
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
                                       fit.draws_pd("mu_beta"), fit.draws_pd("log_std_beta")], axis=-1)
    local_posterior = fit.draws_pd("beta").T
    return global_posterior, local_posterior
