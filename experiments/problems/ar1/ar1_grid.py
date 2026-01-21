import numpy as np
import torch
from scipy.special import expit, logit
from torch.utils.data import Dataset

from diffusion_model.helper_functions import generate_diffusion_time

N_TIME_POINTS = 5

class Simulator:
    def __init__(self, sigma_noise=0.1):
        """
        Simulator for the hierarchical AR(1) model:
            y[t] = alpha + beta y[t-1] + noise[t]
        starting from an initial value.

        Parameters:
            sigma_noise (float): noise standard deviation.
        """
        self.sigma_noise = sigma_noise
        self.initial_is_zero = False

    def __call__(self, params, n_time_points=N_TIME_POINTS):
        eta = np.array(params['eta'])
        alpha = np.array(params['alpha'])
        N = eta.size
        if eta.ndim > 1:
            raise ValueError("eta must be a 1D array.")

        # Generate noise for the increments: shape (N, n_time_points)
        noise = np.random.normal(
            loc=0,
            scale=self.sigma_noise,
            size=(N, n_time_points)
        )

        # Initialize trajectories with the initial condition
        traj = np.zeros((N, n_time_points))

        # Simulate the AR(1) process for each trajectory and each batch
        if not self.initial_is_zero:
            traj[:, 0] = noise[:, 0]
        for t in range(1, n_time_points):
            traj[:, t] = alpha + traj[:, t - 1] * eta + noise[:, t]

        return dict(observable=traj)


class Prior:
    def __init__(self):
        """
        Hierarchical prior for the AR(1) model.
        """
        self.alpha_mean = 0
        self.alpha_std = 1
        self.beta_mean = 0
        self.beta_std = 1
        self.log_sigma_mean = 0
        self.log_sigma_std = 1
        self.n_params_global = 3
        self.n_params_local = 1
        self.global_param_names = [r'$\alpha$', r'$\beta$', r'$\log \sigma$']

        # Build prior parameters as tensors.
        self.hyper_prior_means = torch.tensor(
            [self.alpha_mean,
             self.beta_mean,
             self.log_sigma_mean],
            dtype=torch.float32
        )
        self.hyper_prior_stds = torch.tensor(
            [self.alpha_std,
             self.beta_std,
             self.log_sigma_std],
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
        self.norm_prior_local_mean = torch.mean(test['local_params_raw'], dim=0)
        self.norm_prior_local_std = torch.std(test['local_params_raw'], dim=0)

        self.current_device = 'cpu'


    def __call__(self, batch_size):
        return self.sample(batch_size=batch_size)

    @staticmethod
    def get_local_param_names(n_local_samples):
        return [r'$\eta_{' + str(i) + '}$' for i in range(n_local_samples)]

    def _sample_global(self):
        # Sample global parameters
        self.alpha = np.random.normal(loc=self.alpha_mean, scale=self.alpha_std)
        self.beta = np.random.normal(loc=self.beta_mean, scale=self.beta_std)
        self.log_sigma = np.random.normal(loc=self.log_sigma_mean, scale=self.log_sigma_std)
        return dict(alpha=self.alpha, beta=self.beta, log_sigma=self.log_sigma)

    def _sample_local(self, n_local_samples=1):
        # Sample local parameters
        eta_raw = np.random.normal(loc=0, scale=np.exp(self.log_sigma), size=n_local_samples)
        eta = self.transform_local_params(beta=self.beta, eta_raw=eta_raw)
        return dict(eta=eta, eta_raw=eta_raw)

    @staticmethod
    def transform_local_params(beta, eta_raw):
        # transform raw local parameters
        return 2*expit(beta + eta_raw)-1

    @staticmethod
    def back_transform_local_params(local_params):
        local_params_raw = logit((local_params + 1) / 2)
        local_params_raw[local_params_raw < -100] = -100
        local_params_raw[local_params_raw > 100] = 100
        return local_params_raw

    def sample(self, batch_size, n_local_samples=1, n_time_points=N_TIME_POINTS, get_grid=False):
        # Sample global and local parameters and simulate data
        global_params = np.zeros((batch_size, self.n_params_global))
        local_params_raw = np.zeros((batch_size, n_local_samples))
        local_params = np.zeros((batch_size, n_local_samples))
        data = np.zeros((batch_size, n_local_samples, n_time_points))

        for i in range(batch_size):
            global_sample = self._sample_global()
            local_sample = self._sample_local(n_local_samples=n_local_samples)
            sim_dict = {'alpha': global_sample['alpha'], 'eta': local_sample['eta']}
            sim = self.simulator(sim_dict, n_time_points=n_time_points)

            global_params[i] = [global_sample['alpha'], global_sample['beta'], global_sample['log_sigma']]
            local_params_raw[i] = local_sample['eta_raw']
            local_params[i] = local_sample['eta']
            data[i] = sim['observable']

        # Convert to tensors
        global_params = torch.tensor(global_params, dtype=torch.float32)
        local_params = torch.tensor(local_params, dtype=torch.float32)
        local_params_raw = torch.tensor(local_params_raw, dtype=torch.float32)
        data = torch.tensor(data, dtype=torch.float32)
        if get_grid:
            grid_size = int(np.sqrt(n_local_samples))
            data = data[:, :grid_size ** 2]
            data = data.reshape(batch_size, n_time_points, grid_size, grid_size)
            local_params = local_params[:, :grid_size ** 2]
            local_params_raw = local_params_raw[:, :grid_size ** 2]
            local_params = local_params.reshape(batch_size, grid_size, grid_size)
            local_params_raw = local_params_raw.reshape(batch_size, grid_size, grid_size)
        return dict(global_params=global_params, local_params=local_params,
                    local_params_raw=local_params_raw, data=data)


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
        #local_means = 0
        local_stds = torch.exp(condition[..., 2])

        # Compute the local score.
        score = -theta / (local_stds ** 2)
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
            #print(f"Moving prior to device: {device}")
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
    def __init__(self, n_data, prior, sde, as_set=False, number_of_obs=1,
                 online_learning=False, amortize_time=False, rectified_flow=False):
        # Create model and dataset
        self._prior = prior
        self._sde = sde
        self._as_set = as_set

        self._number_of_obs_list = number_of_obs if isinstance(number_of_obs, list) else [number_of_obs]
        self._amortize_n_conditions = True if isinstance(number_of_obs, list) else False
        self._max_number_of_obs = max(self._number_of_obs_list)
        self._current_n_obs = self._max_number_of_obs

        self._max_number_of_time_points = N_TIME_POINTS
        self._n_time_points = self._max_number_of_time_points
        self._amortize_time = amortize_time

        self._n_data = n_data
        self.online_learning = online_learning
        self._rectified_flow = rectified_flow
        self._generate_data()
        self._generate_diffusion_target()

    def _generate_data(self):
        # Create model and dataset
        sim_dict = self._prior.sample(
            batch_size=self._n_data,
            n_local_samples=self._max_number_of_obs,
            n_time_points=self._max_number_of_time_points
        )
        self._thetas_global = self._prior.normalize_theta(sim_dict['global_params'], global_params=True)
        self._thetas_local = self._prior.normalize_theta(sim_dict['local_params_raw'], global_params=False)
        self._xs = self._prior.normalize_data(sim_dict['data'])

        if self._max_number_of_obs == 1:
            # squeeze obs-dimension
            self._xs = self._xs.squeeze(1)
        else:
            # expand obs-dimension
            self._thetas_local = self._thetas_local.unsqueeze(-1)
        if not self._as_set:
            # add feature dimension for RNN
            self._xs = self._xs.unsqueeze(-1)

        # generate new noise only with new data
        if self._rectified_flow:
            self._noise_global = torch.randn_like(self._thetas_global)
            self._noise_local = torch.randn_like(self._thetas_local)

    def _generate_diffusion_target(self):
        # Generate diffusion time and training target
        self._diffusion_time = generate_diffusion_time(size=self._n_data, return_batch=True)

        # perturb the theta batch
        snr = self._sde.get_snr(t=self._diffusion_time)
        self._alpha, self._sigma = self._sde.kernel(log_snr=snr)

        # generate new noise in each epoch
        if not self._rectified_flow:
            self._noise_global = torch.randn_like(self._thetas_global)
            self._noise_local = torch.randn_like(self._thetas_local)

    def __len__(self):
        # this should return the size of the dataset
        return len(self._thetas_global)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        params_global = self._thetas_global[idx]
        noise_global = self._noise_global[idx]

        if self._amortize_n_conditions:
            params_local = self._thetas_local[idx, :self._current_n_obs]
            noise_local = self._thetas_local[idx, :self._current_n_obs]
            data = self._xs[idx, :self._current_n_obs]
            param_local_noisy = self._alpha[idx].unsqueeze(1) * params_local + self._sigma[idx].unsqueeze(1) * noise_local

        else:
            params_local = self._thetas_local[idx]
            noise_local = self._noise_local[idx]
            data = self._xs[idx]
            param_local_noisy = self._alpha[idx] * params_local + self._sigma[idx] * noise_local

        if self._amortize_time:
            data = data[:, :, :self._n_time_points]

        param_global_noisy = self._alpha[idx] * params_global + self._sigma[idx] * noise_global

        target_global = noise_global  # for e-prediction
        target_local = noise_local
        return param_global_noisy, target_global, param_local_noisy, target_local, data, self._diffusion_time[idx]

    def on_epoch_end(self):  # for online learning
        # Regenerate data at the end of each epoch
        if self.online_learning:
            self._generate_data()
        self._generate_diffusion_target()

    def on_batch_end(self):
        # Called at the end of each batch
        if self._amortize_n_conditions:
            # sample number of observations
            self._current_n_obs = np.random.choice(self._number_of_obs_list)
        if self._amortize_time:
            self._n_time_points = np.random.randint(2, self._max_number_of_time_points + 1)
