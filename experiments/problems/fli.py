import numpy as np
import torch

from scipy.special import expit, logit
from torch.utils.data import Dataset
from diffusion_model.helper_functions import generate_diffusion_time

class Simulator:
    def __init__(self):
        self.gw = 12500/256  # gate width in ps
        try:
            self.noise = np.load('experiments/problems/FLI/noise_micro.npy')
            self.pIRF = np.load('experiments/problems/FLI/irf_micro.npy')
        except FileNotFoundError:
            try:
                self.noise = np.load('noise_micro.npy')
            except FileNotFoundError:
                self.noise = None
            self.pIRF = np.load('irf_micro.npy')

        self.n_time_points = self.pIRF.shape[2]
        self.img_size_full = (self.pIRF.shape[0], self.pIRF.shape[1])

    def __call__(self, params):
        # Convert parameters to numpy arrays.
        batch_tau_L, batch_tau_L_2, batch_A_L = params['tau_L'], params['tau_L_2'], params['A_L']

        sims = []
        for tau_L, tau_L_2, A_L in zip(batch_tau_L, batch_tau_L_2, batch_A_L):
            F_dec_conv = self.decay_gen_single(tau_L=tau_L, tau_L_2=tau_L_2, A_L=A_L)
            sims.append(F_dec_conv)

        F_dec_conv = np.stack(sims)
        return dict(observable=F_dec_conv)

    def _sample_noise(self, data, scale=1):
        if self.noise is None:
            # sample intensity
            img = np.random.randint(0, high=25)
            noisy_data = np.round(np.random.poisson(data * img))
        else:
            # use recorded noise
            i = np.random.choice(self.noise.shape[0])
            j = np.random.choice(self.noise.shape[1])
            noisy_data = data + scale * self.noise[i, j]
        return noisy_data

    def decay_gen_single(self, tau_L, tau_L_2, A_L):
        cropped_pIRF = self._random_crop(self.pIRF, crop_size=(1, 1))
        cropped_pIRF = cropped_pIRF / np.sum(cropped_pIRF)
        a1, b1, c1 = np.shape(cropped_pIRF)
        t = np.linspace(0, c1 * (self.gw * (10 ** -3)), c1)
        A = A_L * np.exp(-t / tau_L)
        B = (1-A_L) * np.exp(-t / tau_L_2)
        dec = A + B
        irf_out = cropped_pIRF[0,0]
        dec_conv = self._conv_dec(dec, irf_out)

        # add noise
        dec_conv = self._sample_noise(dec_conv)

        # truncated from below
        dec_conv = np.maximum(dec_conv, 0)

        # scale output to 1
        dec_conv = self._norm1D(dec_conv)
        return dec_conv

    @staticmethod
    def _random_crop(array, crop_size):
        A, B, C = array.shape
        a, b = crop_size

        if a > A or b > B:
            raise ValueError("Crop size must be smaller than or equal to the original array size")

        # Randomly select the top-left corner of the crop
        top = np.random.randint(0, A - a + 1)
        left = np.random.randint(0, B - b + 1)

        # Crop the subarray
        return array[top:top + a, left:left + b, :]

    @staticmethod
    def _norm1D(fn):
        if np.amax(fn) == 0:
            nfn = fn
        else:
            nfn = fn / np.amax(fn)
        return nfn

    @staticmethod
    def _conv_dec(dec, irf):
        conv = np.convolve(dec, irf)
        conv = conv[:len(dec)]
        return conv


class FLI_Prior:
    """
    Prior parameters for the FLI problem. All hyper-priors are log-normal.
    """
    def __init__(self, parameterization='difference'):
        self.n_params_global = 6
        self.n_params_local = 3
        self.parameterization = parameterization

        self.a_mean_hyperprior_mean = logit(0.6)  # skewed towards higher values
        self.a_mean_hyperprior_std = 1
        self.a_std_hyperprior_mean = -1
        self.a_std_hyperprior_std = 0.5

        if self.parameterization == 'difference':
            self.tau1_mean_hyperprior_mean = np.log(0.7)
            self.tau1_mean_hyperprior_std = 0.7
            self.tau1_std_hyperprior_mean = -1
            self.tau1_std_hyperprior_std = 0.5

            self.delta_tau_mean_hyperprior_mean = np.log(1.)
            self.delta_tau_mean_hyperprior_std = 0.5
            self.delta_tau_std_hyperprior_mean = -2
            self.delta_tau_std_hyperprior_std = 0.5

            self.global_param_names = [r'$\log \tau_1^G$', r'$\log \sigma_{\tau_1^G}$',
                                       r'$\log \Delta\tau_2^G$', r'$\log \sigma_{\Delta\tau_2^G}$',
                                       r'$a^G$', r'$\log \sigma_{a^G}$']

            # Build prior parameters as tensors.
            self.hyper_prior_means = torch.tensor(
                [self.tau1_mean_hyperprior_mean,
                 self.tau1_std_hyperprior_mean,
                 self.delta_tau_mean_hyperprior_mean,
                 self.delta_tau_std_hyperprior_mean,
                 self.a_mean_hyperprior_mean,
                 self.a_std_hyperprior_mean],
                dtype=torch.float32
            )
            self.hyper_prior_stds = torch.tensor(
                [self.tau1_mean_hyperprior_std,
                 self.tau1_std_hyperprior_std,
                 self.delta_tau_mean_hyperprior_std,
                 self.delta_tau_std_hyperprior_std,
                 self.a_mean_hyperprior_std,
                 self.a_std_hyperprior_std],
                dtype=torch.float32)
        else:
            # s = tau1 + tau2
            # r = tau2 / tau1, r = 1 + exp(log_r_L), such that r is always > 1 and tau2 > tau1
            self.log_s_mean_hyperprior_mean = np.log(2.0)
            self.log_s_mean_hyperprior_std = 0.3
            self.log_s_std_hyperprior_mean = -2
            self.log_s_std_hyperprior_std = 0.1

            self.log_r_mean_hyperprior_mean = 0.0  # ratio ~2
            self.log_r_mean_hyperprior_std = 0.3
            self.log_r_std_hyperprior_mean = -2
            self.log_r_std_hyperprior_std = 0.1

            self.global_param_names = [r'$\log s^G$', r'$\log \sigma_{s^G}$',
                                       r'$\log r^G$', r'$\log \sigma_{r^G}$',
                                       r'$a^G$', r'$\log \sigma_{a^G}$']

            # Build prior parameters as tensors.
            self.hyper_prior_means = torch.tensor(
                [self.log_s_mean_hyperprior_mean,
                 self.log_s_std_hyperprior_mean,
                 self.log_r_mean_hyperprior_mean,
                 self.log_r_std_hyperprior_mean,
                 self.a_mean_hyperprior_mean,
                 self.a_std_hyperprior_mean],
                dtype=torch.float32
            )
            self.hyper_prior_stds = torch.tensor(
                [self.log_s_mean_hyperprior_std,
                 self.log_s_std_hyperprior_std,
                 self.log_r_mean_hyperprior_std,
                 self.log_r_std_hyperprior_std,
                 self.a_mean_hyperprior_std,
                 self.a_std_hyperprior_std],
                dtype=torch.float32
            )

        self._sample_global()

        np.random.seed(0)
        self.simulator = Simulator()
        self.n_time_points = self.simulator.n_time_points

        test = self.sample_single(1000)
        self.norm_x_mean = torch.tensor([np.mean(test['data'])], dtype=torch.float32)
        self.norm_x_std = torch.tensor([np.std(test['data'])], dtype=torch.float32)
        self.norm_prior_global_mean = torch.tensor(np.mean(test['global_params'], axis=0), dtype=torch.float32)
        self.norm_prior_global_std = torch.tensor(np.std(test['global_params'], axis=0), dtype=torch.float32)
        self.norm_prior_local_mean = torch.tensor(np.mean(test['local_params'], axis=0), dtype=torch.float32)
        self.norm_prior_local_std = torch.tensor(np.std(test['local_params'], axis=0), dtype=torch.float32)
        self.current_device = 'cpu'

    def get_local_param_names(self, n_local_samples):
        if self.parameterization == 'difference':
            names = [[r'$\log \tau_1^{L' + str(i) + '}$',
                      r'$\log \Delta\tau_2^{L' + str(i) + '}$',
                      r'$\log a^{L' + str(i) + '}$'] for i in range(n_local_samples)]
        else:
            names = [[r'$\log s^{L' + str(i) + '}$',
                      r'$\log r^{L' + str(i) + '}$',
                      r'$\log a^{L' + str(i) + '}$'] for i in range(n_local_samples)]
        return np.concatenate(names)


    def _sample_global(self):

        self.a_mean = np.random.normal(self.a_mean_hyperprior_mean, self.a_mean_hyperprior_std)
        self.a_log_std = np.random.normal(self.a_std_hyperprior_mean, self.a_std_hyperprior_std)

        if self.parameterization == 'difference':
            self.log_tau_G = np.random.normal(self.tau1_mean_hyperprior_mean, self.tau1_mean_hyperprior_std)
            self.log_sigma_tau_G = np.random.normal(self.tau1_std_hyperprior_mean, self.tau1_std_hyperprior_std)

            self.log_delta_tau_G = np.random.normal(self.delta_tau_mean_hyperprior_mean,
                                                    self.delta_tau_mean_hyperprior_std)
            self.log_delta_sigma_tau_G = np.random.normal(self.delta_tau_std_hyperprior_mean,
                                                          self.delta_tau_std_hyperprior_std)

            raw_params = dict(
                log_tau_G=self.log_tau_G, log_sigma_tau_G=self.log_sigma_tau_G,
                log_delta_tau_G=self.log_delta_tau_G, log_delta_sigma_tau_G=self.log_delta_sigma_tau_G,
                a_mean=self.a_mean, a_log_std=self.a_log_std
            )
            # transform parameters
            tau_G, tau_G_2, A_G, tau_G_std, tau_G_2_std, A_G_std = self.transform_raw_params(
                log_tau=self.log_tau_G, log_delta_tau=self.log_delta_tau_G, a=self.a_mean,
                log_tau_std=self.log_sigma_tau_G, log_delta_tau_std=self.log_delta_sigma_tau_G, log_a_std=self.a_log_std
            )
        else:
            self.log_s_G_mean = np.random.normal(self.log_s_mean_hyperprior_mean, self.log_s_mean_hyperprior_std)
            self.log_s_G_std = np.random.normal(self.log_s_std_hyperprior_mean, self.log_s_std_hyperprior_std)

            self.log_r_G_mean = np.random.normal(self.log_r_mean_hyperprior_mean, self.log_r_mean_hyperprior_std)
            self.log_r_G_std = np.random.normal(self.log_r_std_hyperprior_mean, self.log_r_std_hyperprior_std)

            raw_params = dict(
                log_s_G_mean=self.log_s_G_mean, log_s_G_std=self.log_s_G_std,
                log_r_G_mean=self.log_r_G_mean, log_r_G_std=self.log_r_G_std,
                a_mean=self.a_mean, a_log_std=self.a_log_std
            )
            # transform parameters
            tau_G, tau_G_2, A_G, tau_G_std, tau_G_2_std, A_G_std = self.transform_raw_params_ratios(
                log_r_L=self.log_r_G_mean, log_s_L=self.log_s_G_mean, a=self.a_mean,
                log_s_G_std=self.log_s_G_std, log_r_G_std=self.log_r_G_std, log_a_std=self.a_log_std
            )
        trans_params = dict(
            tau_G=tau_G, tau_G_std=tau_G_std,
            tau_G_2=tau_G_2, tau_G_2_std=tau_G_2_std,
            A_G=A_G, A_G_std=A_G_std
        )
        return raw_params, trans_params

    def _sample_local(self, n_local_samples):
        a_l = np.random.normal(self.a_mean, np.exp(self.a_log_std), size=n_local_samples)

        if self.parameterization == 'difference':
            log_tau_L = np.random.normal(self.log_tau_G, np.exp(self.log_sigma_tau_G), size=n_local_samples)
            log_delta_tau_L = np.random.normal(self.log_delta_tau_G, np.exp(self.log_delta_sigma_tau_G),
                                               size=n_local_samples)
            raw_params = dict(
                log_tau_L=log_tau_L, log_delta_tau_L=log_delta_tau_L, a_l=a_l
            )
            # transform parameters
            tau_L, tau_L_2, A_L = self.transform_raw_params(
                log_tau=log_tau_L,
                log_delta_tau=log_delta_tau_L,
                a=a_l
            )
        else:
            log_s_L = np.random.normal(self.log_s_G_mean, np.exp(self.log_s_G_std), size=n_local_samples)
            log_r_L = np.random.normal(self.log_r_G_mean, np.exp(self.log_r_G_std), size=n_local_samples)

            raw_params = dict(
                log_r_L=log_r_L, log_s_L=log_s_L, a_l=a_l
            )
            # transform parameters
            tau_L, tau_L_2, A_L = self.transform_raw_params_ratios(
                log_r_L=log_r_L,
                log_s_L=log_s_L,
                a=a_l
            )

        trans_params = dict(
            tau_L=tau_L, tau_L_2=tau_L_2, A_L=A_L
        )
        return raw_params, trans_params

    @staticmethod
    def transform_raw_params(log_tau, log_delta_tau, a, log_tau_std=None, log_delta_tau_std=None, log_a_std=None):
        tau = np.exp(log_tau)
        tau_2 = tau + np.exp(log_delta_tau)
        A = expit(a)

        if log_tau_std is not None and log_delta_tau is not None and log_a_std is not None:
            tau_std = tau * np.exp(log_tau_std)
            tau_2_std = np.sqrt((tau * np.exp(log_tau_std))**2 + (np.exp(log_delta_tau) * np.exp(log_delta_tau_std))**2)
            A_std = A * (1 - A) * np.exp(log_a_std)
            return tau, tau_2, A, tau_std, tau_2_std, A_std
        return tau, tau_2, A

    @staticmethod
    def transform_raw_params_ratios(log_r_L, log_s_L, a, log_s_G_std=None, log_r_G_std=None, log_a_std=None):
        r = 1 + np.exp(log_r_L)
        s = np.exp(log_s_L)
        A = expit(a)
        tau = s / (1 + r)
        tau_2 = r * tau

        if log_s_G_std is not None and log_r_G_std is not None and log_a_std is not None:
            sigma_r = np.exp(log_r_G_std)
            sigma_s = np.exp(log_s_G_std)
            sigma_a = np.exp(log_a_std)

            # propagate into tau = s/(1+r)
            #   ∂tau/∂s = 1/(1+r)    →   var_s term = (tau * sigma_s)^2
            #   ∂tau/∂r = -s/(1+r)^2 →   var_r term = (tau * r/(1+r) * sigma_r)^2
            tau_std = np.sqrt(
                (tau * sigma_s) ** 2 +
                (tau * r / (1 + r) * sigma_r) ** 2
            )

            # propagate into tau2 = r * tau1
            #   best combined: var = var(r τ1) + var(τ1 r)
            #   ≈ (tau2 * sigma_r)**2 + (tau2 * sigma_s)**2
            tau_2_std = tau_2 * np.sqrt(sigma_r ** 2 + sigma_s ** 2)

            # amplitude A = sigmoid(a), so ∂A/∂a = A(1−A)
            A_std = A * (1 - A) * sigma_a
            return tau, tau_2, A, tau_std, tau_2_std, A_std
        return tau, tau_2, A

    def __call__(self, batch_size):
        return self.sample_single(batch_size)

    def sample_single(self, batch_size, n_local_samples=1, transform_params=False):
        """
        Sample a batch of data. The number of local samples can be specified. The data is returned as a dictionary.
        IRF and noise are randomly sampled for each pixel.
        """
        global_params = np.zeros((batch_size, self.n_params_global))
        local_params = np.zeros((batch_size, n_local_samples, self.n_params_local))
        data = np.zeros((batch_size, n_local_samples, self.n_time_points))

        for i in range(batch_size):
            global_sample_raw, global_sample_trans = self._sample_global()
            local_sample_raw, local_sample_trans = self._sample_local(n_local_samples=n_local_samples)
            sim = self.simulator(local_sample_trans)

            if not transform_params:
                global_params[i] = np.array(list(global_sample_raw.values()))
                local_params[i] = np.stack(list(local_sample_raw.values()), axis=-1)
            else:
                global_params[i] = np.array(list(global_sample_trans.values()))
                local_params[i] = np.stack(list(local_sample_trans.values()), axis=-1)
            data[i] = sim['observable'].reshape(n_local_samples, self.n_time_points)

        if not np.isfinite(global_params).all():
            raise ValueError('Non-finite global parameters')
        if not np.isfinite(local_params).all():
            raise ValueError('Non-finite local parameters')
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

        # Extract the relevant entries:
        # For local parameters: [log_tau_L, log_delta_tau_L, a_l]
        # For conditioning global parameters, we use:
        #   - Mean for log_tau_L: condition[..., 0] (global.log_tau_G)
        #   - Mean for log_delta_tau_L: condition[..., 2] (global.log_delta_tau_G)
        #   - Mean for a_l: condition[..., 4] (global.a_mean)
        # and standard deviations given by exp(global.log_sigma_tau_G), exp(global.log_delta_sigma_tau_G), exp(global.a_log_std)
        local_means = condition[..., [0, 2, 4]]
        local_stds = torch.exp(condition[..., [1, 3, 5]])

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


def generate_synthetic_data(prior, n_data, n_local_samples=1, normalize=False,
                            as_grid=False, transform_params=False, random_seed=None):
    """Generate synthetic data for the hierarchical model.

    Parameters:
        prior (Prior): Prior distribution for the model.
        n_data (int): Number of samples to generate.
        n_local_samples (int): Number of pixels in the grid for each sample.
        normalize (bool): Whether to normalize the data.
        as_grid (bool): Whether to return the data as a grid.
        transform_params (bool): Whether to transform the parameters.
        random_seed (int): Random seed for reproducibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sample_dict = prior.sample_single(n_data, n_local_samples=n_local_samples, transform_params=transform_params)
    param_global = torch.tensor(sample_dict['global_params'], dtype=torch.float32)
    param_local = torch.tensor(sample_dict['local_params'], dtype=torch.float32)
    data = torch.tensor(sample_dict['data'], dtype=torch.float32)
    if as_grid:
        # batch_size, n_time_steps, n_grid, n_grid
        grid_size = int(np.sqrt(n_local_samples))
        n_time_steps = data.shape[-1]
        data = data[:, :grid_size*grid_size]
        data = data.reshape(n_data, n_time_steps, grid_size, grid_size)
        param_local = param_local[:, :grid_size*grid_size]
        param_local = param_local.reshape(n_data, grid_size, grid_size, prior.n_params_local)
    if normalize:
        param_global = prior.normalize_theta(param_global, global_params=True)
        param_local = prior.normalize_theta(param_local, global_params=False)
        data = prior.normalize_data(data)
    # add feature dimension for RNN
    data = data.unsqueeze(3)
    return param_global, param_local, data


class FLIProblem(Dataset):
    def __init__(self, n_data, prior, sde, number_of_obs=1,
                 online_learning=False, rectified_flow=False):
        # Create model and dataset
        self._prior = prior
        self._sde = sde

        self._number_of_obs_list = number_of_obs if isinstance(number_of_obs, list) else [number_of_obs]
        self._amortize_n_conditions = True if isinstance(number_of_obs, list) else False
        self._max_number_of_obs = max(self._number_of_obs_list)
        self._current_n_obs = self._max_number_of_obs

        self._n_data = n_data
        self.online_learning = online_learning
        self._rectified_flow = rectified_flow
        self._generate_data()
        self._generate_diffusion_target()

    def _generate_data(self):
        # Create model and dataset
        self._thetas_global, self._thetas_local, self._xs = generate_synthetic_data(
            self._prior,
            n_local_samples=self._max_number_of_obs,
            n_data=self._n_data,
            normalize=True, transform_params=False,
        )
        if self._max_number_of_obs == 1:
            # squeeze obs-dimension
            self._xs = self._xs.squeeze(1)
            self._thetas_local = self._thetas_local.squeeze(1)

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
