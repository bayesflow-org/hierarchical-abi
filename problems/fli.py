import numpy as np
import torch

from scipy.special import expit
from torch.utils.data import Dataset


class Simulator:
    def __init__(self):
        self.gw = 40  # gate width in ps

        self.noise = np.load('problems/FLI/system_noise.npy')
        self.pIRF = np.load('problems/FLI/IRF.npy')

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

    def sample_augmented_noise(self):
        i = np.random.choice(self.noise.shape[0])
        j = np.random.choice(self.noise.shape[1])

        # augment noise, can be forward or backward
        if np.random.rand() > 0.5:
            noise = self.noise[i, j]
        else:
            noise = self.noise[i, j, ::-1]
        return noise

    def decay_gen_single(self, tau_L, tau_L_2, A_L):
        img = np.random.randint(0, high=1000, size=(1, 1))
        cropped_pIRF = self.random_crop(self.pIRF, crop_size=(1, 1))

        a1, b1, c1 = np.shape(cropped_pIRF)
        t = np.linspace(0, c1 * (self.gw * (10 ** -3)), c1)
        t_minus = np.multiply(t, -1)
        frac2 = 1 - A_L
        A = np.multiply(A_L, np.exp(np.divide(t_minus, tau_L)))
        B = np.multiply(frac2, np.exp(np.divide(t_minus, tau_L_2)))
        dec = A + B
        irf_out = self.norm1D(cropped_pIRF[0,0])
        dec_conv = self.conv_dec(self.norm1D(dec), irf_out)
        dec_conv = self.norm1D(np.squeeze(dec_conv)) * img
        dec_conv += self.sample_augmented_noise()
        return dec_conv

    @staticmethod
    def random_crop(array, crop_size):
        A, B, C = array.shape
        a, b = crop_size

        if a > A or b > B:
            raise ValueError("Crop size must be smaller than or equal to the original array size")

        # Randomly select the top-left corner of the crop
        top = np.random.randint(0, A - a + 1)
        left = np.random.randint(0, B - b + 1)

        # Crop the subarray
        return array[top:top + a, left:left + b, :]

    def decay_gen_full(self, tau_L, tau_L_2, A_L):
        if tau_L.size != self.img_size_full[0] * self.img_size_full[1]:
            raise ValueError("tau_L must be the same size as the image")

        img = np.random.randint(0, high=1000, size=(self.img_size_full[0], self.img_size_full[1]))

        a, b = np.shape(img)
        a1, b1, c1 = np.shape(self.pIRF)
        t = np.linspace(0, c1 * (self.gw * (10 ** -3)), c1)
        t_minus = np.multiply(t, -1)
        tau1 = np.reshape(tau_L, (a, b))
        tau2 = np.reshape(tau_L_2, (a, b))
        frac1 = np.reshape(A_L, (a, b))
        frac2 = 1 - frac1
        dec = np.zeros([a1, b1, c1])
        A = np.zeros([a1, b1, c1])
        B = np.zeros([a1, b1, c1])
        irf_out = np.zeros([a, b, c1])
        dec_conv = np.zeros([a, b, c1])
        for i in range(a):
            for j in range(b):
                if tau1[i, j] != 0:
                    A[i, j, :] = np.multiply(frac1[i, j], np.exp(np.divide(t_minus, tau1[i, j])))
                if tau2[i, j] != 0:
                    B[i, j, :] = np.multiply(frac2[i, j], np.exp(np.divide(t_minus, tau2[i, j])))
                dec[i, j, :] = A[i, j, :] + B[i, j, :]
                irf_out[i, j, :] = self.norm1D(self.pIRF[i, j, :])
                dec_conv[i, j, :] = self.conv_dec(self.norm1D(np.squeeze(dec[i, j, :])), np.squeeze(irf_out[i, j, :]))
                dec_conv[i, j, :] = self.norm1D(np.squeeze(dec_conv[i, j, :])) * img[i, j]
                dec_conv[i, j, :] += self.noise[i, j, :]
                #dec_conv[i, j, :] += self.sample_augmented_noise()
        return dec_conv

    @staticmethod
    def norm1D(fn):
        if np.amax(fn) == 0:
            nfn = fn
        else:
            nfn = np.divide(fn, np.amax(fn))
        return nfn

    @staticmethod
    def conv_dec(dec, irf):
        conv = np.convolve(dec, irf)
        conv = conv[:len(dec)]
        return conv


class FLI_Prior:
    def __init__(self):
        self.tau1_mean_hyperprior_mean = np.log(0.2)
        self.tau1_mean_hyperprior_std = 0.7
        self.tau1_std_hyperprior_mean = -1
        self.tau1_std_hyperprior_std = 0.1

        self.delta_tau_mean_hyperprior_mean = np.log(1.2)
        self.delta_tau_mean_hyperprior_std = 0.5
        self.delta_tau_std_hyperprior_mean = -2
        self.delta_tau_std_hyperprior_std = 0.1

        self.a_mean_hyperprior_mean = 0
        self.a_mean_hyperprior_std = 1
        self.a_std_hyperprior_mean = -1
        self.a_std_hyperprior_std = 1

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
            dtype=torch.float32
        )

        self.sample_global()
        self.n_params_global = 6
        self.n_params_local = 3

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

    @staticmethod
    def get_local_param_names(n_local_samples):
        return np.concatenate([[r'$\log \tau_1^{L' + str(i) + '}$',
                                r'$\log \Delta\tau_2^{L' + str(i) + '}$',
                                r'$\log a^{L' + str(i) + '}$'] for i in range(n_local_samples)])


    def sample_global(self):
        self.log_tau_G = np.random.normal(self.tau1_mean_hyperprior_mean, self.tau1_mean_hyperprior_std)
        self.log_sigma_tau_G = np.random.normal(self.tau1_std_hyperprior_mean, self.tau1_std_hyperprior_std)

        self.log_delta_tau_G = np.random.normal(self.delta_tau_mean_hyperprior_mean, self.delta_tau_mean_hyperprior_std)
        self.log_delta_sigma_tau_G = np.random.normal(self.delta_tau_std_hyperprior_mean, self.delta_tau_std_hyperprior_std)

        self.a_mean = np.random.normal(self.a_mean_hyperprior_mean, self.a_mean_hyperprior_std)
        self.a_log_std = np.random.normal(self.a_std_hyperprior_mean, self.a_std_hyperprior_std)
        self.A_mean = expit(self.a_mean)

        return dict(log_tau_G=self.log_tau_G, log_sigma_tau_G=self.log_sigma_tau_G,
                    log_delta_tau_G=self.log_delta_tau_G, log_delta_sigma_tau_G=self.log_delta_sigma_tau_G,
                    a_mean=self.a_mean, a_log_std=self.a_log_std, A_mean=self.A_mean)

    def get_local_variances(self):
        var_tau_L = np.exp(2 * self.log_tau_G + np.exp(self.log_sigma_tau_G)**2) * (np.exp(np.exp(self.log_sigma_tau_G)**2) - 1)
        var_delta = np.exp(2 * self.log_delta_tau_G + np.exp(self.log_delta_sigma_tau_G)**2) * (np.exp(np.exp(self.log_delta_sigma_tau_G)**2) - 1)
        tau_L1_std = np.sqrt(var_tau_L)
        tau_L2_std = np.sqrt(var_tau_L + var_delta)
        return dict(tau_L1_std=tau_L1_std, tau_L2_std=tau_L2_std)

    def sample_local(self, n_local_samples):
        log_tau_L = np.random.normal(self.log_tau_G, np.exp(self.log_sigma_tau_G), size=(n_local_samples,1))
        log_delta_tau_L = np.random.normal(self.log_delta_tau_G, np.exp(self.log_delta_sigma_tau_G), size=(n_local_samples,1))
        a_l = np.random.normal(self.a_mean, np.exp(self.a_log_std), size=(n_local_samples,1))

        tau_L = np.exp(log_tau_L)
        tau_L_2 = tau_L + np.exp(log_delta_tau_L)
        A_L = expit(a_l)
        return dict(tau_L=tau_L, tau_L_2=tau_L_2, A_L=A_L,
                    log_tau_L=log_tau_L, log_delta_tau_L=log_delta_tau_L, a_l=a_l)

    def __call__(self, batch_size):
        return self.sample_single(batch_size)

    def sample_single(self, batch_size, n_local_samples=1):
        global_params = np.zeros((batch_size, self.n_params_global))
        local_params = np.zeros((batch_size, n_local_samples, self.n_params_local))
        data = np.zeros((batch_size, n_local_samples, self.n_time_points))

        for i in range(batch_size):
            global_sample = self.sample_global()
            local_sample = self.sample_local(n_local_samples=n_local_samples)
            sim = self.simulator(local_sample)

            global_params[i] = [global_sample['log_tau_G'], global_sample['log_sigma_tau_G'],
                                global_sample['log_delta_tau_G'], global_sample['log_delta_sigma_tau_G'],
                                global_sample['a_mean'], global_sample['a_log_std']]

            local_params[i] = np.concatenate((local_sample['log_tau_L'],
                                              local_sample['log_delta_tau_L'],
                                              local_sample['a_l']), axis=-1)
            data[i] = sim['observable'].reshape(n_local_samples, self.n_time_points)

        #if n_local_samples == 1:
        #    local_params = local_params[:, 0]
        #    data = data[:, 0]
        return dict(global_params=global_params, local_params=local_params, data=data)

    def sample_full(self, batch_size):
        n_local_samples = self.simulator.img_size_full.shape[0] * self.simulator.img_size_full.shape[1]

        global_params = np.zeros((batch_size, self.n_params_global))
        local_params = np.zeros((batch_size, n_local_samples, self.n_params_local))
        data = np.zeros((batch_size, n_local_samples, self.n_time_points))

        for i in range(batch_size):
            global_sample = self.sample_global()
            local_sample = self.sample_local(n_local_samples=n_local_samples)
            sim = self.simulator.decay_gen_full(
                tau_L=local_sample['tau_L'],
                tau_L_2=local_sample['tau_L_2'],
                A_L=local_sample['A_L']
            )

            global_params[i] = [global_sample['log_tau_G'], global_sample['log_sigma_tau_G'],
                                global_sample['log_delta_tau_G'], global_sample['log_delta_sigma_tau_G'],
                                global_sample['a_mean'], global_sample['a_log_std']]
            local_params[i] = [local_sample['log_tau_L'], local_sample['log_delta_tau_L'], local_sample['a_l']]
            data[i] = sim['observable'].reshape(n_local_samples, self.n_time_points)
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

def generate_synthetic_data(prior, n_samples, n_local_samples=1, normalize=False,
                            as_grid=False,
                            random_seed=None):
    """Generate synthetic data for the hierarchical model.

    Parameters:
        prior (Prior): Prior distribution for the model.
        n_samples (int): Number of samples to generate.
        n_local_samples (int): Number of pixels in the grid for each sample.
        normalize (bool): Whether to normalize the data.
        as_grid (bool): Whether to return the data as a grid.
        random_seed (int): Random seed for reproducibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sample_dict = prior.sample_single(n_samples, n_local_samples=n_local_samples)
    param_global = torch.tensor(sample_dict['global_params'], dtype=torch.float32)
    param_local = torch.tensor(sample_dict['local_params'], dtype=torch.float32)
    data = torch.tensor(sample_dict['data'], dtype=torch.float32)
    if as_grid:
        # batch_size, n_time_steps, n_grid, n_grid
        grid_size = int(np.sqrt(n_local_samples))
        n_time_steps = data.shape[-1]
        data = data[:, :grid_size*grid_size]
        data = data.reshape(n_samples, n_time_steps, grid_size, grid_size)
    if normalize:
        param_global = prior.normalize_theta(param_global, global_params=True)
        param_local = prior.normalize_theta(param_local, global_params=False)
        data = prior.normalize_data(data)
    return param_global, param_local, data


class FLIProblem(Dataset):
    def __init__(self, n_data, prior, online_learning=False, max_number_of_obs=1):
        # Create model and dataset
        self.prior = prior
        self.n_data = n_data
        self.max_number_of_obs = max_number_of_obs
        self.n_obs = self.max_number_of_obs  # this can change for each batch
        self.online_learning = online_learning
        self.generate_data()

    def generate_data(self):
        # Create model and dataset
        self.thetas_global, self.thetas_local, self.xs = generate_synthetic_data(
            self.prior,
            n_local_samples=self.n_obs,
            n_samples=self.n_data, normalize=True
        )
        if self.max_number_of_obs == 1:
            # squeeze obs-dimension for RNN
            self.xs = self.xs.squeeze(1)
        # add feature dimension for RNN
        self.xs = self.xs.unsqueeze(-1)
        # generate noise
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

