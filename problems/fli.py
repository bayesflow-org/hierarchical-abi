import os

import matplotlib.image as mpimg
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from scipy.special import expit
from torch.utils.data import Dataset


class Simulator:
    def __init__(self, n_time_points, max_time=1):
        self.max_time = max_time
        self.dt = self.max_time / n_time_points
        self.n_time_points = n_time_points

        self.gw = 40  # gate width in ps
        self.img_size = (520, 696)  # full image size

        self.noise = np.load('system_noise.npy')
        self.pIRF = np.load('IRF.npy')

    def __call__(self, params):
        # Convert parameters to numpy arrays.
        tau_L, tau_L_2, A_L = np.array(params['tau_L', 'tau_L_2', 'A_L'])
        F_dec_conv = self.decay_gen(tau_L=tau_L, tau_L_2=tau_L_2, A_L=A_L)
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

    def decay_gen(self, tau_L, tau_L_2, A_L):
        img = np.random.randint(0, high=1000, size=(img_size[0], img_size[1]))

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
                irf_out[i, j, :] = self.norm1D(self.pIRF[np.random.randint(a1), np.random.randint(b1), :])
                dec_conv[i, j, :] = self.conv_dec(self.norm1D(np.squeeze(dec[i, j, :])), np.squeeze(irf_out[i, j, :]))
                dec_conv[i, j, :] = self.norm1D(np.squeeze(dec_conv[i, j, :])) * img[i, j]
                #dec_conv[i, j, :] += self.noise[i, j, :]
                dec_conv[i, j, :] += self.sample_augmented_noise()
        return dec_conv

    @staticmethod
    def tiff2tpsfs(p):
        k = os.listdir(p)
        le = len(k)
        img0 = mpimg.imread(os.path.join(p, k[0]))
        (a, b) = np.shape(img0)
        tpsfs = np.zeros([a, b, le - 1])
        for count, file in enumerate(os.listdir(p)):
            # Check whether file is in text format or not
            if file.endswith(".tif"):
                imt = mpimg.imread(os.path.join(p, file))
                tpsfs[:, :, count] = imt
        inten = np.sum(tpsfs, axis=2)
        return tpsfs, inten

    @staticmethod
    def norm1D(fn):
        if np.amax(fn) == 0:
            nfn = fn
        else:
            nfn = np.divide(fn, np.amax(fn))
        return nfn

    @staticmethod
    def conv_dec(dec, irf):
        c = np.shape(dec)
        #     conv = np.zeros((2 * c - 1,1))
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

        self.sample_global()
        self.n_params_global = 6
        self.n_params_local = 3

        np.random.seed(0)
        test_prior = self.sample_single(1000)
        self.simulator = Simulator(n_time_points=n_time_points)
        test = self.simulator(test_prior)
        self.x_mean = torch.tensor([np.mean(test['observable'])], dtype=torch.float32)
        self.x_std = torch.tensor([np.std(test['observable'])], dtype=torch.float32)
        self.prior_global_mean = torch.tensor(np.array([np.mean(test_prior['mu']), np.mean(test_prior['log_tau'])]),
                                              dtype=torch.float32)
        self.prior_global_std = torch.tensor(np.array([np.std(test_prior['mu']), np.std(test_prior['log_tau'])]),
                                             dtype=torch.float32)
        self.prior_local_mean = torch.tensor(np.array([np.mean(test_prior['theta'])]),
                                             dtype=torch.float32)
        self.prior_local_std = torch.tensor(np.array([np.std(test_prior['theta'])]),
                                            dtype=torch.float32)
        self.device = 'cpu'

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
        log_tau_L = np.random.normal(self.log_tau_G, np.exp(self.log_sigma_tau_G), size=n_local_samples)
        log_delta_tau_L = np.random.normal(self.log_delta_tau_G, np.exp(self.log_delta_sigma_tau_G), size=n_local_samples)
        a_l = np.random.normal(self.a_mean, np.exp(self.a_log_std), size=n_local_samples)

        tau_L = np.exp(log_tau_L)
        tau_L_2 = tau_L + np.exp(log_delta_tau_L)
        A_L = expit(a_l)
        return dict(tau_L=tau_L, tau_L_2=tau_L_2, A_L=A_L,
                    log_tau_L=log_tau_L, log_delta_tau_L=log_delta_tau_L, a_l=a_l)

    def __call__(self, batch_size):
        return self.sample_single(batch_size)

    def sample_single(self, batch_size):
        mu = np.random.normal(loc=self.mu_mean, scale=self.mu_std, size=(batch_size,1))
        log_tau = np.random.normal(loc=self.log_tau_mean, scale=self.log_tau_std, size=(batch_size,1))
        theta = np.random.normal(loc=mu, scale=np.exp(log_tau), size=(batch_size, 1))
        return dict(mu=mu, log_tau=log_tau, theta=theta)

    def sample_full(self, batch_size, n_grid):
        mu = np.random.normal(loc=self.mu_mean, scale=self.mu_std, size=(batch_size, 1))
        log_tau = np.random.normal(loc=self.log_tau_mean, scale=self.log_tau_std, size=(batch_size, 1))
        theta = np.random.normal(loc=mu[:, np.newaxis], scale=np.exp(log_tau)[:, np.newaxis],
                                 size=(batch_size, n_grid, n_grid))
        return dict(mu=mu, log_tau=log_tau, theta=theta)

    def score_global_batch(self, theta_batch_norm, condition_norm=None):
        """ Computes the global score for a batch of parameters."""
        theta_batch = self.denormalize_theta(theta_batch_norm, global_params=True)
        mu, log_tau = theta_batch[..., 0], theta_batch[..., 1]
        grad_logp_mu = -(mu - self.mu_mean) / (self.mu_std**2)
        grad_logp_tau = -(log_tau - self.log_tau_mean) / (self.log_tau_std**2)
        # correct the score for the normalization
        score = torch.stack([grad_logp_mu, grad_logp_tau], dim=-1)
        return score * self.prior_global_std

    def score_local_batch(self, theta_batch_norm, condition_norm):
        """ Computes the local score for a batch of samples. """
        theta = self.denormalize_theta(theta_batch_norm, global_params=False)
        condition = self.denormalize_theta(condition_norm, global_params=True)
        mu, log_tau = condition[..., 0], condition[..., 1]
        # Gradient w.r.t theta conditioned on mu and log_tau
        grad_logp_theta = -(theta - mu) / torch.exp(log_tau*2)
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

    param_global = torch.tensor(np.concatenate((batch_params['mu'], batch_params['log_tau']), axis=1),
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


def petcolormap(m=256):
    """
    Generates a PET colormap similar to the MATLAB version.

    Args:
        m (int): Number of colors in the colormap.

    Returns:
        ListedColormap: A matplotlib colormap.
    """
    # Base colormap array (each row is an [R, G, B] triplet).
    # This is taken directly from your MATLAB function.
    c = np.array([
        [0,	0,	0],
        [0,	0,	2],
        [0,	0,	4],
        [0,	0,	6],
        [0,	0,	8],
        [0,	0,	10],
        [0,	0,	12],
        [0,	0,	14],
        [0,	0,	16],
        [0,	0,	17],
        [0,	0,	19],
        [0,	0,	21],
        [0,	0,	23],
        [0,	0,	25],
        [0,	0,	27],
        [0,	0,	29],
        [0,	0,	31],
        [0,	0,	33],
        [0,	0,	35],
        [0,	0,	37],
        [0,	0,	39],
        [0,	0,	41],
        [0,	0,	43],
        [0,	0,	45],
        [0,	0,	47],
        [0,	0,	49],
        [0,	0,	51],
        [0,	0,	53],
        [0,	0,	55],
        [0,	0,	57],
        [0,	0,	59],
        [0,	0,	61],
        [0,	0,	63],
        [0,	0,	65],
        [0,	0,	67],
        [0,	0,	69],
        [0,	0,	71],
        [0,	0,	73],
        [0,	0,	75],
        [0,	0,	77],
        [0,	0,	79],
        [0,	0,	81],
        [0,	0,	83],
        [0,	0,	84],
        [0,	0,	86],
        [0,	0,	88],
        [0,	0,	90],
        [0,	0,	92],
        [0,	0,	94],
        [0,	0,	96],
        [0,	0,	98],
        [0,	0,	100],
        [0,	0,	102],
        [0,	0,	104],
        [0,	0,	106],
        [0,	0,	108],
        [0,	0,	110],
        [0,	0,	112],
        [0,	0,	114],
        [0,	0,	116],
        [0,	0,	117],
        [0,	0,	119],
        [0,	0,	121],
        [0,	0,	123],
        [0,	0,	125],
        [0,	0,	127],
        [0,	0,	129],
        [0,	0,	131],
        [0,	0,	133],
        [0,	0,	135],
        [0,	0,	137],
        [0,	0,	139],
        [0,	0,	141],
        [0,	0,	143],
        [0,	0,	145],
        [0,	0,	147],
        [0,	0,	149],
        [0,	0,	151],
        [0,	0,	153],
        [0,	0,	155],
        [0,	0,	157],
        [0,	0,	159],
        [0,	0,	161],
        [0,	0,	163],
        [0,	0,	165],
        [0,	0,	167],
        [3,	0,	169],
        [6,	0,	171],
        [9,	0,	173],
        [12,	0,	175],
        [15,	0,	177],
        [18,	0,	179],
        [21,	0,	181],
        [24,	0,	183],
        [26,	0,	184],
        [29,	0,	186],
        [32,	0,	188],
        [35,	0,	190],
        [38,	0,	192],
        [41,	0,	194],
        [44,	0,	196],
        [47,	0,	198],
        [50,	0,	200],
        [52,	0,	197],
        [55,	0,	194],
        [57,	0,	191],
        [59,	0,	188],
        [62,	0,	185],
        [64,	0,	182],
        [66,	0,	179],
        [69,	0,	176],
        [71,	0,	174],
        [74,	0,	171],
        [76,	0,	168],
        [78,	0,	165],
        [81,	0,	162],
        [83,	0,	159],
        [85,	0,	156],
        [88,	0,	153],
        [90,	0,	150],
        [93,	2,	144],
        [96,	4,	138],
        [99,	6,	132],
        [102,	8,	126],
        [105,	9,	121],
        [108,	11,	115],
        [111,	13,	109],
        [114,	15,	103],
        [116,	17,	97],
        [119,	19,	91],
        [122,	21,	85],
        [125,	23,	79],
        [128,	24,	74],
        [131,	26,	68],
        [134,	28,	62],
        [137,	30,	56],
        [140,	32,	50],
        [143,	34,	47],
        [146,	36,	44],
        [149,	38,	41],
        [152,	40,	38],
        [155,	41,	35],
        [158,	43,	32],
        [161,	45,	29],
        [164,	47,	26],
        [166,	49,	24],
        [169,	51,	21],
        [172,	53,	18],
        [175,	55,	15],
        [178,	56,	12],
        [181,	58,	9],
        [184,	60,	6],
        [187,	62,	3],
        [190,	64,	0],
        [194,	66,	0],
        [198,	68,	0],
        [201,	70,	0],
        [205,	72,	0],
        [209,	73,	0],
        [213,	75,	0],
        [217,	77,	0],
        [221,	79,	0],
        [224,	81,	0],
        [228,	83,	0],
        [232,	85,	0],
        [236,	87,	0],
        [240,	88,	0],
        [244,	90,	0],
        [247,	92,	0],
        [251,	94,	0],
        [255,	96,	0],
        [255,	98,	3],
        [255,	100,	6],
        [255,	102,	9],
        [255,	104,	12],
        [255,	105,	15],
        [255,	107,	18],
        [255,	109,	21],
        [255,	111,	24],
        [255,	113,	26],
        [255,	115,	29],
        [255,	117,	32],
        [255,	119,	35],
        [255,	120,	38],
        [255,	122,	41],
        [255,	124,	44],
        [255,	126,	47],
        [255,	128,	50],
        [255,	130,	53],
        [255,	132,	56],
        [255,	134,	59],
        [255,	136,	62],
        [255,	137,	65],
        [255,	139,	68],
        [255,	141,	71],
        [255,	143,	74],
        [255,	145,	76],
        [255,	147,	79],
        [255,	149,	82],
        [255,	151,	85],
        [255,	152,	88],
        [255,	154,	91],
        [255,	156,	94],
        [255,	158,	97],
        [255,	160,	100],
        [255,	162,	103],
        [255,	164,	106],
        [255,	166,	109],
        [255,	168,	112],
        [255,	169,	115],
        [255,	171,	118],
        [255,	173,	121],
        [255,	175,	124],
        [255,	177,	126],
        [255,	179,	129],
        [255,	181,	132],
        [255,	183,	135],
        [255,	184,	138],
        [255,	186,	141],
        [255,	188,	144],
        [255,	190,	147],
        [255,	192,	150],
        [255,	194,	153],
        [255,	196,	156],
        [255,	198,	159],
        [255,	200,	162],
        [255,	201,	165],
        [255,	203,	168],
        [255,	205,	171],
        [255,	207,	174],
        [255,	209,	176],
        [255,	211,	179],
        [255,	213,	182],
        [255,	215,	185],
        [255,	216,	188],
        [255,	218,	191],
        [255,	220,	194],
        [255,	222,	197],
        [255,	224,	200],
        [255,	226,	203],
        [255,	228,	206],
        [255,	229,	210],
        [255,	231,	213],
        [255,	233,	216],
        [255,	235,	219],
        [255,	237,	223],
        [255,	239,	226],
        [255,	240,	229],
        [255,	242,	232],
        [255,	244,	236],
        [255,	246,	239],
        [255,	248,	242],
        [255,	250,	245],
        [255,	251,	249],
        [255,	253,	252],
        [255,	255,	255]
    ])

    n = c.shape[0]
    # Create a mapping from the original colormap positions to the new m points.
    xp = np.linspace(0, m-1, n)
    x = np.arange(m)
    r = np.interp(x, xp, c[:, 0])
    g = np.interp(x, xp, c[:, 1])
    b = np.interp(x, xp, c[:, 2])
    colormap = np.stack([r, g, b], axis=1) / 255.0
    # Normalize so that the maximum value is 1 (this mimics the MATLAB division by max(map(:))).
    colormap = colormap / colormap.max()
    return ListedColormap(colormap)
