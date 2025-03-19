import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class Simulator:

    def __init__(self):
        self.scale = 0.1  # The scale of the Gaussian likelihood.

    def __call__(self, theta, n_obs=None):
        """Generates batched draws from a D-dimenional Gaussian distributions given a batch of
        location (mean) parameters of D dimensions. Assumes a spherical convariance matrix given
        by scale * I_D.

        Parameters
        ----------
        theta  : np.ndarray of shape (theta, D) or dict
            The location parameters of the Gaussian likelihood.
        n_obs  : int or None, optional, default: None
            The number of observations to draw from the likelihood given the location
            parameter `theta`. If `n obs is None`, a single draw is produced.

        Returns
        -------
        x : np.ndarray of shape (theta.shape[0], theta.shape[1]) if n_obs is None,
            else np.ndarray of shape (theta.shape[0], n_obs, theta.shape[1])
            A single draw or a sample from a batch of Gaussians.
        """
        # Generate prior predictive samples, possibly a single if n_obs is None
        if isinstance(theta, dict):
            theta = theta['theta']
        if n_obs is None:
            return dict(observable=np.random.normal(loc=theta, scale=self.scale))
        x = np.random.normal(loc=theta, scale=self.scale, size=(n_obs, theta.shape[0], theta.shape[1]))
        return dict(observable=np.transpose(x, (1, 0, 2)))


class Prior:
    def __init__(self):
        self.D = 10  # The dimensionality of the Gaussian prior distribution.
        self.scale = 0.1 * np.ones(self.D)  # The scale of the Gaussian prior.
        self.scale_tensor = torch.tensor(self.scale, dtype=torch.float32)

        self.n_params_global = self.D
        self.n_params_local = 0  # not a hierarchical model

        np.random.seed(0)
        test_prior = self.sample(1000)
        self.simulator = Simulator()
        test = self.simulator(test_prior)
        self.norm_x_mean = torch.tensor(np.mean(test['observable'], axis=0), dtype=torch.float32)
        self.norm_x_std = torch.tensor(np.std(test['observable'], axis=0), dtype=torch.float32)
        self.norm_prior_global_mean = torch.tensor(np.mean(test_prior['theta'], axis=0), dtype=torch.float32)
        self.norm_prior_global_std = torch.tensor(np.std(test_prior['theta'], axis=0), dtype=torch.float32)
        self.current_device = 'cpu'

    def __call__(self, batch_size):
        return self.sample(batch_size=batch_size)

    def sample(self, batch_size):
        """Generates a random draw from a D-dimensional Gaussian prior distribution with a
        spherical scale matrix given by sigma * I_D. Represents the location vector of
        a (conjugate) Gaussian likelihood.

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        theta : np.ndarray of shape (D, )
            A single draw from the D-dimensional Gaussian prior.
        """
        return dict(theta=np.random.normal(loc=np.zeros(self.D), scale=self.scale, size=(batch_size, self.D)))


    def score_batch(self, theta_batch_norm):
        """ Computes the global score for a batch of parameters."""
        theta_batch = self.denormalize_theta(theta_batch_norm, global_params=True)
        score = -theta_batch / torch.square(self.scale_tensor)
        # correct the score for the normalization
        return score * self.norm_prior_global_std

    def score_global_batch(self, theta_batch_norm, condition_norm=None):
        return self.score_batch(theta_batch_norm)

    def normalize_theta(self, theta, global_params=True):
        self._move_to_device(theta.device)
        if global_params:
            return (theta - self.norm_prior_global_mean) / self.norm_prior_global_std
        raise ValueError('This is not a hierarchical model.')

    def denormalize_theta(self, theta, global_params=True):
        self._move_to_device(theta.device)
        if global_params:
            return theta * self.norm_prior_global_std + self.norm_prior_global_mean
        raise ValueError('This is not a hierarchical model.')

    def normalize_data(self, x):
        self._move_to_device(x.device)
        return (x - self.norm_x_mean) / self.norm_x_std

    def _move_to_device(self, device):
        if self.current_device != device:
            print(f"Moving prior to device: {device}")
            self.norm_prior_global_mean = self.norm_prior_global_mean.to(device)
            self.norm_prior_global_std = self.norm_prior_global_std.to(device)
            self.norm_x_mean = self.norm_x_mean.to(device)
            self.norm_x_std = self.norm_x_std.to(device)
            self.current_device = device
        return


def sample_posterior(x, prior_sigma, sigma, n_samples=1):
    """
    Returns samples from the posterior distribution given observations,
    where each dimension can have its own likelihood standard deviation.

    Parameters
    ----------
    x : np.ndarray
        Observations with shape (n_obs, D). Each row is a D-dimensional observation.
    prior_sigma : np.ndarray
        1D array of length D containing the standard deviations of the Gaussian prior
        for each dimension (prior: theta_d ~ N(0, prior_sigma[d]**2)).
    sigma : np.ndarray
        1D array of length D containing the standard deviations of the likelihood
        for each dimension (likelihood: x_d | theta_d ~ N(theta_d, sigma[d]**2)).
    n_samples : int, optional
        Number of posterior samples to draw (default is 1).

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, D) containing samples from the posterior.
    """
    # Ensure x is a 2D array.
    x = np.atleast_2d(x)
    n_obs, D = x.shape

    # Compute variances for prior and likelihood.
    prior_var = prior_sigma ** 2
    likelihood_var = sigma ** 2

    # Compute the posterior variance for each dimension.
    post_var = 1 / (1 / prior_var + n_obs / likelihood_var)

    # Compute the posterior mean for each dimension.
    post_mean = post_var * (np.sum(x, axis=0) / likelihood_var)

    # Draw samples from the posterior (each dimension is independent).
    samples = np.random.normal(loc=post_mean, scale=np.sqrt(post_var), size=(n_samples, D))
    return samples


def analytical_posterior_median(x, prior_std, likelihood_std):
    """
    Computes the analytical median of the posterior for a Gaussian prior and likelihood.

    The model is:
        theta ~ N(prior_mean, prior_std^2)
        x | theta ~ N(theta, likelihood_std^2)

    For n independent observations x, the posterior is:
        sigma_post^2 = 1 / (1/prior_std^2 + n/likelihood_std^2)
        mu_post = sigma_post^2 * (prior_mean/prior_std^2 + (sum x_i)/likelihood_std^2)

    Since the posterior is Gaussian, its median equals its mean.

    Parameters
    ----------
    x : np.ndarray
        Observations. Should have shape (n,) for scalar or (n, D) for D-dimensional parameters.
    prior_std : float or np.ndarray
        The prior standard deviation. For a D-dimensional case, shape should be (D,).
    likelihood_std : float or np.ndarray
        The likelihood standard deviation. For a D-dimensional case, shape should be (D,).

    Returns
    -------
    median : float or np.ndarray
        The analytical median of the posterior, which is the posterior mean.
    """
    # Ensure x is 2D: shape (n_obs, D)
    x = np.atleast_2d(x)
    n_obs = x.shape[0]

    # Convert parameters to variances
    prior_var = np.square(prior_std)
    likelihood_var = np.square(likelihood_std)

    # Compute the posterior variance
    post_var = 1.0 / (1.0 / prior_var + n_obs / likelihood_var)

    # Compute the posterior mean (the median, since Gaussian)
    post_mean = post_var * (0 / prior_var + np.sum(x, axis=0) / likelihood_var)
    return post_mean


def analytical_posterior_mean_std(x, prior_std, likelihood_std):
    """
    Computes the analytical mean and std of the posterior for a Gaussian prior and likelihood.

    The model is:
        theta ~ N(prior_mean, prior_std^2)
        x | theta ~ N(theta, likelihood_std^2)

    For n independent observations x, the posterior is:
        sigma_post^2 = 1 / (1/prior_std^2 + n/likelihood_std^2)
        mu_post = sigma_post^2 * (prior_mean/prior_std^2 + (sum x_i)/likelihood_std^2)

    Since the posterior is Gaussian, its median equals its mean.

    Parameters
    ----------
    x : np.ndarray
        Observations. Should have shape (n,) for scalar or (n, D) for D-dimensional parameters.
    prior_std : float or np.ndarray
        The prior standard deviation. For a D-dimensional case, shape should be (D,).
    likelihood_std : float or np.ndarray
        The likelihood standard deviation. For a D-dimensional case, shape should be (D,).

    Returns
    -------
    median : float or np.ndarray
        The analytical median of the posterior, which is the posterior mean.
    """
    # Ensure x is 2D: shape (n_obs, D)
    x = np.atleast_2d(x)
    n_obs = x.shape[0]

    # Convert parameters to variances
    prior_var = np.square(prior_std)
    likelihood_var = np.square(likelihood_std)

    # Compute the posterior variance
    post_var = 1.0 / (1.0 / prior_var + n_obs / likelihood_var)

    # Compute the posterior mean (the median, since Gaussian)
    post_mean = post_var * (0 / prior_var + np.sum(x, axis=0) / likelihood_var)
    return post_mean, np.sqrt(post_var)


def posterior_contraction(prior_std, likelihood_std, n_obs):
    """
    Computes the posterior standard deviation and contraction factor given a Gaussian prior and likelihood.

    Parameters
    ----------
    prior_std : float or np.ndarray
        The standard deviation(s) of the prior distribution.
    likelihood_std : float or np.ndarray
        The standard deviation(s) of the likelihood.
    n_obs : int
        The number of independent observations.

    Returns
    -------
    contraction : float or np.ndarray
        The contraction factor, i.e. the ratio posterior_std / prior_std.
    """
    # Compute posterior variance: sigma_post^2 = 1 / (1/prior_std^2 + n_obs/likelihood_std^2)
    # which can be rearranged to: sigma_post^2 = (prior_std^2 * likelihood_std^2) / (likelihood_std^2 + n_obs * prior_std^2)
    prior_var = np.square(prior_std)
    likelihood_var = np.square(likelihood_std)

    posterior_var = (prior_var * likelihood_var) / (likelihood_var + n_obs * prior_var)

    # Contraction factor is posterior_std divided by prior_std.
    contraction =  1 - posterior_var / prior_var
    return contraction


def generate_synthetic_data(prior, n_samples, data_size=None, normalize=False, random_seed=None):
    """Generate synthetic data for the flat model.

    Parameters:
        prior (Prior): Prior distribution for the model.
        n_samples (int): Number of samples to generate.
        data_size (int): Number of observations per sample.
        normalize (bool): Whether to normalize the data.
        random_seed (int): Random seed for reproducibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    batch_params = prior.sample(n_samples)
    simulator = Simulator()
    if data_size is not None:
        sim_batch = simulator(batch_params, n_obs=data_size)
    else:
        sim_batch = simulator(batch_params)

    param_global = torch.tensor(batch_params['theta'], dtype=torch.float32)
    data = torch.tensor(sim_batch['observable'], dtype=torch.float32)
    if normalize:
        param_global = prior.normalize_theta(param_global, global_params=True)
        data = prior.normalize_data(data)
    return param_global, data


def visualize_simulation_output(sim_output):
    if sim_output.ndim == 1:
        data = sim_output
        # Calculate the mean and standard deviation for each dimension.
        mean_vals = data
        std_vals = None
    elif sim_output.ndim == 2:
        data = sim_output
        # Calculate the mean and standard deviation for each dimension.
        mean_vals = data.mean(axis=0)
        std_vals = data.std(axis=0)
    else:
        data = sim_output[0]
        # Calculate the mean and standard deviation for each dimension.
        mean_vals = data.mean(axis=0)
        std_vals = data.std(axis=0)

    # Create an array for the dimension indices.
    dimensions = np.arange(1, data.shape[1] + 1)

    # Plot the mean values with error bars representing the standard deviation.
    plt.figure(figsize=(6, 4))
    plt.errorbar(dimensions, mean_vals, yerr=std_vals, fmt='o-', capsize=5, label='Mean ± STD')
    plt.xlabel("Dimension")
    plt.ylabel("Mean Value")
    plt.title("Mean and Standard Deviation with Error Bars")
    plt.legend()
    plt.show()
    return


class GaussianProblem(Dataset):
    def __init__(self, n_data, prior, online_learning=False, max_number_of_obs=1):
        # Create model and dataset
        self.prior = prior
        self.n_data = n_data
        self.max_number_of_obs = max_number_of_obs
        self.n_obs = max_number_of_obs
        self.online_learning = online_learning
        self._generate_data()

    def _generate_data(self):
        # Create model and dataset
        self.thetas_global, self.xs = generate_synthetic_data(
            self.prior,
            data_size=self.max_number_of_obs if self.max_number_of_obs > 1 else None,
            n_samples=self.n_data, normalize=True
        )
        self.epsilon_global = torch.randn_like(self.thetas_global, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.thetas_global)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.thetas_global[idx]
        if self.max_number_of_obs > 1:
            target = self.xs[idx, :self.n_obs]
        else:
            target = self.xs[idx]
        noise = self.epsilon_global[idx]
        return features, noise, target

    def on_epoch_end(self):  # for online learning
        # Regenerate data at the end of each epoch
        if self.online_learning:
            self._generate_data()

    def on_batch_end(self):
        # Regenerate data at the end of each epoch
        if self.max_number_of_obs > 1:
            # sample number of observations
            self.n_obs = np.random.choice([1, 5, 10, 20, 50, 100])
