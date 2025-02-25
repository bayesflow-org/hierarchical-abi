import matplotlib.pyplot as plt
import numpy as np
import torch


class Simulator:
    def __init__(self, n_grid=8):
        self.n_grid = n_grid
        self.max_time = 100
        self.n_time_points = 10  # number of observation points to return
        self.dt = 0.1            # simulation time step
        self.n_time_steps = int(self.max_time / self.dt)  # number of simulation steps

    def __call__(self, params):
        """
        Simulate Brownian motion with drift.

        The SDE is:
            dx(t) = theta * dt + 1 * sqrt(dt) * dW(t)
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
            if theta.shape[1] != self.n_grid or theta.shape[2] != self.n_grid:
                raise ValueError("For 3D 'theta', the second and third dimensions must equal n_grid.")
            grid_shape = (self.n_grid, self.n_grid)
        else:
            raise ValueError("Parameter 'theta' must be provided as a scalar, 2D array, or 3D array.")
        batch_size = theta.shape[0]

        # Simulate the full trajectory.
        # The noise will have shape: (batch_size, n_time_steps, *grid_shape)
        noise_shape = (batch_size, self.n_time_steps) + grid_shape
        noise = np.random.normal(loc=0, scale=1, size=noise_shape)

        # Expand mu and tau to include a time axis.
        if theta.ndim in (1, 2):
            # mu and tau have shape (batch_size, grid) in the 2D case
            # For a scalar, we already set them to shape (1,)
            # Expand to (batch_size, 1, grid)
            if batch_size == 1:
                # Ensure shape is (1, 1, grid)
                theta_expanded = theta[np.newaxis, np.newaxis, :]
            else:
                theta_expanded = theta[:, np.newaxis, np.newaxis, :]
        else:
            # For 3D parameters, mu and tau have shape (batch_size, n_grid, n_grid)
            # Expand to (batch_size, 1, n_grid, n_grid)
            theta_expanded = theta[:, np.newaxis, :, :]

        # Compute increments:
        #   increment = theta * dt + sqrt(dt) * noise
        increments = theta_expanded * self.dt + 1 * np.sqrt(self.dt) * noise

        # Initial condition: zeros with shape (batch_size, 1, *grid_shape)
        x0 = np.zeros((batch_size, 1) + grid_shape)
        # Full trajectory: shape (batch_size, n_time_steps+1, *grid_shape)
        traj_full = np.concatenate([x0, np.cumsum(increments, axis=1)], axis=1)

        # Sample self.n_time_points evenly spaced indices from the full trajectory.
        # These indices span from n_time_points to max_time (0 would be the initial condition).
        indices = np.linspace(self.n_time_points, self.max_time, self.n_time_points, dtype=int)
        traj_sampled = traj_full[:, indices, ...]  # shape: (batch_size, n_time_points, *grid_shape)

        if theta.ndim == 2:  # just one grid element
            traj_sampled = traj_sampled.reshape(batch_size, self.n_time_points, 1)
        return dict(observable=traj_sampled)

class Prior:
    def __init__(self):
        self.mu_mean = 0
        self.mu_std = 3
        self.log_tau_mean = 0
        self.log_tau_std = 1
        self.n_params_global = 2
        self.n_params_local = 1

        np.random.seed(0)
        test_prior = self.sample_single(1000)
        self.simulator = Simulator()
        test = self.simulator(test_prior,)
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

    def __call__(self, batch_size):
        return self.sample_single(batch_size)

    def sample_single(self, batch_size):
        mu = np.random.normal(loc=self.mu_mean, scale=self.mu_std, size=(batch_size,1))
        log_tau = np.random.normal(loc=self.log_tau_mean, scale=self.log_tau_std, size=(batch_size,1))
        theta = np.random.normal(loc=mu, scale=np.exp(log_tau), size=(batch_size, 1))
        return dict(mu=mu, log_tau=log_tau, theta=theta)

    def sample_full(self, batch_size):
        mu = np.random.normal(loc=self.mu_mean, scale=self.mu_std, size=(batch_size, 1))
        log_tau = np.random.normal(loc=self.log_tau_mean, scale=self.log_tau_std, size=(batch_size, 1))
        theta = np.random.normal(loc=mu[:, np.newaxis], scale=np.exp(log_tau)[:, np.newaxis],
                                 size=(batch_size, self.simulator.n_grid, self.simulator.n_grid))
        return dict(mu=mu, log_tau=log_tau, theta=theta)

    def score_global_batch(self, theta_batch_norm, condition_norm=None):
        """ Computes the global score for a batch of parameters."""
        theta_batch = self.denormalize_theta(theta_batch_norm, global_params=True)
        mu, log_tau = theta_batch[..., 0], theta_batch[..., 1]
        grad_logp_mu = -(mu - self.mu_mean) / (self.mu_std**2)
        grad_logp_tau = -(log_tau - self.log_tau_mean) / (self.log_tau_std**2)
        # correct the score for the normalization
        score = torch.stack([grad_logp_mu, grad_logp_tau], dim=-1)
        return score / self.prior_global_std

    def score_local_batch(self, theta_batch_norm, condition_norm):
        """ Computes the local score for a batch of samples. """
        theta = self.denormalize_theta(theta_batch_norm, global_params=False)
        condition = self.denormalize_theta(condition_norm, global_params=True)
        mu, log_tau = condition[..., 0], condition[..., 1]
        # Gradient w.r.t theta conditioned on mu and log_tau
        grad_logp_theta = -(theta - mu) / torch.exp(log_tau*2)
        # correct the score for the normalization
        score = grad_logp_theta / self.prior_local_std
        return score

    def normalize_theta(self, theta, global_params):
        if self.device != theta.device:
            self.prior_global_mean = self.prior_global_mean.to(theta.device)
            self.prior_global_std = self.prior_global_std.to(theta.device)
            self.prior_local_mean = self.prior_local_mean.to(theta.device)
            self.prior_local_std = self.prior_local_std.to(theta.device)
            self.device = theta.device
            print(f"Moving prior to device {theta.device}")
        if global_params:
            return (theta - self.prior_global_mean) / self.prior_global_std
        return (theta - self.prior_local_mean) / self.prior_local_std

    def denormalize_theta(self, theta, global_params):
        if self.device != theta.device:
            self.prior_global_mean = self.prior_global_mean.to(theta.device)
            self.prior_global_std = self.prior_global_std.to(theta.device)
            self.prior_local_mean = self.prior_local_mean.to(theta.device)
            self.prior_local_std = self.prior_local_std.to(theta.device)
            self.device = theta.device
            print(f"Moving prior to device {theta.device}")
        if global_params:
            return theta * self.prior_global_std + self.prior_global_mean
        return theta * self.prior_local_std + self.prior_local_mean

    def normalize_data(self, x):
        return (x - self.x_mean) / self.x_std


def generate_synthetic_data(prior, n_data, grid_size=8, full_grid=False, normalize=False, random_seed=None):
    """Generate synthetic data for the hierarchical model.

    Parameters:
        prior (Prior): Prior distribution for the model.
        n_data (int): Number of samples to generate.
        grid_size (int): Size of the grid for the simulator.
        full_grid (bool): Whether to sample the full grid or a single element.
        normalize (bool): Whether to normalize the data.
        random_seed (int): Random seed for reproducibility.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if full_grid:
        batch_params = prior.sample_full(n_data)
    else:
        batch_params = prior.sample_single(n_data)
    simulator = Simulator(n_grid=grid_size)
    sim_batch = simulator(batch_params)

    param_global = torch.tensor(np.concatenate((batch_params['mu'], batch_params['log_tau']), axis=1),
                                dtype=torch.float32)
    param_local = torch.tensor(batch_params['theta'], dtype=torch.float32)
    data = torch.tensor(sim_batch['observable'], dtype=torch.float32)
    if normalize:
        param_global = prior.normalize_theta(param_global, global_params=True)
        param_local = prior.normalize_theta(param_local, global_params=False)
        data = prior.normalize_data(data)
    return param_global, param_local, data


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
