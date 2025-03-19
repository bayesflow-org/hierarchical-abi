import numpy as np
import torch
from scipy import stats
from scipy.integrate import solve_ivp
from tqdm import tqdm

from diffusion_model.helper_functions import generate_diffusion_time

sampling_defaults = {
    'damping_factor': lambda t: torch.ones_like(t),
    'sample_same_obs': None,
    #'type': 'mean',  # mean, median, pareto, huber_mean, small_mean
    #'tail_fraction': 0.2,
    #'huber_delta': 1.,
    'size': np.inf,  # for mini-batch
    'damping_factor_prior': lambda t: torch.ones_like(t),
    #'small_size': 10
}


def prepare_observations(x_obs, model, n_post_samples, device):
    """
    Prepare and normalize observations.

    Converts x_obs (batch_size, n_time_steps, n_grid, n_grid) or (batch_size, n_obs, n_features) to a
     tensor if necessary, normalizes via model.prior, reshapes to
     (batch_size*n_post_samples*n_obs, n_time_steps, n_features) and n_time_steps might be 1.
    If scores is conditioned on multiple observations, the data is returned as
    (batch_size*n_post_samples * reduced_data_obs, current_number_of_obs, n_time_steps, n_features)

    Returns:
        x_obs_norm: Tensor of shape (n_post_samples*n_obs, n_time_steps, n_features)
        n_obs: Number of observations
    """
    if not isinstance(x_obs, torch.Tensor):
        x_obs = torch.tensor(x_obs, dtype=torch.float32, device=device)
    else:
        x_obs = x_obs.to(device)
    x_obs_norm = model.prior.normalize_data(x_obs)

    if x_obs_norm.ndim == 4:
        # input is (batch_size, n_time_steps, grid, grid)
        batch_size = x_obs_norm.shape[0]
        n_time_steps = x_obs_norm.shape[1]
        # Reshape to (batch_size, n_time_steps, n_obs)
        x_obs_norm = x_obs_norm.contiguous().view(batch_size, n_time_steps, -1, 1)
        # Transpose to (batch_size, n_obs, n_time_steps, 1)
        x_obs_norm = x_obs_norm.permute(0, 2, 1, 3)
        n_obs = x_obs_norm.shape[1]
    elif x_obs_norm.ndim == 3:  # data is not time series
        # input is (batch_size, n_obs, n_dim)
        batch_size = x_obs_norm.shape[0]
        n_obs = x_obs_norm.shape[1]
        n_time_steps = 1
        # Reshape to (batch_size, n_obs, 1, n_dim)
        x_obs_norm = x_obs_norm.unsqueeze(2)
    else:
        raise ValueError('x_obs_norm must be 2 or 3 dimensional')

    ##########################

    # Reshape observations to (batch_size*n_post_samples*n_obs, n_time_steps, n_features)
    if model.max_number_of_obs == 1:
        # the score is always conditioned on only one observation
        # expand to number of posterior samples
        x_exp = x_obs_norm.unsqueeze(1).expand(batch_size, n_post_samples, n_obs, n_time_steps, -1)
        x_expanded = x_exp.contiguous().view(batch_size * n_post_samples * n_obs, n_time_steps, -1)
    else:
        # the score is conditioned on multiple observations
        # factorize data into (batch_size*n_post_samples * reduced_data, current_number_of_obs, n_time_steps, n_features)
        n_obs_reduced = n_obs // model.current_number_of_obs
        if n_obs % model.current_number_of_obs != 0:
            print('warning: number of observations is not a multiple of current_number_of_obs '
                  f'dropping last {n_obs % model.current_number_of_obs} observations.')
            n_obs = n_obs_reduced * model.current_number_of_obs
            x_obs_norm = x_obs_norm[:, :n_obs]
        x_exp = x_obs_norm.contiguous().view(batch_size, n_obs_reduced, model.current_number_of_obs, n_time_steps, -1)
        # expand to number of posterior samples
        x_expanded = x_exp.unsqueeze(1).expand(batch_size, n_post_samples, n_obs_reduced, model.current_number_of_obs,
                                               n_time_steps, -1)
        x_expanded = x_expanded.contiguous().view(batch_size * n_post_samples * n_obs_reduced,
                                        model.current_number_of_obs, n_time_steps, -1)

    return x_expanded, batch_size, n_obs


def initialize_sampling(model, x_obs, n_post_samples, conditions, mini_batch_arg, random_seed, device):
    """
    Initialize common parameters for sampling methods.

    Args:
        model (ScoreModel): Score-based model with prior and SDE attributes
        x_obs (tensor or array): Observations to be normalized and processed
        n_post_samples (int): Number of posterior samples to generate
        conditions (tensor or None): Conditioning parameters for local sampling
        mini_batch_arg (dict or None): Dict with arguments for the mini-batch algorithm
        random_seed (int or None): Random seed for reproducibility
        device: Computation device for tensor operations

    Returns:
        tuple: A tuple containing:
            - n_obs (int): Number of observations
            - n_scores_update (int): Number of scores to update
            - theta_init (tensor): Initial parameter samples
            - conditions_exp (tensor or None): Expanded conditions
            - x_expanded (tensor): Expanded observations
            - mini_batch_dict (dict): Mini-batch parameters
            - subsample (bool): Whether to subsample or not
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Preprocess observations
    x_expanded, batch_size, n_obs = prepare_observations(x_obs=x_obs, model=model, n_post_samples=n_post_samples, device=device)

    if n_obs % model.current_number_of_obs != 0:
        raise ValueError("'n_obs' must be a multiple of 'model.current_number_of_obs'")

    # Preprocess conditions
    if not conditions is None:
        global_sampling = False

        if not isinstance(conditions, torch.Tensor):
            conditions = torch.tensor(conditions, dtype=torch.float32, device=device)
        else:
            conditions = conditions.to(device)
        conditions_norm = model.prior.normalize_theta(conditions, global_params=True)

        # if condition for each posterior sample provided use it, otherwise expand
        if conditions_norm.ndim == 3:  # (batch_size, n_post_samples, model.prior.n_params_global)
            # expand conditions for every observation
            conditions_exp = conditions_norm.unsqueeze(2).expand(batch_size, n_post_samples, n_obs, -1)
        elif conditions_norm.ndim == 2:  # (batch_size, model.prior.n_params_global)
            # expand conditions for every observation and each posterior sample
            conditions_exp = conditions_norm.unsqueeze(1).expand(batch_size, n_post_samples, n_obs, -1)
        else:
            raise ValueError('Conditions must be 2 or 3 dimensional')
        # collapse conditions to shape (batch_size*n_post_samples*n_obs, n_params_local)
        conditions_collapsed = conditions_exp.contiguous().view(-1, model.prior.n_params_global)
    else:
        global_sampling = True  # global sampling or flat model without extra conditions
        conditions_collapsed = None

    ##########################

    # Number of observations to use for the score update
    if global_sampling:
        n_scores_update = n_obs // model.current_number_of_obs
    else:
        n_scores_update = n_obs

    ###### Prepare Mini-Batching
    mini_batch_dict = sampling_defaults.copy()
    if mini_batch_arg is not None:
        mini_batch_dict.update(mini_batch_arg)
    mini_batch_dict['size'] = min(n_scores_update, mini_batch_dict['size'])
    if mini_batch_dict['size'] < n_scores_update:
        subsample = True
    else:
        subsample = False

    # sample from latent prior for diffusion model
    if global_sampling:
        theta_init = torch.randn((batch_size * n_post_samples, model.prior.n_params_global), dtype=torch.float32,
                                 device=device)
        theta_init = theta_init / np.sqrt(n_obs)

        theta_init = theta_init / torch.sqrt(
            mini_batch_dict['damping_factor_prior'](t=torch.tensor(1, dtype=torch.float32,
                                                                   device=device)))
    else:
        theta_init = torch.randn((batch_size * n_post_samples, n_obs, model.prior.n_params_local), dtype=torch.float32,
                                 device=device)

    return batch_size, n_obs, n_scores_update, theta_init, conditions_collapsed, x_expanded, mini_batch_dict, subsample


def sub_sample_observations(x, batch_size_full, n_scores_update, mini_batch_dict):
    """
    Subsample observations for the score update by computing correct indices directly.

      - x shape: (batch_shape * n_scores_update, ...)
      - For each posterior sample, randomly subsample mini_batch_dict['size'] observations.
      - Final output shape: (batch_shape * mini_batch_dict['size'], ...)

    Parameters:
        x (Tensor): Normalized and expanded observations with shape (n_obs*posterior_samples, n_obs_time_steps, d) or
            (n_obs*posterior_samples, current_number_of_obs, n_obs_time_steps, d)
        batch_size_full (int): Batch size * Number of posterior samples.
        n_scores_update (int): Total number of observations (or groups, if factorized) per posterior sample.
        mini_batch_dict (dict): Dictionary with keys 'size' and 'sample_same_obs'.

    Returns:
        Tensor: Sub-sampled observations (n_post_samples * mini_batch_size, n_obs_time_steps, d)
    """
    if mini_batch_dict['sample_same_obs'] is not None:
        # sample the same observations in each batch
        torch.manual_seed(mini_batch_dict['sample_same_obs'])

    # x has shape: (batch_size * n_post_samples * n_scores_update, ...)
    # For each posterior sample i, the observations occupy a contiguous block of size n_scores_update.
    # Generate random indices in [0, n_scores_update) for each sample.
    rand_idx = torch.argsort(torch.rand(batch_size_full, n_scores_update, dtype=x.dtype, device=x.device),
                             dim=1)[:, :mini_batch_dict['size']]

    # Compute the offset for each posterior sample.
    sample_offset = torch.arange(batch_size_full, device=x.device) * n_scores_update  # shape: (batch_size * n_post_samples,)
    sample_offset = sample_offset.unsqueeze(1).expand(-1, mini_batch_dict['size'])  # shape: (batch_size * n_post_samples, mini_batch_dict['size'])

    # Compute final indices into T.
    final_idx = (sample_offset + rand_idx).reshape(-1)  # shape: (batch_size * n_post_samples * mini_batch_dict['size'],)

    # Index into T along the first dimension.
    x_sub = x[final_idx]  # shape: (batch_size * n_post_samples * mini_batch_dict['size'], ....)
    return x_sub


def pareto_smooth_weights(weights, tail_fraction):
    """
    Smooth a 1D array of raw importance weights using a Pareto tail fit.

    Parameters:
        weights (np.ndarray): 1D array of raw importance weights.
        tail_fraction (float): Fraction of weights considered as the tail (e.g. 0.2 for top 20%).

    Returns:
        np.ndarray: The smoothed importance weights.
    """
    # Determine threshold: weights above this quantile are in the tail.
    threshold = np.quantile(weights, 1 - tail_fraction)
    tail_mask = weights > threshold

    # If no weights exceed the threshold, nothing to smooth.
    if np.sum(tail_mask) == 0:
        #print('pareto smoothing: no weights exceed the threshold')
        return np.ones_like(weights)

    # Fit a generalized Pareto distribution (GPD) to the excesses (weight - threshold)
    tail_excess = weights[tail_mask] - threshold
    # Force location = 0 for the excesses. Returns shape c and scale.
    c, loc, scale = stats.genpareto.fit(tail_excess, floc=0)

    # A simple smoothing: cap the extreme weights by a computed maximum value.
    # For a GPD with c < 1, the expected maximum is threshold + scale / (1 - c).
    # (If c >= 1, we fall back to no smoothing.)
    if c < 1:
        max_allowed = threshold + scale / (1 - c)
    else:
        max_allowed = np.max(weights)

    # Replace tail weights with the minimum of their original value and the cap.
    smoothed_weights = np.copy(weights)
    smoothed_weights[tail_mask] = np.minimum(weights[tail_mask], max_allowed)
    return smoothed_weights


def pareto_smooth_sum(scores, tail_fraction):
    """
    Compute a Pareto-smoothed weighted sum of scores over batches.

    Each row in scores (and corresponding raw_weights) is processed independently.
    The raw_weights for each batch are smoothed using a Pareto tail fit,
    then normalized to sum to one before computing the weighted sum.

    Parameters:
        scores (torch.tensor): Array of scores of shape (B, N) where B is the batch size.
        tail_fraction (float): Fraction of the highest weights in each batch to smooth.

    Returns:
        np.ndarray: A 1D array of weighted sums (one per batch).
    """
    B, N, D = scores.shape
    smoothed_sums = torch.zeros((scores.shape[0], scores.shape[2]), dtype=scores.dtype, device=scores.device)

    for b in range(B):
        # Extract the b-th batch
        s = scores[b]  # shape (N, D)
        magnitudes = torch.norm(s, p=2, dim=1).cpu().numpy()

        # Apply Pareto smoothing to the raw weights in this batch.
        smoothed_magnitudes = pareto_smooth_weights(magnitudes, tail_fraction=tail_fraction)
        weights = torch.tensor(smoothed_magnitudes, dtype=scores.dtype, device=scores.device)

        # Compute normalized directions for all vectors
        norms = torch.norm(s, p=2, dim=1, keepdim=True) + 1e-8
        directions = s / norms

        # Apply smoothed magnitudes to original directions
        scaled_vectors = weights.unsqueeze(1) * directions

        # Sum the smoothed vectors
        smoothed_sums[b] = torch.sum(scaled_vectors, dim=0)
    return smoothed_sums


def small_mean(x, size, dim):
    sorted_x, _ = torch.sort(x, dim=dim)
    trimmed_x = sorted_x[:, :size]  # Remove extremes
    return trimmed_x.mean(dim=dim)


def huber_mean(x, delta, dim):
    abs_diff = torch.abs(x - x.mean(dim=dim, keepdim=True))
    mask = abs_diff <= delta
    squared_loss = 0.5 * (x ** 2)
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = torch.where(mask, squared_loss, linear_loss)
    return loss.mean(dim=dim)


def eval_compositional_score(model, theta, diffusion_time, x_exp, conditions_exp, batch_size_full,
                             n_scores_update_full, mini_batch_dict):
    """
    Compute the (global or local) compositional score.

    For global scores:
        - theta is expanded to (n_post_samples, n_obs, model.prior.n_params_global)
        - The score is computed per observation and then summed.
        - The prior score is also incorporated.

    For local scores:
        - theta is reshaped to (n_post_samples*n_obs, model.prior.n_params_local)
        - The computed score reshaped back to (n_post_samples, n_scores_update, -1).

    Returns:
        model_scores: Computed score tensor.
    """
    # Expand diffusion_time to shape (n_post_samples*n_scores_update, 1)
    t_exp = diffusion_time.unsqueeze(1).expand(-1, mini_batch_dict['size'], -1).contiguous().view(-1, 1)

    if conditions_exp is None:
        theta_exp = theta.unsqueeze(1).expand(-1, mini_batch_dict['size'], -1).contiguous().view(-1, model.prior.n_params_global)
        model_indv_scores = model.forward_global(
            theta_global=theta_exp,
            time=t_exp,
            x=x_exp,
            pred_score=True,
            clip_x=True
        )
        # Reshape to (batch_size_full, n_obs, -1) and sum over observations
        model_sum_scores_indv = model_indv_scores.contiguous().view(batch_size_full, mini_batch_dict['size'], -1)

        # add prior to the individual score, this is more stable than adding it to the sum
        prior_scores = (1 - diffusion_time) * model.prior.score_global_batch(theta)
        # expand prior scores to match the individual scores
        prior_scores_indv = prior_scores.unsqueeze(1)
        # (1 - n_scores_update) * (1 - diffusion_time) * model.prior.score_global_batch(theta)
        model_sum_scores_indv = model_sum_scores_indv - prior_scores_indv

        scores_mean = torch.mean(model_sum_scores_indv, dim=1)
        damping_factor = mini_batch_dict['damping_factor'](diffusion_time)
        model_sum_scores = damping_factor * n_scores_update_full * scores_mean

        # (1 - n_scores_update) * (1 - diffusion_time) * model.prior.score_global_batch(theta)
        # just the plus 1 is missing
        damping_factor_prior = mini_batch_dict['damping_factor_prior'](diffusion_time)
        model_scores = damping_factor_prior * prior_scores + model_sum_scores
    else:
        theta_exp = theta.contiguous().view(-1, model.prior.n_params_local)

        model_scores = model.forward_local(
            theta_local=theta_exp,
            time=t_exp,
            x=x_exp,
            theta_global=conditions_exp,
            pred_score=True,
            clip_x=True
        )
        model_scores = model_scores.contiguous().view(batch_size_full, n_scores_update_full, -1)

    return model_scores


def euler_maruyama_step(model, x, score, t, dt, noise=None):
    """
    Perform one Euler-Maruyama update step for the reverse SDE.

    The backward SDE update is:
        x_next = x - [f(x,t) - g(t)^2 * score] * dt + g(t)*sqrt(dt)*noise
    """
    f, g = model.sde.get_f_g(t=t, x=x)
    drift = f - torch.square(g) * score
    if noise is None:
        noise = torch.randn_like(x)
    x_next = x - drift * dt + torch.sqrt(dt) * g * noise
    return x_next


def euler_maruyama_sampling(model, x_obs, n_post_samples=1, conditions=None,
                            diffusion_steps=1000, t_end=0, mini_batch_arg=None,
                            random_seed=None, device=None, verbose=False):
    """
    Generate posterior samples using Euler-Maruyama sampling. Expects un-normalized observations.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: (batch_size, n_post_samples, model.prior.n_params_global)
            - Local: (batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    """
    # Initialize sampling
    batch_size, n_obs, n_scores_update, theta, conditions_exp, x_exp, mini_batch_dict, subsample = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        mini_batch_arg=mini_batch_arg,
        random_seed=random_seed, device=device
    )
    diffusion_time = generate_diffusion_time(size=diffusion_steps+1, epsilon=t_end, device=device)
    batch_size_full = batch_size * n_post_samples

    with torch.no_grad():
        model.to(device)
        model.eval()
        # Reverse iterate over diffusion steps.
        for t in tqdm(reversed(range(1, diffusion_steps + 1)), total=diffusion_steps, disable=not verbose):
            t_tensor = torch.full((batch_size_full, 1), diffusion_time[t],
                                  dtype=torch.float32, device=device)

            if subsample:
                # subsample observations for the score update
                sub_x_expanded = sub_sample_observations(
                    x=x_exp, batch_size_full=batch_size_full,
                    n_scores_update=n_scores_update,
                    mini_batch_dict=mini_batch_dict
                )
            else:
                sub_x_expanded = x_exp

            scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                              x_exp=sub_x_expanded, conditions_exp=conditions_exp,
                                              batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                              mini_batch_dict=mini_batch_dict)
            theta = euler_maruyama_step(model, theta, score=scores, t=diffusion_time[t],
                                        dt=diffusion_time[t] - diffusion_time[t - 1])
            if torch.isnan(theta).any():
                print(f"NaNs in theta at time {diffusion_time[t]} with step size:", diffusion_time[t] - diffusion_time[t - 1])
                break

        # Denormalize and reshape theta.
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    return theta


def sde_sampling(model, x_obs, n_post_samples=1, conditions=None,
                 diffusion_steps=100, t_end=0, mini_batch_arg=None,
                 method='euler',
                 random_seed=None, device=None, verbose=False):
    """
    Generate posterior samples using any SDE solving method for sampling. Expects un-normalized observations.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: (batch_size, n_post_samples, model.prior.n_params_global)
            - Local: (batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    """
    from brainpy import sdeint, IntegratorRunner
    import brainpy.math as bm

    # Initialize sampling
    batch_size, n_obs, n_scores_update, theta, conditions_exp, x_exp, mini_batch_dict, subsample = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        mini_batch_arg=mini_batch_arg,
        random_seed=random_seed, device=device
    )
    batch_size_full = batch_size * n_post_samples

    with torch.no_grad():
        model.to(device)
        model.eval()

        #  dx = [f(x,t) - g(t)^2 * score] * dt + g(t) * dWt
        def sde_f_part(theta, t, data):
            t = 1 - t.item()
            t_tensor = torch.full((batch_size_full, 1), t, dtype=torch.float32, device=device)

            if subsample:
                data = torch.tensor(data, dtype=torch.float32, device=device)

            if conditions is None:
                theta_torch = torch.tensor(theta.reshape(batch_size_full, model.prior.n_params_global),
                                       dtype=torch.float32, device=device)
            else:
                theta_torch = torch.tensor(theta.reshape(batch_size_full, n_obs, model.prior.n_params_local),
                                       dtype=torch.float32, device=device)

            scores = eval_compositional_score(model=model, theta=theta_torch, diffusion_time=t_tensor,
                                              x_exp=data, conditions_exp=conditions_exp,
                                              batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                              mini_batch_dict=mini_batch_dict)

            f, g = model.sde.get_f_g(t=t_tensor, x=theta_torch)
            drift = f - torch.square(g) * scores

            if conditions is None:
                drift_reshaped = drift.cpu().numpy().reshape(batch_size_full * model.prior.n_params_global)
            else:
                drift_reshaped = drift.cpu().numpy().reshape(batch_size_full * n_obs * model.prior.n_params_local)
            return -bm.ndarray(drift_reshaped)

        def sde_g_part(theta, t, data):
            t = 1 - t.item()
            if conditions is None:
                t_tensor = torch.ones((batch_size_full * model.prior.n_params_global),
                                      dtype=torch.float32, device=device) * t
            else:
                t_tensor = torch.ones((batch_size_full * n_obs * model.prior.n_params_local),
                                      dtype=torch.float32, device=device) * t

            g = model.sde.get_f_g(t=t_tensor, x=None)
            g_reshaped = g.cpu().numpy()
            return bm.ndarray(g_reshaped)

        if conditions is None:
            x0 = theta.cpu().numpy().reshape(batch_size_full * model.prior.n_params_global)
        else:
            x0 = theta.cpu().numpy().reshape(batch_size_full * n_obs * model.prior.n_params_local)

        if subsample:
            # subsample observations for the score update
            # create a tensor for each time step
            sub_x_expanded = [sub_sample_observations(x=x_exp, batch_size_full=batch_size_full,
                                                      n_scores_update=n_scores_update,
                                                      mini_batch_dict=mini_batch_dict).cpu() for _ in range(diffusion_steps)]
            sub_x_expanded = np.stack(sub_x_expanded, axis=0)
        else:
            sub_x_expanded = x_exp

        integral = sdeint(f=sde_f_part, g=sde_g_part, method=method)
        runner = IntegratorRunner(
            integral,  # the simulation target
            dt=1. / diffusion_steps,  # the time step
            monitors=['theta'],  # the variables to monitor
            inits={'theta': x0},  # the initial values
            jit=False,
            numpy_mon_after_run=True,
            progress_bar=verbose,
        )
        runner.run(
            duration=1.-t_end, start_t=0.,  # reverse SDE
            dyn_args={'data': sub_x_expanded} if subsample else None,
            args={'data': sub_x_expanded} if not subsample else None
        )
        theta = runner.mon['theta'][-1]

        if np.isnan(theta).any():
            print("NaNs in theta, increase number of steps.")

        if conditions is None:
            theta = torch.tensor(theta.reshape(batch_size_full, model.prior.n_params_global),
                                 dtype=torch.float32, device=device)
        else:
            theta = torch.tensor(theta.reshape(batch_size_full, n_obs, model.prior.n_params_local),
                                 dtype=torch.float32, device=device)

        # Denormalize and reshape theta.
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    return theta


def adaptive_sampling(model, x_obs,
                      n_post_samples=1,
                      conditions=None,
                      e_abs: float = None,  # abs error tolerance should grow with the dimension
                      e_rel: float = 0.5,
                      h_init: float = 0.5,
                      r: float = 0.9,
                      adapt_safety: float = 0.9,
                      max_evals: int = 10000,
                      t_start: float = 1.0,
                      t_end: float = 0.,
                      mini_batch_arg=None,
                      run_sampling_in_parallel=True,
                      random_seed=None,
                      device=None,
                      return_steps=False,
                      verbose=False
                      ):
    """
    Generate posterior samples using an adaptive, Heun-style sampling scheme. Expects un-normalized observations.
    Based on "Gotta Go Fast When Generating Data with Score-Based Models" by Jolicoeur-Martineau et. al. (2021).

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: (batch_size, n_post_samples, model.prior.n_params_global)
            - Local: (batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    """
    if not run_sampling_in_parallel:
        post_samples = []
        list_accepted_steps = []
        for i in range(n_post_samples):
            ps, ls = adaptive_sampling(model=model, x_obs=x_obs, n_post_samples=1,
                                       conditions=conditions[:, i][:, None] if conditions is not None else None,
                                  e_abs=e_abs, e_rel=e_rel, h_init=h_init, r=r, adapt_safety=adapt_safety,
                                  max_evals=max_evals, t_start=t_start, t_end=t_end, mini_batch_arg=mini_batch_arg,
                                  run_sampling_in_parallel=True,
                                  random_seed=random_seed+i if random_seed is not None else None,
                                  device=device, return_steps=True,
                                  verbose=verbose)
            post_samples.append(ps)
            list_accepted_steps.append(ls)
            if len(ls) == max_evals / 2:
                print('maximum steps reached, not computing any more posterior samples.')
                break
            if np.isnan(ps).any():
                print("NaNs in theta, increase number of steps.")
                break
        post_samples = np.concatenate(post_samples, axis=1)
        if return_steps:
            return post_samples, list_accepted_steps
        return post_samples

    # Initialize sampling
    batch_size, n_obs, n_scores_update, theta, conditions_exp, x_exp, mini_batch_dict, subsample = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        mini_batch_arg=mini_batch_arg,
        random_seed=random_seed, device=device
    )
    batch_size_full = batch_size * n_post_samples

    current_t = torch.tensor(t_start, dtype=torch.float32, device=device)
    if isinstance(t_end, float):
        t_end = torch.tensor(t_end, dtype=torch.float32, device=device)
    h = torch.tensor(h_init, dtype=torch.float32, device=device)
    if e_abs is None:
        # abs error tolerance grows with the dimension
        e_abs = 0.01 * np.sqrt(theta.shape[-1])
    e_abs_tensor = torch.full((batch_size_full, 1), e_abs, dtype=torch.float32, device=device)
    if not conditions is None:
        e_abs_tensor = e_abs_tensor[:, None]
    theta_prev = theta

    error_scale = 1 / np.sqrt(theta[0].numel()).item()  # 1 / sqrt(n_params)
    list_accepted_steps = []
    with torch.no_grad():
        model.to(device)
        model.eval()
        total_steps = 0
        for _ in tqdm(range(max_evals // 2), disable=not verbose):
            total_steps += 1
            z = torch.randn_like(theta)  # same noise for both steps
            t_tensor = torch.full((batch_size_full, 1), current_t, dtype=torch.float32, device=device)

            if subsample:
                # subsample observations for the score update
                sub_x_expanded = sub_sample_observations(
                    x=x_exp, batch_size_full=batch_size_full,
                    n_scores_update=n_scores_update,
                    mini_batch_dict=mini_batch_dict
                )
            else:
                sub_x_expanded = x_exp

            # Euler-Maruyama step.
            scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                              x_exp=sub_x_expanded, conditions_exp=conditions_exp,
                                              batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                              mini_batch_dict=mini_batch_dict)
            if conditions is None:
                theta_eul = euler_maruyama_step(model, theta, score=scores, t=t_tensor, dt=h, noise=z)
            else:
                theta_eul = euler_maruyama_step(model, theta, score=scores, t=t_tensor[:, None], dt=h, noise=z)

            # Heun-style improved step.
            t_mid = t_tensor - h
            scores_mid = eval_compositional_score(model=model, theta=theta_eul, diffusion_time=t_mid,
                                              x_exp=sub_x_expanded, conditions_exp=conditions_exp,
                                              batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                              mini_batch_dict=mini_batch_dict)
            if conditions is None:
                theta_eul_mid = euler_maruyama_step(model, theta_eul, score=scores_mid, t=t_mid, dt=h, noise=z)
            else:
                theta_eul_mid = euler_maruyama_step(model, theta_eul, score=scores_mid, t=t_mid[:, None], dt=h, noise=z)

            # Average the two steps.
            theta_eul_sec = 0.5 * (theta_eul + theta_eul_mid)

            # Error estimation.
            delta = torch.maximum(e_abs_tensor,
                                  e_rel * torch.maximum(torch.abs(theta_eul), torch.abs(theta_prev)))
            sample_error = torch.max(torch.abs((theta_eul - theta_eul_sec) / delta), dim=1)[0]
            error_norm = sample_error.max()  # max of posterior samples

            E2 = error_scale * error_norm.item() #/ (1.0 + model.sde.kernel(log_snr=model.sde.get_snr(t=current_t))[1])

            # Accept/reject step.
            if E2 <= 1.0+10*model.sde.kernel(log_snr=model.sde.get_snr(t=current_t))[1]:
                theta = theta_eul_sec
                current_t = current_t - h
                theta_prev = theta_eul
                list_accepted_steps.append(h.cpu().numpy())
            elif np.isnan(E2):
                print('delta', delta, 'error_scale', error_scale, 'sample_error', sample_error)
                print("NaNs in E2")
                break
            elif torch.isnan(theta).any():
                print("NaNs in theta")
                theta = theta_prev
                break
            else:
                list_accepted_steps.append(np.nan)

            if E2 == 0:
                E2 = 1e-10
            h = torch.minimum(current_t - t_end, h * adapt_safety * (E2 ** (-r)))
            if current_t <= t_end:
                break

        if verbose or (current_t != t_end):
            print(f"Finished after {total_steps} steps ({total_steps*2} score evals) at time {current_t}.")
            print(f"Mean step size: {np.nanmean(list_accepted_steps)}, "
                  f"min: {np.nanmin(list_accepted_steps)}, "
                  f"max: {np.nanmax(list_accepted_steps)}")
        # Denormalize and reshape theta.
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    if return_steps:
        return theta, list_accepted_steps
    return theta


def probability_ode_solving(model, x_obs, n_post_samples=1, conditions=None,
                            method='RK45',
                            t_end=0, mini_batch_arg=None, random_seed=None, run_sampling_in_parallel=True,
                            device=None, verbose=False):
    """
    Solve the probability ODE (Song et al., 2021) to generate posterior samples. Expects un-normalized observations.
    Solving the ODE can be done either jointly for all posterior samples or separately for each sample. If jointly, then
    the error is computed for all samples at once which might lead to less accurate results.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: (batch_size, n_post_samples, model.prior.n_params_global)
            - Local: (batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    """
    if not run_sampling_in_parallel:
        post_samples = []
        for i in range(n_post_samples):
            post_samples.append(
                probability_ode_solving(model=model, x_obs=x_obs, n_post_samples=1,
                                        conditions=conditions[:, i][:, None] if conditions is not None else None,
                                        t_end=t_end, mini_batch_arg=mini_batch_arg,
                                        run_sampling_in_parallel=True,
                                        random_seed=random_seed+i if random_seed is not None else None,
                                        device=device, verbose=verbose)
            )
            if torch.isnan(post_samples[-1]).any():
                print("NaNs in theta, increase number of steps.")
                break
        return np.concatenate(post_samples, axis=1)

    # Initialize sampling
    batch_size, n_obs, n_scores_update, theta, conditions_exp, x_exp, mini_batch_dict, subsample = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        mini_batch_arg=mini_batch_arg,
        random_seed=random_seed, device=device
    )
    batch_size_full = batch_size * n_post_samples

    with torch.no_grad():
        model.to(device)
        model.eval()
        def probability_ode(t, x):
            t_tensor = torch.full((batch_size_full, 1), t, dtype=torch.float32, device=device)
            if subsample:
                # subsample observations for the score update
                sub_x_expanded = sub_sample_observations(
                    x=x_exp, batch_size_full=batch_size_full,
                    n_scores_update=n_scores_update,
                    mini_batch_dict=mini_batch_dict
                )
            else:
                sub_x_expanded = x_exp

            if conditions_exp is None:
                x_torch = torch.tensor(x.reshape(batch_size_full, model.prior.n_params_global),
                                       dtype=torch.float32, device=device)
            else:
                x_torch = torch.tensor(x.reshape(batch_size_full, n_obs, model.prior.n_params_local),
                                       dtype=torch.float32, device=device)
            scores = eval_compositional_score(model=model, theta=x_torch, diffusion_time=t_tensor,
                                              x_exp=sub_x_expanded, conditions_exp=conditions_exp,
                                              batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                              mini_batch_dict=mini_batch_dict)
            t_exp = t_tensor if conditions_exp is None else t_tensor.unsqueeze(1).expand(-1, n_obs, -1)
            f, g = model.sde.get_f_g(x=x_torch, t=t_exp)
            drift = f - 0.5 * torch.square(g) * scores
            if conditions_exp is None:
                return drift.cpu().numpy().reshape(batch_size_full * model.prior.n_params_global)
            return drift.cpu().numpy().reshape(batch_size_full * n_obs * model.prior.n_params_local)

        # Solve the ODE for all posterior samples at once.
        if conditions_exp is None:
            x0 = theta.cpu().numpy().reshape(batch_size_full * model.prior.n_params_global)
        else:
            x0 = theta.cpu().numpy().reshape(batch_size_full * n_obs * model.prior.n_params_local)
        sol = solve_ivp(probability_ode, t_span=[1, t_end], y0=x0, method=method, t_eval=[t_end])
        if verbose:
            print(f'ODE solved: {sol.success} with #score evals: {sol.nfev}')

        if conditions is None:
            theta = torch.tensor(sol.y[:, -1].reshape(batch_size_full, model.prior.n_params_global),
                                 dtype=torch.float32, device=device)
        else:
            theta = torch.tensor(sol.y[:, -1].reshape(batch_size_full, n_obs, model.prior.n_params_local),
                                 dtype=torch.float32, device=device)
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    return theta


def langevin_sampling(model, x_obs, n_post_samples, conditions=None,
                      diffusion_steps=1000, langevin_steps=10, step_size_factor=0.3, t_end=0,
                      mini_batch_arg=None,
                      random_seed=None, device=None, verbose=False):
    """
    Annealed Langevin Dynamics sampling. Expects un-normalized observations. Based on Song et al., 2020.

    Parameters:
        model: Score-based model.
        x_obs: Observations (assumed to be a 2D tensor or array of shape (n_obs, d)).
        n_post_samples: Number of posterior samples.
        conditions: Conditioning parameters (if None, global sampling is performed).
        diffusion_steps: Number of diffusion steps.
        langevin_steps: Number of inner Langevin steps per diffusion time.
        step_size_factor: Factor to scale the step size by.
        t_end: End time for diffusion
        mini_batch_arg: Dict with number of observations to use for the score update.
            If None, all observations are used. Default: None.
        random_seed: Optional random seed for reproducibility.
        device: Computation device.
        verbose: If True, display progress bar.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: shape (n_post_samples, model.prior.n_params_global)
            - Local: shape (n_post_samples, n_obs * model.prior.n_params_local)
    """
    # Initialize sampling
    batch_size, n_obs, n_scores_update, theta, conditions_exp, x_exp, mini_batch_dict, subsample = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        mini_batch_arg=mini_batch_arg,
        random_seed=random_seed, device=device
    )
    batch_size_full = batch_size * n_post_samples
    diffusion_time = generate_diffusion_time(size=diffusion_steps, epsilon=t_end, device=device)

    # generate steps in reverse
    snr = model.sde.get_snr(t=diffusion_time)
    alpha_t, sigma_t = model.sde.kernel(log_snr=snr)
    max_as = torch.max(sigma_t)
    annealing_step_size = step_size_factor * torch.square(sigma_t / max_as)
    with torch.no_grad():
        model.to(device)
        model.eval()
        # Annealed Langevin dynamics: iterate over time steps in reverse
        progress_bar = tqdm(total=diffusion_steps*langevin_steps, disable=not verbose)
        for t in reversed(range(diffusion_steps)):
            t_tensor = torch.full((batch_size_full, 1), diffusion_time[t],
                                  dtype=torch.float32, device=device)
            step_size = annealing_step_size[t]

            for _ in range(langevin_steps):
                eps = torch.randn_like(theta)

                if subsample:
                    # subsample observations for the score update
                    sub_x_expanded = sub_sample_observations(
                        x=x_exp, batch_size_full=batch_size_full,
                        n_scores_update=n_scores_update,
                        mini_batch_dict=mini_batch_dict
                    )
                else:
                    sub_x_expanded = x_exp

                # Compute model scores
                scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                                  x_exp=sub_x_expanded, conditions_exp=conditions_exp,
                                                  batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                                  mini_batch_dict=mini_batch_dict)
                # Langevin update step
                theta = theta + (step_size / 2) * scores + torch.sqrt(step_size) * eps

                progress_bar.update(1)
            if torch.isnan(theta).any():
                print("NaNs in theta, stopping here.")
                break

        # Denormalize theta using the prior's statistics.
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    return theta
