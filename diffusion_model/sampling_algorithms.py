import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm

from diffusion_model.helper_functions import generate_diffusion_time

sampling_defaults = {
    'damping_factor': lambda t: torch.ones_like(t),
    'size': np.inf,  # for mini-batch
    'noisy_condition': {
        'apply': False,
        'noise_scale': 0.1,
        'tau_1': 0.6,
        'tau_2': 0.8,
        'mixing_factor': 1.
    },
    'sampling_chunk_size': 2048,  # to prevent out of memory errors
    'MC-dropout': False,
}


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def expand_obs(x_obs, model, n_post_samples):
    """
    Expand and normalize observations.
    Input is (batch_size, n_obs, n_time_steps, n_features) or (batch_size, n_obs, n_features)
    """
    if not isinstance(x_obs, torch.Tensor):
        x_obs = torch.tensor(x_obs, dtype=torch.float32)
    x_obs_norm = model.prior.normalize_data(x_obs)
    batch_size = x_obs_norm.shape[0]
    n_obs = x_obs_norm.shape[1]

    ##########################

    if x_obs_norm.ndim == 4:
        # Reshape observations to (batch_size*n_post_samples*n_obs, n_time_steps, n_features)
        n_time_steps = x_obs_norm.shape[2]

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
            x_exp = x_obs_norm.contiguous().view(batch_size, n_obs_reduced, model.current_number_of_obs, n_time_steps,
                                                 -1)
            # expand to number of posterior samples
            x_expanded = x_exp.unsqueeze(1).expand(batch_size, n_post_samples, n_obs_reduced,
                                                   model.current_number_of_obs,
                                                   n_time_steps, -1)
            x_expanded = x_expanded.contiguous().view(batch_size * n_post_samples * n_obs_reduced,
                                                      model.current_number_of_obs, n_time_steps, -1)
    else:
        # Reshape observations to (batch_size*n_post_samples*n_obs, n_features)
        if model.max_number_of_obs == 1:
            # the score is always conditioned on only one observation
            # expand to number of posterior samples
            x_exp = x_obs_norm.unsqueeze(1).expand(batch_size, n_post_samples, n_obs, -1)
            x_expanded = x_exp.contiguous().view(batch_size * n_post_samples * n_obs, -1)
        else:
            # the score is conditioned on multiple observations
            # factorize data into (batch_size*n_post_samples * reduced_data, current_number_of_obs, n_features)
            n_obs_reduced = n_obs // model.current_number_of_obs
            if n_obs % model.current_number_of_obs != 0:
                print('warning: number of observations is not a multiple of current_number_of_obs '
                      f'dropping last {n_obs % model.current_number_of_obs} observations.')
                n_obs = n_obs_reduced * model.current_number_of_obs
                x_obs_norm = x_obs_norm[:, :n_obs]
            x_exp = x_obs_norm.contiguous().view(batch_size, n_obs_reduced, model.current_number_of_obs, -1)
            # expand to number of posterior samples
            x_expanded = x_exp.unsqueeze(1).expand(batch_size, n_post_samples, n_obs_reduced,
                                                   model.current_number_of_obs, -1)
            x_expanded = x_expanded.contiguous().view(batch_size * n_post_samples * n_obs_reduced,
                                                      model.current_number_of_obs, -1)
    return x_expanded


def get_n_obs(x_obs, model):
    """
        Get number of observations from x_obs.

        Returns:
            x_obs_norm: Tensor of shape (n_post_samples*n_obs, n_time_steps, n_features)
            n_obs: Number of observations
    """
    if x_obs.ndim == 4:
        # input is (batch_size, n_obs, n_time_steps, n_features)
        n_obs = x_obs.shape[1]
    elif x_obs.ndim == 3:
        # input is (batch_size, n_obs, n_features)
        n_obs = x_obs.shape[1]
    else:
        raise ValueError('x_obs must have shape (batch_size, n_obs, n_time_steps, n_features) or '
                         '(batch_size, n_obs, n_features)')

    ##########################

    if model.max_number_of_obs > 1:
        # the score is conditioned on multiple observations
        # factorize data into (batch_size*n_post_samples * reduced_data, current_number_of_obs, n_features)
        n_obs_reduced = n_obs // model.current_number_of_obs
        if n_obs % model.current_number_of_obs != 0:
            print('warning: number of observations is not a multiple of current_number_of_obs '
                 f'dropping last {n_obs % model.current_number_of_obs} observations.')
            n_obs = n_obs_reduced * model.current_number_of_obs
    return n_obs


def initialize_sampling(model, x_obs, n_post_samples, conditions, sampling_arg, random_seed):
    """
    Initialize common parameters for sampling methods.

    Args:
        model (ScoreModel): Score-based model with prior and SDE attributes
        x_obs (np.array): Batch of observations.
        n_post_samples (int): Number of posterior samples to generate
        conditions (tensor or None): Conditioning parameters for local sampling
        sampling_arg (dict or None): Dict with arguments for the mini-batch algorithm
        random_seed (int or None): Random seed for reproducibility

    Returns:
        tuple: A tuple containing:
            - n_obs (int): Number of observations
            - n_scores_update (int): Number of scores to update
            - theta_init (tensor): Initial parameter samples
            - conditions_exp (tensor or None): Expanded conditions
            - x_expanded (tensor): Expanded observations
            - sampling_arg_dict (dict): Mini-batch parameters
            - subsample (bool): Whether to subsample or not
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    global_sampling = True
    if not conditions is None:
        global_sampling = False
        model.current_number_of_obs = 1  # local sampling, so only one observation per posterior sample

    # Preprocess observations
    batch_size = x_obs.shape[0]
    n_obs = get_n_obs(x_obs=x_obs, model=model)

    if n_obs % model.current_number_of_obs != 0:
        raise ValueError("'n_obs' must be a multiple of 'model.current_number_of_obs'")

    # Preprocess conditions
    if not global_sampling:
        if not isinstance(conditions, torch.Tensor):
            conditions = torch.tensor(conditions, dtype=torch.float32)
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
        conditions_collapsed = None

    ##########################

    # Number of observations to use for the score update
    if global_sampling:
        n_scores_update = n_obs // model.current_number_of_obs
    else:
        n_scores_update = n_obs

    ###### Prepare Mini-Batching
    sampling_arg_dict = sampling_defaults.copy()
    if sampling_arg is not None:
        sampling_arg_dict.update(sampling_arg)
    sampling_arg_dict['size'] = min(n_scores_update, sampling_arg_dict['size'])
    if sampling_arg_dict['size'] < n_scores_update:
        sampling_arg_dict['subsample'] = True
    else:
        sampling_arg_dict['subsample'] = False

    if not sampling_arg_dict['subsample']:
        x_obs = expand_obs(x_obs=x_obs, model=model, n_post_samples=n_post_samples)
        # otherwise x_obs is expanded later on a batch-wise basis

    # sample from latent prior for diffusion model
    if global_sampling:
        theta_init = torch.randn((batch_size * n_post_samples, model.prior.n_params_global), dtype=torch.float32)
        theta_init = theta_init / np.sqrt(n_obs)

        if 'damping_factor_prior' in sampling_arg_dict:
            damping_factor = torch.tensor(sampling_arg_dict['damping_factor_prior'], dtype=torch.float32)
        else:
            damping_factor = sampling_arg_dict['damping_factor'](t=torch.tensor(1, dtype=torch.float32))
        theta_init = theta_init / torch.sqrt(damping_factor)
    else:
        theta_init = torch.randn((batch_size * n_post_samples, n_obs, model.prior.n_params_local), dtype=torch.float32)

    return theta_init, x_obs, conditions_collapsed, sampling_arg_dict, batch_size, n_obs, n_scores_update


def sub_sample_observations(data, sampling_arg_dict):
    """
    Subsample observations for the score update with x shape: (batch_shape, n_obs, ...)
    """
    # sample indicies
    rand_obs_indx = np.random.permutation(data.shape[1])
    # get subsampled data
    data_sub = data[:, rand_obs_indx[:sampling_arg_dict['size']]]
    return data_sub


def piecewise_condition_function(t, tau_1, tau_2):
    """
    Computes a piecewise function on tensor t:
      - If t < tau_1, returns 1.
      - Elif t < tau_2, returns (tau_2 - t) / (tau_2 - tau_1).
      - Else, returns 0.

    Note: With tau_1 = 1 and tau_2 = 0, the first condition (t < 1) always holds
    for t < 1, so the second branch is never reached.

    Args:
        t (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with the piecewise function applied elementwise.
    """
    # Use torch.where to apply conditions elementwise.
    result = torch.where(
        t <= tau_1,
        torch.tensor(1.0, dtype=t.dtype, device=t.device),
        torch.where(
            t < tau_2,
            (tau_2 - t) / (tau_2 - tau_1),
            torch.tensor(0.0, dtype=t.dtype, device=t.device)
        )
    )
    return result


def eval_compositional_score(model, theta, diffusion_time, x_obs, conditions_exp, batch_size_full,
                             n_scores_update_full, sampling_arg_dict, n_post_samples, clip_x=True):
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
    # subsample observations for the score update
    if sampling_arg_dict['subsample']:
        sub_x_obs = sub_sample_observations(data=x_obs, sampling_arg_dict=sampling_arg_dict)
        # prepare data
        x_exp = expand_obs(x_obs=sub_x_obs, model=model, n_post_samples=n_post_samples)
        x_exp = x_exp.to(theta.device)
    else:
        # data is already prepared
        x_exp = x_obs

    # Expand diffusion_time to shape (n_post_samples*n_scores_update, 1)
    t_exp = diffusion_time.unsqueeze(1).expand(-1, sampling_arg_dict['size'], -1).contiguous().view(-1, 1)

    if sampling_arg_dict['noisy_condition']['apply']:
        # mean and std on clean data
        x_mean_clean = torch.mean(x_exp, dim=1, keepdim=True)
        x_std_clean = torch.std(x_exp, dim=1, keepdim=True)

        # get noisy condition
        cond_t = piecewise_condition_function(t_exp,
                                              tau_1=sampling_arg_dict['noisy_condition']['tau_1'],
                                              tau_2=sampling_arg_dict['noisy_condition']['tau_2'])
        if x_exp.dim() == 3:
            cond_t = cond_t[:, None]
        x_noisy = torch.sqrt(cond_t) * x_exp
        x_noisy = x_noisy + sampling_arg_dict['noisy_condition']['noise_scale'] * torch.sqrt(1 - cond_t) * torch.randn_like(x_exp)
        x_rescaled = (x_noisy - torch.mean(x_noisy, dim=1, keepdim=True)) / torch.std(x_noisy, dim=1, keepdim=True)
        # rescale noisy condition
        x_rescaled = x_rescaled * x_std_clean + x_mean_clean
        # final corrupted output
        x_exp = sampling_arg_dict['noisy_condition']['mixing_factor'] * x_rescaled + (1 - sampling_arg_dict['noisy_condition']['mixing_factor']) * x_exp

    if conditions_exp is None and n_scores_update_full > 1:
        theta_exp = theta.unsqueeze(1).expand(-1, sampling_arg_dict['size'], -1).contiguous().view(-1, model.prior.n_params_global)

        # pass chunks through the model
        n_samples_global = theta_exp.shape[0]
        model_scores_list = []
        for start_idx in range(0, n_samples_global, sampling_arg_dict['sampling_chunk_size']):
            end_idx = min(start_idx + sampling_arg_dict['sampling_chunk_size'], n_samples_global)
            # Run the sampling on the chunk
            model_scores_chunk = model.forward_global(
                theta_global=theta_exp[start_idx:end_idx],
                time=t_exp[start_idx:end_idx],
                x=x_exp[start_idx:end_idx],
                pred_score=True,
                clip_x=clip_x
            )
            model_scores_list.append(model_scores_chunk)
        # Concatenate all sample chunks along the batch dimension.
        model_indv_scores = torch.cat(model_scores_list, dim=0)

        # Reshape to (batch_size_full, n_obs, -1) and sum over observations
        model_sum_scores_indv = model_indv_scores.contiguous().view(batch_size_full, sampling_arg_dict['size'], -1)

        # add prior to the individual score, this is more stable than adding it to the sum
        prior_scores = (1 - diffusion_time) * model.prior.score_global_batch(theta)
        # expand prior scores to match the individual scores
        prior_scores_indv = prior_scores.unsqueeze(1)
        # (1 - n_scores_update) * (1 - diffusion_time) * model.prior.score_global_batch(theta)
        model_sum_scores_indv = model_sum_scores_indv - prior_scores_indv
        model_sum_scores = n_scores_update_full * torch.mean(model_sum_scores_indv, dim=1)

        # (1 - n_scores_update) * (1 - diffusion_time) * model.prior.score_global_batch(theta)
        # just the plus 1 is missing
        damping_factor = sampling_arg_dict['damping_factor'](diffusion_time)
        model_scores = damping_factor * (prior_scores + model_sum_scores)
    elif conditions_exp is None and n_scores_update_full == 1:

        # pass chunks through the model
        n_samples_global = theta.shape[0]
        model_scores_list = []
        for start_idx in range(0, n_samples_global, sampling_arg_dict['sampling_chunk_size']):
            end_idx = min(start_idx + sampling_arg_dict['sampling_chunk_size'], n_samples_global)
            # Run the sampling on the chunk
            model_scores_chunk = model.forward_global(
                theta_global=theta[start_idx:end_idx],
                time=t_exp[start_idx:end_idx],
                x=x_exp[start_idx:end_idx],
                pred_score=True,
                clip_x=clip_x
            )
            model_scores_list.append(model_scores_chunk)
        # Concatenate all sample chunks along the batch dimension.
        model_indv_scores = torch.cat(model_scores_list, dim=0)

        # apply damping
        damping_factor = sampling_arg_dict['damping_factor'](diffusion_time)
        model_scores = damping_factor * model_indv_scores
    else:
        theta_exp = theta.contiguous().view(-1, model.prior.n_params_local)

        # pass chunks through the model
        n_samples_local = theta_exp.shape[0]
        model_scores_list = []
        for start_idx in range(0, n_samples_local, sampling_arg_dict['sampling_chunk_size']):
            end_idx = min(start_idx + sampling_arg_dict['sampling_chunk_size'], n_samples_local)
            # Run the sampling on the chunk
            model_scores_chunk = model.forward_local(
                theta_local=theta_exp[start_idx:end_idx],
                time=t_exp[start_idx:end_idx],
                x=x_exp[start_idx:end_idx],
                theta_global=conditions_exp[start_idx:end_idx],
                pred_score=True,
                clip_x=clip_x
            )
            model_scores_list.append(model_scores_chunk)
        # Concatenate all sample chunks along the batch dimension.
        model_scores = torch.cat(model_scores_list, dim=0)

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
                            diffusion_steps=1000, t_end=0, sampling_arg=None,
                            random_seed=None, device=None, verbose=False):
    """
    Generate posterior samples using Euler-Maruyama sampling. Expects un-normalized observations.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: (batch_size, n_post_samples, model.prior.n_params_global)
            - Local: (batch_size, n_post_samples, n_obs, model.prior.n_params_local)
    """
    # Initialize sampling
    theta, x_obs, conditions_exp, sampling_arg_dict, batch_size, n_obs, n_scores_update = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        sampling_arg=sampling_arg, random_seed=random_seed
    )
    diffusion_time = generate_diffusion_time(size=diffusion_steps+1, epsilon=t_end, device=device)
    batch_size_full = batch_size * n_post_samples

    with torch.no_grad():
        model.to(device)
        model.eval()
        if sampling_arg_dict['MC-dropout']:
            enable_dropout(model)
        theta = theta.to(device)
        if not sampling_arg_dict['subsample']:
            x_obs = x_obs.to(device)
        if conditions_exp is not None:
            conditions_exp = conditions_exp.to(device)
        # Reverse iterate over diffusion steps.
        for t in tqdm(reversed(range(1, diffusion_steps + 1)), total=diffusion_steps, disable=not verbose):
            t_tensor = torch.full((batch_size_full, 1), diffusion_time[t],
                                  dtype=torch.float32, device=device)

            scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                              x_obs=x_obs, conditions_exp=conditions_exp,
                                              batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                              n_post_samples=n_post_samples,
                                              sampling_arg_dict=sampling_arg_dict)
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

#
# def sde_sampling(model, x_obs, n_post_samples=1, conditions=None,
#                  diffusion_steps=100, t_end=0, sampling_arg=None,
#                  method='euler',
#                  random_seed=None, device=None, verbose=False):
#     """
#     Generate posterior samples using any SDE solving method for sampling. Expects un-normalized observations.
#
#     Returns:
#         theta: Posterior samples as a NumPy array.
#             - Global: (batch_size, n_post_samples, model.prior.n_params_global)
#             - Local: (batch_size, n_post_samples, n_obs, model.prior.n_params_local)
#     """
#     from brainpy import sdeint, IntegratorRunner
#     import brainpy.math as bm
#
#     # Initialize sampling
#     batch_size, n_obs, n_scores_update, theta, conditions_exp, sampling_arg_dict = initialize_sampling(
#         model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
#         sampling_arg=sampling_arg, random_seed=random_seed
#     )
#     batch_size_full = batch_size * n_post_samples
#
#     with torch.no_grad():
#         model.to(device)
#         model.eval()
#         if sampling_arg_dict['MC-dropout']:
#             enable_dropout(model)
#         if conditions_exp is not None:
#             conditions_exp = conditions_exp.to(device)
#         #  dx = [f(x,t) - g(t)^2 * score] * dt + g(t) * dWt
#         def sde_f_part(theta, t, data):
#             t = 1 - t.item()
#             t_tensor = torch.full((batch_size_full, 1), t, dtype=torch.float32, device=device)
#
#             if subsample:
#                 data = torch.tensor(data, dtype=torch.float32, device=device)
#
#             if conditions is None:
#                 theta_torch = torch.tensor(theta.reshape(batch_size_full, model.prior.n_params_global),
#                                        dtype=torch.float32, device=device)
#             else:
#                 theta_torch = torch.tensor(theta.reshape(batch_size_full, n_obs, model.prior.n_params_local),
#                                        dtype=torch.float32, device=device)
#
#             scores = eval_compositional_score(model=model, theta=theta_torch, diffusion_time=t_tensor,
#                                               x_exp=data, conditions_exp=conditions_exp,
#                                               batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
#                                               sampling_arg_dict=sampling_arg_dict)
#
#             f, g = model.sde.get_f_g(t=t_tensor, x=theta_torch)
#             drift = f - torch.square(g) * scores
#
#             if conditions is None:
#                 drift_reshaped = drift.cpu().numpy().reshape(batch_size_full * model.prior.n_params_global)
#             else:
#                 drift_reshaped = drift.cpu().numpy().reshape(batch_size_full * n_obs * model.prior.n_params_local)
#             return -bm.ndarray(drift_reshaped)
#
#         def sde_g_part(theta, t, data):
#             t = 1 - t.item()
#             if conditions is None:
#                 t_tensor = torch.ones((batch_size_full * model.prior.n_params_global),
#                                       dtype=torch.float32, device=device) * t
#             else:
#                 t_tensor = torch.ones((batch_size_full * n_obs * model.prior.n_params_local),
#                                       dtype=torch.float32, device=device) * t
#
#             g = model.sde.get_f_g(t=t_tensor, x=None)
#             g_reshaped = g.cpu().numpy()
#             return bm.ndarray(g_reshaped)
#
#         if conditions is None:
#             x0 = theta.cpu().numpy().reshape(batch_size_full * model.prior.n_params_global)
#         else:
#             x0 = theta.cpu().numpy().reshape(batch_size_full * n_obs * model.prior.n_params_local)
#
#         if subsample:
#             # subsample observations for the score update
#             # create a tensor for each time step
#             sub_x_expanded = [sub_sample_observations(x=x_exp, batch_size_full=batch_size_full,
#                                                       n_scores_update=n_scores_update,
#                                                       sampling_arg_dict=sampling_arg_dict).cpu() for _ in range(diffusion_steps)]
#             sub_x_expanded = np.stack(sub_x_expanded, axis=0)
#         else:
#             sub_x_expanded = x_exp
#             sub_x_expanded = sub_x_expanded.to(device)
#
#         integral = sdeint(f=sde_f_part, g=sde_g_part, method=method)
#         runner = IntegratorRunner(
#             integral,  # the simulation target
#             dt=1. / diffusion_steps,  # the time step
#             monitors=['theta'],  # the variables to monitor
#             inits={'theta': x0},  # the initial values
#             jit=False,
#             numpy_mon_after_run=True,
#             progress_bar=verbose,
#         )
#         runner.run(
#             duration=1.-t_end, start_t=0.,  # reverse SDE
#             dyn_args={'data': sub_x_expanded} if subsample else None,
#             args={'data': sub_x_expanded} if not subsample else None
#         )
#         theta = runner.mon['theta'][-1]
#
#         if np.isnan(theta).any():
#             print("NaNs in theta, increase number of steps.")
#
#         if conditions is None:
#             theta = torch.tensor(theta.reshape(batch_size_full, model.prior.n_params_global),
#                                  dtype=torch.float32, device=device)
#         else:
#             theta = torch.tensor(theta.reshape(batch_size_full, n_obs, model.prior.n_params_local),
#                                  dtype=torch.float32, device=device)
#
#         # Denormalize and reshape theta.
#         if conditions is None:
#             theta = model.prior.denormalize_theta(theta, global_params=True)
#             theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, model.prior.n_params_global)
#         else:
#             theta = model.prior.denormalize_theta(theta, global_params=False)
#             theta = theta.detach().cpu().numpy().reshape(batch_size, n_post_samples, n_obs, model.prior.n_params_local)
#     return theta


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
                      sampling_arg=None,
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
        for i in tqdm(range(n_post_samples), disable=not verbose):
            ps, ls = adaptive_sampling(model=model, x_obs=x_obs, n_post_samples=1,
                                       conditions=conditions[:, i][:, None] if conditions is not None else None,
                                  e_abs=e_abs, e_rel=e_rel, h_init=h_init, r=r, adapt_safety=adapt_safety,
                                  max_evals=max_evals, t_start=t_start, t_end=t_end, sampling_arg=sampling_arg,
                                  run_sampling_in_parallel=True,
                                  random_seed=random_seed+i if random_seed is not None else None,
                                  device=device, return_steps=True,
                                  verbose=False)
            post_samples.append(ps)
            list_accepted_steps.append(ls)
            if len(ls) == max_evals / 2:
                print('maximum steps reached, increase number of steps.')
                break
            if np.isnan(ps).any():
                print("NaNs in theta, increase number of steps.")
                break
        post_samples = np.concatenate(post_samples, axis=1)
        if return_steps:
            return post_samples, list_accepted_steps
        return post_samples

    # Initialize sampling
    theta, x_obs, conditions_exp, sampling_arg_dict, batch_size, n_obs, n_scores_update = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        sampling_arg=sampling_arg, random_seed=random_seed
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
        if sampling_arg_dict['MC-dropout']:
            enable_dropout(model)
        theta = theta.to(device)
        theta_prev = theta_prev.to(device)
        if not sampling_arg_dict['subsample']:
            x_obs = x_obs.to(device)
        if conditions_exp is not None:
            conditions_exp = conditions_exp.to(device)
        total_steps = 0
        for _ in tqdm(range(max_evals // 2), disable=not verbose):
            total_steps += 1
            z = torch.randn_like(theta)  # same noise for both steps
            t_tensor = torch.full((batch_size_full, 1), current_t, dtype=torch.float32, device=device)

            # Euler-Maruyama step.
            scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                              x_obs=x_obs, conditions_exp=conditions_exp,
                                              batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                              n_post_samples=n_post_samples,
                                              sampling_arg_dict=sampling_arg_dict)
            if conditions is None:
                theta_eul = euler_maruyama_step(model, theta, score=scores, t=t_tensor, dt=h, noise=z)
            else:
                theta_eul = euler_maruyama_step(model, theta, score=scores, t=t_tensor[:, None], dt=h, noise=z)

            # Heun-style improved step.
            t_mid = t_tensor - h
            scores_mid = eval_compositional_score(model=model, theta=theta_eul, diffusion_time=t_mid,
                                              x_obs=x_obs, conditions_exp=conditions_exp,
                                              batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                              n_post_samples=n_post_samples,
                                              sampling_arg_dict=sampling_arg_dict)
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
                            t_end=0, sampling_arg=None, random_seed=None, run_sampling_in_parallel=True,
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
        for i in tqdm(range(n_post_samples), disable=not verbose):
            post_samples.append(
                probability_ode_solving(model=model, x_obs=x_obs, n_post_samples=1,
                                        conditions=conditions[:, i][:, None] if conditions is not None else None,
                                        t_end=t_end, sampling_arg=sampling_arg,
                                        run_sampling_in_parallel=True,
                                        random_seed=random_seed+i if random_seed is not None else None,
                                        device=device, verbose=False)
            )
            if np.isnan(post_samples[-1]).any():
                print("NaNs in theta, increase number of steps.")
                break
        return np.concatenate(post_samples, axis=1)

    # Initialize sampling
    theta, x_obs, conditions_exp, sampling_arg_dict, batch_size, n_obs, n_scores_update = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        sampling_arg=sampling_arg, random_seed=random_seed
    )
    batch_size_full = batch_size * n_post_samples

    with torch.no_grad():
        model.to(device)
        model.eval()
        if sampling_arg_dict['MC-dropout']:
            enable_dropout(model)
        theta = theta.to(device)
        if not sampling_arg_dict['subsample']:
            x_obs = x_obs.to(device)
        if conditions_exp is not None:
            conditions_exp = conditions_exp.to(device)
        def probability_ode(t, x):
            t_tensor = torch.full((batch_size_full, 1), t, dtype=torch.float32, device=device)

            if conditions_exp is None:
                x_torch = torch.tensor(x.reshape(batch_size_full, model.prior.n_params_global),
                                       dtype=torch.float32, device=device)
            else:
                x_torch = torch.tensor(x.reshape(batch_size_full, n_obs, model.prior.n_params_local),
                                       dtype=torch.float32, device=device)
            scores = eval_compositional_score(model=model, theta=x_torch, diffusion_time=t_tensor,
                                                  x_obs=x_obs, conditions_exp=conditions_exp,
                                                  batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                                  n_post_samples=n_post_samples,
                                                  sampling_arg_dict=sampling_arg_dict)

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
                      sampling_arg=None,
                      random_seed=None, device=None, verbose=False):
    """
    Annealed Langevin Dynamics sampling. Expects un-normalized observations. Based on Song et al., 2020.

    Parameters:
        model: Score-based model.
        x_obs: Batch of observations.
        n_post_samples: Number of posterior samples.
        conditions: Conditioning parameters (if None, global sampling is performed).
        diffusion_steps: Number of diffusion steps.
        langevin_steps: Number of inner Langevin steps per diffusion time.
        step_size_factor: Factor to scale the step size by.
        t_end: End time for diffusion
        sampling_arg: Dict with number of observations to use for the score update.
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
    theta, x_obs, conditions_exp, sampling_arg_dict, batch_size, n_obs, n_scores_update = initialize_sampling(
        model=model, x_obs=x_obs, n_post_samples=n_post_samples, conditions=conditions,
        sampling_arg=sampling_arg, random_seed=random_seed
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
        if sampling_arg_dict['MC-dropout']:
            enable_dropout(model)
        theta = theta.to(device)
        if not sampling_arg_dict['subsample']:
            x_obs = x_obs.to(device)
        if not conditions_exp is None:
            conditions_exp = conditions_exp.to(device)
        # Annealed Langevin dynamics: iterate over time steps in reverse
        progress_bar = tqdm(total=diffusion_steps*langevin_steps, disable=not verbose)
        for t in reversed(range(diffusion_steps)):
            t_tensor = torch.full((batch_size_full, 1), diffusion_time[t],
                                  dtype=torch.float32, device=device)
            step_size = annealing_step_size[t]

            for _ in range(langevin_steps):
                eps = torch.randn_like(theta)

                # Compute model scores
                scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                                  x_obs=x_obs, conditions_exp=conditions_exp,
                                                  batch_size_full=batch_size_full, n_scores_update_full=n_scores_update,
                                                  n_post_samples=n_post_samples,
                                                  sampling_arg_dict=sampling_arg_dict)
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
