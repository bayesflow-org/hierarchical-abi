import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm
from helper_functions import generate_diffusion_time


def prepare_observations(x_obs, model, device):
    """
    Prepare and normalize observations.

    Converts x_obs to a tensor if necessary, normalizes via model.prior,
    reshapes to (n_time_steps, n_obs) then transposes to
    (n_obs, n_time_steps, d) where d is the feature dimension.

    Returns:
        x_obs_norm: Tensor of shape (n_obs, n_time_steps, d)
        n_obs: Number of observations
        n_time_steps: Number of time steps
    """
    if not isinstance(x_obs, torch.Tensor):
        x_obs = torch.tensor(x_obs, dtype=torch.float32, device=device)
    x_obs_norm = model.prior.normalize_data(x_obs)
    # Reshape to (n_time_steps, n_obs)
    x_obs_norm = x_obs_norm.reshape(x_obs_norm.shape[0], -1)
    n_time_steps = x_obs_norm.shape[0]
    n_obs = x_obs_norm.shape[-1]
    # Transpose to (n_obs, n_time_steps) and add feature dimension: (n_obs, n_time_steps, 1)
    x_obs_norm = x_obs_norm.T[:, :, None]
    return x_obs_norm, n_obs, n_time_steps


def initialize_parameters(n_post_samples, n_obs, model, conditions, device, random_seed):
    """
    Initialize parameter samples theta and condition expansions.

    Global case:
        theta: shape (n_post_samples, model.prior.n_params_global)
    Local case:
        theta: shape (n_post_samples, n_obs, model.prior.n_params_local)

    Returns:
        theta, conditions_exp (None in the global case)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    if conditions is None:
        theta = torch.randn(n_post_samples, model.prior.n_params_global, dtype=torch.float32, device=device)
        # Adapt theta for compositional score matching
        theta = theta * torch.tensor(1/np.sqrt(n_obs), dtype=torch.float32, device=device)
        conditions_collapsed = None
    else:
        if not isinstance(conditions, torch.Tensor):
            conditions = torch.tensor(conditions, dtype=torch.float32, device=device)
        conditions_norm = model.prior.normalize_theta(conditions, global_params=True)
        theta = torch.randn(n_post_samples, n_obs, model.prior.n_params_local,
                            dtype=torch.float32, device=device)
        # if condition for each posterior sample provided use it, otherwise expand
        if conditions_norm.ndim == 2:  # (n_post_samples, model.prior.n_params_global)
            # expand conditions for every observation
            conditions_exp = conditions_norm.unsqueeze(1).expand(n_post_samples, n_obs, -1)
        else:  # (model.prior.n_params_global)
            # expand conditions for every observation and each posterior sample
            conditions_exp = conditions_norm.unsqueeze(0).expand(n_post_samples, n_obs, -1)
        # collapse conditions to shape (n_post_samples*n_obs, n_params_local)
        conditions_collapsed = conditions_exp.reshape(-1, model.prior.n_params_global)
    return theta, conditions_collapsed


def expand_observations(x_obs_norm, n_post_samples, n_obs, n_time_steps):
    """
    Expand normalized observations to shape (n_post_samples*n_obs, n_time_steps, d).

    x_obs_norm is expected to have shape (n_obs, n_time_steps, d).
    """
    x_exp = x_obs_norm.unsqueeze(0).expand(n_post_samples, n_obs, n_time_steps, -1)
    x_expanded = x_exp.reshape(n_post_samples * n_obs, n_time_steps, -1)
    return x_expanded



def pareto_smooth_sum(scores, tail_fraction, alpha, dim):
    """
    Compute a Pareto smoothed sum along the specified dimension.

    Instead of summing the scores directly, the largest 'tail' fraction
    of the scores is replaced by their mean value before summing.

    Parameters:
        scores (torch.Tensor): Tensor of scores with shape
            (n_post_samples, n_obs, d) or similar.
        tail_fraction (float): Fraction of the tail to smooth.
        alpha (float): Damping factor for the tail values.
        dim (int): Dimension along which to perform the smoothing.

    Returns:
        torch.Tensor: The Pareto smoothed sum along the specified dimension,
            with the dimension removed.
    """
    # Compute threshold for the tail: the value below which (1 - tail_fraction) of the scores lie.
    threshold = torch.quantile(scores, 1 - tail_fraction, dim=dim, keepdim=True)
    # Create a mask: True for elements in the tail (>= threshold)
    tail_mask = scores >= threshold
    # Apply damping: for tail elements, multiply by alpha; else leave unchanged.
    damped_scores = torch.where(tail_mask, alpha * scores, scores)
    return damped_scores.sum(dim=dim)


def eval_compositional_score(model, theta, diffusion_time, x_expanded, n_post_samples, n_obs, conditions_exp,
                             pareto_smooth_sum_dict):
    """
    Compute the (global or local) compositional score.

    For global scores:
        - theta is expanded to (n_post_samples, n_obs, model.prior.n_params_global)
        - The score is computed per observation and then summed.
        - The prior score is also incorporated.

    For local scores:
        - theta is reshaped to (n_post_samples*n_obs, model.prior.n_params_local)
        - The score is computed and then reshaped back to (n_post_samples, n_obs, -1).

    Returns:
        model_scores: Computed score tensor.
    """
    # Expand diffusion_time to shape (n_post_samples*n_obs, 1)
    t_exp = diffusion_time.unsqueeze(1).expand(-1, n_obs, -1).reshape(-1, 1)

    if conditions_exp is None:
        theta_exp = theta.unsqueeze(1).expand(-1, n_obs, -1).reshape(-1, model.prior.n_params_global)
        model_indv_scores = model.forward_global(
            theta_global=theta_exp,
            time=t_exp,
            x=x_expanded,
            pred_score=True,
            clip_x=True
        )
        # Reshape to (n_post_samples, n_obs, -1) and sum over observations
        model_sum_scores_indv = model_indv_scores.reshape(n_post_samples, n_obs, -1)
        if pareto_smooth_sum_dict is None:
            model_sum_scores = model_sum_scores_indv.sum(dim=1)
        else:
            # Instead of a simple sum, perform Pareto smoothing over observations.
            model_sum_scores = pareto_smooth_sum(model_sum_scores_indv,
                                                 tail_fraction=pareto_smooth_sum_dict['tail_fraction'],
                                                 alpha=pareto_smooth_sum_dict['alpha'],
                                                 dim=1)

        prior_score = model.prior.score_global_batch(theta)
        model_scores = (1 - n_obs) * (1 - diffusion_time) * prior_score + model_sum_scores
    else:
        theta_exp = theta.reshape(-1, model.prior.n_params_local)
        model_scores = model.forward_local(
            theta_local=theta_exp,
            time=t_exp,
            x=x_expanded,
            theta_global=conditions_exp,
            pred_score=True,
            clip_x=True
        )
        model_scores = model_scores.reshape(n_post_samples, n_obs, -1)

    return model_scores


def euler_maruyama_step(model, x, score, t, dt, noise=None):
    """
    Perform one Euler-Maruyama update step for the SDE.

    The backward SDE update is:
        x_next = x - [f(x,t) - g(t)^2 * score] * dt + g(t)*sqrt(dt)*noise
    """
    f, g = model.sde.get_f_g(t=t, x=x)
    drift = f - torch.square(g) * score
    if noise is None:
        noise = torch.randn_like(x, dtype=x.dtype, device=x.device)
    x_next = x - drift * dt + torch.sqrt(dt) * g * noise
    return x_next


def euler_maruyama_sampling(model, x_obs, n_post_samples=1, conditions=None,
                            diffusion_steps=1000, t_end=0, pareto_smooth_sum_dict=None, random_seed=None, device=None):
    """
    Generate posterior samples using Euler-Maruyama sampling. Expects un-normalized observations.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: (n_post_samples, model.prior.n_params_global)
            - Local: (n_post_samples, n_obs, model.prior.n_params_local)
    """
    # Prepare observations and initialize parameters.
    x_obs_norm, n_obs, n_obs_time_steps = prepare_observations(x_obs, model, device)
    theta, conditions_exp = initialize_parameters(n_post_samples, n_obs, model, conditions, device, random_seed)
    diffusion_time = generate_diffusion_time(size=diffusion_steps + 1, epsilon=t_end, device=device)
    x_expanded = expand_observations(x_obs_norm, n_post_samples, n_obs, n_obs_time_steps)

    with torch.no_grad():
        model.to(device)
        model.eval()
        # Reverse iterate over diffusion steps.
        for t in tqdm(reversed(range(1, diffusion_steps + 1)), total=diffusion_steps):
            t_tensor = torch.full((n_post_samples, 1), diffusion_time[t],
                                  dtype=torch.float32, device=device)
            scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                              x_expanded=x_expanded, n_post_samples=n_post_samples, n_obs=n_obs,
                                              conditions_exp=conditions_exp, pareto_smooth_sum_dict=pareto_smooth_sum_dict)
            theta = euler_maruyama_step(model, theta, score=scores, t=diffusion_time[t],
                                        dt=diffusion_time[t] - diffusion_time[t - 1])
            if torch.isnan(theta).any():
                print("NaNs in theta")
                break

        # Denormalize and reshape theta.
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().numpy().reshape(n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().numpy().reshape(n_post_samples, n_obs, model.prior.n_params_local)
    return theta


def adaptive_sampling(model, x_obs,
                      n_post_samples=1,
                      conditions=None,
                      e_abs: float = 0.00078,
                      e_rel: float = 1e-3,
                      h_init: float = 0.01,
                      r: float = 0.9,
                      adapt_safety: float = 0.9,
                      max_steps: int = 2000,
                      t_start: float = 1.0,
                      t_end: float = 0.,
                      pareto_smooth_sum_dict=None,
                      random_seed=None,
                      device=None,):
    """
    Generate posterior samples using an adaptive, Heun-style sampling scheme. Expects un-normalized observations.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: (n_post_samples, model.prior.n_params_global)
            - Local: (n_post_samples, n_obs, model.prior.n_params_local)
    """

    # Prepare observations and initialize parameters.
    x_obs_norm, n_obs, n_obs_time_steps = prepare_observations(x_obs, model, device)
    theta, conditions_exp = initialize_parameters(n_post_samples, n_obs, model, conditions, device, random_seed)
    x_expanded = expand_observations(x_obs_norm, n_post_samples, n_obs, n_obs_time_steps)

    current_t = torch.tensor(t_start, dtype=torch.float32, device=device)
    if isinstance(t_end, float):
        t_end = torch.tensor(t_end, dtype=torch.float32, device=device)
    h = torch.tensor(h_init, dtype=torch.float32, device=device)
    e_abs_tensor = torch.full((n_post_samples, 1), e_abs, dtype=torch.float32, device=device)
    theta_prev = theta

    with torch.no_grad():
        model.to(device)
        model.eval()
        for steps in tqdm(range(max_steps)):
            z = torch.randn_like(theta, dtype=torch.float32, device=device)  # same noise for both steps
            t_tensor = torch.full((n_post_samples, 1), current_t, dtype=torch.float32, device=device)

            # Euler-Maruyama step.
            scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                              x_expanded=x_expanded, n_post_samples=n_post_samples, n_obs=n_obs,
                                              conditions_exp=conditions_exp, pareto_smooth_sum_dict=pareto_smooth_sum_dict)
            theta_eul = euler_maruyama_step(model, theta, score=scores, t=t_tensor, dt=h, noise=z)

            # Heun-style improved step.
            t_mid = t_tensor - h
            scores_mid = eval_compositional_score(model=model, theta=theta_eul, diffusion_time=t_mid, x_expanded=x_expanded,
                                                  n_post_samples=n_post_samples, n_obs=n_obs, conditions_exp=conditions_exp,
                                                  pareto_smooth_sum_dict=pareto_smooth_sum_dict)
            theta_eul_mid = euler_maruyama_step(model, theta_eul, score=scores_mid, t=t_mid, dt=h, noise=z)

            # Average the two steps.
            theta_eul_sec = 0.5 * (theta_eul + theta_eul_mid)

            # Error estimation.
            delta = torch.maximum(e_abs_tensor,
                                  e_rel * torch.maximum(torch.abs(theta_eul), torch.abs(theta_prev)))
            error_norm = torch.linalg.matrix_norm((theta_eul - theta_eul_sec) / delta, ord=torch.inf)
            E2 = (1 / np.sqrt(n_obs * n_post_samples)) * error_norm.item()

            # Accept/reject step.
            if E2 <= 1.0:
                theta = theta_eul_sec
                current_t = current_t - h
                theta_prev = theta_eul
            elif np.isnan(E2):
                print("NaNs in E2")
                break
            elif torch.isnan(theta).any():
                print("NaNs in theta")
                theta = theta_prev
                break

            if E2 == 0:
                E2 = 1e-10
            h = torch.minimum(current_t - t_end, h * adapt_safety * (E2 ** (-r)))
            if current_t <= t_end:
                break

        print(f"Finished after {steps+1} ({(steps+1)*2} score evals) steps at time {current_t}.")
        # Denormalize and reshape theta.
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().numpy().reshape(n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().numpy().reshape(n_post_samples, n_obs, model.prior.n_params_local)
    return theta


def probability_ode_solving(model, x_obs, n_post_samples=1, conditions=None,
                            t_end=0, pareto_smooth_sum_dict=None, random_seed=None, device=None):
    """
    Solve the probability ODE (Song et al., 2021) to generate posterior samples. Expects un-normalized observations.
    Solving the ODE can be done either jointly for all posterior samples or separately for each sample. If jointly, then
    the error is computed for all samples at once which might lead to less accurate results.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: (n_post_samples, model.prior.n_params_global)
            - Local: (n_post_samples, n_obs, model.prior.n_params_local)
    """
    # Prepare observations and initialize parameters.
    x_obs_norm, n_obs, n_obs_time_steps = prepare_observations(x_obs, model, device)
    theta, conditions_exp = initialize_parameters(n_post_samples, n_obs, model, conditions, device, random_seed)
    x_expanded = expand_observations(x_obs_norm, n_post_samples, n_obs, n_obs_time_steps)

    with torch.no_grad():
        model.to(device)
        model.eval()
        def probability_ode(t, x, n_samples):
            t_tensor = torch.full((n_samples, 1), t, dtype=torch.float32, device=device)
            if conditions is None:
                x_torch = torch.tensor(x.reshape(n_samples, model.prior.n_params_global),
                                       dtype=torch.float32, device=device)
            else:
                x_torch = torch.tensor(x.reshape(n_samples, n_obs, model.prior.n_params_local),
                                       dtype=torch.float32, device=device)
            scores = eval_compositional_score(model=model, theta=x_torch, diffusion_time=t_tensor, x_expanded=x_expanded,
                                              n_post_samples=n_samples, n_obs=n_obs, conditions_exp=conditions_exp,
                                              pareto_smooth_sum_dict=pareto_smooth_sum_dict)
            t_exp = t_tensor if conditions is None else t_tensor.unsqueeze(1).expand(-1, n_obs, -1)
            f, g = model.sde.get_f_g(x=x_torch, t=t_exp)
            drift = f - 0.5 * torch.square(g) * scores
            if conditions is None:
                return drift.cpu().numpy().reshape(n_samples * model.prior.n_params_global)
            return drift.cpu().numpy().reshape(n_samples * n_obs * model.prior.n_params_local)

        # Solve the ODE for all posterior samples at once.
        if conditions is None:
            x0 = theta.cpu().numpy().reshape(n_post_samples * model.prior.n_params_global)
        else:
            x0 = theta.cpu().numpy().reshape(n_post_samples * n_obs * model.prior.n_params_local)
        sol = solve_ivp(probability_ode, t_span=[1, t_end], y0=x0, args=(n_post_samples,),
                            method='RK45', t_eval=[t_end])
        print(f'ODE solved: {sol.success} with #score evals: {sol.nfev}')

        if conditions is None:
            theta = torch.tensor(sol.y[:, -1].reshape(n_post_samples, model.prior.n_params_global),
                                 dtype=torch.float32, device=device)
        else:
            theta = torch.tensor(sol.y[:, -1].reshape(n_post_samples, n_obs, model.prior.n_params_local),
                                 dtype=torch.float32, device=device)
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().numpy().reshape(n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().numpy().reshape(n_post_samples, n_obs, model.prior.n_params_local)
    return theta

def langevin_sampling(model, x_obs, n_post_samples, conditions=None,
                      diffusion_steps=1000, langevin_steps=10, t_end=0,
                      pareto_smooth_sum_dict=None,
                      random_seed=None, device=None):
    """
    Annealed Langevin Dynamics sampling. Expects un-normalized observations.

    Parameters:
        model: Score-based model.
        x_obs: Observations (assumed to be a 2D tensor or array of shape (n_obs, d)).
        n_post_samples: Number of posterior samples.
        conditions: Conditioning parameters (if None, global sampling is performed).
        diffusion_steps: Number of diffusion steps.
        langevin_steps: Number of inner Langevin steps per diffusion time.
        t_end: End time for diffusion
        pareto_smooth_sum_dict: Dictionary with keys 'tail_fraction' and 'alpha' for Pareto smoothing. Default: None.
        random_seed: Optional random seed for reproducibility.
        device: Computation device.

    Returns:
        theta: Posterior samples as a NumPy array.
            - Global: shape (n_post_samples, model.prior.n_params_global)
            - Local: shape (n_post_samples, n_obs * model.prior.n_params_local)
    """
    # Prepare observations and initialize parameters.
    x_obs_norm, n_obs, n_obs_time_steps = prepare_observations(x_obs, model, device)
    theta, conditions_exp = initialize_parameters(n_post_samples, n_obs, model, conditions, device, random_seed)
    x_expanded = expand_observations(x_obs_norm, n_post_samples, n_obs, n_obs_time_steps)
    diffusion_time = generate_diffusion_time(size=diffusion_steps + 1, epsilon=t_end, device=device)

    with torch.no_grad():
        model.to(device)
        model.eval()
        # Annealed Langevin dynamics: iterate over time steps in reverse
        for t in tqdm(reversed(range(1, diffusion_steps + 1)), total=diffusion_steps):
            t_tensor = torch.full((n_post_samples, 1), diffusion_time[t],
                                  dtype=torch.float32, device=device)
            step_size = diffusion_time[t] - diffusion_time[t - 1]

            for _ in range(langevin_steps):
                eps = torch.randn_like(theta, dtype=torch.float32, device=device)

                # Compute model scores
                scores = eval_compositional_score(model=model, theta=theta, diffusion_time=t_tensor,
                                                  x_expanded=x_expanded, n_post_samples=n_post_samples, n_obs=n_obs,
                                                  conditions_exp=conditions_exp,
                                                  pareto_smooth_sum_dict=pareto_smooth_sum_dict)
                # Langevin update step
                theta = theta + (step_size / 2) * scores + torch.sqrt(step_size) * eps

            if torch.isnan(theta).any():
                print("NaNs in theta")
                break

        # Denormalize theta using the prior's statistics.
        if conditions is None:
            theta = model.prior.denormalize_theta(theta, global_params=True)
            theta = theta.detach().numpy().reshape(n_post_samples, model.prior.n_params_global)
        else:
            theta = model.prior.denormalize_theta(theta, global_params=False)
            theta = theta.detach().numpy().reshape(n_post_samples, n_obs, model.prior.n_params_local)
    return theta
