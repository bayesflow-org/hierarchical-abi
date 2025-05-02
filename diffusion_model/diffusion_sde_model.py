import numpy as np
import torch
import torch.nn as nn

from diffusion_model.helper_functions import count_parameters
from diffusion_model.helper_functions import sech
from diffusion_model.helper_networks import MLP, FiLMResidualBlock


class ScoreModel(nn.Module):
    """
        Neural network model that computes score estimates.

        Args:
            input_dim_theta (int): Input dimension for theta.
            input_dim_x (int): Input dimension for x. (e.g., output dimension of the summary network)
            hidden_dim (int): Hidden dimension for theta network.
            n_blocks (int): Number of residual blocks.
            sde (SDE): SDE model. Needed to calculate the kernel.
            prior (Prior): Prior model. Only needed for sampling etc.
            max_number_of_obs (int): Maximal number of observations grouped together. Default is 1.
            input_dim_condition (int): Input dimension for the condition. Default is 0.
            weighting_type (str): Type of weighting to use. Default is None.
            prediction_type (str): Type of prediction to perform. Can be 'score', 'e', 'x', or 'v'.
            loss_type (str): Prediction type to compute the loss on. Can be 'e' or 'v'.
            use_film (bool): Whether to use FiLM-residual blocks. Default is False.
            dropout_rate (float): Dropout rate. Default is 0.05.
            use_spectral_norm (bool): Whether to use spectral normalization. Default is False.
            name_prefix (str): Prefix for the name of the model. Default is ''.
            time_embedding (nn.Module): Time embedding module. Default is nn.Identity.
            summary_net (nn.Module, None): Summary network. Default is nn.Identity.
            full_res_layer (bool): Whether to connect the last layer as a residual block. Default is False.
    """
    def __init__(self,
                 input_dim_theta, input_dim_x,
                 hidden_dim, n_blocks,
                 sde, prior, max_number_of_obs=1, input_dim_condition=0,
                 weighting_type=None,
                 prediction_type='v', loss_type='e',
                 use_film=False, dropout_rate=0.05, use_spectral_norm=False,
                 name_prefix='',
                 time_embedding=None, summary_net=None, full_res_layer=False):
        super().__init__()

        if prediction_type not in ['score', 'e', 'x', 'v', 'F']:
            raise ValueError("Invalid prediction type. Must be one of 'score', 'e', 'x', 'v' or 'F'.")
        if loss_type not in ['e', 'v']:
            raise ValueError("Invalid loss type. Must be one of 'e', 'v'.")
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.sde = sde
        self.use_film = use_film
        self.weighting_type = weighting_type

        self.max_number_of_obs = max_number_of_obs
        self.current_number_of_obs = max_number_of_obs
        self.amortize_n_conditions = False if max_number_of_obs == 1 else True

        self.name = (f'{name_prefix}score_model_{prediction_type}_{sde.kernel_type}_{sde.noise_schedule}'
                     f'_{weighting_type}')
        self.n_params_global = prior.n_params_global
        self.prior = prior

        if time_embedding is None:
            self.time_embedding = nn.Identity()
            time_embed_dim = 1
        else:
            self.time_embedding = time_embedding
            dummy_input = torch.zeros(1, 1)
            dummy_output = time_embedding(dummy_input)
            time_embed_dim = dummy_output.shape[1]

        if summary_net is None:
            self.summary_net = nn.Identity()
        else:
            self.summary_net = summary_net

        # Define the dimension of the conditioning vector
        cond_dim = input_dim_x + input_dim_condition + time_embed_dim

        self.full_res_layer = full_res_layer
        if full_res_layer:
            if input_dim_condition > 0:
                self.projection_layer = nn.Linear(input_dim_condition, hidden_dim)
            else:
                self.projection_layer = nn.Linear(input_dim_theta, hidden_dim)
            nn.init.zeros_(self.projection_layer.bias)

        if not self.use_film:
            # Create a sequence of residual blocks
            self.blocks = MLP(
                input_shape=input_dim_theta + cond_dim,
                widths=[hidden_dim]*n_blocks,
                dropout=dropout_rate,
                spectral_normalization=use_spectral_norm
            )
        else:
            # Create a series of FiLM-residual blocks
            self.blocks = nn.ModuleList([
                FiLMResidualBlock(in_dim=input_dim_theta if b == 0 else hidden_dim,
                                  out_dim=hidden_dim,
                                  cond_dim=cond_dim, dropout_rate=dropout_rate, use_spectral_norm=use_spectral_norm)
                for b in range(n_blocks)
            ])

        # Final layer to get back to the theta dimension
        if self.prediction_type == 'v' and self.sde.kernel_type == 'sub_variance_preserving':
            self.final_projection_linear = nn.Linear(hidden_dim, input_dim_theta * 2)  # e- and x-prediction
        else:
            self.final_projection_linear = nn.Linear(hidden_dim, input_dim_theta)
        nn.init.zeros_(self.final_projection_linear.bias)

        # Apply spectral normalization
        if use_spectral_norm:
            self.final_projection_linear = nn.utils.parametrizations.spectral_norm(self.final_projection_linear)
            self.input_layer = nn.utils.parametrizations.spectral_norm(self.input_layer)
            if full_res_layer:
                self.projection_layer = nn.utils.parametrizations.spectral_norm(self.projection_layer)

        count_parameters(self)
        print(self.name)

    def _transform_log_snr(self, log_snr):
        """Transform the log_snr to the range [-1, 1] for the diffusion process."""
        return (
            2
            * (log_snr - self.sde.log_snr_min)
            / (self.sde.log_snr_max - self.sde.log_snr_min)
            - 1
        )

    def summary_forward(self, x, chunk_size, device):
        """Forward pass through the summary network. If amortize_n_conditions, the network should be able to handle it."""
        # pass only chunks through the model
        batch_size, n_obs = x.shape[:2]
        x_list = []
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            x_temp = x[start_idx:end_idx]
            x_temp = x_temp.to(device)
            # pass through the summary network
            x_emb = self.summary_net(x_temp)
            x_list.append(x_emb)
        x_emb = torch.cat(x_list, dim=0)
        return x_emb

    def forward_global(self, theta_global, time, x, pred_score, clip_x=False, x_emb=None):
        return self.forward(theta=theta_global, time=time, x=x, x_emb=x_emb,
                            conditions=None, pred_score=pred_score, clip_x=clip_x)

    def forward(self, theta, time, x, conditions=None, pred_score=False, clip_x=False, x_emb=None):
        """
        Forward pass of the ScoreModel.

        Args:
            theta (torch.Tensor): Input theta tensor of shape (batch_size, input_dim_theta).
            time (torch.Tensor): Input time tensor of shape (batch_size, 1).
            x (torch.Tensor or None): Input x tensor of shape (batch_size, input_dim_x).
            x_emb (torch.Tensor or None): Input x tensor of shape (batch_size, input_dim_x). Can be None.
            conditions (torch.Tensor or None): Input condition tensor of shape (batch_size, input_dim_condition). Can be None.
            pred_score (bool): Whether to predict the score (True) or whatever is specified in prediction_type (False).
            clip_x (bool): Whether to clip the output x.

        Returns:
            torch.Tensor: Output of the network (dependent on prediction_type) or the score of shape (batch_size, input_dim_theta).
        """
        # Compute the summary of x, in the hierarchical case this is just the identity as x is already embedded
        if x_emb is None:
            x_emb = self.summary_net(x)

        # Compute a time embedding (shape: [batch, time_embed_dim])
        log_snr = self.sde.get_snr(t=time)
        if isinstance(self.time_embedding, nn.Identity):
            t_emb = self._transform_log_snr(log_snr)
        else:
            t_emb = self.time_embedding(log_snr)

        # Form the conditioning vector. If conditions is None, only x and time are used.
        if conditions is not None:
            cond = torch.cat([x_emb, conditions, t_emb], dim=-1)
            h_update = conditions
        else:
            cond = torch.cat([x_emb, t_emb], dim=-1)
            h_update = theta.clone()

        # initial input
        if self.use_film:
            # Pass through each block, injecting the same cond at each layer
            h = theta
            for block in self.blocks:
                h = block(h, cond)
        else:
            # Pass through each block, concatenating the cond at the beginning
            h_cond = torch.cat([theta, cond], dim=-1)
            h = self.blocks(h_cond)

        # If conditions are provided, compute an update from them else use skip connection
        if self.full_res_layer:
            h = h + self.projection_layer(h_update)
        h = self.final_projection_linear(h)

        if pred_score:
            return self.convert_to_score(pred=h, z=theta, log_snr=log_snr, clip_x=clip_x)

        # convert the prediction to e or v
        return self.convert_to_output(pred=h, z=theta, log_snr=log_snr, clip_x=clip_x)


    def convert_to_x(self, pred, z, alpha, sigma, log_snr, clip_x):
        if self.prediction_type == 'v':
            # convert v into x
            if self.sde.kernel_type == 'variance_preserving':
                x = alpha * z - sigma * pred
            elif self.sde.kernel_type == 'sub_variance_preserving':
                e_tilde, x_tilde = torch.tensor_split(pred, 2, dim=-1)
                x = torch.square(sigma) * x_tilde + alpha * (z - sigma * e_tilde)
            else:
                raise ValueError('For v-prediction unknown sde.kernel_type {}'.format(self.sde.kernel_type))
        elif self.prediction_type == 'e':
            # prediction is the error
            x = (z - sigma * pred) / alpha
        elif self.prediction_type == 'x':
            x = pred
        elif self.prediction_type == 'score':
            x = (z + sigma ** 2 * pred) / alpha
        elif self.prediction_type == 'F':  # EDM
            sigma_data = 1.
            x1 = (sigma_data ** 2 * alpha) / (torch.exp(-log_snr) + sigma_data ** 2)
            x2 = torch.exp(-log_snr / 2) * sigma_data / torch.sqrt(torch.exp(-log_snr) + sigma_data ** 2)
            x = x1 * z + x2 * pred
        else:
            raise ValueError("Invalid prediction type. Must be one of 'score', 'e', 'x', or 'v'.")
        if clip_x:
            x = torch.clamp(x, -5, 5)
        return x

    def convert_to_output(self, pred, z, log_snr, clip_x):
        if self.loss_type == 'v':
            alpha, sigma = self.sde.kernel(log_snr=log_snr)
            x_pred = self.convert_to_x(pred=pred, z=z, alpha=alpha, sigma=sigma, log_snr=log_snr, clip_x=clip_x)
            out = (alpha * z - x_pred) / sigma
        elif self.loss_type == 'e':
            alpha, sigma = self.sde.kernel(log_snr=log_snr)
            x_pred = self.convert_to_x(pred=pred, z=z, alpha=alpha, sigma=sigma, log_snr=log_snr, clip_x=clip_x)
            out = (z - x_pred * alpha) / sigma
        else:
            raise ValueError('Unknown loss type')
        return out

    def convert_to_score(self, pred, z, log_snr, clip_x):
        alpha, sigma = self.sde.kernel(log_snr=log_snr)
        x_pred = self.convert_to_x(pred=pred, z=z, alpha=alpha, sigma=sigma, log_snr=log_snr, clip_x=clip_x)
        score = (alpha * x_pred - z) / torch.square(sigma)
        return score


class HierarchicalScoreModel(nn.Module):
    """
        Neural network model that computes score estimates for a hierarchical model.

        Args:
            input_dim_theta_global (int): Input dimension for global theta.
            input_dim_theta_local (int): Input dimension for local theta.
            input_dim_x_global (int): Input dimension for x. (e.g., output dimension of the summary network)
            input_dim_x_local (int): Input dimension for x. (e.g., output dimension of the summary network)
            hidden_dim (int): Hidden dimension for theta network.
            n_blocks (int): Number of residual blocks.
            sde (SDE): SDE model. Needed to calculate the kernel.
            prior (Prior): Prior model. Only needed for sampling etc.
            max_number_of_obs (int): Maximal number of observations grouped together. Default is 1.
            weighting_type (str): Type of weighting to use. Default is None.
            prediction_type (str): Type of prediction to perform. Can be 'score', 'e', 'x', or 'v'.
            loss_type (str): Prediction type to compute the loss on. Can be 'e' or 'v'.
            use_film (bool): Whether to use FiLM-residual blocks. Default is False.
            dropout_rate (float): Dropout rate. Default is 0.05.
            use_spectral_norm (bool): Whether to use spectral normalization. Default is False.
            name_prefix (str): Prefix for the name of the model. Default is ''.
            time_embedding_global (nn.Module): Time embedding module for the global network. Default is nn.Identity.
            time_embedding_local (nn.Module): Time embedding module for the local network. Default is time_embedding_global.
            summary_net (nn.Module, None): Summary network. Default is nn.Identity.
            global_summary_net (nn.Module, None): Summary network for the global part. It is applied after the other
            summary net. Default is nn.Identity.
            full_res_layer (bool): Whether to connect the last layer as residual block. Default is False.
            split_summary_vector (bool): Whether to split the summary vector into two parts for local and global models. Default is False.
    """
    def __init__(self,
                 input_dim_theta_global, input_dim_theta_local, input_dim_x_global, input_dim_x_local,
                 hidden_dim, n_blocks,
                 sde, prior, max_number_of_obs=1,
                 weighting_type=None,
                 prediction_type='v', loss_type='e',
                 use_film=False, dropout_rate=0.05, use_spectral_norm=False,
                 name_prefix='',
                 time_embedding_global=None, time_embedding_local=None,
                 summary_net=None, global_summary_net=None,
                 full_res_layer=False, split_summary_vector=False):
        super().__init__()
        if prediction_type not in ['score', 'e', 'x', 'v']:
            raise ValueError("Invalid prediction type. Must be one of 'score', 'e', 'x', or 'v'.")
        if loss_type not in ['e', 'v']:
            raise ValueError("Invalid loss type. Must be one of 'e', 'v'.")
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.sde = sde
        self.use_film = use_film
        self.weighting_type = weighting_type

        self.max_number_of_obs = max_number_of_obs
        self.current_number_of_obs = max_number_of_obs
        self.amortize_n_conditions = False if max_number_of_obs == 1 else True

        self.name = (f'{name_prefix}hierarchical_score_model_{prediction_type}_{sde.kernel_type}_{sde.noise_schedule}'
                     f'_{weighting_type}')
        self.n_params_global = prior.n_params_global
        self.prior = prior

        if time_embedding_global is None:
            time_embedding_global = nn.Identity()
        if time_embedding_local is None:
            time_embedding_local = time_embedding_global

        if summary_net is None:
            self.summary_net = nn.Identity()
        else:
            self.summary_net = summary_net
        self.split_summary_vector = split_summary_vector

        self.n_params_global = input_dim_theta_global
        self.global_model = ScoreModel(
            input_dim_theta=input_dim_theta_global,
            input_dim_x=input_dim_x_global if not self.split_summary_vector else int(np.ceil(input_dim_x_global / 2)),
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            sde=sde,
            prior=prior,
            max_number_of_obs=1,
            input_dim_condition=0,
            weighting_type=weighting_type,
            prediction_type=prediction_type,
            loss_type=loss_type,
            use_film=use_film,
            dropout_rate=dropout_rate,
            use_spectral_norm=use_spectral_norm,
            name_prefix=name_prefix+'global_',
            time_embedding=time_embedding_global,
            summary_net=global_summary_net,
            full_res_layer=full_res_layer
        )
        self.n_params_local = input_dim_theta_local
        self.local_model = ScoreModel(
            input_dim_theta=input_dim_theta_local,
            input_dim_x=input_dim_x_local if not self.split_summary_vector else int(np.floor(input_dim_x_local / 2)),
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            sde=sde,
            prior=prior,
            max_number_of_obs=1,
            input_dim_condition=input_dim_theta_global,
            weighting_type=weighting_type,
            prediction_type=prediction_type,
            loss_type=loss_type,
            use_film=use_film,
            dropout_rate=dropout_rate,
            use_spectral_norm=use_spectral_norm,
            name_prefix=name_prefix+'local_',
            time_embedding=time_embedding_local,
            summary_net=None,
            full_res_layer=full_res_layer
        )

        count_parameters(self)
        print(self.name)

    def forward(self, theta_global, theta_local, time, x, pred_score, clip_x=False):  # __call__ method for the model
        """Forward pass through the global and local model. This usually only used during training."""
        x_emb = self.summary_forward(x, chunk_size=x.shape[0], device=theta_global.device)
        if self.split_summary_vector:
            # split vector at last dimension
            local_summary = x_emb[..., :x_emb.shape[-1] // 2]
            global_summary = x_emb[..., x_emb.shape[-1] // 2:]
        else:
            global_summary = x_emb
            local_summary = x_emb

        if self.amortize_n_conditions:
            global_out = self.global_model.forward(theta=theta_global, time=time, x=global_summary,
                                                   conditions=None, pred_score=pred_score, clip_x=clip_x)

            # Flatten the observation dimension for theta_local and local_summary.
            batch_size, n_obs = local_summary.shape[:2]
            theta_local_flat = theta_local.contiguous().view(batch_size * n_obs, *theta_local.shape[2:])
            x_emb_flat = local_summary.contiguous().view(batch_size * n_obs, *local_summary.shape[2:])

            # Expand time and theta_global so they match the flattened observations.
            # time: from [batch_size, time_dim] -> [batch_size, n_obs, time_dim] -> [batch_size * n_obs, time_dim]
            time_expanded = time.unsqueeze(1).expand(batch_size, n_obs, 1).contiguous().view(batch_size * n_obs, 1)
            # theta_global: from [batch_size, global_theta_dim] -> [batch_size, n_obs, global_theta_dim] -> [batch_size * n_obs, global_theta_dim]
            theta_global_expanded = theta_global.unsqueeze(1).expand(batch_size, n_obs, theta_global.shape[-1]).contiguous().view(
                batch_size * n_obs, theta_global.shape[-1])

            # Pass all observations at once through the local model.
            local_out_flat = self.local_model.forward(
                theta=theta_local_flat,
                time=time_expanded,
                x=x_emb_flat,
                conditions=theta_global_expanded,
                pred_score=pred_score,
                clip_x=clip_x
            )

            # Reshape back to [batch_size, n_obs, ...]
            local_out = local_out_flat.contiguous().view(batch_size, n_obs, *theta_local.shape[2:])
        else:
            global_out = self.global_model.forward(theta=theta_global, time=time, x=global_summary,
                                                   conditions=None, pred_score=pred_score, clip_x=clip_x)
            local_out = self.local_model.forward(theta=theta_local, time=time, x=local_summary,
                                                 conditions=theta_global, pred_score=pred_score, clip_x=clip_x)
        return global_out, local_out

    def forward_global(self, theta_global, time, x, pred_score, x_emb=None, clip_x=False):
        """Forward pass through the global model. Usually we want the score, not the predicting task from training."""
        if x_emb is None:
            x_emb = self.summary_forward(x, chunk_size=x.shape[0], device=theta_global.device)
        if self.split_summary_vector:
            # split vector at last dimension
            global_summary = x_emb[..., x_emb.shape[-1] // 2:]
        else:
            global_summary = x_emb
        global_out = self.global_model.forward(theta=theta_global, time=time, x=global_summary,
                                               conditions=None, pred_score=pred_score, clip_x=clip_x)
        return global_out

    def forward_local(self, theta_local, theta_global, time, x, pred_score, x_emb=None, clip_x=False):
        """Forward pass through the local model. Usually we want the score, not the predicting task from training."""
        if x_emb is None:
            x_emb = self.summary_forward(x, chunk_size=x.shape[0], device=theta_global.device)
        if self.split_summary_vector:
            # split vector at the last dimension
            local_summary = x_emb[..., :x_emb.shape[-1] // 2]
        else:
            local_summary = x_emb

        # during training, we looped over observations, here we expect that only one is passed
        if self.amortize_n_conditions:
            if local_summary.shape[1] > 1:
                batch_size, n_obs = local_summary.shape[:2]
                local_summary = local_summary.contiguous().view(batch_size*n_obs, *local_summary.shape[2:])
            else:
                local_summary = local_summary.squeeze(1)
        local_out = self.local_model.forward(theta=theta_local, time=time, x=local_summary,
                                             conditions=theta_global, pred_score=pred_score, clip_x=clip_x)
        return local_out

    def summary_forward(self, x, chunk_size, device):
        """Forward pass through the summary network. This network is always applied to the single input x."""
        # pass only chunks through the model
        batch_size, n_obs = x.shape[:2]
        x_list = []
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            x_temp = x[start_idx:end_idx]
            x_temp = x_temp.to(device)
            # pass through the summary network
            if self.amortize_n_conditions:
                # reshape obs such that the summary can work with it
                x_temp = x_temp.contiguous().view(batch_size * n_obs, *x_temp.shape[2:])
                x_emb = self.summary_net(x_temp)
                x_emb = x_emb.contiguous().view(batch_size, n_obs, *x_emb.shape[1:])
            else:
                x_emb = self.summary_net(x_temp)
            x_list.append(x_emb)
        x_emb = torch.cat(x_list, dim=0)
        return x_emb


class SDE:
    def __init__(self, kernel_type='variance_preserving', noise_schedule='cosine',
                 log_snr_min=-15, log_snr_max=15, s_shift_cosine=0.):
        if kernel_type not in ['variance_preserving', 'sub_variance_preserving']:
            raise ValueError("Invalid kernel type.")
        if noise_schedule not in ['linear', 'cosine', 'flow_matching', 'edm-training', 'edm-sampling']:
            raise ValueError("Invalid noise schedule.")

        if noise_schedule == 'flow_matching' and kernel_type == 'variance_preserving':
            raise ValueError("Variance-preserving kernel is not compatible with 'flow_matching' noise schedule.")

        if noise_schedule == 'edm-training':
            print('Remember to switch the noise schedule for sampling!')

        self.kernel_type = kernel_type
        self.noise_schedule = noise_schedule
        self.s_shift_cosine = s_shift_cosine

        # needed for schedule
        self._log_snr_min = log_snr_min
        self._log_snr_max = log_snr_max

        # for edm: lambda \in [− log σ^2_max, − log σ^2_min]
        self.sigma_max = torch.tensor(80)
        self.sigma_min = torch.tensor(0.002)
        self.p_mean = -1.2
        self.p_std = 1.2
        self.rho = 7
        if noise_schedule in ['edm-training', 'edm-sampling']:
            self._log_snr_min = -2 * torch.log(self.sigma_max)
            self._log_snr_max = -2 * torch.log(self.sigma_min)
            self.t_min = self.get_t_from_snr(self._log_snr_max)
            self.t_max = self.get_t_from_snr(self._log_snr_min)
            print(f"Using EDM noise schedule with log_snr_min: {self._log_snr_min}, log_snr_max: {self._log_snr_max}")
        else:
            self.t_min = self.get_t_from_snr(torch.tensor(self._log_snr_max))
            self.t_max = self.get_t_from_snr(torch.tensor(self._log_snr_min))

        print(f"Kernel type: {self.kernel_type}, noise schedule: {self.noise_schedule}")
        print(f"t_min: {self.t_min}, t_max: {self.t_max}")
        print('alpha, sigma:',
              self.kernel(log_snr=self.get_snr(t=0)),
              self.kernel(log_snr=self.get_snr(t=1)))

    @property
    def log_snr_min(self):
        return self._log_snr_min

    @log_snr_min.setter
    def log_snr_min(self, value):
        self._log_snr_min = value
        # Update t_max because it depends on log_snr_min.
        self.t_max = self.get_t_from_snr(torch.tensor(self._log_snr_min))

    @property
    def log_snr_max(self):
        return self._log_snr_max

    @log_snr_max.setter
    def log_snr_max(self, value):
        self._log_snr_max = value
        # Update t_min because it depends on log_snr_max.
        self.t_min = self.get_t_from_snr(torch.tensor(self._log_snr_max))

    def kernel(self, log_snr):
        if self.kernel_type == 'variance_preserving':
            return self._variance_preserving_kernel(log_snr=log_snr)
        if self.kernel_type == 'sub_variance_preserving':
            return self._sub_variance_preserving_kernel(log_snr=log_snr)
        raise ValueError("Invalid kernel type.")

    def grad_log_kernel(self, x, x0, t):  # t is not truncated
        if self.kernel_type == 'variance_preserving':
            return self._grad_log_variance_preserving_kernel(x, x0, t)
        if self.kernel_type == 'sub_variance_preserving':
            return self._grad_log_sub_variance_preserving_kernel(x, x0, t)
        raise ValueError("Invalid kernel type.")

    def get_snr(self, t):  # t is not truncated
        t_trunc = self.t_min + (self.t_max - self.t_min) * t
        if self.noise_schedule == 'linear':
            return -torch.log(torch.exp(torch.square(t_trunc)) - 1)
        if self.noise_schedule == 'cosine':  # this is usually used with variance_preserving
            return -2 * torch.log(torch.tan(torch.pi * t_trunc / 2)) + 2 * self.s_shift_cosine
        if self.noise_schedule == 'flow_matching':  # this usually used with sub_variance_preserving
            return 2 * torch.log((1 - t_trunc) / t_trunc)
        if self.noise_schedule == 'edm-training':
            # training
            dist = torch.distributions.Normal(loc=-2*self.p_mean, scale=2*self.p_std)
            snr = dist.icdf(t_trunc)
            snr = snr.clamp(min=self._log_snr_min.to(snr.device), max=self._log_snr_max.to(snr.device))
            return snr
        if self.noise_schedule == 'edm-sampling':
            # sampling
            snr = -2 * self.rho * torch.log(self.sigma_max ** (1/self.rho) + (1 - t_trunc) * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho)))
            return snr
        raise ValueError("Invalid 'noise_schedule'.")

    def get_t_from_snr(self, snr):  # not truncated
        # Invert the noise scheduling to recover t
        if self.noise_schedule == 'linear':
            # SNR = -log(exp(t^2) - 1)
            # => t = sqrt(log(1 + exp(-snr)))
            return torch.sqrt(torch.log(1 + torch.exp(-snr)))
        if self.noise_schedule == 'cosine':
            # SNR = -2 * log(tan(pi*t/2))
            # => t = 2/pi * arctan(exp(-snr/2))
            return 2 / torch.pi * torch.atan(torch.exp((2 * self.s_shift_cosine - snr) / 2))
        if self.noise_schedule == 'flow_matching':
            # SNR = 2 * log((1-t)/t)
            # => t = 1 / (1 + exp(snr/2))
            return 1 / (1 + torch.exp(snr / 2))
        if self.noise_schedule == 'edm-training':
            # SNR = dist.icdf(t_trunc)
            # => t = dist.cdf(snr)
            dist = torch.distributions.Normal(loc=-2*self.p_mean, scale=2*self.p_std)
            return dist.cdf(snr)
        if self.noise_schedule == 'edm-sampling':
            # SNR = -2 * rho * log(sigma_max ** (1/rho) + (1 - t) * (sigma_min ** (1/rho) - sigma_max ** (1/rho)))
            # => t = 1 - ((torch.exp(-snr/(2*rho)) - sigma_max ** (1/rho)) / (sigma_min ** (1/rho) - sigma_max ** (1/rho)))
            return 1 - ((torch.exp(-snr/(2*self.rho)) - self.sigma_max ** (1/self.rho)) / (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho)))
        raise ValueError("Invalid 'noise_schedule'.")

    def _get_snr_derivative(self, t):  # t is not truncated
        """
        Compute d/dt log(1 + e^(-snr(t))) for the truncated schedules.
        """
        # Compute the truncated time t_trunc
        t_trunc = self.t_min + (self.t_max - self.t_min) * t
        # Compute snr(t) using the existing function
        snr = self.get_snr(t=t)

        # Compute d/dx snr(x) based on the noise schedule
        if self.noise_schedule == 'linear':
            # d/dx snr(x) = - 2*x*exp(x^2) / (exp(x^2) - 1)
            dsnr_dx = - (2 * t_trunc * torch.exp(t_trunc**2)) / (torch.exp(t_trunc**2) - 1)
        elif self.noise_schedule == 'cosine':
            # d/dx snr(x) = -2*pi/sin(pi*x)
            dsnr_dx = - (2 * torch.pi) / torch.sin(torch.pi * t_trunc)
        elif self.noise_schedule == 'flow_matching':
            # d/dx snr(x) = -2/(x*(1-x))
            dsnr_dx = - 2 / (t_trunc * (1 - t_trunc))
        elif self.noise_schedule == 'edm-training':
            raise ValueError("EDM-training noise schedule should be used for training only.")
        elif self.noise_schedule == 'edm-sampling':
            # SNR = -2*rho*log(s_max + (1 - x)*(s_min - s_max))
            s_max = self.sigma_max ** (1 / self.rho)
            s_min = self.sigma_min ** (1 / self.rho)
            u = s_max + (1 - t_trunc) * (s_min - s_max)
            # d/dx snr = 2*rho*(s_min - s_max) / u
            dsnr_dx = 2 * self.rho * (s_min - s_max) / u
        else:
            raise ValueError("Invalid 'noise_schedule'.")

        # Chain rule: d/dt snr(t) = d/dx snr(x) * (t_max - t_min)
        dsnr_dt = dsnr_dx * (self.t_max - self.t_min)

        # f(t) = log(1 + e^{-snr(t)})  =>  f'(t) = - (e^{-snr}/(1+e^{-snr})) * dsnr_dt
        factor = torch.exp(-snr) / (1 + torch.exp(-snr))
        return -factor * dsnr_dt

    def _beta_integral(self, t):  # t is not truncated
        """Song et al. (2021) defined this as integral over the beta, here we express is equivalently via the snr"""
        return torch.log(1 + torch.exp(-self.get_snr(t)))

    @staticmethod
    def _variance_preserving_kernel(log_snr):
        """
        Computes the variance-preserving kernel p(x_t | x_0) for the diffusion process.

        Args:
            log_snr (torch.Tensor): The log of the signal-to-noise ratio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation of the kernel at time t.
        """
        alpha_t = torch.sqrt(torch.sigmoid(log_snr))
        sigma_t = torch.sqrt(torch.sigmoid(-log_snr))
        return alpha_t, sigma_t

    def _grad_log_variance_preserving_kernel(self, x, x0, t):  # t is not truncated
        """
        Computes the gradient of the log probability density of the variance-preserving kernel.

        Given the kernel p(x_t|x_0) = N(x_t; mean = x0 * exp(-0.5 * I), variance = 1 - exp(-I))
        where I = ∫₀ᵗ beta(s) ds, the gradient is:

            ∇ₓ log p(x_t|x_0) = -(x_t - x0 * exp(-0.5 * I)) / (1 - exp(-I))

        Args:
            x (torch.Tensor): The current state x_t.
            x0 (torch.Tensor): The initial state.
            t (torch.Tensor): The time parameter.

        Returns:
            torch.Tensor: The gradient ∇ₓ log p(x_t|x0).
        """
        if self.noise_schedule == 'flow_matching':
            raise ValueError("Variance-preserving kernel is not compatible with 'flow_matching' noise schedule.")
        integral = self._beta_integral(t)
        mean_factor = torch.exp(-0.5 * integral)
        variance = 1 - torch.exp(-integral)         # variance for the VP kernel
        grad = -(x - mean_factor * x0) / variance
        return grad

    def _sub_variance_preserving_kernel(self, log_snr):
        """
        Computes the sub-variance-preserving kernel p(x_t | x_0) for the diffusion process.
        Args:
            t (torch.Tensor): The time at which to evaluate the kernel in [0,1]. Should be not too close to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation of the kernel at time t.
        """
        if self.noise_schedule == 'flow_matching':
            t = self.get_t_from_snr(snr=log_snr)
            x = self.t_min + (self.t_max - self.t_min) * t
            alpha_t = 1 - x
            sigma_t = x
        else:
            alpha_t = 1 / (1 + torch.exp(-log_snr))
            sigma_t = 1  - torch.square(alpha_t)
        return alpha_t, sigma_t

    def _grad_log_sub_variance_preserving_kernel(self, x, x0, t):  # t is not truncated
        """
        Computes the gradient of the log probability density of the sub-variance-preserving kernel.

        For this kernel, we use the convention that:
            mean = x0 * exp(-0.5 * I)
            "std" = 1 - exp(-I)  -> variance = (1 - exp(-I))^2
        so that:

            ∇ₓ log p(x_t|x_0) = -(x_t - x0 * exp(-0.5 * I)) / ((1 - exp(-I))^2)

        Args:
            x (torch.Tensor): The current state x_t.
            x0 (torch.Tensor): The initial state.
            t (torch.Tensor): The time parameter.

        Returns:
            torch.Tensor: The gradient ∇ₓ log p(x_t|x0).
        """
        integral = self._beta_integral(t)
        mean_factor = torch.exp(-0.5 * integral)
        variance = (1 - torch.exp(-integral))**2  # variance for the sub-VP kernel
        grad = -(x - mean_factor * x0) / variance
        return grad

    def get_f_g(self, t, x=None):  # t is not truncated
        if self.kernel_type == 'variance_preserving':
            # Compute beta(t)
            beta_t = self._get_snr_derivative(t)

            # Compute drift f(x, t) = -0.5 * beta(t) * x
            if x is not None:
                f = -0.5 * beta_t * x

            # Compute diffusion coefficient g(t) = sqrt(beta(t))
            g = torch.sqrt(beta_t)
        elif self.kernel_type == 'sub_variance_preserving':
            # Compute beta(t)
            beta_t = self._get_snr_derivative(t)

            # Compute drift f(x, t) = -0.5 * beta(t) * x
            if x is not None:
                f = -0.5 * beta_t * x

            # Compute diffusion coefficient g(t) = sqrt(beta(t) * (1 - exp(-2 \int_0^t beta(s) ds)))
            g = torch.sqrt(beta_t * (1 - torch.exp(-2 * self._beta_integral(t))))
        else:
            raise ValueError("Invalid kernel type.")
        if x is None:
            return g
        return f, g


def weighting_function(t, sde, weighting_type, prediction_type='error'):

    if prediction_type == 'score':
        # likelihood weighting, since beta(t) = g(t)^2
        g_t = sde.get_f_g(t=t)
        return torch.square(g_t)

    if weighting_type is None:
        return torch.ones_like(t)

    if weighting_type == 'likelihood_weighting':
        g_t = sde.get_f_g(t=t)
        log_snr = sde.get_snr(t=t)
        sigma_t = sde.kernel(log_snr=log_snr)[1]  # divide by sigma^2, since the loss is on the score
        return torch.square(g_t / sigma_t)

    log_snr = sde.get_snr(t=t)
    if weighting_type == 'flow_matching':
        alpha = sde.kernel(log_snr=log_snr)[0]
        if sde.noise_schedule == 'flow_matching':
            return torch.exp(-log_snr / 2)
        elif sde.noise_schedule == 'cosine':
            return torch.square(alpha * (torch.exp(-log_snr) +1)) / (2 * torch.pi * torch.cosh(-log_snr/2))  # flow matching equivalent
        else:
            raise NotImplementedError("Flow matching not implemented for this noise schedule.")

    if weighting_type == 'sigmoid':
        return torch.sigmoid(-log_snr + 2)

    if weighting_type == 'min-snr':
        gamma = 5
        return sech(log_snr / 2) * torch.minimum(torch.ones_like(log_snr), gamma * torch.exp(-log_snr))

    if weighting_type == 'edm':
        return torch.exp(-log_snr) + 1
        #sigma_data = 1.
        #return torch.exp(-log_snr) / torch.square(sigma_data) + 1

    raise ValueError("Invalid weighting...")
