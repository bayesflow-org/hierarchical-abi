import numpy as np
import torch
import torch.nn as nn

from diffusion_model.helper_functions import count_parameters
from diffusion_model.helper_networks import MLP


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
            dropout_rate (float): Dropout rate. Default is 0.05.
            use_spectral_norm (bool): Whether to use spectral normalization. Default is False.
            name_prefix (str): Prefix for the name of the model. Default is ''.
            time_embedding (nn.Module): Time embedding module. Default is nn.Identity.
            summary_net (nn.Module, None): Summary network. Default is nn.Identity.
    """
    def __init__(self,
                 input_dim_theta, input_dim_x,
                 hidden_dim, n_blocks,
                 sde, prior, max_number_of_obs=1, input_dim_condition=0,
                 weighting_type=None,
                 prediction_type='v', loss_type='e',
                 dropout_rate=0.05, use_spectral_norm=False,
                 name_prefix='',
                 time_embedding=None, summary_net=None):
        super().__init__()

        if prediction_type not in ['score', 'e', 'x', 'v', 'F']:
            raise ValueError("Invalid prediction type. Must be one of 'score', 'e', 'x', 'v' or 'F'.")
        if loss_type not in ['e', 'v']:
            raise ValueError("Invalid loss type. Must be one of 'e', 'v'.")
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.sde = sde
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

        # Create a sequence of residual blocks
        self.blocks = MLP(
            input_shape=input_dim_theta + cond_dim,
            widths=[hidden_dim]*n_blocks,
            dropout=dropout_rate,
            spectral_normalization=use_spectral_norm
        )

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

        # Pass through each block, concatenating the cond at the beginning
        h_cond = torch.cat([theta, cond], dim=-1)
        h = self.blocks(h_cond)

        # If conditions are provided, compute an update from them else use skip connection
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
            dropout_rate (float): Dropout rate. Default is 0.05.
            use_spectral_norm (bool): Whether to use spectral normalization. Default is False.
            name_prefix (str): Prefix for the name of the model. Default is ''.
            time_embedding_global (nn.Module): Time embedding module for the global network. Default is nn.Identity.
            time_embedding_local (nn.Module): Time embedding module for the local network. Default is time_embedding_global.
            summary_net (nn.Module, None): Summary network. Default is nn.Identity.
            global_summary_net (nn.Module, None): Summary network for the global part. It is applied after the other
            summary net. Default is nn.Identity.
            split_summary_vector (bool): Whether to split the summary vector into two parts for local and global models. Default is False.
    """
    def __init__(self,
                 input_dim_theta_global, input_dim_theta_local, input_dim_x_global, input_dim_x_local,
                 hidden_dim, n_blocks,
                 sde, prior, max_number_of_obs=1,
                 weighting_type=None,
                 prediction_type='v', loss_type='e',
                 dropout_rate=0.05, use_spectral_norm=False,
                 name_prefix='',
                 time_embedding_global=None, time_embedding_local=None,
                 summary_net=None, global_summary_net=None,
                 split_summary_vector=False):
        super().__init__()
        if prediction_type not in ['score', 'e', 'x', 'v']:
            raise ValueError("Invalid prediction type. Must be one of 'score', 'e', 'x', or 'v'.")
        if loss_type not in ['e', 'v']:
            raise ValueError("Invalid loss type. Must be one of 'e', 'v'.")
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.sde = sde
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
            dropout_rate=dropout_rate,
            use_spectral_norm=use_spectral_norm,
            name_prefix=name_prefix+'global_',
            time_embedding=time_embedding_global,
            summary_net=global_summary_net,
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
            dropout_rate=dropout_rate,
            use_spectral_norm=use_spectral_norm,
            name_prefix=name_prefix+'local_',
            time_embedding=time_embedding_local,
            summary_net=None,
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
            current_chunk_size = end_idx - start_idx
            x_temp = x[start_idx:end_idx]
            x_temp = x_temp.to(device)
            # pass through the summary network
            if self.amortize_n_conditions:
                # reshape obs such that the summary can work with it
                x_temp = x_temp.contiguous().view(current_chunk_size * n_obs, *x_temp.shape[2:])
                x_emb = self.summary_net(x_temp)
                x_emb = x_emb.contiguous().view(current_chunk_size, n_obs, *x_emb.shape[1:])
            else:
                x_emb = self.summary_net(x_temp)
            x_list.append(x_emb)
        x_emb = torch.cat(x_list, dim=0)
        return x_emb
