import numpy as np
import torch
import torch.nn as nn


# Helper: map activation strings to PyTorch modules.
def get_activation(activation: str):
    act = activation.lower()
    if act == "mish":
        return nn.Mish()
    elif act == "relu":
        return nn.ReLU()
    elif act == "tanh":
        return nn.Tanh()
    # Add more activations if needed; otherwise use identity.
    else:
        return nn.Identity()


class ConfigurableHiddenBlock(nn.Module):
    def __init__(
        self,
        input_shape: int,
        units: int = 256,
        activation: str = "mish",
        kernel_initializer: str = "he_normal",
        residual: bool = True,
        dropout: float | None = 0.05,
        spectral_normalization: bool = False,
    ):
        super().__init__()
        self.units = units
        self.residual = residual
        self.kernel_initializer = kernel_initializer

        # Create the dense layer.
        self.dense = nn.Linear(input_shape, units)
        # Optionally wrap with spectral normalization.
        if spectral_normalization:
            self.dense = nn.utils.spectral_norm(self.dense)

        # Set up dropout if rate > 0.
        self.dropout = nn.Dropout(dropout) if (dropout is not None and dropout > 0.0) else None

        self.activation_fn = get_activation(activation)

        # For residual connection: projector will be created lazily if needed.
        self.projector = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute dense output.
        out = self.dense(x)
        # (Optional) Apply dropout.
        if self.dropout is not None:
            out = self.dropout(out)

        # If using residual connections, add the input (projected if dimensions differ)
        if self.residual:
            if x.shape[-1] != self.units:
                # Lazily create a linear projector if not already created.
                if self.projector is None:
                    # Create a linear mapping with no bias.
                    self.projector = nn.Linear(x.shape[-1], self.units, bias=False)
                    # Initialize using Xavier/Glorot uniform.
                    nn.init.xavier_uniform_(self.projector.weight)
                    # Register the projector as a submodule.
                    self.add_module("projector", self.projector)
                res = self.projector(x)
            else:
                res = x
            out = out + res

        return self.activation_fn(out)


class MLP(nn.Module):
    """
    A configurable MLP with optional residual connections, dropout, and spectral normalization.
    The network is built from a sequence of hidden blocks.
    """
    def __init__(
        self,
        *,
        input_shape: int,  # Input dimension for the input_shape
        depth: int | None = None,
        width: int | None = None,
        widths: list[int] | None = None,
        activation: str = "mish",
        kernel_initializer: str = "he_normal",
        residual: bool = True,
        dropout: float | None = 0.05,
        spectral_normalization: bool = False,
    ):
        super().__init__()
        # Either provide explicit widths or specify depth and width.
        if widths is not None:
            if depth is not None or width is not None:
                raise ValueError("Either specify 'widths' or 'depth' and 'width', not both.")
        else:
            # If neither is provided, use defaults.
            if depth is None or width is None:
                depth, width = 5, 256
            widths = [width] * depth

        # Build a ModuleList of hidden blocks.
        self.res_blocks = nn.ModuleList([
            ConfigurableHiddenBlock(
                input_shape=input_shape if i == 0 else widths[i - 1],
                units=w,
                activation=activation,
                kernel_initializer=kernel_initializer,
                residual=residual,
                dropout=dropout,
                spectral_normalization=spectral_normalization,
            )
            for i, w in enumerate(widths)  # Iterate over widths
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequentially pass the input through each hidden block.
        for layer in self.res_blocks:
            x = layer(x)
        return x


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, init_scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.register_buffer('W', torch.randn(embed_dim // 2) * 2 * np.pi)
        self.scale = nn.Parameter(torch.tensor(init_scale), requires_grad=True)

    def forward(self, x):
        x_proj = x * self.W * self.scale
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, dropout_rate, use_spectral_norm):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.film_gamma = nn.Linear(cond_dim, out_dim)
        self.film_beta = nn.Linear(cond_dim, out_dim)
        self.activation = nn.Mish()
        self.dropout = nn.Dropout(dropout_rate)
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

        # Zero-initialize the fc and film_beta layers so that:
        #   self.fc(h) -> 0, and self.film_beta(cond) -> 0.
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # Apply spectral normalization if specified.
        if use_spectral_norm:
            self.fc = nn.utils.parametrizations.spectral_norm(self.fc)
            if in_dim != out_dim:
                self.skip = nn.utils.parametrizations.spectral_norm(self.skip)

    def forward(self, h, cond):
        # h: [batch, in_dim], cond: [batch, cond_dim]
        x = self.fc(h)
        # Compute modulation parameters
        gamma = self.film_gamma(cond)
        beta = self.film_beta(cond)
        # Apply FiLM modulation
        x = gamma * x + beta
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm(x)
        return self.activation(x + self.skip(h))


class TimeDistributedGRU(nn.Module):
    def __init__(self, module):
        super(TimeDistributedGRU, self).__init__()
        self.module = module
        # check if module is GRU
        if isinstance(module, nn.GRU):
            pass
        else:
            raise ValueError("Module must be an instance of nn.GRU")

    def forward(self, x):
        # Assume input x is of shape (batch_size, n_obs, n_time_steps, n_features)
        batch_size, n_obs, n_time_steps, n_features = x.size(0), x.size(1), x.size(2), x.size(3)
        # Merge batch and time dimensions
        x_reshaped = x.contiguous().view(batch_size * n_obs, n_time_steps, n_features)
        # Apply the module
        _, y = self.module(x_reshaped)
        # Reshape back to (n_rnn_layers, batch_size, n_obs, hidden_dim)
        y = y.contiguous().view(y.shape[0], batch_size, n_obs, y.shape[2])
        return None, y  # to match output of GRU


class TimeDistributedLSTM(nn.Module):
    def __init__(self, module):
        super(TimeDistributedLSTM, self).__init__()
        self.module = module
        # Check that the module is an instance of nn.LSTM.
        if not isinstance(module, nn.LSTM):
            raise ValueError("Module must be an instance of nn.LSTM")

    def forward(self, x):
        # x: (batch_size, n_obs, n_time_steps, n_features)
        batch_size, n_obs, n_time_steps, n_features = x.size()
        # Merge the batch and observation dimensions
        x_reshaped = x.contiguous().view(batch_size * n_obs, n_time_steps, n_features)
        # Apply the LSTM module; LSTM returns (output, (h_n, c_n))
        _, (h_n, c_n) = self.module(x_reshaped)
        # h_n and c_n have shape (n_rnn_layers, batch_size * n_obs, hidden_dim)
        # Reshape them back to (n_rnn_layers, batch_size, n_obs, hidden_dim)
        h_n = h_n.contiguous().view(h_n.size(0), batch_size, n_obs, h_n.size(2))
        c_n = c_n.contiguous().view(c_n.size(0), batch_size, n_obs, c_n.size(2))
        return None, (h_n, c_n)


class TimeDistributedDense(nn.Module):
    def __init__(self, module):
        super(TimeDistributedDense, self).__init__()
        self.module = module

    def forward(self, x):
        # Assume input x is of shape (batch_size, n_obs, n_features)
        batch_size, n_obs, n_features = x.size(0), x.size(1), x.size(2)
        # Merge batch and time dimensions
        x_reshaped = x.contiguous().view(batch_size * n_obs, n_features)
        # Apply the module
        y = self.module(x_reshaped)
        # Reshape back to (batch_size, n_obs, hidden_dim)
        y = y.contiguous().view(batch_size, n_obs, y.shape[1])
        return y

