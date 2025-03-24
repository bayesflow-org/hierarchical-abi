import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if input_shape != units:
            self.projector = nn.Linear(input_shape, self.units, bias=False)
            # Initialize using Xavier/Glorot uniform.
            nn.init.xavier_uniform_(self.projector.weight)
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute dense output.
        out = self.dense(x)

        # (Optional) Apply dropout.
        if self.dropout is not None:
            out = self.dropout(out)

        # If using residual connections, add the input (projected if dimensions differ)
        if self.residual:
            res = self.projector(x)
            out = out + res

        return self.activation_fn(out)


class MLP(nn.Module):  # copied from the bayesflow 2.0 implementation
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

#############
# taken from: https://github.com/juho-lee/set_transformer/blob/master/

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class ShallowSet(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128):
        super(ShallowSet, self).__init__()
        #self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                #nn.Linear(dim_hidden, num_outputs*dim_output))
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X) #.reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
