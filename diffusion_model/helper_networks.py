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


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int = 1, max_batch_size: int = 512):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        total = x.size(0)

        # Process in chunks if needed
        for i in range(0, total, self.max_batch_size):
            chunk = x[i:i + self.max_batch_size]

            if chunk.ndim == 4:
                # Rearrange: assume time is not the second dimension yet
                chunk = chunk.permute(0, 2, 1, 3)
                bs, n_time_steps, n_obs, n_features = chunk.size()
                # Reshape into 3D tensor for LSTM: combine batch and observation dimensions
                chunk = chunk.contiguous().view(bs * n_obs, n_time_steps, n_features)
                out, (h_n, c_n) = self.lstm(chunk)
                # Retrieve the last hidden state and reshape back
                h_n = h_n.contiguous().view(bs, n_obs, self.hidden_dim * self.num_layers)
            else:
                # Assume chunk is already 3D: [batch, time, features]
                out, (h_n, c_n) = self.lstm(chunk)
                h_n = h_n.contiguous().view(chunk.size(0), self.hidden_dim * self.num_layers)

            outputs.append(h_n)

        # Concatenate along the batch dimension
        return torch.cat(outputs, dim=0)


class GRUEncoder(nn.Module):
    def __init__(self, input_size, summary_dim=32, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True,
                          dropout=0.05)
        self.projector = nn.Linear(hidden_size * 2, summary_dim)
        self.name = "GRUEncoder"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)

        fwd = h_n[-2]
        bwd = h_n[-1]

        embedding = self.projector(torch.cat((fwd, bwd), dim=-1))

        return embedding #, h_n


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

#############
# taken from: https://github.com/juho-lee/set_transformer/blob/master/

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
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
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class ShallowSet(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128):
        super().__init__()
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
        X = self.enc(X)
        X = X.mean(1)
        X = self.dec(X) #.reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super().__init__()
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X)).reshape(-1, self.dim_output)
