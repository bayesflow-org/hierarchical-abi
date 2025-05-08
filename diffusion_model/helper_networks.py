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

#############
# taken from: https://github.com/juho-lee/set_transformer/blob/master/

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


### implementation taken from bayesflow 2.0
class SkipRecurrentNet(nn.Module):
    """
    Skip‐recurrent layer:
      - direct recurrent path
      - skip‐conv + recurrent path
      - concat their final outputs
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        recurrent_type: str = "gru",
        bidirectional: bool = True,
        input_channels: int = 64,
        skip_steps: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()

        # 1) skip‐conv: in C→C*skip_steps, kernel/stride = skip_steps, same‐padding
        self.skip_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels * skip_steps,
            kernel_size=skip_steps,
            stride=skip_steps,
            padding=skip_steps // 2,
        )

        # choose GRU or LSTM
        Rec = nn.GRU if recurrent_type.lower() == "gru" else nn.LSTM
        r_hid = hidden_dim // 2 if bidirectional else hidden_dim

        # 2) direct and skip recurrent modules
        self.recurrent = Rec(
            input_size=input_channels,
            hidden_size=r_hid,
            num_layers=1,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.skip_recurrent = Rec(
            input_size=input_channels * skip_steps,
            hidden_size=r_hid,
            num_layers=1,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_channels)
        returns: (batch, hidden_dim*2)   # concat of direct & skip summaries
        """
        # --- direct path ---
        # out_direct: (batch, seq_len, r_out)
        out_direct, _ = self.recurrent(x)
        # take last time‐step
        direct_summary = out_direct[:, -1, :]  # → (batch, hidden_dim)

        # --- skip path ---
        # to (batch, C, L) for conv1d
        skip = x.transpose(1, 2)
        skip = self.skip_conv(skip)            # → (batch, C*skip_steps, L//skip_steps)
        skip = skip.transpose(1, 2)            # → (batch, L//skip_steps, C*skip_steps)
        out_skip, _ = self.skip_recurrent(skip)
        skip_summary = out_skip[:, -1, :]      # → (batch, hidden_dim)

        # --- concat ---
        return torch.cat([direct_summary, skip_summary], dim=-1)  # (batch, hidden_dim*2)


class TimeSeriesNetwork(nn.Module):
    """
    Hybrid conv + SkipRecurrentNet → final summary_dim
    """

    def __init__(
        self,
        input_dim: int,
        summary_dim: int = 16,
        filters: int | list[int] = 32,
        kernel_sizes: int | list[int] = 3,
        strides: int | list[int] = 1,
        activation: str = "mish",
        kernel_initializer: str = "glorot_uniform",
        groups: int = None,
        recurrent_type: str = "gru",
        recurrent_dim: int = 128,
        bidirectional: bool = True,
        dropout: float = 0.,
        skip_steps: int = 4,
    ):
        super().__init__()
        self.name = "TimeSeriesNetwork"

        # ensure tuples
        if not isinstance(filters, (list, tuple)):
            filters = (filters,)
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = (kernel_sizes,)
        if not isinstance(strides, (list, tuple)):
            strides = (strides,)

        # build conv‐blocks as ModuleList
        self.conv_blocks = nn.ModuleList()
        in_channels = input_dim
        for f, k, s in zip(filters, kernel_sizes, strides):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=f,
                kernel_size=k,
                stride=s,
                padding=k // 2,
                groups=1 if groups is None else groups
            )
            # we'll lazy-initialize in forward since in_channels is dynamic
            self.conv_blocks.append(conv)

            if groups is not None:
                self.conv_blocks.append(nn.GroupNorm(num_groups=groups, num_channels=f))

            self.conv_blocks.append(get_activation(activation))

            # Update in_channels for the next layer
            in_channels = f

        # skip‐recurrent backbone
        self.recurrent = SkipRecurrentNet(
            hidden_dim=recurrent_dim,
            recurrent_type=recurrent_type,
            bidirectional=bidirectional,
            input_channels=filters[-1],
            skip_steps=skip_steps,
            dropout=dropout,
        )

        # final linear projector
        # note: SkipRecurrentNet returns size hidden_dim * 2
        self.output_projector = nn.Linear(recurrent_dim * 2, summary_dim)

        # optionally apply Glorot init
        if kernel_initializer == "glorot_uniform":
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, channels_in)
        returns: (batch, summary_dim)
        """
        # conv1d wants (B, C, L)
        x = x.transpose(1, 2)

        # apply conv_blocks
        for layer in self.conv_blocks:
            x = layer(x)

        # back to (B, L, C)
        x = x.transpose(1, 2)

        # skip‐recurrent summary
        summary = self.recurrent(x)         # → (batch, recurrent_dim*2)

        # final projection
        out = self.output_projector(summary)  # → (batch, summary_dim)
        return out
