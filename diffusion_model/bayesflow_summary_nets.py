from collections.abc import Sequence

import keras
import torch
import torch.nn as nn
from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs


def get_activation(name: str):
    name = name.lower()
    if name == "mish":
        return nn.Mish()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


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
        for f, k, s in zip(filters, kernel_sizes, strides):
            conv = nn.Conv1d(
                in_channels=input_dim,
                out_channels=f,
                kernel_size=k,
                stride=s,
                padding=k // 2,
            )
            # we'll lazy-initialize in forward since in_channels is dynamic
            self.conv_blocks.append(conv)

            if groups is not None:
                self.conv_blocks.append(nn.GroupNorm(num_groups=groups, num_channels=f))

            self.conv_blocks.append(get_activation(activation))

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
        return self.output_projector(summary)  # → (batch, summary_dim)


class MambaBlock(nn.Module):
    """
    Wraps the original Mamba module from, with added functionality for bidirectional processing:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

    Copyright (c) 2023, Tri Dao, Albert Gu.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        conv_dim: int,
        feature_dim: int = 16,
        expand: int = 2,
        bidirectional: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        device: str = "cuda",
        **kwargs,
    ):
        """
        A Keras layer implementing a Mamba-based sequence processing block.

        This layer applies a Mamba model for sequence modeling, preceded by a
        convolutional projection and followed by layer normalization.

        Parameters
        ----------
        state_dim : int
            The dimension of the state space in the Mamba model.
        conv_dim : int
            The dimension of the convolutional layer used in Mamba.
        feature_dim : int, optional
            The feature dimension for input projection and Mamba processing (default is 16).
        expand : int, optional
            Expansion factor for Mamba's internal dimension (default is 1).
        dt_min : float, optional
            Minimum delta time for Mamba (default is 0.001).
        dt_max : float, optional
            Maximum delta time for Mamba (default is 0.1).
        device : str, optional
            The device to which the Mamba model is moved, typically "cuda" or "cpu" (default is "cuda").
        **kwargs :
            Additional keyword arguments passed to the `keras.layers.Layer` initializer.
        """

        super().__init__(**keras_kwargs(kwargs))

        if keras.backend.backend() != "torch":
            raise RuntimeError("Mamba is only available using torch backend.")

        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError("Could not import Mamba. Please install it via `pip install mamba-ssm`") from e

        self.bidirectional = bidirectional

        self.mamba = Mamba(
            d_model=feature_dim, d_state=state_dim, d_conv=conv_dim, expand=expand, dt_min=dt_min, dt_max=dt_max
        ).to(device)

        #self.input_projector = keras.layers.Conv1D(
        #    feature_dim,
        #    kernel_size=1,
        #    strides=1,
        #)
        #self.layer_norm = keras.layers.LayerNormalization()

        self.input_projector = nn.Conv1d(
            in_channels=input_dim,
            out_channels=feature_dim,
            kernel_size=1,
            stride=1
        )
        # LayerNorm over the feature dimension
        self.layer_norm = nn.LayerNorm(feature_dim)

    def call(self, x: Tensor) -> Tensor:
        """
        Applies the Mamba layer to the input tensor `x`, optionally in a bidirectional manner.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape `(batch_size, sequence_length, input_dim)`.

        Returns
        -------
        Tensor
            Output tensor of shape `(batch_size, sequence_length, feature_dim)` if unidirectional,
            or `(batch_size, sequence_length, 2 * feature_dim)` if bidirectional.
        """

        out_forward = self._call(x)
        if self.bidirectional:
            out_backward = self._call(keras.ops.flip(x, axis=-2))
            return keras.ops.concatenate((out_forward, out_backward), axis=-1)
        return out_forward

    def _call(self, x: Tensor) -> Tensor:
        x = self.input_projector(x)
        h = self.mamba(x)
        out = self.layer_norm(h + x)
        return out


class Mamba(nn.Module):
    """
    Wraps a sequence of Mamba modules using the simple Mamba module from:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

    Copyright (c) 2023, Tri Dao, Albert Gu.

    Example usage in a BayesFlow workflow as a summary network:

    `summary_net = bayesflow.wrappers.Mamba(summary_dim=32)`
    """

    def __init__(
        self,
        input_dim: int,
        summary_dim: int = 16,
        feature_dims: Sequence[int] = (64, 64),
        state_dims: Sequence[int] = (64, 64),
        conv_dims: Sequence[int] = (64, 64),
        expand_dims: Sequence[int] = (2, 2),
        bidirectional: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.05,
        device: str = "cuda",
        **kwargs,
    ):
        """
        A time-series summarization network using Mamba-based State Space Models (SSM). This model processes
        sequential input data using a sequence of Mamba SSM layers (determined by the length of the tuples),
        followed by optional pooling, dropout, and a dense layer for extracting summary statistics.

        Parameters
        ----------
        summary_dim : Sequence[int], optional
            The output dimensionality of the summary statistics layer (default is 16).
        feature_dims : Sequence[int], optional
            The feature dimension for each mamba block, default is (64, 64),
        state_dims : Sequence[int], optional
            The dimensionality of the internal state in each Mamba block, default is (64, 64)
        conv_dims : Sequence[int], optional
            The dimensionality of the convolutional layer in each Mamba block, default is (32, 32)
        expand_dims : Sequence[int], optional
            The expansion factors for the hidden state in each Mamba block, default is (2, 2)
        dt_min : float, optional
            Minimum dynamic state evolution over time (default is 0.001).
        dt_max : float, optional
            Maximum dynamic state evolution over time (default is 0.1).
        pooling : bool, optional
            Whether to apply global average pooling (default is True).
        dropout : int, float, or None, optional
            Dropout rate applied before the summary layer (default is 0.5).
        dropout: float, optional
            Dropout probability; dropout is applied to the pooled summary vector.
        device : str, optional
            The computing device. Currently, only "cuda" is supported (default is "cuda").
        **kwargs :
            Additional keyword arguments passed to the `SummaryNetwork` parent class.
        """

        super().__init__(**keras_kwargs(kwargs))
        self.name = "Mamba"

        if device != "cuda":
            raise NotImplementedError("MambaSSM only supports cuda as `device`.")

        self.mamba_blocks = []
        in_dim = input_dim
        for feature_dim, state_dim, conv_dim, expand in zip(feature_dims, state_dims, conv_dims, expand_dims):
            mamba = MambaBlock(in_dim, feature_dim, state_dim, conv_dim, expand, bidirectional, dt_min, dt_max, device)
            self.mamba_blocks.append(mamba)
            in_dim = feature_dim

        self.dropout = nn.Dropout(p=dropout)
        self.summary_stats = nn.Linear(feature_dims[-1], summary_dim)

    def call(self, time_series: Tensor) -> Tensor:
        """
        Apply a sequence of Mamba blocks, followed by pooling, dropout, and summary statistics.

        Parameters
        ----------
        time_series : Tensor
            Input tensor representing the time series data, typically of shape
            (batch_size, sequence_length, feature_dim).

        Returns
        -------
        Tensor
            Output tensor after applying Mamba blocks, pooling, dropout, and summary statistics.
        """

        summary = time_series
        for mamba_block in self.mamba_blocks:
            summary = mamba_block(summary)

        #summary = self.pooling_layer(summary)
        summary = summary.mean(dim=1)  # todo: Stefan please check this here
        summary = self.dropout(summary)
        summary = self.summary_stats(summary)

        return summary
