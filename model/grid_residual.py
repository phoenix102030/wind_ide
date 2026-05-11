from __future__ import annotations

import torch
from torch import Tensor, nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels >= groups and channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            nn.GroupNorm(num_groups=_group_count(channels), num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class GridResidualCNN(nn.Module):
    """Independent analysis-correction CNN for full-grid residual fields.

    The model keeps the original VectorMIDE untouched. It consumes engineered
    grid features with shape ``[B,T,C,H,W]`` and predicts scalar residual fields
    with shape ``[B,T,1,H,W]``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_blocks: int = 5,
        dropout: float = 0.0,
        max_residual: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.max_residual = float(max_residual)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        dilation_cycle = [1, 2, 4, 2, 1]
        self.blocks = nn.Sequential(
            *[
                ResidualConvBlock(
                    hidden_dim,
                    dilation=dilation_cycle[index % len(dilation_cycle)],
                    dropout=dropout,
                )
                for index in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=_group_count(hidden_dim), num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2 if hidden_dim >= 2 else hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2 if hidden_dim >= 2 else hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"Expected x [B,T,C,H,W] or [B,C,H,W], got {tuple(x.shape)}")
        bsz, steps, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")
        flat = x.reshape(bsz * steps, channels, height, width)
        out = self.head(self.blocks(self.stem(flat)))
        if self.max_residual > 0.0:
            out = self.max_residual * torch.tanh(out / self.max_residual)
        return out.reshape(bsz, steps, 1, height, width)
