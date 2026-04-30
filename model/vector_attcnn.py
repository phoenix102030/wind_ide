from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .covariance import covariance_from_cholesky_raw, inverse_softplus


def build_cholesky_4x4(raw: Tensor, jitter: float = 1.0e-4) -> Tensor:
    """Build a 4x4 lower Cholesky factor from 10 raw parameters."""
    L, _ = covariance_from_cholesky_raw(raw, dim=4, jitter=jitter)
    return L


class ConvBackbone(nn.Module):
    """Compact CNN backbone for 40x40 NWP maps."""

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        mid_dim = max(hidden_dim // 2, 16)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=mid_dim),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(mid_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class ChannelSpatialAttention(nn.Module):
    """Lightweight attention gate over CNN feature maps."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        squeeze_dim = max(hidden_dim // 4, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, squeeze_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(squeeze_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.channel(x) * self.spatial(x)


class VectorAdvectionNet(nn.Module):
    """Map NWP maps to ``mu``, full 4x4 covariance, and 2x2 mixing weights."""

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 18,
        mu_scale_init: float = 1.0,
        chol_jitter: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if output_dim != 18:
            raise ValueError("VectorAdvectionNet expects output_dim=18")

        self.backbone = ConvBackbone(in_channels=in_channels, hidden_dim=hidden_dim)
        self.attention = ChannelSpatialAttention(hidden_dim=hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.raw_mu_scale = nn.Parameter(
            torch.tensor(inverse_softplus(mu_scale_init), dtype=torch.float32)
        )
        self.chol_jitter = chol_jitter

    @property
    def mu_scale(self) -> Tensor:
        return F.softplus(self.raw_mu_scale) + 1.0e-6

    def head_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.attention.parameters()
        yield from self.heads.parameters()
        yield self.raw_mu_scale

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [T,C,H,W] or [C,H,W], got {tuple(x.shape)}")

        features = self.backbone(x)
        features = self.attention(features)
        pooled = self.pool(features)
        raw = self.heads(pooled)

        raw_mu = raw[..., 0:4]
        raw_chol = raw[..., 4:14]
        raw_A = raw[..., 14:18]

        mu = torch.tanh(raw_mu) * self.mu_scale
        L, sigma = covariance_from_cholesky_raw(
            raw_chol,
            dim=4,
            jitter=self.chol_jitter,
        )
        A_logits = raw_A.reshape(-1, 2, 2)
        A = torch.softmax(A_logits, dim=-1)

        return {
            "raw": raw,
            "mu": mu,
            "L": L,
            "Sigma": sigma,
            "A": A,
            "A_logits": A_logits,
        }
