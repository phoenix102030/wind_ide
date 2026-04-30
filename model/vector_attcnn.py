from __future__ import annotations

import math
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


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal temporal position encoding for sequence features."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.pe.shape[0]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds max positional length {self.pe.shape[0]}"
            )
        return x + self.pe[: x.shape[1]].to(device=x.device, dtype=x.dtype).unsqueeze(0)


class VectorAdvectionNet(nn.Module):
    """Map NWP maps to ``mu``, full 4x4 covariance, and 2x2 mixing weights.

    ``network_type='cnn'`` treats each time independently. ``'cnn_transformer'``
    first encodes each NWP map with a CNN and then models temporal context with
    a Transformer encoder. The causal setting is the online-safe default.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 18,
        mu_scale_init: float = 1.0,
        chol_jitter: float = 1.0e-4,
        network_type: str = "cnn_transformer",
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        transformer_dropout: float = 0.1,
        transformer_causal: bool = True,
        transformer_max_len: int = 4096,
    ) -> None:
        super().__init__()
        if output_dim != 18:
            raise ValueError("VectorAdvectionNet expects output_dim=18")
        if network_type not in {"cnn", "cnn_transformer"}:
            raise ValueError("network_type must be 'cnn' or 'cnn_transformer'")
        if transformer_d_model % transformer_nhead != 0:
            raise ValueError("transformer_d_model must be divisible by transformer_nhead")

        self.network_type = network_type
        self.transformer_causal = transformer_causal
        self.backbone = ConvBackbone(in_channels=in_channels, hidden_dim=hidden_dim)
        self.attention = ChannelSpatialAttention(hidden_dim=hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

        feature_dim = hidden_dim
        if network_type == "cnn_transformer":
            self.temporal_proj = nn.Linear(hidden_dim, transformer_d_model)
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model=transformer_d_model,
                max_len=transformer_max_len,
            )
            layer = nn.TransformerEncoderLayer(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            feature_dim = transformer_d_model
        else:
            self.temporal_proj = None
            self.positional_encoding = None
            self.temporal_encoder = None

        self.head_norm = nn.LayerNorm(feature_dim)
        self.head_shared = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(feature_dim, 4)
        self.chol_head = nn.Linear(feature_dim, 10)
        self.A_head = nn.Linear(feature_dim, 4)
        self.raw_mu_scale = nn.Parameter(
            torch.tensor(inverse_softplus(mu_scale_init), dtype=torch.float32)
        )
        self.chol_jitter = chol_jitter

    @property
    def mu_scale(self) -> Tensor:
        return F.softplus(self.raw_mu_scale) + 1.0e-6

    def head_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.attention.parameters()
        if self.temporal_proj is not None:
            yield from self.temporal_proj.parameters()
        if self.temporal_encoder is not None:
            yield from self.temporal_encoder.parameters()
        yield from self.head_norm.parameters()
        yield from self.head_shared.parameters()
        yield from self.mu_head.parameters()
        yield from self.chol_head.parameters()
        yield from self.A_head.parameters()
        yield self.raw_mu_scale

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def encode_features(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        features = self.attention(features)
        pooled = self.pool(features).flatten(1)

        if self.network_type == "cnn":
            return pooled

        seq = self.temporal_proj(pooled).unsqueeze(0)
        seq = self.positional_encoding(seq)
        mask = self._causal_mask(seq.shape[1], seq.device) if self.transformer_causal else None
        encoded = self.temporal_encoder(seq, mask=mask)
        return encoded.squeeze(0)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [T,C,H,W] or [C,H,W], got {tuple(x.shape)}")

        features = self.encode_features(x)
        features = self.head_shared(self.head_norm(features))

        raw_mu = self.mu_head(features)
        raw_chol = self.chol_head(features)
        raw_A = self.A_head(features)
        raw = torch.cat([raw_mu, raw_chol, raw_A], dim=-1)

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
