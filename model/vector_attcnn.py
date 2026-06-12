from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .covariance import covariance_from_cholesky_raw, inverse_softplus, safe_cholesky


def _logit(value: float) -> float:
    value = min(max(value, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def _bounded_logit(value: float, lower: float, upper: float) -> float:
    if not lower < upper:
        raise ValueError("Require lower < upper for bounded scalar output")
    return _logit((value - lower) / (upper - lower))


def build_cholesky_4x4(raw: Tensor, jitter: float = 1.0e-4) -> Tensor:
    """Build a 4x4 lower Cholesky factor from 10 raw parameters."""
    L, _ = covariance_from_cholesky_raw(raw, dim=4, jitter=jitter)
    return L


def _cholesky_raw_from_diag(stds: list[float], jitter: float) -> Tensor:
    if len(stds) != 4:
        raise ValueError("Expected four covariance regime standard deviations")
    raw = torch.zeros(10, dtype=torch.float32)
    diag_positions = [0, 2, 5, 9]
    for pos, std in zip(diag_positions, stds):
        raw[pos] = inverse_softplus(max(float(std), jitter) - jitter)
    return raw


def _bounded_cross_correlation(raw: Tensor, limit: float) -> Tensor:
    """Return a 2x2 contraction matrix for stable block covariance coupling."""
    if limit <= 0.0:
        return raw.new_zeros(raw.shape[:-1] + (2, 2))
    matrix = torch.tanh(raw).reshape(raw.shape[:-1] + (2, 2))
    fro = torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1), keepdim=True)
    factor = torch.clamp(float(limit) / fro.clamp_min(1.0e-6), max=1.0)
    return matrix * factor


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
    """Map NWP maps to component-wise spatial advection moments.

    In ``advection_mode='component'``, the network predicts ``mu`` in the order
    ``[u_x, u_y, v_x, v_y]`` with a full 4x4 covariance. The two pairs are 2D
    spatial displacements for the U and V component fields, not four scalar
    component-pair displacements.

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
        component_mixing_floor: float = 0.0,
        network_type: str = "cnn_transformer",
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        transformer_dropout: float = 0.1,
        transformer_causal: bool = True,
        transformer_max_len: int = 4096,
        component_specific_mu: bool = False,
        advection_mode: str = "component",
        deformation_scale: float = 0.3,
        anchored_advection: bool = False,
        advection_residual_scale: float = 1.0,
        transition_kernel_weight: bool = False,
        transition_kernel_weight_init: float = 1.0,
        transition_kernel_weight_min: float = 0.0,
        transition_kernel_weight_max: float = 1.0,
        transition_residual_decay: bool = False,
        transition_residual_decay_init: float = 1.0,
        transition_residual_decay_min: float = 0.0,
        transition_residual_decay_max: float = 1.0,
        transition_control_dim: int = 0,
        transition_control_scale: float = 0.0,
        covariance_mode: str = "free_cholesky",
        covariance_regimes: int = 3,
        covariance_floor: float = 0.0,
        covariance_regime_stds: list[float] | tuple[float, ...] | None = None,
        covariance_dynamic_scale: bool = False,
        covariance_scale_init: float = 1.0,
        covariance_scale_min: float = 0.25,
        covariance_scale_max: float = 4.0,
        covariance_cross_corr_limit: float = 0.6,
        advection_component_scale: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if output_dim != 18:
            raise ValueError("VectorAdvectionNet expects output_dim=18")
        if network_type not in {"cnn", "cnn_transformer"}:
            raise ValueError("network_type must be 'cnn' or 'cnn_transformer'")
        if advection_mode not in {"component", "shared_flow_deformation", "shared_flow_component_kernel"}:
            raise ValueError(
                "advection_mode must be 'component', 'shared_flow_deformation', "
                "or 'shared_flow_component_kernel'"
            )
        if transformer_d_model % transformer_nhead != 0:
            raise ValueError("transformer_d_model must be divisible by transformer_nhead")
        if not 0.0 <= component_mixing_floor < 0.5:
            raise ValueError("component_mixing_floor must be in [0, 0.5)")

        self.network_type = network_type
        self.transformer_causal = transformer_causal
        self.component_mixing_floor = component_mixing_floor
        self.transition_kernel_weight_enabled = transition_kernel_weight
        self.transition_kernel_weight_min = float(transition_kernel_weight_min)
        self.transition_kernel_weight_max = float(transition_kernel_weight_max)
        self.transition_residual_decay_enabled = transition_residual_decay
        self.transition_residual_decay_min = float(transition_residual_decay_min)
        self.transition_residual_decay_max = float(transition_residual_decay_max)
        self.transition_control_dim = int(transition_control_dim)
        self.transition_control_scale = float(transition_control_scale)
        self.component_specific_mu = bool(component_specific_mu)
        self.advection_mode = advection_mode
        self.deformation_scale = float(deformation_scale)
        self.anchored_advection = bool(anchored_advection)
        self.advection_residual_scale = float(advection_residual_scale)
        self.covariance_mode = covariance_mode.replace("-", "_").lower()
        if self.covariance_mode not in {"free_cholesky", "regime", "block_cholesky", "coupled_block_cholesky"}:
            raise ValueError(
                "covariance_mode must be 'free_cholesky', 'regime', "
                "'block_cholesky', or 'coupled_block_cholesky'"
            )
        self.covariance_regimes = int(covariance_regimes)
        if self.covariance_regimes <= 0:
            raise ValueError("covariance_regimes must be positive")
        self.covariance_floor = float(covariance_floor)
        self.covariance_dynamic_scale = bool(covariance_dynamic_scale)
        self.covariance_scale_init = float(covariance_scale_init)
        self.covariance_scale_min = float(covariance_scale_min)
        self.covariance_scale_max = float(covariance_scale_max)
        if self.covariance_scale_min <= 0.0 or self.covariance_scale_max <= self.covariance_scale_min:
            raise ValueError("Require 0 < covariance_scale_min < covariance_scale_max")
        if not self.covariance_scale_min <= self.covariance_scale_init <= self.covariance_scale_max:
            raise ValueError("Require covariance_scale_init within [covariance_scale_min, covariance_scale_max]")
        self.covariance_cross_corr_limit = float(covariance_cross_corr_limit)
        if not 0.0 <= self.covariance_cross_corr_limit < 1.0:
            raise ValueError("covariance_cross_corr_limit must be in [0, 1)")
        if advection_component_scale is None:
            advection_component_scale = (1.0, 1.0, 1.0, 1.0)
        if len(advection_component_scale) != 4:
            raise ValueError("advection_component_scale must contain four values")
        self.register_buffer(
            "advection_component_scale",
            torch.tensor(advection_component_scale, dtype=torch.float32),
        )

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
        use_component_heads = self.advection_mode == "component"
        self.mu_head = None if (self.component_specific_mu or not use_component_heads) else nn.Linear(feature_dim, 4)
        if use_component_heads and self.component_specific_mu:
            self.mu_u_head_shared = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.SiLU(),
            )
            self.mu_v_head_shared = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.SiLU(),
            )
            self.mu_u_head = nn.Linear(feature_dim, 2)
            self.mu_v_head = nn.Linear(feature_dim, 2)
        else:
            self.mu_u_head_shared = None
            self.mu_v_head_shared = None
            self.mu_u_head = None
            self.mu_v_head = None
        if use_component_heads and self.covariance_mode == "regime":
            self.chol_head = nn.Linear(feature_dim, self.covariance_regimes)
            if covariance_regime_stds is None:
                covariance_regime_stds = (0.25, 1.0, 3.0)
            if len(covariance_regime_stds) != self.covariance_regimes:
                raise ValueError("covariance_regime_stds length must match covariance_regimes")
            raw_regimes = []
            for std in covariance_regime_stds:
                raw_regimes.append(_cholesky_raw_from_diag([float(std)] * 4, chol_jitter))
            self.regime_chol_raw = nn.Parameter(torch.stack(raw_regimes, dim=0))
            if self.covariance_dynamic_scale:
                self.covariance_scale_head = nn.Linear(feature_dim, 1)
                nn.init.zeros_(self.covariance_scale_head.weight)
                nn.init.constant_(
                    self.covariance_scale_head.bias,
                    _bounded_logit(
                        self.covariance_scale_init,
                        self.covariance_scale_min,
                        self.covariance_scale_max,
                    ),
                )
            else:
                self.covariance_scale_head = None
        elif use_component_heads and self.covariance_mode in {"block_cholesky", "coupled_block_cholesky"}:
            self.chol_head = nn.Linear(feature_dim, 10 if self.covariance_mode == "coupled_block_cholesky" else 6)
            self.regime_chol_raw = None
            if self.covariance_dynamic_scale:
                self.covariance_scale_head = nn.Linear(feature_dim, 1)
                nn.init.zeros_(self.covariance_scale_head.weight)
                nn.init.constant_(
                    self.covariance_scale_head.bias,
                    _bounded_logit(
                        self.covariance_scale_init,
                        self.covariance_scale_min,
                        self.covariance_scale_max,
                    ),
                )
            else:
                self.covariance_scale_head = None
        else:
            self.chol_head = nn.Linear(feature_dim, 10) if use_component_heads else None
            self.regime_chol_raw = None
            self.covariance_scale_head = None
        self.alpha_head = nn.Linear(feature_dim, 4) if use_component_heads else None
        if self.advection_mode in {"shared_flow_deformation", "shared_flow_component_kernel"}:
            self.flow_head = nn.Linear(feature_dim, 2)
            self.flow_chol_head = nn.Linear(
                feature_dim,
                10 if self.advection_mode == "shared_flow_component_kernel" else 3,
            )
            self.deformation_head = nn.Linear(feature_dim, 4)
            nn.init.zeros_(self.deformation_head.weight)
            nn.init.zeros_(self.deformation_head.bias)
        else:
            self.flow_head = None
            self.flow_chol_head = None
            self.deformation_head = None
        if self.transition_kernel_weight_enabled:
            self.kernel_weight_head = nn.Linear(feature_dim, 1)
            nn.init.zeros_(self.kernel_weight_head.weight)
            nn.init.constant_(
                self.kernel_weight_head.bias,
                _bounded_logit(
                    float(transition_kernel_weight_init),
                    self.transition_kernel_weight_min,
                    self.transition_kernel_weight_max,
                ),
            )
        else:
            self.kernel_weight_head = None
        if self.transition_residual_decay_enabled:
            self.residual_decay_head = nn.Linear(feature_dim, 1)
            nn.init.zeros_(self.residual_decay_head.weight)
            nn.init.constant_(
                self.residual_decay_head.bias,
                _bounded_logit(
                    float(transition_residual_decay_init),
                    self.transition_residual_decay_min,
                    self.transition_residual_decay_max,
                ),
            )
        else:
            self.residual_decay_head = None
        if self.transition_control_dim > 0:
            self.control_head = nn.Linear(feature_dim, self.transition_control_dim)
            nn.init.zeros_(self.control_head.weight)
            nn.init.zeros_(self.control_head.bias)
        else:
            self.control_head = None
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
        if self.mu_head is not None:
            yield from self.mu_head.parameters()
        if self.mu_u_head_shared is not None:
            yield from self.mu_u_head_shared.parameters()
        if self.mu_v_head_shared is not None:
            yield from self.mu_v_head_shared.parameters()
        if self.mu_u_head is not None:
            yield from self.mu_u_head.parameters()
        if self.mu_v_head is not None:
            yield from self.mu_v_head.parameters()
        if self.chol_head is not None:
            yield from self.chol_head.parameters()
        if self.regime_chol_raw is not None:
            yield self.regime_chol_raw
        if self.covariance_scale_head is not None:
            yield from self.covariance_scale_head.parameters()
        if self.alpha_head is not None:
            yield from self.alpha_head.parameters()
        if self.flow_head is not None:
            yield from self.flow_head.parameters()
        if self.flow_chol_head is not None:
            yield from self.flow_chol_head.parameters()
        if self.deformation_head is not None:
            yield from self.deformation_head.parameters()
        if self.kernel_weight_head is not None:
            yield from self.kernel_weight_head.parameters()
        if self.residual_decay_head is not None:
            yield from self.residual_decay_head.parameters()
        if self.control_head is not None:
            yield from self.control_head.parameters()
        yield self.raw_mu_scale

    def output_head_parameters(self) -> Iterable[nn.Parameter]:
        """Lightweight online adaptation parameters after frozen feature encoding."""
        yield from self.head_norm.parameters()
        yield from self.head_shared.parameters()
        if self.mu_head is not None:
            yield from self.mu_head.parameters()
        if self.mu_u_head_shared is not None:
            yield from self.mu_u_head_shared.parameters()
        if self.mu_v_head_shared is not None:
            yield from self.mu_v_head_shared.parameters()
        if self.mu_u_head is not None:
            yield from self.mu_u_head.parameters()
        if self.mu_v_head is not None:
            yield from self.mu_v_head.parameters()
        if self.chol_head is not None:
            yield from self.chol_head.parameters()
        if self.regime_chol_raw is not None:
            yield self.regime_chol_raw
        if self.covariance_scale_head is not None:
            yield from self.covariance_scale_head.parameters()
        if self.alpha_head is not None:
            yield from self.alpha_head.parameters()
        if self.flow_head is not None:
            yield from self.flow_head.parameters()
        if self.flow_chol_head is not None:
            yield from self.flow_chol_head.parameters()
        if self.deformation_head is not None:
            yield from self.deformation_head.parameters()
        if self.kernel_weight_head is not None:
            yield from self.kernel_weight_head.parameters()
        if self.residual_decay_head is not None:
            yield from self.residual_decay_head.parameters()
        if self.control_head is not None:
            yield from self.control_head.parameters()
        yield self.raw_mu_scale

    @staticmethod
    def _bounded_sigmoid(raw: Tensor, lower: float, upper: float) -> Tensor:
        return lower + (upper - lower) * torch.sigmoid(raw)

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

    def predict_raw_mu(self, shared_features: Tensor) -> Tensor:
        if not self.component_specific_mu:
            if self.mu_head is None:
                raise RuntimeError("mu_head is not initialized")
            return self.mu_head(shared_features)

        features_u = self.mu_u_head_shared(shared_features)
        features_v = self.mu_v_head_shared(shared_features)
        return torch.cat([self.mu_u_head(features_u), self.mu_v_head(features_v)], dim=-1)

    @staticmethod
    def _expand_flow_cholesky_raw(raw_flow_chol: Tensor) -> Tensor:
        raw_chol = raw_flow_chol.new_zeros(raw_flow_chol.shape[:-1] + (10,))
        raw_chol[..., 0] = raw_flow_chol[..., 0]
        raw_chol[..., 1] = raw_flow_chol[..., 1]
        raw_chol[..., 2] = raw_flow_chol[..., 2]
        raw_chol[..., 5] = raw_flow_chol[..., 0]
        raw_chol[..., 8] = raw_flow_chol[..., 1]
        raw_chol[..., 9] = raw_flow_chol[..., 2]
        return raw_chol

    def forward_shared_flow_deformation(self, features: Tensor) -> dict[str, Tensor]:
        if self.flow_head is None or self.flow_chol_head is None or self.deformation_head is None:
            raise RuntimeError("Shared-flow advection heads are not initialized")

        raw_flow = self.flow_head(features)
        raw_flow_chol = self.flow_chol_head(features)
        raw_B = self.deformation_head(features)
        flow_mu = torch.tanh(raw_flow) * self.mu_scale
        flow_L, flow_sigma = covariance_from_cholesky_raw(
            raw_flow_chol,
            dim=2,
            jitter=self.chol_jitter,
        )

        n_time = flow_mu.shape[0]
        mu = torch.cat([flow_mu, flow_mu], dim=-1)
        L = flow_L.new_zeros(n_time, 4, 4)
        L[:, :2, :2] = flow_L
        L[:, 2:, 2:] = flow_L
        sigma = flow_sigma.new_zeros(n_time, 4, 4)
        sigma[:, :2, :2] = flow_sigma
        sigma[:, 2:, 2:] = flow_sigma

        eye2 = torch.eye(2, device=features.device, dtype=features.dtype).unsqueeze(0)
        B_delta = self.deformation_scale * torch.tanh(raw_B.reshape(-1, 2, 2))
        B = eye2 + B_delta

        raw_mu = torch.cat([raw_flow, raw_flow], dim=-1)
        raw_chol = self._expand_flow_cholesky_raw(raw_flow_chol)
        raw = torch.cat([raw_mu, raw_chol, raw_B], dim=-1)
        B_logits = raw_B.reshape(-1, 2, 2)

        return {
            "raw": raw,
            "mu": mu,
            "L": L,
            "Sigma": sigma,
            "alpha": B,
            "alpha_logits": B_logits,
            "B": B,
            "B_delta": B_delta,
            "flow_mu": flow_mu,
            "flow_L": flow_L,
            "flow_Sigma": flow_sigma,
        }

    @staticmethod
    def _component_kernel_moments(flow_mu: Tensor, scalar_sigma: Tensor, B: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Expand one learned 2D flow into component-pair flow moments.

        ``pair_flow_mu[:, i, j]`` is the spatial shift used by the separate
        kernel for source component ``j`` contributing to target component ``i``.
        The 4D covariance is over scalar displacements along the learned joint
        flow direction for ``[UU, UV, VU, VV]``.
        """
        flow_norm = flow_mu.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        flow_dir = flow_mu / flow_norm
        cross_dir = torch.stack([-flow_dir[:, 1], flow_dir[:, 0]], dim=-1)

        scalar_mu = (B * flow_norm[:, None, :]).reshape(flow_mu.shape[0], 4)
        pair_flow_mu = scalar_mu.reshape(flow_mu.shape[0], 2, 2, 1) * flow_dir[:, None, None, :]

        scalar_var = torch.diagonal(scalar_sigma, dim1=-2, dim2=-1).clamp_min(1.0e-8)
        pair_scalar_var = scalar_var.reshape(flow_mu.shape[0], 2, 2)
        along = flow_dir[:, :, None] @ flow_dir[:, None, :]
        cross = cross_dir[:, :, None] @ cross_dir[:, None, :]
        crosswind_floor = 0.05 * pair_scalar_var.mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        pair_flow_sigma = (
            pair_scalar_var[:, :, :, None, None] * along[:, None, None, :, :]
            + crosswind_floor[:, :, :, None, None] * cross[:, None, None, :, :]
        )

        return pair_flow_mu, pair_flow_sigma, scalar_mu, scalar_sigma

    def forward_shared_flow_component_kernel(self, features: Tensor) -> dict[str, Tensor]:
        if self.flow_head is None or self.flow_chol_head is None or self.deformation_head is None:
            raise RuntimeError("Shared-flow component-kernel heads are not initialized")

        raw_flow = self.flow_head(features)
        raw_scalar_chol = self.flow_chol_head(features)
        raw_B = self.deformation_head(features)
        flow_mu = torch.tanh(raw_flow) * self.mu_scale
        scalar_L, scalar_sigma = covariance_from_cholesky_raw(
            raw_scalar_chol,
            dim=4,
            jitter=self.chol_jitter,
        )

        eye2 = torch.eye(2, device=features.device, dtype=features.dtype).unsqueeze(0)
        B_delta = self.deformation_scale * torch.tanh(raw_B.reshape(-1, 2, 2))
        B = eye2 + B_delta
        pair_flow_mu, pair_flow_sigma, mu, sigma = self._component_kernel_moments(flow_mu, scalar_sigma, B)

        n_time = flow_mu.shape[0]
        flow_dir = flow_mu / flow_mu.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        flow_var = torch.einsum("ti,tij,tj->t", flow_dir, pair_flow_sigma[:, 0, 0], flow_dir).clamp_min(1.0e-8)
        flow_sigma = flow_var[:, None, None] * (flow_dir[:, :, None] @ flow_dir[:, None, :])
        flow_sigma = flow_sigma + self.chol_jitter * torch.eye(2, device=features.device, dtype=features.dtype).unsqueeze(0)
        flow_L = torch.linalg.cholesky(flow_sigma)
        L = scalar_L
        raw_mu = torch.cat([raw_flow, raw_flow], dim=-1)
        raw = torch.cat([raw_mu, raw_scalar_chol, raw_B], dim=-1)
        B_logits = raw_B.reshape(-1, 2, 2)

        return {
            "raw": raw,
            "mu": mu,
            "L": L,
            "Sigma": sigma,
            "alpha": torch.ones_like(B),
            "alpha_logits": B_logits,
            "B": B,
            "B_delta": B_delta,
            "flow_mu": flow_mu,
            "flow_L": flow_L,
            "flow_Sigma": flow_sigma,
            "pair_flow_mu": pair_flow_mu,
            "pair_flow_Sigma": pair_flow_sigma,
        }

    @staticmethod
    def _expand_anchor(advection_anchor: Tensor, target: Tensor) -> Tensor:
        anchor = advection_anchor.to(device=target.device, dtype=target.dtype)
        if anchor.ndim == 1:
            anchor = anchor.unsqueeze(0).expand(target.shape[0], -1)
        if anchor.shape != target.shape:
            raise ValueError(f"Expected advection_anchor shape {tuple(target.shape)}, got {tuple(anchor.shape)}")
        return anchor

    def _scaled_component_advection(self, value: Tensor) -> Tensor:
        scale = self.advection_component_scale.to(device=value.device, dtype=value.dtype)
        return value * scale

    def component_covariance_from_raw(
        self,
        raw_chol: Tensor,
        raw_scale: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.covariance_mode in {"block_cholesky", "coupled_block_cholesky"}:
            raw_u = raw_chol[..., :3]
            raw_v = raw_chol[..., 3:]
            L_u, sigma_u = covariance_from_cholesky_raw(raw_u, dim=2, jitter=self.chol_jitter)
            L_v, sigma_v = covariance_from_cholesky_raw(raw_v, dim=2, jitter=self.chol_jitter)
            scale = None
            if raw_scale is not None:
                scale = self._bounded_sigmoid(
                    raw_scale.squeeze(-1),
                    self.covariance_scale_min,
                    self.covariance_scale_max,
                )
                sigma_u = sigma_u * scale[:, None, None]
                sigma_v = sigma_v * scale[:, None, None]
            if self.covariance_floor > 0.0:
                eye2 = torch.eye(2, device=raw_chol.device, dtype=raw_chol.dtype).unsqueeze(0)
                sigma_u = sigma_u + self.covariance_floor**2 * eye2
                sigma_v = sigma_v + self.covariance_floor**2 * eye2
            L_u = safe_cholesky(sigma_u)
            L_v = safe_cholesky(sigma_v)
            n_time = raw_chol.shape[0]
            L = raw_chol.new_zeros(n_time, 4, 4)
            sigma = raw_chol.new_zeros(n_time, 4, 4)
            sigma[:, :2, :2] = sigma_u
            sigma[:, 2:, 2:] = sigma_v
            extra = {
                "covariance_block_Sigma": torch.stack([sigma_u, sigma_v], dim=1),
                "covariance_block_L": torch.stack([L_u, L_v], dim=1),
            }
            if self.covariance_mode == "coupled_block_cholesky":
                raw_cross = raw_chol[..., 6:10]
                cross_corr = _bounded_cross_correlation(raw_cross, self.covariance_cross_corr_limit)
                cross_sigma = torch.matmul(torch.matmul(L_u, cross_corr), L_v.transpose(-1, -2))
                sigma[:, :2, 2:] = cross_sigma
                sigma[:, 2:, :2] = cross_sigma.transpose(-1, -2)
                sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
                extra["covariance_cross_correlation"] = cross_corr
                extra["covariance_cross_Sigma"] = cross_sigma
            L = safe_cholesky(sigma)
            if scale is not None:
                extra["raw_covariance_scale"] = raw_scale
                extra["covariance_scale"] = scale
            return L, sigma, extra

        if self.covariance_mode == "regime":
            if self.regime_chol_raw is None:
                raise RuntimeError("regime_chol_raw is not initialized")
            regime_L, regime_sigma = covariance_from_cholesky_raw(
                self.regime_chol_raw.to(device=raw_chol.device, dtype=raw_chol.dtype),
                dim=4,
                jitter=self.chol_jitter,
            )
            regime_probs = torch.softmax(raw_chol, dim=-1)
            base_sigma = torch.einsum("tr,rij->tij", regime_probs, regime_sigma)
            scale = None
            if raw_scale is not None:
                scale = self._bounded_sigmoid(
                    raw_scale.squeeze(-1),
                    self.covariance_scale_min,
                    self.covariance_scale_max,
                )
                sigma = base_sigma * scale[:, None, None]
            else:
                sigma = base_sigma
            if self.covariance_floor > 0.0:
                eye = torch.eye(4, device=raw_chol.device, dtype=raw_chol.dtype).unsqueeze(0)
                sigma = sigma + self.covariance_floor**2 * eye
            L = safe_cholesky(sigma)
            extra = {
                "covariance_regime_logits": raw_chol,
                "covariance_regime_probs": regime_probs,
                "covariance_regime_Sigma": regime_sigma,
                "covariance_regime_L": regime_L,
                "covariance_regime_base_Sigma": base_sigma,
            }
            if scale is not None:
                extra["raw_covariance_scale"] = raw_scale
                extra["covariance_scale"] = scale
            return L, sigma, extra

        L, sigma = covariance_from_cholesky_raw(
            raw_chol,
            dim=4,
            jitter=self.chol_jitter,
        )
        if self.covariance_floor > 0.0:
            eye = torch.eye(4, device=raw_chol.device, dtype=raw_chol.dtype).unsqueeze(0)
            sigma = sigma + self.covariance_floor**2 * eye
            L = safe_cholesky(sigma)
        return L, sigma, {}

    def forward(self, x: Tensor, advection_anchor: Tensor | None = None) -> dict[str, Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [T,C,H,W] or [C,H,W], got {tuple(x.shape)}")

        features = self.encode_features(x)
        features = self.head_shared(self.head_norm(features))

        if self.advection_mode == "shared_flow_deformation":
            result = self.forward_shared_flow_deformation(features)
            if self.kernel_weight_head is not None:
                raw_kernel_weight = self.kernel_weight_head(features)
                result["raw_kernel_weight"] = raw_kernel_weight
                result["kernel_weight"] = self._bounded_sigmoid(
                    raw_kernel_weight,
                    self.transition_kernel_weight_min,
                    self.transition_kernel_weight_max,
                )
            if self.residual_decay_head is not None:
                raw_residual_decay = self.residual_decay_head(features)
                result["raw_residual_decay"] = raw_residual_decay
                result["residual_decay"] = self._bounded_sigmoid(
                    raw_residual_decay,
                    self.transition_residual_decay_min,
                    self.transition_residual_decay_max,
                )
            if self.control_head is not None:
                raw_control = self.control_head(features)
                result["raw_transition_control"] = raw_control
                result["transition_control"] = torch.tanh(raw_control) * self.transition_control_scale
            return result

        if self.advection_mode == "shared_flow_component_kernel":
            result = self.forward_shared_flow_component_kernel(features)
            if self.kernel_weight_head is not None:
                raw_kernel_weight = self.kernel_weight_head(features)
                result["raw_kernel_weight"] = raw_kernel_weight
                result["kernel_weight"] = self._bounded_sigmoid(
                    raw_kernel_weight,
                    self.transition_kernel_weight_min,
                    self.transition_kernel_weight_max,
                )
            if self.residual_decay_head is not None:
                raw_residual_decay = self.residual_decay_head(features)
                result["raw_residual_decay"] = raw_residual_decay
                result["residual_decay"] = self._bounded_sigmoid(
                    raw_residual_decay,
                    self.transition_residual_decay_min,
                    self.transition_residual_decay_max,
                )
            if self.control_head is not None:
                raw_control = self.control_head(features)
                result["raw_transition_control"] = raw_control
                result["transition_control"] = torch.tanh(raw_control) * self.transition_control_scale
            return result

        raw_mu = self.predict_raw_mu(features)
        if self.chol_head is None or self.alpha_head is None:
            raise RuntimeError("Component advection heads are not initialized")
        raw_chol = self.chol_head(features)
        raw_covariance_scale = self.covariance_scale_head(features) if self.covariance_scale_head is not None else None
        raw_alpha = self.alpha_head(features)
        raw = torch.cat([raw_mu, raw_chol, raw_alpha], dim=-1)

        if self.anchored_advection:
            delta_mu = torch.tanh(raw_mu) * self.advection_residual_scale
            if advection_anchor is None:
                anchor = torch.zeros_like(delta_mu)
            else:
                anchor = self._expand_anchor(advection_anchor, delta_mu)
            anchor = self._scaled_component_advection(anchor)
            mu = anchor + delta_mu
        else:
            delta_mu = None
            anchor = None
            mu = self._scaled_component_advection(torch.tanh(raw_mu) * self.mu_scale)
        L, sigma, covariance_extra = self.component_covariance_from_raw(raw_chol, raw_scale=raw_covariance_scale)
        alpha_logits = raw_alpha.reshape(-1, 2, 2)
        alpha = torch.softmax(alpha_logits, dim=-1)
        if self.component_mixing_floor > 0.0:
            alpha = self.component_mixing_floor + (1.0 - 2.0 * self.component_mixing_floor) * alpha

        result = {
            "raw": raw,
            "mu": mu,
            "L": L,
            "Sigma": sigma,
            "alpha": alpha,
            "alpha_logits": alpha_logits,
            "advection_component_scale": self.advection_component_scale.to(device=mu.device, dtype=mu.dtype),
            **covariance_extra,
        }
        if delta_mu is not None and anchor is not None:
            result["delta_mu"] = delta_mu
            result["advection_anchor"] = anchor
        if self.kernel_weight_head is not None:
            raw_kernel_weight = self.kernel_weight_head(features)
            result["raw_kernel_weight"] = raw_kernel_weight
            result["kernel_weight"] = self._bounded_sigmoid(
                raw_kernel_weight,
                self.transition_kernel_weight_min,
                self.transition_kernel_weight_max,
            )
        if self.residual_decay_head is not None:
            raw_residual_decay = self.residual_decay_head(features)
            result["raw_residual_decay"] = raw_residual_decay
            result["residual_decay"] = self._bounded_sigmoid(
                raw_residual_decay,
                self.transition_residual_decay_min,
                self.transition_residual_decay_max,
            )
        if self.control_head is not None:
            raw_control = self.control_head(features)
            result["raw_transition_control"] = raw_control
            result["transition_control"] = torch.tanh(raw_control) * self.transition_control_scale
        return result
