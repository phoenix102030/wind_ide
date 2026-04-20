import math

import torch
from torch import nn


def inverse_softplus(value, eps=1e-6):
    value = max(float(value), float(eps))
    return value + math.log(-math.expm1(-value))


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=32, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        h = self.net(x)
        return h.squeeze(-1).squeeze(-1)


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, num_layers=2, ff_dim=64, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)[:, -1, :]


class AdvectionMeanNet(nn.Module):
    """
    Predict theorem-driven advection parameters for the two directional fields:

      [A_uu, A_vv] ~ N(mu_t, Sigma_t),

    where
      A_uu, A_vv in R^2,
      mu_t in R^4,
      Sigma_t in R^(4x4).

    The flattened ordering is:
      mu_t = [A_uu_x, A_uu_y, A_vv_x, A_vv_y].
    """

    def __init__(
        self,
        in_channels=6,
        hidden_dim=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        dropout=0.1,
        mu_scale=0.5,
        chol_offdiag_scale=0.15,
        chol_diag_max=1.5,
        chol_eps=1e-4,
        mix_scale=0.35,
        mu_mode="free",
        sigma_mode="network",
        init_global_sigma_diag=0.2,
        init_base_scale_par=1.0,
        init_base_scale_perp=1.0,
        base_scale_min=0.15,
        base_scale_max=2.5,
        wind_anchor_indices=None,
    ):
        super().__init__()
        del mix_scale

        self.in_channels = int(in_channels)
        self.mu_scale = float(mu_scale)
        self.chol_offdiag_scale = float(chol_offdiag_scale)
        self.chol_diag_max = float(chol_diag_max)
        self.chol_eps = float(chol_eps)
        self.mu_mode = str(mu_mode).lower()
        self.sigma_mode = str(sigma_mode).lower()
        self.base_scale_min = max(float(base_scale_min), 1e-4)
        self.base_scale_max = max(float(base_scale_max), self.base_scale_min)
        self.num_advections = 2
        self.advection_dim = 2
        self.mu_dim = self.num_advections * self.advection_dim
        self.chol_param_dim = self.mu_dim * (self.mu_dim + 1) // 2

        if self.mu_mode not in {"free", "anchored"}:
            raise ValueError(f"Unsupported mu_mode={mu_mode!r}; expected 'free' or 'anchored'.")
        if self.sigma_mode not in {"network", "global"}:
            raise ValueError(f"Unsupported sigma_mode={sigma_mode!r}; expected 'network' or 'global'.")

        if wind_anchor_indices is None:
            wind_anchor_indices = (8, 9) if self.in_channels >= 10 else (2, 3)
        if len(wind_anchor_indices) != 2:
            raise ValueError("wind_anchor_indices must contain exactly two channel indices.")
        self.wind_anchor_indices = tuple(int(idx) for idx in wind_anchor_indices)

        self.spatial = SpatialEncoder(in_channels=self.in_channels, hidden_dim=hidden_dim, out_dim=embed_dim)
        self.temporal = TemporalEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        mu_head_dim = self.mu_dim if self.mu_mode == "free" else self.mu_dim * self.advection_dim
        self.mu_head = nn.Linear(embed_dim, mu_head_dim)
        self.base_scale_head = nn.Linear(embed_dim, 2)
        self.chol_head = nn.Linear(embed_dim, self.chol_param_dim)

        init_base = torch.tensor(
            [
                inverse_softplus(max(float(init_base_scale_par) - self.base_scale_min, 1e-4)),
                inverse_softplus(max(float(init_base_scale_perp) - self.base_scale_min, 1e-4)),
            ]
        )
        with torch.no_grad():
            self.base_scale_head.bias.copy_(init_base)

        init_diag_raw = inverse_softplus(max(float(init_global_sigma_diag) - self.chol_eps, 1e-4))
        global_chol_init = torch.zeros(self.chol_param_dim)
        diag_positions = self._diag_param_positions()
        global_chol_init[diag_positions] = init_diag_raw
        self.global_chol_params = nn.Parameter(global_chol_init)
        self._configure_parameter_usage()

    def _diag_param_positions(self):
        positions = []
        cursor = 0
        for row in range(self.mu_dim):
            positions.append(cursor + row)
            cursor += row + 1
        return positions

    def _configure_parameter_usage(self):
        chol_head_trainable = self.sigma_mode == "network"
        for param in self.chol_head.parameters():
            param.requires_grad_(chol_head_trainable)
        self.global_chol_params.requires_grad_(self.sigma_mode == "global")

    def _expand_legacy_mu_head(self, tensor, target_shape):
        expanded = torch.zeros(target_shape, device=tensor.device, dtype=tensor.dtype)
        rows = min(tensor.shape[0], target_shape[0])
        expanded[:rows] = tensor[:rows]
        return expanded

    def _expand_legacy_chol(self, tensor):
        if tensor.ndim == 1:
            expanded = tensor.new_zeros(self.chol_param_dim)
            if tensor.shape[0] == 3:
                expanded[0] = tensor[0]
                expanded[1] = tensor[1]
                expanded[2] = tensor[2]
                expanded[5] = tensor[0]
                expanded[8] = tensor[1]
                expanded[9] = tensor[2]
                return expanded
            expanded[: min(tensor.shape[0], self.chol_param_dim)] = tensor[: min(tensor.shape[0], self.chol_param_dim)]
            return expanded

        expanded = tensor.new_zeros(self.chol_param_dim, tensor.shape[1])
        if tensor.shape[0] == 3:
            expanded[0] = tensor[0]
            expanded[1] = tensor[1]
            expanded[2] = tensor[2]
            expanded[5] = tensor[0]
            expanded[8] = tensor[1]
            expanded[9] = tensor[2]
            return expanded
        rows = min(tensor.shape[0], self.chol_param_dim)
        expanded[:rows] = tensor[:rows]
        return expanded

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for legacy_key in ("mix_head.weight", "mix_head.bias"):
            state_dict.pop(f"{prefix}{legacy_key}", None)

        mu_weight_key = f"{prefix}mu_head.weight"
        mu_bias_key = f"{prefix}mu_head.bias"
        if mu_weight_key in state_dict and state_dict[mu_weight_key].shape != self.mu_head.weight.shape:
            state_dict[mu_weight_key] = self._expand_legacy_mu_head(state_dict[mu_weight_key], self.mu_head.weight.shape)
        if mu_bias_key in state_dict and state_dict[mu_bias_key].shape != self.mu_head.bias.shape:
            state_dict[mu_bias_key] = self._expand_legacy_mu_head(
                state_dict[mu_bias_key].reshape(-1, 1),
                self.mu_head.bias.reshape(-1, 1).shape,
            ).reshape(-1)

        for key in ("weight", "bias"):
            full_key = f"{prefix}base_scale_head.{key}"
            if full_key not in state_dict:
                state_dict[full_key] = getattr(self.base_scale_head, key).detach().clone()

        for key in ("weight", "bias"):
            full_key = f"{prefix}chol_head.{key}"
            expected = getattr(self.chol_head, key)
            if full_key in state_dict and state_dict[full_key].shape != expected.shape:
                state_dict[full_key] = self._expand_legacy_chol(state_dict[full_key])

        global_key = f"{prefix}global_chol_params"
        if global_key in state_dict and state_dict[global_key].shape != self.global_chol_params.shape:
            state_dict[global_key] = self._expand_legacy_chol(state_dict[global_key])
        if global_key not in state_dict:
            state_dict[global_key] = self.global_chol_params.detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._configure_parameter_usage()

    def _build_cholesky(self, chol_raw):
        batch_size = chol_raw.shape[0]
        chol = chol_raw.new_zeros(batch_size, self.mu_dim, self.mu_dim)

        cursor = 0
        for row in range(self.mu_dim):
            row_width = row + 1
            row_raw = chol_raw[:, cursor:cursor + row_width]
            diag_raw = row_raw[:, -1]
            offdiag_raw = row_raw[:, :-1]

            chol[:, row, row] = torch.nn.functional.softplus(diag_raw).clamp_max(self.chol_diag_max) + self.chol_eps
            if row > 0:
                chol[:, row, :row] = self.chol_offdiag_scale * torch.tanh(offdiag_raw)
            cursor += row_width

        return chol

    def _extract_wind_anchor(self, x_seq):
        latest_frame = x_seq[:, -1]
        u_idx, v_idx = self.wind_anchor_indices
        if latest_frame.shape[1] <= max(u_idx, v_idx):
            raise ValueError(
                f"Expected at least {max(u_idx, v_idx) + 1} NWP channels, got {latest_frame.shape[1]}"
            )
        wind_uv = latest_frame[:, [u_idx, v_idx]]
        return wind_uv.mean(dim=(-1, -2))

    def _predict_mu(self, h, x_seq):
        mu_raw = self.mu_scale * torch.tanh(self.mu_head(h))
        wind_anchor = self._extract_wind_anchor(x_seq)

        if self.mu_mode == "anchored":
            mu_coeff = mu_raw.reshape(-1, self.mu_dim, self.advection_dim)
            mu = torch.einsum("bij,bj->bi", mu_coeff, wind_anchor)
        else:
            mu_coeff = None
            mu = mu_raw

        return mu, mu_coeff, wind_anchor

    def _predict_base_scales(self, h):
        raw = self.base_scale_head(h)
        max_delta = self.base_scale_max - self.base_scale_min
        scales = self.base_scale_min + torch.nn.functional.softplus(raw)
        if math.isfinite(max_delta):
            scales = torch.clamp(scales, max=self.base_scale_max)
        return scales

    def forward(self, x_seq):
        # x_seq: [B, L, C, Y, X]
        batch_size, seq_len, channels, height, width = x_seq.shape
        x = x_seq.reshape(batch_size * seq_len, channels, height, width)
        feat = self.spatial(x)
        embed_dim = feat.shape[-1]
        feat = feat.reshape(batch_size, seq_len, embed_dim)

        h = self.temporal(feat)
        mu, mu_coeff_matrix, wind_anchor = self._predict_mu(h, x_seq)
        base_scales = self._predict_base_scales(h)

        if self.sigma_mode == "global":
            chol_raw = self.global_chol_params.unsqueeze(0).expand(batch_size, -1)
        else:
            chol_raw = self.chol_head(h)
        chol_factor = self._build_cholesky(chol_raw)
        sigma = chol_factor @ chol_factor.transpose(-1, -2)

        return {
            "mu": mu,
            "mu_pairs": mu.reshape(batch_size, self.num_advections, self.advection_dim),
            "mu_coeff_matrix": mu_coeff_matrix,
            "wind_anchor": wind_anchor,
            "base_scales": base_scales,
            "sigma": sigma,
            "chol_factor": chol_factor,
        }
