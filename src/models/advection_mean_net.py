import math

import torch
from torch import nn
import torch.nn.functional as F


def inverse_softplus(value, eps=1e-6):
    value = max(float(value), float(eps))
    return value + math.log(-math.expm1(-value))


def inverse_sigmoid(value, eps=1e-6):
    value = min(max(float(value), float(eps)), 1.0 - float(eps))
    return math.log(value / (1.0 - value))


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
        num_sites=3,
        init_transport_gate=0.05,
        transport_gate_max=0.35,
        state_bias_scale=1.0,
        wind_anchor_indices=None,
    ):
        super().__init__()
        del mix_scale

        self.in_channels = int(in_channels)
        self.mu_scale = float(mu_scale)
        self.chol_offdiag_scale = float(chol_offdiag_scale)
        self.chol_diag_max = float(chol_diag_max)
        self.chol_eps = float(chol_eps)
        self.mu_bias_max = min(max(float(mu_scale), 0.1), 0.5)
        self.sigma_cross_scale = min(max(float(chol_offdiag_scale), 1e-3), 0.25)
        self.mu_mode = str(mu_mode).lower()
        self.sigma_mode = str(sigma_mode).lower()
        self.base_scale_min = max(float(base_scale_min), 1e-4)
        self.base_scale_max = max(float(base_scale_max), self.base_scale_min)
        self.num_sites = max(int(num_sites), 1)
        self.vec_dim = 2
        self.state_dim = self.num_sites * self.vec_dim
        self.transport_gate_max = max(float(transport_gate_max), 0.0)
        self.state_bias_scale = max(float(state_bias_scale), 0.0)
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

        self.mu_head = nn.Linear(embed_dim, self.mu_dim)
        self.mu_alpha_head = nn.Linear(embed_dim, self.mu_dim)
        self.mu_bias_head = nn.Linear(embed_dim, self.mu_dim)
        self.base_scale_head = nn.Linear(embed_dim, 2)
        self.transport_gate_head = nn.Linear(embed_dim, self.vec_dim)
        self.state_bias_head = nn.Linear(embed_dim, self.state_dim)
        self.sigma11_head = nn.Linear(embed_dim, 3)
        self.sigma22_head = nn.Linear(embed_dim, 3)
        self.sigma12_head = nn.Linear(embed_dim, 4)

        init_base = torch.tensor(
            [
                inverse_softplus(max(float(init_base_scale_par) - self.base_scale_min, 1e-4)),
                inverse_softplus(max(float(init_base_scale_perp) - self.base_scale_min, 1e-4)),
            ]
        )
        with torch.no_grad():
            self.base_scale_head.bias.copy_(init_base)
            self.mu_alpha_head.weight.zero_()
            self.mu_alpha_head.bias.zero_()
            self.mu_bias_head.weight.zero_()
            self.mu_bias_head.bias.zero_()
            self.transport_gate_head.weight.zero_()
            self.state_bias_head.weight.zero_()
            init_gate_ratio = 0.0
            if self.transport_gate_max > 0.0:
                init_gate_ratio = min(max(float(init_transport_gate) / self.transport_gate_max, 1e-4), 1.0 - 1e-4)
            self.transport_gate_head.bias.fill_(inverse_sigmoid(init_gate_ratio) if self.transport_gate_max > 0.0 else -20.0)
            self.state_bias_head.bias.zero_()

        init_diag_raw = inverse_softplus(max(float(init_global_sigma_diag) - self.chol_eps, 1e-4))
        global_sigma_diag_init = torch.tensor([init_diag_raw, 0.0, init_diag_raw], dtype=torch.float32)
        self.global_sigma11_params = nn.Parameter(global_sigma_diag_init.clone())
        self.global_sigma22_params = nn.Parameter(global_sigma_diag_init.clone())
        self.global_sigma12_params = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.force_gate_value = None
        self._configure_parameter_usage()

    def _diag_param_positions(self):
        positions = []
        cursor = 0
        for row in range(self.mu_dim):
            positions.append(cursor + row)
            cursor += row + 1
        return positions

    def _configure_parameter_usage(self):
        mu_head_trainable = self.mu_mode == "free"
        for param in self.mu_head.parameters():
            param.requires_grad_(mu_head_trainable)
        for param in self.mu_alpha_head.parameters():
            param.requires_grad_(self.mu_mode == "anchored")
        for param in self.mu_bias_head.parameters():
            param.requires_grad_(self.mu_mode == "anchored")

        sigma_head_trainable = self.sigma_mode == "network"
        for layer in (self.sigma11_head, self.sigma22_head, self.sigma12_head):
            for param in layer.parameters():
                param.requires_grad_(sigma_head_trainable)
        self.global_sigma11_params.requires_grad_(self.sigma_mode == "global")
        self.global_sigma22_params.requires_grad_(self.sigma_mode == "global")
        self.global_sigma12_params.requires_grad_(self.sigma_mode == "global")

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

    def _legacy_chol_to_sigma_params(self, tensor, prefix):
        expanded = self._expand_legacy_chol(tensor)
        if expanded.ndim == 1:
            expanded = expanded.unsqueeze(1)

        if expanded.ndim == 2 and expanded.shape[1] == 1:
            expanded = expanded[:, 0]
            is_vector = True
        else:
            is_vector = False

        def unpack(vec):
            out = []
            cursor = 0
            for row in range(self.mu_dim):
                row_width = row + 1
                out.append(vec[cursor:cursor + row_width])
                cursor += row_width
            return out

        if is_vector:
            rows = unpack(expanded)
            sigma11 = torch.stack([rows[0][0], rows[1][0], rows[1][1]])
            sigma22 = torch.stack([rows[2][2], rows[3][2], rows[3][3]])
            sigma12 = torch.stack([rows[2][0], rows[2][1], rows[3][0], rows[3][1]])
            return {
                f"{prefix}global_sigma11_params": sigma11,
                f"{prefix}global_sigma22_params": sigma22,
                f"{prefix}global_sigma12_params": sigma12,
            }

        rows = unpack(expanded)
        sigma11 = torch.stack([rows[0][0], rows[1][0], rows[1][1]], dim=0)
        sigma22 = torch.stack([rows[2][2], rows[3][2], rows[3][3]], dim=0)
        sigma12 = torch.stack([rows[2][0], rows[2][1], rows[3][0], rows[3][1]], dim=0)
        return {
            f"{prefix}sigma11_head.weight": sigma11,
            f"{prefix}sigma22_head.weight": sigma22,
            f"{prefix}sigma12_head.weight": sigma12,
        }

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
        for layer_name in ("mu_alpha_head", "mu_bias_head"):
            for key in ("weight", "bias"):
                full_key = f"{prefix}{layer_name}.{key}"
                if full_key not in state_dict:
                    state_dict[full_key] = getattr(getattr(self, layer_name), key).detach().clone()

        for key in ("weight", "bias"):
            full_key = f"{prefix}base_scale_head.{key}"
            if full_key not in state_dict:
                state_dict[full_key] = getattr(self.base_scale_head, key).detach().clone()

        for layer_name in ("transport_gate_head", "state_bias_head"):
            for key in ("weight", "bias"):
                full_key = f"{prefix}{layer_name}.{key}"
                if full_key not in state_dict:
                    state_dict[full_key] = getattr(getattr(self, layer_name), key).detach().clone()

        legacy_sigma_head_weight = f"{prefix}chol_head.weight"
        legacy_sigma_head_bias = f"{prefix}chol_head.bias"
        if legacy_sigma_head_weight in state_dict:
            converted = self._legacy_chol_to_sigma_params(state_dict[legacy_sigma_head_weight], prefix)
            for key, value in converted.items():
                if key not in state_dict:
                    state_dict[key] = value
            state_dict.pop(legacy_sigma_head_weight, None)
        if legacy_sigma_head_bias in state_dict:
            converted = self._legacy_chol_to_sigma_params(state_dict[legacy_sigma_head_bias], prefix)
            for key, value in converted.items():
                if key not in state_dict:
                    state_dict[key.replace(".weight", ".bias")] = value
            state_dict.pop(legacy_sigma_head_bias, None)

        for layer_name in ("sigma11_head", "sigma22_head", "sigma12_head"):
            for key in ("weight", "bias"):
                full_key = f"{prefix}{layer_name}.{key}"
                if full_key not in state_dict:
                    state_dict[full_key] = getattr(getattr(self, layer_name), key).detach().clone()

        legacy_global_key = f"{prefix}global_chol_params"
        if legacy_global_key in state_dict:
            converted = self._legacy_chol_to_sigma_params(state_dict[legacy_global_key], prefix)
            for key, value in converted.items():
                if key not in state_dict:
                    state_dict[key] = value
            state_dict.pop(legacy_global_key, None)

        for name in ("global_sigma11_params", "global_sigma22_params", "global_sigma12_params"):
            full_key = f"{prefix}{name}"
            if full_key not in state_dict:
                state_dict[full_key] = getattr(self, name).detach().clone()

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

    def _build_2x2_cholesky(self, raw):
        batch_size = raw.shape[0]
        chol = raw.new_zeros(batch_size, 2, 2)
        chol[:, 0, 0] = F.softplus(raw[:, 0]).clamp_max(self.chol_diag_max) + self.chol_eps
        chol[:, 1, 0] = self.chol_offdiag_scale * torch.tanh(raw[:, 1])
        chol[:, 1, 1] = F.softplus(raw[:, 2]).clamp_max(self.chol_diag_max) + self.chol_eps
        return chol

    def _make_spd_4x4(self, sigma, min_eig=1e-4):
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
        evals, evecs = torch.linalg.eigh(sigma)
        clipped = evals.clamp_min(min_eig)
        spd = evecs @ torch.diag_embed(clipped) @ evecs.transpose(-1, -2)
        return 0.5 * (spd + spd.transpose(-1, -2))

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
        wind_anchor = self._extract_wind_anchor(x_seq)

        if self.mu_mode == "anchored":
            alpha_raw = self.mu_alpha_head(h)
            bias_raw = self.mu_bias_head(h)
            alpha = 2.0 * torch.sigmoid(alpha_raw)
            bias = self.mu_bias_max * torch.tanh(bias_raw)
            mu_pairs = alpha.reshape(-1, self.num_advections, self.advection_dim) * wind_anchor[:, None, :] + bias.reshape(
                -1, self.num_advections, self.advection_dim
            )
            mu = mu_pairs.reshape(-1, self.mu_dim)
            mu_coeff = alpha.reshape(-1, self.num_advections, self.advection_dim)
            mu_bias = bias.reshape(-1, self.num_advections, self.advection_dim)
        else:
            mu_raw = self.mu_scale * torch.tanh(self.mu_head(h))
            mu = mu_raw
            mu_coeff = None
            mu_bias = None

        return mu, mu_coeff, mu_bias, wind_anchor

    def _predict_base_scales(self, h):
        raw = self.base_scale_head(h)
        max_delta = self.base_scale_max - self.base_scale_min
        scales = self.base_scale_min + torch.nn.functional.softplus(raw)
        if math.isfinite(max_delta):
            scales = torch.clamp(scales, max=self.base_scale_max)
        return scales

    def _predict_transport_gates(self, h):
        if self.transport_gate_max <= 0.0:
            return h.new_zeros(h.shape[0], self.vec_dim)
        return self.transport_gate_max * torch.sigmoid(self.transport_gate_head(h))

    def _predict_state_bias(self, h):
        if self.state_bias_scale <= 0.0:
            return h.new_zeros(h.shape[0], self.state_dim)
        return self.state_bias_scale * torch.tanh(self.state_bias_head(h))

    def forward(self, x_seq):
        # x_seq: [B, L, C, Y, X]
        batch_size, seq_len, channels, height, width = x_seq.shape
        x = x_seq.reshape(batch_size * seq_len, channels, height, width)
        feat = self.spatial(x)
        embed_dim = feat.shape[-1]
        feat = feat.reshape(batch_size, seq_len, embed_dim)

        h = self.temporal(feat)
        mu, mu_alpha, mu_bias, wind_anchor = self._predict_mu(h, x_seq)
        base_scales = self._predict_base_scales(h)
        transport_gates = self._predict_transport_gates(h)
        state_bias = self._predict_state_bias(h)
        if self.training and self.force_gate_value is not None:
            transport_gates = torch.full_like(transport_gates, float(self.force_gate_value))

        if self.sigma_mode == "global":
            raw11 = self.global_sigma11_params.unsqueeze(0).expand(batch_size, -1)
            raw22 = self.global_sigma22_params.unsqueeze(0).expand(batch_size, -1)
            raw12 = self.global_sigma12_params.unsqueeze(0).expand(batch_size, -1)
        else:
            raw11 = self.sigma11_head(h)
            raw22 = self.sigma22_head(h)
            raw12 = self.sigma12_head(h)
        l11 = self._build_2x2_cholesky(raw11)
        l22 = self._build_2x2_cholesky(raw22)
        sigma11 = l11 @ l11.transpose(-1, -2)
        sigma22 = l22 @ l22.transpose(-1, -2)
        sigma12 = self.sigma_cross_scale * torch.tanh(raw12).reshape(-1, self.advection_dim, self.advection_dim)
        top = torch.cat([sigma11, sigma12], dim=-1)
        bot = torch.cat([sigma12.transpose(-1, -2), sigma22], dim=-1)
        sigma = self._make_spd_4x4(torch.cat([top, bot], dim=-2), min_eig=self.chol_eps)
        chol_factor = torch.linalg.cholesky(sigma)

        return {
            "mu": mu,
            "mu_pairs": mu.reshape(batch_size, self.num_advections, self.advection_dim),
            "mu_coeff_matrix": mu_alpha,
            "mu_alpha": mu_alpha,
            "mu_bias": mu_bias,
            "wind_anchor": wind_anchor,
            "base_scales": base_scales,
            "transport_gates": transport_gates,
            "state_bias": state_bias,
            "sigma": sigma,
            "chol_factor": chol_factor,
        }
