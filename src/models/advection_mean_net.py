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
    Learn time-varying advection matrix parameters from NWP context.

    The network predicts a random 2x2 advection/mixing matrix M_t:
        vec(M_t) ~ N(mu_t, Sigma_t)

    where mu_t has 4 entries and Sigma_t is constructed from a 4x4 Cholesky
    factor to guarantee positive definiteness.
    """

    def __init__(
        self,
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
        mu_mode="free",
        sigma_mode="network",
        init_global_sigma_diag=0.2,
    ):
        super().__init__()
        self.mu_scale = mu_scale
        self.chol_offdiag_scale = chol_offdiag_scale
        self.chol_diag_max = chol_diag_max
        self.chol_eps = chol_eps
        self.mu_mode = str(mu_mode).lower()
        self.sigma_mode = str(sigma_mode).lower()
        if self.mu_mode not in {"free", "anchored"}:
            raise ValueError(f"Unsupported mu_mode={mu_mode!r}; expected 'free' or 'anchored'.")
        if self.sigma_mode not in {"network", "global"}:
            raise ValueError(f"Unsupported sigma_mode={sigma_mode!r}; expected 'network' or 'global'.")

        self.spatial = SpatialEncoder(in_channels=6, hidden_dim=hidden_dim, out_dim=embed_dim)
        self.temporal = TemporalEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.mu_head = nn.Linear(embed_dim, 4)
        self.chol_head = nn.Linear(embed_dim, 10)
        init_diag_raw = inverse_softplus(max(float(init_global_sigma_diag) - self.chol_eps, 1e-4))
        global_chol_init = torch.zeros(10)
        global_chol_init[:4] = init_diag_raw
        self.global_chol_params = nn.Parameter(global_chol_init)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        global_key = f"{prefix}global_chol_params"
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

    def _build_cholesky(self, chol_raw):
        batch_size = chol_raw.shape[0]
        chol = chol_raw.new_zeros(batch_size, 4, 4)

        diag_raw = chol_raw[:, :4]
        offdiag_raw = chol_raw[:, 4:]

        diag = torch.nn.functional.softplus(diag_raw).clamp_max(self.chol_diag_max) + self.chol_eps
        offdiag = self.chol_offdiag_scale * torch.tanh(offdiag_raw)

        chol[:, 0, 0] = diag[:, 0]
        chol[:, 1, 0] = offdiag[:, 0]
        chol[:, 1, 1] = diag[:, 1]
        chol[:, 2, 0] = offdiag[:, 1]
        chol[:, 2, 1] = offdiag[:, 2]
        chol[:, 2, 2] = diag[:, 2]
        chol[:, 3, 0] = offdiag[:, 3]
        chol[:, 3, 1] = offdiag[:, 4]
        chol[:, 3, 2] = offdiag[:, 5]
        chol[:, 3, 3] = diag[:, 3]
        return chol

    def _extract_wind_anchor(self, x_seq):
        latest_frame = x_seq[:, -1]
        wind_uv = latest_frame[:, 2:4]
        return wind_uv.mean(dim=(-1, -2))

    def forward(self, x_seq):
        # x_seq: [B, L, 6, Y, X]
        batch_size, seq_len, channels, height, width = x_seq.shape
        x = x_seq.reshape(batch_size * seq_len, channels, height, width)
        feat = self.spatial(x)
        embed_dim = feat.shape[-1]
        feat = feat.reshape(batch_size, seq_len, embed_dim)

        h = self.temporal(feat)
        mu_coeff_matrix = self.mu_scale * torch.tanh(self.mu_head(h)).reshape(batch_size, 2, 2)
        if self.mu_mode == "anchored":
            wind_anchor = self._extract_wind_anchor(x_seq)
            mu_matrix = mu_coeff_matrix * wind_anchor.unsqueeze(1)
        else:
            wind_anchor = self._extract_wind_anchor(x_seq)
            mu_matrix = mu_coeff_matrix

        if self.sigma_mode == "global":
            chol_factor = self._build_cholesky(self.global_chol_params.unsqueeze(0).expand(batch_size, -1))
        else:
            chol_factor = self._build_cholesky(self.chol_head(h))
        sigma = chol_factor @ chol_factor.transpose(-1, -2)
        return {
            "mu": mu_matrix.reshape(batch_size, 4),
            "mu_matrix": mu_matrix,
            "mu_coeff_matrix": mu_coeff_matrix,
            "wind_anchor": wind_anchor,
            "sigma": sigma,
            "chol_factor": chol_factor,
        }
