import torch
from torch import nn


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
        # x: [B, C, Y, X]
        h = self.net(x)              # [B, out_dim, 1, 1]
        return h.squeeze(-1).squeeze(-1)  # [B, out_dim]


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
        # x: [B, L, E]
        h = self.encoder(x)
        return h[:, -1, :]  # final token


class GlobalAdvectionNet(nn.Module):
    """
    Output:
      mu:    [B, 2]
      Sigma: [B, 2, 2]  SPD via Cholesky
    """
    def __init__(self, hidden_dim=32, embed_dim=32, num_heads=4, num_layers=2, ff_dim=64, dropout=0.1):
        super().__init__()
        self.spatial = SpatialEncoder(in_channels=6, hidden_dim=hidden_dim, out_dim=embed_dim)
        self.temporal = TemporalEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.mu_head = nn.Linear(embed_dim, 2)
        self.cov_head = nn.Linear(embed_dim, 3)

    def forward(self, x_seq):
        # x_seq: [B, L, 6, Y, X]
        B, L, C, Y, X = x_seq.shape
        x = x_seq.reshape(B * L, C, Y, X)
        feat = self.spatial(x)              # [B*L, E]
        E = feat.shape[-1]
        feat = feat.reshape(B, L, E)        # [B, L, E]

        h = self.temporal(feat)             # [B, E]

        mu = self.mu_head(h)                # [B, 2]
        raw = self.cov_head(h)              # [B, 3]

        a = torch.nn.functional.softplus(raw[:, 0]) + 1e-4
        b = raw[:, 1]
        c = torch.nn.functional.softplus(raw[:, 2]) + 1e-4

        Lmat = torch.zeros(B, 2, 2, device=h.device, dtype=h.dtype)
        Lmat[:, 0, 0] = a
        Lmat[:, 1, 0] = b
        Lmat[:, 1, 1] = c

        Sigma = Lmat @ Lmat.transpose(-1, -2)
        return mu, Sigma