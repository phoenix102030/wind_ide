import torch
from torch import nn


class SpatialCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, 1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TemporalTransformerHead(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, num_layers=2, ff_dim=64, dropout=0.1, out_channels=2):
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
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x):
        # x: [B, L, C, Y, X]
        b, l, c, y, xdim = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).reshape(b * y * xdim, l, c)
        enc = self.encoder(tokens)
        last = enc[:, -1, :]
        out = self.head(last)  # [B*Y*X, 2]
        out = out.reshape(b, y, xdim, 2).permute(0, 3, 1, 2)  # [B, 2, Y, X]
        return out


class CNNTransformerAdjuster(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dim=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        dropout=0.1,
        delta_scale=0.2,
    ):
        super().__init__()
        self.delta_scale = delta_scale
        self.cnn = SpatialCNNEncoder(in_channels=in_channels, hidden_dim=hidden_dim, out_dim=embed_dim)
        self.temporal = TemporalTransformerHead(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            out_channels=2,
        )

    def forward(self, z_seq):
        # z_seq: [B, L, 3, Y, X]
        b, l, c, y, x = z_seq.shape
        z = z_seq.reshape(b * l, c, y, x)
        feat = self.cnn(z)
        e = feat.shape[1]
        feat = feat.reshape(b, l, e, y, x)

        raw = self.temporal(feat)              # [B, 2, Y, X]
        delta = self.delta_scale * torch.tanh(raw)
        # delta[:,0] = delta_log_ell_par
        # delta[:,1] = delta_log_ell_perp
        return delta