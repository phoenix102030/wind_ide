from torch import nn
from .cnn_transformer import CNNTransformerAdjuster
from .ide_kernel import IDEBaselineKernel


class WindSpeedIDEModel(nn.Module):
    def __init__(
        self,
        dt=1.0,
        local_radius=2,
        hidden_dim=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        dropout=0.1,
        delta_scale=0.2,
        init_log_amp=0.0,
        init_log_ell_par=1.2,
        init_log_ell_perp=0.8,
        init_log_sigma_eps=-2.0,
    ):
        super().__init__()
        self.adjuster = CNNTransformerAdjuster(
            in_channels=3,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            delta_scale=delta_scale,
        )
        self.ide = IDEBaselineKernel(
            dt=dt,
            local_radius=local_radius,
            init_log_amp=init_log_amp,
            init_log_ell_par=init_log_ell_par,
            init_log_ell_perp=init_log_ell_perp,
            init_log_sigma_eps=init_log_sigma_eps,
        )

    def forward(self, batch):
        delta = self.adjuster(batch["z_seq"])
        pred, diag = self.ide.forward_step(
            y_t=batch["y_t"],
            u_t=batch["u_t"],
            v_t=batch["v_t"],
            lat=batch["lat"],
            lon=batch["lon"],
            delta=delta,
        )
        return pred, delta, diag