from torch import nn
from .global_advection_net import GlobalAdvectionNet
from .measurement_ide import MeasurementIDE


class MeasurementWindModel(nn.Module):
    def __init__(
        self,
        dt=1.0,
        hidden_dim=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        dropout=0.1,
        init_log_amp=0.0,
        init_log_ell_par=0.5,
        init_log_ell_perp=0.0,
        init_log_sigma_eps=-2.0,
    ):
        super().__init__()
        self.advection_net = GlobalAdvectionNet(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.ide = MeasurementIDE(
            dt=dt,
            init_log_amp=init_log_amp,
            init_log_ell_par=init_log_ell_par,
            init_log_ell_perp=init_log_ell_perp,
            init_log_sigma_eps=init_log_sigma_eps,
        )

    def forward(self, batch):
        mu_v, Sigma_v = self.advection_net(batch["nwp_seq"])
        y_pred, K = self.ide(
            y_t=batch["y_t"],
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            mu_v=mu_v,
            Sigma_v=Sigma_v,
        )
        return y_pred, mu_v, Sigma_v, K