import torch
from torch import nn


def rotation_from_mu(mu):
    # mu: [B,2]
    angle = torch.atan2(mu[:, 1:2], mu[:, 0:1])
    cos_t = torch.cos(angle)
    sin_t = torch.sin(angle)

    R = torch.zeros(mu.shape[0], 2, 2, device=mu.device, dtype=mu.dtype)
    R[:, 0, 0] = cos_t[:, 0]
    R[:, 0, 1] = -sin_t[:, 0]
    R[:, 1, 0] = sin_t[:, 0]
    R[:, 1, 1] = cos_t[:, 0]
    return R


class MeasurementIDE(nn.Module):
    def __init__(
        self,
        dt=1.0,
        init_log_amp=0.0,
        init_log_ell_par=0.5,
        init_log_ell_perp=0.0,
        init_log_sigma_eps=-2.0,
    ):
        super().__init__()
        self.dt = dt

        self.log_amp = nn.Parameter(torch.tensor(float(init_log_amp)))
        self.log_ell_par = nn.Parameter(torch.tensor(float(init_log_ell_par)))
        self.log_ell_perp = nn.Parameter(torch.tensor(float(init_log_ell_perp)))
        self.log_sigma_eps = nn.Parameter(torch.tensor(float(init_log_sigma_eps)))

    @property
    def amp(self):
        return torch.exp(self.log_amp)

    @property
    def ell_par(self):
        return torch.exp(self.log_ell_par)

    @property
    def ell_perp(self):
        return torch.exp(self.log_ell_perp)

    @property
    def sigma_eps(self):
        return torch.exp(self.log_sigma_eps)

    def build_kernel(self, site_lon, site_lat, mu_v, Sigma_v):
        """
        site_lon/site_lat: [B,3] or [3]
        mu_v: [B,2]
        Sigma_v: [B,2,2]
        return K: [B,3,3]
        """
        if site_lon.ndim == 1:
            site_lon = site_lon.unsqueeze(0).expand(mu_v.shape[0], -1)
        if site_lat.ndim == 1:
            site_lat = site_lat.unsqueeze(0).expand(mu_v.shape[0], -1)

        B = mu_v.shape[0]
        coords = torch.stack([site_lon, site_lat], dim=-1)   # [B,3,2]

        h = coords[:, :, None, :] - coords[:, None, :, :]    # [B,3,3,2]
        drift = self.dt * mu_v[:, None, None, :]             # [B,1,1,2]
        diff = h - drift

        R = rotation_from_mu(mu_v)                           # [B,2,2]
        D0 = torch.zeros(B, 2, 2, device=mu_v.device, dtype=mu_v.dtype)
        D0[:, 0, 0] = self.ell_par ** 2
        D0[:, 1, 1] = self.ell_perp ** 2

        D = R @ D0 @ R.transpose(-1, -2) + (self.dt ** 2) * Sigma_v
        D_inv = torch.linalg.inv(D)

        q = torch.einsum("bijd,bdf,bijf->bij", diff, D_inv, diff)
        K = self.amp * torch.exp(-0.5 * q)

        # 行归一化
        K = K / K.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return K

    def forward(self, y_t, site_lon, site_lat, mu_v, Sigma_v):
        """
        y_t: [B,3,2]
        return y_pred: [B,3,2]
        """
        K = self.build_kernel(site_lon, site_lat, mu_v, Sigma_v)   # [B,3,3]
        y_pred = torch.einsum("bij,bjc->bic", K, y_t)              # same K for U/V
        return y_pred, K
