from dataclasses import dataclass
import torch
from torch import nn


def rotation_matrix_from_uv(u, v, eps=1e-6):
    norm = torch.sqrt(u**2 + v**2 + eps)
    cos_t = u / norm
    sin_t = v / norm
    return torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1),
    ], dim=-2)


@dataclass
class KernelDiagnostics:
    delta_par_mean: torch.Tensor
    delta_perp_mean: torch.Tensor
    ell_par_mean: torch.Tensor
    ell_perp_mean: torch.Tensor
    sigma_eps: torch.Tensor


class IDEBaselineKernel(nn.Module):
    def __init__(
        self,
        dt=1.0,
        local_radius=2,
        init_log_amp=0.0,
        init_log_ell_par=1.2,
        init_log_ell_perp=0.8,
        init_log_sigma_eps=-2.0,
    ):
        super().__init__()
        self.dt = dt
        self.local_radius = local_radius

        # 当前基线参数
        self.log_amp = nn.Parameter(torch.tensor(float(init_log_amp)))
        self.log_ell_par = nn.Parameter(torch.tensor(float(init_log_ell_par)))
        self.log_ell_perp = nn.Parameter(torch.tensor(float(init_log_ell_perp)))
        self.log_sigma_eps = nn.Parameter(torch.tensor(float(init_log_sigma_eps)))

        # 上一轮/上一时刻参考参数（不参与梯度）
        self.register_buffer("prev_log_amp", torch.tensor(float(init_log_amp)))
        self.register_buffer("prev_log_ell_par", torch.tensor(float(init_log_ell_par)))
        self.register_buffer("prev_log_ell_perp", torch.tensor(float(init_log_ell_perp)))
        self.register_buffer("prev_log_sigma_eps", torch.tensor(float(init_log_sigma_eps)))

    @property
    def amp(self):
        return torch.exp(self.log_amp)

    @property
    def ell_par_base(self):
        return torch.exp(self.log_ell_par)

    @property
    def ell_perp_base(self):
        return torch.exp(self.log_ell_perp)

    @property
    def sigma_eps(self):
        return torch.exp(self.log_sigma_eps)

    def temporal_smoothness_penalty(self):
        return (
            (self.log_amp - self.prev_log_amp) ** 2
            + (self.log_ell_par - self.prev_log_ell_par) ** 2
            + (self.log_ell_perp - self.prev_log_ell_perp) ** 2
            + (self.log_sigma_eps - self.prev_log_sigma_eps) ** 2
        )

    def update_prev_params(self):
        self.prev_log_amp.copy_(self.log_amp.detach())
        self.prev_log_ell_par.copy_(self.log_ell_par.detach())
        self.prev_log_ell_perp.copy_(self.log_ell_perp.detach())
        self.prev_log_sigma_eps.copy_(self.log_sigma_eps.detach())

    def forward_step(self, y_t, u_t, v_t, lat, lon, delta):
        # delta: [B, 2, Y, X]
        if lat.ndim == 2:
            lat = lat.unsqueeze(0).expand_as(y_t)
        if lon.ndim == 2:
            lon = lon.unsqueeze(0).expand_as(y_t)

        B, Y, X = y_t.shape
        pred = torch.zeros_like(y_t)

        delta_par = delta[:, 0]
        delta_perp = delta[:, 1]

        ell_par = self.ell_par_base * torch.exp(delta_par)
        ell_perp = self.ell_perp_base * torch.exp(delta_perp)

        R = rotation_matrix_from_uv(u_t, v_t)

        for iy in range(Y):
            y0, y1 = max(0, iy - self.local_radius), min(Y, iy + self.local_radius + 1)
            for ix in range(X):
                x0, x1 = max(0, ix - self.local_radius), min(X, ix + self.local_radius + 1)

                s = torch.stack([lon[:, iy, ix], lat[:, iy, ix]], dim=-1)
                mu = self.dt * torch.stack([u_t[:, iy, ix], v_t[:, iy, ix]], dim=-1)

                xcoord = torch.stack([lon[:, y0:y1, x0:x1], lat[:, y0:y1, x0:x1]], dim=-1)
                diff = s[:, None, None, :] - xcoord - mu[:, None, None, :]

                epar = ell_par[:, iy, ix]
                eperp = ell_perp[:, iy, ix]

                diag = torch.zeros(B, 2, 2, device=y_t.device, dtype=y_t.dtype)
                diag[:, 0, 0] = epar**2
                diag[:, 1, 1] = eperp**2

                Rt = R[:, iy, ix]
                D = Rt @ diag @ Rt.transpose(-1, -2)
                D_inv = torch.linalg.inv(D)

                q = torch.einsum("bpqc,bcf,bpqf->bpq", diff, D_inv, diff)
                K = self.amp * torch.exp(-0.5 * q)

                y_patch = y_t[:, y0:y1, x0:x1]
                denom = K.sum(dim=(1, 2)).clamp_min(1e-6)
                pred[:, iy, ix] = (K * y_patch).sum(dim=(1, 2)) / denom

        diag = KernelDiagnostics(
            delta_par_mean=delta_par.mean().detach(),
            delta_perp_mean=delta_perp.mean().detach(),
            ell_par_mean=ell_par.mean().detach(),
            ell_perp_mean=ell_perp.mean().detach(),
            sigma_eps=self.sigma_eps.detach(),
        )
        return pred, diag