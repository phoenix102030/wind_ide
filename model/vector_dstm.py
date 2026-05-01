from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .covariance import (
    QRParameters,
    advection_nll_loss,
    l2_regularization,
    safe_cholesky,
    solve_linear_system,
    smoothness_loss,
)
from .vector_attcnn import VectorAdvectionNet
from .vector_kernel import VectorLagrangianKernel


class VectorDSTM(nn.Module):
    """Kalman filtering for the 6D vector-wind state."""

    def __init__(
        self,
        n_sites: int = 3,
        q_init: float = 0.2,
        r_init: float = 0.2,
        jitter: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.state_dim = 2 * n_sites
        self.jitter = jitter
        self.qr_params = QRParameters(
            n_sites=n_sites,
            q_init=q_init,
            r_init=r_init,
            jitter=jitter,
        )

    def process_covariance(self) -> Tensor:
        return self.qr_params.process()

    def observation_covariance(self) -> Tensor:
        return self.qr_params.observation()

    def _default_initial_state(self, z: Tensor) -> Tensor:
        y0 = torch.zeros(self.state_dim, device=z.device, dtype=z.dtype)
        if z.numel() > 0 and z.shape[-1] == self.state_dim:
            finite = torch.isfinite(z[0])
            y0[finite] = z[0, finite]
        return y0

    def kalman_filter(
        self,
        z: Tensor,
        M_seq: Tensor,
        H: Optional[Tensor] = None,
        y0: Optional[Tensor] = None,
        P0: Optional[Tensor] = None,
        reduction: str = "mean",
        return_history: bool = False,
    ) -> dict[str, Tensor]:
        """Run a Cholesky-based Kalman filter and innovation NLL."""
        if z.ndim != 2:
            raise ValueError(f"Expected z shape [T,d], got {tuple(z.shape)}")
        if M_seq.ndim != 3:
            raise ValueError(f"Expected M_seq shape [T,m,m], got {tuple(M_seq.shape)}")
        T, obs_dim = z.shape
        if M_seq.shape[0] != T:
            raise ValueError("M_seq and z must have matching time dimension")
        if M_seq.shape[-2:] != (self.state_dim, self.state_dim):
            raise ValueError(f"M_seq must have trailing shape {(self.state_dim, self.state_dim)}")

        dtype = z.dtype
        device = z.device
        eye_state = torch.eye(self.state_dim, device=device, dtype=dtype)
        H_full = H if H is not None else eye_state
        H_full = H_full.to(device=device, dtype=dtype)
        if H_full.shape != (obs_dim, self.state_dim):
            raise ValueError(f"Expected H shape {(obs_dim, self.state_dim)}, got {tuple(H_full.shape)}")

        Q = self.process_covariance().to(device=device, dtype=dtype)
        R = self.observation_covariance().to(device=device, dtype=dtype)
        if R.shape != (obs_dim, obs_dim):
            raise ValueError(f"Expected R shape {(obs_dim, obs_dim)}, got {tuple(R.shape)}")

        mean = y0.to(device=device, dtype=dtype) if y0 is not None else self._default_initial_state(z)
        cov = P0.to(device=device, dtype=dtype) if P0 is not None else eye_state
        total_nll = z.new_tensor(0.0)
        total_obs = z.new_tensor(0.0)

        pred_means = []
        pred_covs = []
        filt_means = []
        filt_covs = []

        for t in range(T):
            M_t = M_seq[t].to(device=device, dtype=dtype)
            pred_mean = M_t @ mean
            pred_cov = M_t @ cov @ M_t.T + Q
            pred_cov = 0.5 * (pred_cov + pred_cov.T)

            obs_mask = torch.isfinite(z[t])
            if obs_mask.any():
                H_t = H_full[obs_mask]
                z_t = z[t, obs_mask]
                R_t = R[obs_mask][:, obs_mask]
                innovation = z_t - H_t @ pred_mean
                F_t = H_t @ pred_cov @ H_t.T + R_t
                d_t = F_t.shape[0]
                F_t = F_t + self.jitter * torch.eye(d_t, device=device, dtype=dtype)
                L_t = safe_cholesky(F_t)
                alpha = solve_linear_system(F_t, innovation.unsqueeze(-1)).squeeze(-1)
                quad = innovation @ alpha
                logdet = 2.0 * torch.log(torch.diagonal(L_t)).sum()
                total_nll = total_nll + 0.5 * (
                    logdet + quad + d_t * math.log(2.0 * math.pi)
                )
                total_obs = total_obs + d_t

                gain = solve_linear_system(F_t, H_t @ pred_cov).T
                mean = pred_mean + gain @ innovation
                cov = pred_cov - gain @ H_t @ pred_cov
                cov = 0.5 * (cov + cov.T)
            else:
                mean = pred_mean
                cov = pred_cov

            if return_history:
                pred_means.append(pred_mean)
                pred_covs.append(pred_cov)
                filt_means.append(mean)
                filt_covs.append(cov)

        if reduction == "mean":
            nll = total_nll / total_obs.clamp_min(1.0)
        elif reduction == "sum":
            nll = total_nll
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        result = {
            "loss": nll,
            "nll_sum": total_nll,
            "obs_count": total_obs,
            "filter_mean": mean,
            "filter_cov": cov,
        }
        if return_history:
            result.update(
                {
                    "pred_means": torch.stack(pred_means),
                    "pred_covs": torch.stack(pred_covs),
                    "filter_means": torch.stack(filt_means),
                    "filter_covs": torch.stack(filt_covs),
                }
            )
        return result

    def kalman_nll(self, z: Tensor, M_seq: Tensor, H: Optional[Tensor] = None) -> Tensor:
        return self.kalman_filter(z=z, M_seq=M_seq, H=H, reduction="mean")["loss"]

    def get_filter_dist(self, z: Tensor, M_seq: Tensor, H: Optional[Tensor] = None) -> dict[str, Tensor]:
        return self.kalman_filter(z=z, M_seq=M_seq, H=H, return_history=True)

    def get_forecast_dist(
        self,
        filter_mean: Tensor,
        filter_cov: Tensor,
        future_M: Tensor,
    ) -> tuple[Tensor, Tensor]:
        Q = self.process_covariance().to(device=filter_mean.device, dtype=filter_mean.dtype)
        mean = future_M @ filter_mean
        cov = future_M @ filter_cov @ future_M.T + Q
        return mean, 0.5 * (cov + cov.T)

    def torch_get_kf(self, z: Tensor, M_seq: Tensor, H: Optional[Tensor] = None) -> dict[str, Tensor]:
        return self.get_filter_dist(z=z, M_seq=M_seq, H=H)

    def torch_e_step(self, z: Tensor, M_seq: Tensor, H: Optional[Tensor] = None) -> dict[str, Tensor]:
        return self.get_filter_dist(z=z, M_seq=M_seq, H=H)

    def torch_multi_step_forecast(
        self,
        filter_mean: Tensor,
        filter_cov: Tensor,
        future_M_seq: Tensor,
    ) -> dict[str, Tensor]:
        means = []
        covs = []
        mean = filter_mean
        cov = filter_cov
        for M_t in future_M_seq:
            mean, cov = self.get_forecast_dist(mean, cov, M_t)
            means.append(mean)
            covs.append(cov)
        return {"means": torch.stack(means), "covs": torch.stack(covs)}


class VectorMIDE(nn.Module):
    """End-to-end VectorMIDE model."""

    def __init__(
        self,
        n_sites: int = 3,
        in_channels: int = 6,
        hidden_dim: int = 64,
        mu_scale_init: float = 1.0,
        component_mixing_floor: float = 0.0,
        network_type: str = "cnn_transformer",
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        transformer_dim_feedforward: int = 256,
        transformer_dropout: float = 0.1,
        transformer_causal: bool = True,
        transformer_max_len: int = 4096,
        dt: float = 1.0,
        gamma: float = 0.0,
        row_normalize: bool = True,
        use_spectral_scaling: bool = False,
        kernel_jitter: float = 1.0e-5,
        ell_init: float = 1.0,
        ell_min: float = 0.05,
        ell_max: float = 10.0,
        learnable_gamma: bool = False,
        q_init: float = 0.2,
        r_init: float = 0.2,
        kalman_jitter: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.state_dim = 2 * n_sites
        self.net = VectorAdvectionNet(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            mu_scale_init=mu_scale_init,
            component_mixing_floor=component_mixing_floor,
            network_type=network_type,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_dropout=transformer_dropout,
            transformer_causal=transformer_causal,
            transformer_max_len=transformer_max_len,
        )
        self.kernel = VectorLagrangianKernel(
            n_dim=n_sites,
            dt=dt,
            gamma=gamma,
            row_normalize=row_normalize,
            use_spectral_scaling=use_spectral_scaling,
            jitter=kernel_jitter,
            ell_init=ell_init,
            ell_min=ell_min,
            ell_max=ell_max,
            learnable_gamma=learnable_gamma,
        )
        self.dstm = VectorDSTM(
            n_sites=n_sites,
            q_init=q_init,
            r_init=r_init,
            jitter=kalman_jitter,
        )
        self.qr_params = self.dstm.qr_params

    def forward(self, x: Tensor, coords: Tensor) -> dict[str, Tensor]:
        outputs = self.net(x)
        M = self.kernel(coords, outputs["mu"], outputs["Sigma"], outputs["A"])
        outputs["M"] = M
        return outputs

    def kalman_nll(self, x: Tensor, z: Tensor, coords: Tensor, H: Optional[Tensor] = None) -> Tensor:
        outputs = self.forward(x, coords)
        return self.dstm.kalman_nll(z=z, M_seq=outputs["M"], H=H)

    def multi_step_forecast_loss(
        self,
        z: Tensor,
        M_seq: Tensor,
        filter_means: Tensor,
        horizons: Sequence[int],
        H: Optional[Tensor] = None,
        max_origins: int = 0,
    ) -> Tensor:
        """Masked MSE for forecasts rolled forward from filtered states."""
        if not horizons:
            return z.new_tensor(0.0)

        T = z.shape[0]
        H_full = H.to(device=z.device, dtype=z.dtype) if H is not None else None
        losses = []
        for horizon in horizons:
            h = int(horizon)
            if h < 1 or h >= T:
                continue

            n_origins = T - h
            if max_origins > 0 and n_origins > max_origins:
                origin_idx = torch.linspace(
                    0,
                    n_origins - 1,
                    max_origins,
                    device=z.device,
                ).round().long().unique()
            else:
                origin_idx = torch.arange(n_origins, device=z.device)

            mean = filter_means[origin_idx]
            for step in range(1, h + 1):
                M_batch = M_seq[origin_idx + step].to(device=z.device, dtype=z.dtype)
                mean = torch.bmm(M_batch, mean.unsqueeze(-1)).squeeze(-1)

            pred = mean if H_full is None else mean @ H_full.T
            target = z[origin_idx + h]
            mask = torch.isfinite(pred) & torch.isfinite(target)
            if mask.any():
                losses.append((pred[mask] - target[mask]).pow(2).mean())

        if not losses:
            return z.new_tensor(0.0)
        return torch.stack(losses).mean()

    def training_losses(
        self,
        x: Tensor,
        z: Tensor,
        coords: Tensor,
        v_star: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
        lambda_adv: float = 0.1,
        lambda_smooth: float = 0.001,
        lambda_reg: float = 0.0001,
        lambda_multistep: float = 0.0,
        multistep_horizons: Optional[Sequence[int]] = None,
        multistep_max_origins: int = 0,
    ) -> dict[str, Tensor]:
        outputs = self.forward(x, coords)
        use_multistep = lambda_multistep > 0.0 and bool(multistep_horizons)
        if use_multistep:
            kf = self.dstm.kalman_filter(
                z=z,
                M_seq=outputs["M"],
                H=H,
                return_history=True,
            )
            loss_kf = kf["loss"]
            loss_multistep = self.multi_step_forecast_loss(
                z=z,
                M_seq=outputs["M"],
                filter_means=kf["filter_means"],
                horizons=multistep_horizons or (),
                H=H,
                max_origins=multistep_max_origins,
            )
        else:
            loss_kf = self.dstm.kalman_nll(z=z, M_seq=outputs["M"], H=H)
            loss_multistep = loss_kf.new_tensor(0.0)
        loss_adv = advection_nll_loss(v_star, outputs["mu"], outputs["Sigma"])
        loss_smooth = smoothness_loss(outputs["mu"], outputs["A"])
        reg_params = list(self.kernel.parameters()) + list(self.qr_params.parameters())
        loss_reg = l2_regularization(reg_params)
        total = (
            loss_kf
            + lambda_adv * loss_adv
            + lambda_smooth * loss_smooth
            + lambda_reg * loss_reg
            + lambda_multistep * loss_multistep
        )
        loss_forecast = loss_kf + lambda_multistep * loss_multistep
        return {
            "loss": total,
            "loss_forecast": loss_forecast,
            "loss_kf": loss_kf,
            "loss_adv": loss_adv,
            "loss_smooth": loss_smooth,
            "loss_reg": loss_reg,
            "loss_multistep": loss_multistep,
            **outputs,
        }
