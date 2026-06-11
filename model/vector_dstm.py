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
        bounded_qr: bool = False,
        q_std_min: float = 0.01,
        q_std_max: float = 1.0,
        r_std_min: float = 0.01,
        r_std_max: float = 1.0,
        q_corr_limit: float = 0.95,
        r_corr_limit: float = 0.95,
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
            bounded=bounded_qr,
            q_std_min=q_std_min,
            q_std_max=q_std_max,
            r_std_min=r_std_min,
            r_std_max=r_std_max,
            q_corr_limit=q_corr_limit,
            r_corr_limit=r_corr_limit,
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
        control_seq: Optional[Tensor] = None,
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
        if control_seq is not None:
            if control_seq.shape != (T, self.state_dim):
                raise ValueError(f"Expected control_seq shape {(T, self.state_dim)}, got {tuple(control_seq.shape)}")
            control_seq = control_seq.to(device=z.device, dtype=z.dtype)

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
            control_t = control_seq[t] if control_seq is not None else 0.0
            pred_mean = M_t @ mean + control_t
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
                solved_innovation = solve_linear_system(F_t, innovation.unsqueeze(-1)).squeeze(-1)
                quad = innovation @ solved_innovation
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

    def kalman_nll(
        self,
        z: Tensor,
        M_seq: Tensor,
        control_seq: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
    ) -> Tensor:
        return self.kalman_filter(
            z=z,
            M_seq=M_seq,
            control_seq=control_seq,
            H=H,
            reduction="mean",
        )["loss"]

    def get_filter_dist(
        self,
        z: Tensor,
        M_seq: Tensor,
        control_seq: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        return self.kalman_filter(
            z=z,
            M_seq=M_seq,
            control_seq=control_seq,
            H=H,
            return_history=True,
        )

    def get_forecast_dist(
        self,
        filter_mean: Tensor,
        filter_cov: Tensor,
        future_M: Tensor,
        future_control: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        Q = self.process_covariance().to(device=filter_mean.device, dtype=filter_mean.dtype)
        control = 0.0 if future_control is None else future_control.to(
            device=filter_mean.device,
            dtype=filter_mean.dtype,
        )
        mean = future_M @ filter_mean + control
        cov = future_M @ filter_cov @ future_M.T + Q
        return mean, 0.5 * (cov + cov.T)

    def torch_get_kf(
        self,
        z: Tensor,
        M_seq: Tensor,
        control_seq: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        return self.get_filter_dist(z=z, M_seq=M_seq, control_seq=control_seq, H=H)

    def torch_e_step(
        self,
        z: Tensor,
        M_seq: Tensor,
        control_seq: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        return self.get_filter_dist(z=z, M_seq=M_seq, control_seq=control_seq, H=H)

    def torch_multi_step_forecast(
        self,
        filter_mean: Tensor,
        filter_cov: Tensor,
        future_M_seq: Tensor,
        future_control_seq: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        means = []
        covs = []
        mean = filter_mean
        cov = filter_cov
        if future_control_seq is not None and future_control_seq.shape[0] != future_M_seq.shape[0]:
            raise ValueError("future_control_seq and future_M_seq must have matching time dimension")
        for step, M_t in enumerate(future_M_seq):
            control_t = None if future_control_seq is None else future_control_seq[step]
            mean, cov = self.get_forecast_dist(mean, cov, M_t, future_control=control_t)
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
        component_specific_mu: bool = False,
        advection_mode: str = "component",
        deformation_scale: float = 0.3,
        anchored_advection: bool = False,
        advection_residual_scale: float = 1.0,
        dt: float = 1.0,
        gamma: float = 0.0,
        row_normalize: bool = True,
        use_spectral_scaling: bool = False,
        kernel_jitter: float = 1.0e-5,
        ell_init: float = 1.0,
        ell_min: float = 0.05,
        ell_max: float = 10.0,
        learnable_ell: bool = True,
        learnable_gamma: bool = False,
        q_init: float = 0.2,
        r_init: float = 0.2,
        bounded_qr: bool = False,
        q_std_min: float = 0.01,
        q_std_max: float = 1.0,
        r_std_min: float = 0.01,
        r_std_max: float = 1.0,
        q_corr_limit: float = 0.95,
        r_corr_limit: float = 0.95,
        kalman_jitter: float = 1.0e-5,
        transition_kernel_weight: bool = False,
        transition_kernel_weight_init: float = 1.0,
        transition_kernel_weight_min: float = 0.0,
        transition_kernel_weight_max: float = 1.0,
        transition_residual_decay: bool = False,
        transition_residual_decay_init: float = 1.0,
        transition_residual_decay_min: float = 0.0,
        transition_residual_decay_max: float = 1.0,
        transition_control: bool = False,
        transition_control_scale: float = 0.0,
        sigma_scale_init: float = 1.0,
        sigma_scale_min: float = 1.0,
        sigma_scale_max: float = 1.0,
        learnable_sigma_scale: bool = False,
        covariance_mode: str = "free_cholesky",
        covariance_regimes: int = 3,
        covariance_floor: float = 0.0,
        covariance_regime_stds: Sequence[float] | None = None,
        covariance_dynamic_scale: bool = False,
        covariance_scale_init: float = 1.0,
        covariance_scale_min: float = 0.25,
        covariance_scale_max: float = 4.0,
        advection_component_scale: Sequence[float] | None = None,
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
            component_specific_mu=component_specific_mu,
            advection_mode=advection_mode,
            deformation_scale=deformation_scale,
            anchored_advection=anchored_advection,
            advection_residual_scale=advection_residual_scale,
            transition_kernel_weight=transition_kernel_weight,
            transition_kernel_weight_init=transition_kernel_weight_init,
            transition_kernel_weight_min=transition_kernel_weight_min,
            transition_kernel_weight_max=transition_kernel_weight_max,
            transition_residual_decay=transition_residual_decay,
            transition_residual_decay_init=transition_residual_decay_init,
            transition_residual_decay_min=transition_residual_decay_min,
            transition_residual_decay_max=transition_residual_decay_max,
            transition_control_dim=self.state_dim if transition_control else 0,
            transition_control_scale=transition_control_scale,
            covariance_mode=covariance_mode,
            covariance_regimes=covariance_regimes,
            covariance_floor=covariance_floor,
            covariance_regime_stds=covariance_regime_stds,
            covariance_dynamic_scale=covariance_dynamic_scale,
            covariance_scale_init=covariance_scale_init,
            covariance_scale_min=covariance_scale_min,
            covariance_scale_max=covariance_scale_max,
            advection_component_scale=advection_component_scale,
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
            learnable_ell=learnable_ell,
            learnable_gamma=learnable_gamma,
            sigma_scale_init=sigma_scale_init,
            sigma_scale_min=sigma_scale_min,
            sigma_scale_max=sigma_scale_max,
            learnable_sigma_scale=learnable_sigma_scale,
        )
        self.dstm = VectorDSTM(
            n_sites=n_sites,
            q_init=q_init,
            r_init=r_init,
            jitter=kalman_jitter,
            bounded_qr=bounded_qr,
            q_std_min=q_std_min,
            q_std_max=q_std_max,
            r_std_min=r_std_min,
            r_std_max=r_std_max,
            q_corr_limit=q_corr_limit,
            r_corr_limit=r_corr_limit,
        )
        self.qr_params = self.dstm.qr_params

    def shape_transition_matrix(self, base_M: Tensor, outputs: dict[str, Tensor]) -> Tensor:
        M = base_M
        if "kernel_weight" in outputs:
            eye = torch.eye(self.state_dim, device=M.device, dtype=M.dtype).unsqueeze(0)
            weight = outputs["kernel_weight"].to(device=M.device, dtype=M.dtype).view(-1, 1, 1)
            M = (1.0 - weight) * eye + weight * M
        if "residual_decay" in outputs:
            decay = outputs["residual_decay"].to(device=M.device, dtype=M.dtype).view(-1, 1, 1)
            M = decay * M
        return M

    @staticmethod
    def transition_modifier_smoothness(outputs: dict[str, Tensor]) -> Tensor:
        terms = []
        for key in ("kernel_weight", "residual_decay", "transition_control"):
            value = outputs.get(key)
            if value is not None and value.shape[0] > 1:
                terms.append((value[1:] - value[:-1]).pow(2).mean())
        if not terms:
            return outputs["mu"].new_tensor(0.0)
        return torch.stack(terms).mean()

    def forward(self, x: Tensor, coords: Tensor, advection_anchor: Optional[Tensor] = None) -> dict[str, Tensor]:
        outputs = self.net(x, advection_anchor=advection_anchor)
        if "pair_flow_mu" in outputs and "pair_flow_Sigma" in outputs:
            base_M = self.kernel.forward_pairwise_flow(
                coords,
                outputs["pair_flow_mu"],
                outputs["pair_flow_Sigma"],
            )
        elif "flow_mu" in outputs and "flow_Sigma" in outputs and "B" in outputs:
            base_M = self.kernel.forward_shared_flow(
                coords,
                outputs["flow_mu"],
                outputs["flow_Sigma"],
                outputs["B"],
            )
        else:
            base_M = self.kernel(coords, outputs["mu"], outputs["Sigma"], outputs["alpha"])
        M = self.shape_transition_matrix(base_M, outputs)
        outputs["M_base"] = base_M
        outputs["M"] = M
        outputs["kernel_sigma_scale"] = self.kernel.sigma_scale_value(M.device, M.dtype)
        return outputs

    def kalman_nll(
        self,
        x: Tensor,
        z: Tensor,
        coords: Tensor,
        H: Optional[Tensor] = None,
        advection_anchor: Optional[Tensor] = None,
    ) -> Tensor:
        outputs = self.forward(x, coords, advection_anchor=advection_anchor)
        return self.dstm.kalman_nll(
            z=z,
            M_seq=outputs["M"],
            control_seq=outputs.get("transition_control"),
            H=H,
        )

    def multi_step_forecast_loss(
        self,
        z: Tensor,
        M_seq: Tensor,
        filter_means: Tensor,
        horizons: Sequence[int],
        control_seq: Optional[Tensor] = None,
        nwp_baseline: Optional[Tensor] = None,
        hybrid_first_horizon_direct: bool = False,
        hybrid_direct_horizon_steps: int = 1,
        H: Optional[Tensor] = None,
        max_origins: int = 0,
    ) -> Tensor:
        """Masked MSE for forecasts rolled forward from filtered states."""
        if not horizons:
            return z.new_tensor(0.0)
        direct_steps = max(int(hybrid_direct_horizon_steps), 1)
        if hybrid_first_horizon_direct:
            if nwp_baseline is None:
                raise ValueError("hybrid_first_horizon_direct requires nwp_baseline")
            if nwp_baseline.shape != z.shape:
                raise ValueError(
                    f"Expected nwp_baseline shape {tuple(z.shape)}, got {tuple(nwp_baseline.shape)}"
                )
            nwp_baseline = nwp_baseline.to(device=z.device, dtype=z.dtype)

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
                if control_seq is not None:
                    mean = mean + control_seq[origin_idx + step].to(device=z.device, dtype=z.dtype)

            pred = mean if H_full is None else mean @ H_full.T
            if hybrid_first_horizon_direct and h <= direct_steps:
                # In residual-NWP mode, direct measurement persistence at t+1 is
                # (measurement_t - nwp_{t+h}) = z_t + nwp_t - nwp_{t+h}.
                target = z[origin_idx] + nwp_baseline[origin_idx] - nwp_baseline[origin_idx + h]
            else:
                target = z[origin_idx + h]
            mask = torch.isfinite(pred) & torch.isfinite(target)
            if mask.any():
                losses.append((pred[mask] - target[mask]).pow(2).mean())

        if not losses:
            return z.new_tensor(0.0)
        return torch.stack(losses).mean()

    @staticmethod
    def shared_flow_target(v_star: Optional[Tensor]) -> Optional[Tensor]:
        if v_star is None:
            return None
        if v_star.shape[-1] == 2:
            return v_star
        if v_star.shape[-1] == 4:
            return 0.5 * (v_star[..., :2] + v_star[..., 2:])
        raise ValueError(f"Expected v_star trailing size 2 or 4, got {v_star.shape[-1]}")

    def advection_supervision_loss(
        self,
        v_star: Optional[Tensor],
        outputs: dict[str, Tensor],
    ) -> Tensor:
        if v_star is not None and "advection_component_scale" in outputs and v_star.shape[-1] == 4:
            scale = outputs["advection_component_scale"].to(device=v_star.device, dtype=v_star.dtype)
            v_star = v_star * scale
        if "flow_mu" in outputs and "flow_Sigma" in outputs:
            target = self.shared_flow_target(v_star)
            return advection_nll_loss(target, outputs["flow_mu"], outputs["flow_Sigma"])
        return advection_nll_loss(v_star, outputs["mu"], outputs["Sigma"])

    @staticmethod
    def deformation_supervision_loss(
        B_star: Optional[Tensor],
        outputs: dict[str, Tensor],
    ) -> Tensor:
        B = outputs.get("B")
        if B_star is None or B is None:
            return outputs["mu"].new_tensor(0.0)
        valid = torch.isfinite(B_star).all(dim=(-1, -2))
        if valid.sum() == 0:
            return B.new_tensor(0.0)
        return (B[valid] - B_star[valid].to(device=B.device, dtype=B.dtype)).pow(2).mean()

    def projected_sigma_ratio(self, outputs: dict[str, Tensor]) -> Tensor:
        sigma = outputs.get("Sigma")
        if sigma is None or sigma.shape[-2:] != (4, 4):
            return outputs["mu"].new_zeros(outputs["mu"].shape[0])
        device = sigma.device
        dtype = sigma.dtype
        selectors = self.kernel.selectors(device, dtype)
        ell = self.kernel.get_ell().to(device=device, dtype=dtype)
        gamma = self.kernel.gamma_value(device, dtype)
        sigma_scale = self.kernel.sigma_scale_value(device, dtype)
        ratios = []
        for i in range(2):
            for j in range(2):
                projection = self.kernel.component_projection(i, j, selectors, gamma)
                projected = torch.matmul(projection.unsqueeze(0), sigma)
                projected = torch.matmul(projected, projection.T)
                projected_trace = sigma_scale * 2.0 * torch.diagonal(projected, dim1=-2, dim2=-1).sum(dim=-1)
                base_trace = 2.0 * ell[i, j].pow(2)
                ratios.append(projected_trace / base_trace.clamp_min(1.0e-12))
        return torch.stack(ratios, dim=-1).mean(dim=-1)

    @staticmethod
    def _normalize_proxy(proxy: Tensor) -> Tensor:
        proxy = proxy.clamp_min(0.0)
        finite = torch.isfinite(proxy)
        if finite.sum() == 0:
            return torch.zeros_like(proxy)
        values = proxy[finite]
        lo = values.amin()
        hi = values.amax()
        norm = (proxy - lo) / (hi - lo).clamp_min(1.0e-8)
        return torch.where(finite, norm.clamp(0.0, 1.0), torch.zeros_like(norm))

    def sigma_ratio_loss(
        self,
        outputs: dict[str, Tensor],
        covariance_proxy: Optional[Tensor],
        ratio_min: float,
        ratio_max: float,
        eps: float = 1.0e-8,
    ) -> Tensor:
        ratio = self.projected_sigma_ratio(outputs)
        if covariance_proxy is None:
            target = ratio.new_full(ratio.shape, float(ratio_min))
        else:
            proxy = covariance_proxy.to(device=ratio.device, dtype=ratio.dtype).reshape(-1)
            if proxy.shape[0] != ratio.shape[0]:
                raise ValueError(f"Expected covariance_proxy length {ratio.shape[0]}, got {proxy.shape[0]}")
            proxy_norm = self._normalize_proxy(proxy)
            target = float(ratio_min) + (float(ratio_max) - float(ratio_min)) * proxy_norm
        valid = torch.isfinite(ratio) & torch.isfinite(target) & (target > 0)
        if valid.sum() == 0:
            return ratio.new_tensor(0.0)
        return (torch.log(ratio[valid].clamp_min(eps)) - torch.log(target[valid].clamp_min(eps))).pow(2).mean()

    def covariance_proxy_loss(
        self,
        outputs: dict[str, Tensor],
        covariance_proxy_primary: Optional[Tensor],
        covariance_proxy_secondary: Optional[Tensor],
        floor: float = 1.0e-4,
        eps: float = 1.0e-8,
    ) -> Tensor:
        sigma = outputs.get("Sigma")
        if sigma is None or covariance_proxy_primary is None or covariance_proxy_secondary is None:
            return outputs["mu"].new_tensor(0.0)
        primary = covariance_proxy_primary.to(device=sigma.device, dtype=sigma.dtype)
        secondary = covariance_proxy_secondary.to(device=sigma.device, dtype=sigma.dtype)
        if "advection_component_scale" in outputs and primary.shape[-1] == 4:
            scale = outputs["advection_component_scale"].to(device=sigma.device, dtype=sigma.dtype)
            primary = primary * scale
            secondary = secondary * scale
        valid = torch.isfinite(primary).all(dim=-1) & torch.isfinite(secondary).all(dim=-1)
        if valid.sum() == 0:
            return sigma.new_tensor(0.0)
        diff = primary[valid] - secondary[valid]
        target_trace = diff.pow(2).sum(dim=-1) + float(floor) ** 2 * diff.shape[-1]
        pred_trace = torch.diagonal(sigma[valid], dim1=-2, dim2=-1).sum(dim=-1)
        return (torch.log(pred_trace.clamp_min(eps)) - torch.log(target_trace.clamp_min(eps))).pow(2).mean()

    @staticmethod
    def _advection_disagreement(
        outputs: dict[str, Tensor],
        covariance_proxy_primary: Optional[Tensor],
        covariance_proxy_secondary: Optional[Tensor],
        residual_scale: float,
    ) -> tuple[Tensor, Tensor] | None:
        sigma = outputs.get("Sigma")
        if sigma is None or covariance_proxy_primary is None or covariance_proxy_secondary is None:
            return None
        primary = covariance_proxy_primary.to(device=sigma.device, dtype=sigma.dtype)
        secondary = covariance_proxy_secondary.to(device=sigma.device, dtype=sigma.dtype)
        if primary.shape[-1] != 4 or secondary.shape[-1] != 4:
            return None
        if "advection_component_scale" in outputs:
            scale = outputs["advection_component_scale"].to(device=sigma.device, dtype=sigma.dtype)
            primary = primary * scale
            secondary = secondary * scale
        valid = (
            torch.isfinite(primary).all(dim=-1)
            & torch.isfinite(secondary).all(dim=-1)
            & torch.isfinite(sigma).all(dim=(-1, -2))
        )
        if valid.sum() == 0:
            return None
        scaled_residual = (primary[valid] - secondary[valid]) / max(float(residual_scale), 1.0e-8)
        return sigma[valid], scaled_residual

    def covariance_residual_nll_loss(
        self,
        outputs: dict[str, Tensor],
        covariance_proxy_primary: Optional[Tensor],
        covariance_proxy_secondary: Optional[Tensor],
        residual_scale: float = 10.0,
        eps: float = 1.0e-6,
    ) -> Tensor:
        prepared = self._advection_disagreement(
            outputs,
            covariance_proxy_primary=covariance_proxy_primary,
            covariance_proxy_secondary=covariance_proxy_secondary,
            residual_scale=residual_scale,
        )
        if prepared is None:
            return outputs["mu"].new_tensor(0.0)
        sigma, residual = prepared
        terms = []
        eye2 = torch.eye(2, device=sigma.device, dtype=sigma.dtype).unsqueeze(0)
        for block_idx, start in enumerate((0, 2)):
            cov = sigma[:, start : start + 2, start : start + 2] + eps * eye2
            resid = residual[:, start : start + 2]
            L = safe_cholesky(cov)
            solved = solve_linear_system(cov, resid.unsqueeze(-1)).squeeze(-1)
            quad = (resid * solved).sum(dim=-1)
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            terms.append(0.5 * (logdet + quad))
        return torch.cat(terms, dim=0).mean()

    def covariance_shape_loss(
        self,
        outputs: dict[str, Tensor],
        covariance_proxy_primary: Optional[Tensor],
        covariance_proxy_secondary: Optional[Tensor],
        residual_scale: float = 10.0,
        floor: float = 0.05,
        eps: float = 1.0e-8,
    ) -> Tensor:
        prepared = self._advection_disagreement(
            outputs,
            covariance_proxy_primary=covariance_proxy_primary,
            covariance_proxy_secondary=covariance_proxy_secondary,
            residual_scale=residual_scale,
        )
        if prepared is None:
            return outputs["mu"].new_tensor(0.0)
        sigma, residual = prepared
        losses = []
        eye2 = torch.eye(2, device=sigma.device, dtype=sigma.dtype).unsqueeze(0)
        for start in (0, 2):
            pred = sigma[:, start : start + 2, start : start + 2]
            resid = residual[:, start : start + 2]
            target = resid.unsqueeze(-1) @ resid.unsqueeze(-2)
            target = target + float(floor) ** 2 * eye2
            pred_trace = torch.diagonal(pred, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(eps)
            target_trace = torch.diagonal(target, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(eps)
            pred_shape = pred / pred_trace[:, None, None]
            target_shape = target / target_trace[:, None, None]
            losses.append((pred_shape - target_shape).pow(2).mean(dim=(-1, -2)))
        return torch.cat(losses, dim=0).mean()

    def covariance_correlation_loss(
        self,
        outputs: dict[str, Tensor],
        covariance_proxy: Optional[Tensor],
        eps: float = 1.0e-8,
    ) -> Tensor:
        if covariance_proxy is None or outputs.get("Sigma") is None:
            return outputs["mu"].new_tensor(0.0)
        ratio = self.projected_sigma_ratio(outputs)
        proxy = covariance_proxy.to(device=ratio.device, dtype=ratio.dtype).reshape(-1)
        if proxy.shape[0] != ratio.shape[0]:
            raise ValueError(f"Expected covariance_proxy length {ratio.shape[0]}, got {proxy.shape[0]}")
        valid = torch.isfinite(ratio) & torch.isfinite(proxy)
        if valid.sum() < 3:
            return ratio.new_tensor(0.0)
        pred = torch.log(ratio[valid].clamp_min(eps))
        target = self._normalize_proxy(proxy[valid])
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        pred_std = pred_centered.pow(2).mean().sqrt()
        target_std = target_centered.pow(2).mean().sqrt()
        if float(target_std.detach().cpu()) <= eps:
            return ratio.new_tensor(0.0)
        corr = (pred_centered * target_centered).mean() / (pred_std * target_std).clamp_min(eps)
        return 1.0 - corr.clamp(-1.0, 1.0)

    @staticmethod
    def regime_usage_loss(outputs: dict[str, Tensor], eps: float = 1.0e-8) -> Tensor:
        probs = outputs.get("covariance_regime_probs")
        if probs is None:
            return outputs["mu"].new_tensor(0.0)
        probs = probs.clamp_min(eps)
        mean_probs = probs.mean(dim=0)
        target = mean_probs.new_full(mean_probs.shape, 1.0 / float(mean_probs.numel()))
        return (mean_probs * (torch.log(mean_probs.clamp_min(eps)) - torch.log(target))).sum()

    @staticmethod
    def innovation_calibration_loss(kf: dict[str, Tensor], z: Tensor, R: Tensor, H: Optional[Tensor]) -> Tensor:
        pred_means = kf.get("pred_means")
        pred_covs = kf.get("pred_covs")
        if pred_means is None or pred_covs is None:
            return z.new_tensor(0.0)
        obs_dim = z.shape[-1]
        state_dim = pred_means.shape[-1]
        if H is None:
            H_full = torch.eye(state_dim, device=z.device, dtype=z.dtype)
        else:
            H_full = H.to(device=z.device, dtype=z.dtype)
        terms = []
        for t in range(z.shape[0]):
            obs_mask = torch.isfinite(z[t])
            if not obs_mask.any():
                continue
            H_t = H_full[obs_mask]
            R_t = R[obs_mask][:, obs_mask]
            innovation = z[t, obs_mask] - H_t @ pred_means[t]
            F_t = H_t @ pred_covs[t] @ H_t.T + R_t
            F_t = F_t + 1.0e-5 * torch.eye(F_t.shape[0], device=z.device, dtype=z.dtype)
            solved = solve_linear_system(F_t, innovation.unsqueeze(-1)).squeeze(-1)
            normalized_quad = (innovation * solved).sum() / float(F_t.shape[0])
            terms.append((normalized_quad - 1.0).pow(2))
        if not terms:
            return z.new_tensor(0.0)
        return torch.stack(terms).mean()

    def training_losses(
        self,
        x: Tensor,
        z: Tensor,
        coords: Tensor,
        v_star: Optional[Tensor] = None,
        advection_anchor: Optional[Tensor] = None,
        B_star: Optional[Tensor] = None,
        H: Optional[Tensor] = None,
        lambda_adv: float = 0.1,
        lambda_deform: float = 0.0,
        lambda_smooth: float = 0.001,
        lambda_advection_residual: float = 0.0,
        lambda_reg: float = 0.0001,
        lambda_multistep: float = 0.0,
        lambda_sigma_ratio: float = 0.0,
        sigma_ratio_min: float = 0.05,
        sigma_ratio_max: float = 0.5,
        lambda_covariance_proxy: float = 0.0,
        covariance_proxy_floor: float = 1.0e-4,
        lambda_covariance_correlation: float = 0.0,
        lambda_regime_usage: float = 0.0,
        lambda_covariance_residual_nll: float = 0.0,
        lambda_covariance_shape: float = 0.0,
        covariance_residual_scale: float = 10.0,
        covariance_shape_floor: float = 0.05,
        lambda_calibration: float = 0.0,
        multistep_horizons: Optional[Sequence[int]] = None,
        multistep_max_origins: int = 0,
        nwp_baseline: Optional[Tensor] = None,
        covariance_proxy: Optional[Tensor] = None,
        covariance_proxy_primary: Optional[Tensor] = None,
        covariance_proxy_secondary: Optional[Tensor] = None,
        hybrid_first_horizon_direct: bool = False,
        hybrid_direct_horizon_steps: int = 1,
    ) -> dict[str, Tensor]:
        outputs = self.forward(x, coords, advection_anchor=advection_anchor)
        use_multistep = lambda_multistep > 0.0 and bool(multistep_horizons)
        need_history = use_multistep or lambda_calibration > 0.0
        if need_history:
            kf = self.dstm.kalman_filter(
                z=z,
                M_seq=outputs["M"],
                control_seq=outputs.get("transition_control"),
                H=H,
                return_history=True,
            )
            loss_kf = kf["loss"]
            loss_multistep = self.multi_step_forecast_loss(
                z=z,
                M_seq=outputs["M"],
                filter_means=kf["filter_means"],
                horizons=multistep_horizons or (),
                control_seq=outputs.get("transition_control"),
                nwp_baseline=nwp_baseline,
                hybrid_first_horizon_direct=hybrid_first_horizon_direct,
                hybrid_direct_horizon_steps=hybrid_direct_horizon_steps,
                H=H,
                max_origins=multistep_max_origins,
            )
        else:
            loss_kf = self.dstm.kalman_nll(
                z=z,
                M_seq=outputs["M"],
                control_seq=outputs.get("transition_control"),
                H=H,
            )
            loss_multistep = loss_kf.new_tensor(0.0)
        loss_adv = self.advection_supervision_loss(v_star, outputs)
        loss_deform = self.deformation_supervision_loss(B_star, outputs)
        delta_mu = outputs.get("delta_mu")
        loss_advection_residual = (
            delta_mu.pow(2).mean() if delta_mu is not None else loss_kf.new_tensor(0.0)
        )
        smooth_mu = outputs.get("flow_mu", outputs["mu"])
        smooth_matrix = outputs.get("B", outputs["alpha"])
        loss_smooth = smoothness_loss(smooth_mu, smooth_matrix) + self.transition_modifier_smoothness(outputs)
        reg_params = list(self.kernel.parameters()) + list(self.qr_params.parameters())
        loss_reg = l2_regularization(reg_params)
        loss_sigma_ratio = self.sigma_ratio_loss(
            outputs,
            covariance_proxy=covariance_proxy,
            ratio_min=sigma_ratio_min,
            ratio_max=sigma_ratio_max,
        )
        loss_covariance_proxy = self.covariance_proxy_loss(
            outputs,
            covariance_proxy_primary=covariance_proxy_primary,
            covariance_proxy_secondary=covariance_proxy_secondary,
            floor=covariance_proxy_floor,
        )
        loss_covariance_correlation = self.covariance_correlation_loss(
            outputs,
            covariance_proxy=covariance_proxy,
        )
        loss_regime_usage = self.regime_usage_loss(outputs)
        loss_covariance_residual_nll = self.covariance_residual_nll_loss(
            outputs,
            covariance_proxy_primary=covariance_proxy_primary,
            covariance_proxy_secondary=covariance_proxy_secondary,
            residual_scale=covariance_residual_scale,
        )
        loss_covariance_shape = self.covariance_shape_loss(
            outputs,
            covariance_proxy_primary=covariance_proxy_primary,
            covariance_proxy_secondary=covariance_proxy_secondary,
            residual_scale=covariance_residual_scale,
            floor=covariance_shape_floor,
        )
        if lambda_calibration > 0.0 and need_history:
            R = self.dstm.observation_covariance().to(device=z.device, dtype=z.dtype)
            loss_calibration = self.innovation_calibration_loss(kf, z=z, R=R, H=H)
        else:
            loss_calibration = loss_kf.new_tensor(0.0)
        total = (
            loss_kf
            + lambda_adv * loss_adv
            + lambda_deform * loss_deform
            + lambda_smooth * loss_smooth
            + lambda_advection_residual * loss_advection_residual
            + lambda_reg * loss_reg
            + lambda_multistep * loss_multistep
            + lambda_sigma_ratio * loss_sigma_ratio
            + lambda_covariance_proxy * loss_covariance_proxy
            + lambda_covariance_correlation * loss_covariance_correlation
            + lambda_regime_usage * loss_regime_usage
            + lambda_covariance_residual_nll * loss_covariance_residual_nll
            + lambda_covariance_shape * loss_covariance_shape
            + lambda_calibration * loss_calibration
        )
        loss_forecast = loss_kf + lambda_multistep * loss_multistep
        return {
            "loss": total,
            "loss_forecast": loss_forecast,
            "loss_kf": loss_kf,
            "loss_adv": loss_adv,
            "loss_deform": loss_deform,
            "loss_advection_residual": loss_advection_residual,
            "loss_smooth": loss_smooth,
            "loss_reg": loss_reg,
            "loss_multistep": loss_multistep,
            "loss_sigma_ratio": loss_sigma_ratio,
            "loss_covariance_proxy": loss_covariance_proxy,
            "loss_covariance_correlation": loss_covariance_correlation,
            "loss_regime_usage": loss_regime_usage,
            "loss_covariance_residual_nll": loss_covariance_residual_nll,
            "loss_covariance_shape": loss_covariance_shape,
            "loss_calibration": loss_calibration,
            **outputs,
        }
