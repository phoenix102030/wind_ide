import math

import torch
from torch import nn


def project_lon_lat(site_lon, site_lat):
    if site_lon.ndim == 1:
        site_lon = site_lon.unsqueeze(0)
    if site_lat.ndim == 1:
        site_lat = site_lat.unsqueeze(0)

    lat0 = site_lat.mean(dim=-1, keepdim=True)
    lon0 = site_lon.mean(dim=-1, keepdim=True)

    km_per_deg = site_lat.new_tensor(111.32)
    lon_scale = km_per_deg * torch.cos(torch.deg2rad(lat0))

    x = (site_lon - lon0) * lon_scale
    y = (site_lat - lat0) * km_per_deg
    coords = torch.stack([x, y], dim=-1)

    h = coords[:, :, None, :] - coords[:, None, :, :]
    dist = torch.sqrt((h ** 2).sum(dim=-1) + 1e-12)
    positive = dist > 1e-6
    scale = dist.sum(dim=(-1, -2)) / positive.sum(dim=(-1, -2)).clamp_min(1)
    coords = coords / scale[:, None, None].clamp_min(1e-6)
    return coords


class IDEStateSpaceModel(nn.Module):
    """
    State Y_t is stored as [u(s1), v(s1), ..., u(sS), v(sS)].

    The ML model supplies time-varying advection parameters
        vec(M_t) ~ N_4(mu_t, Sigma_t)

    and Sigma_t enters the propagation shape directly through D_ij.

    The IDE parameters are also time-varying, but are learned directly as
    piecewise-constant sequences over absolute time rather than being driven by
    a separate NWP network.
    """

    def __init__(
        self,
        dt=1.0,
        num_sites=3,
        total_steps=1,
        param_window=1,
        init_log_ell_par=0.5,
        init_log_ell_perp=0.0,
        init_log_q_proc=-2.0,
        init_log_r_obs=-2.0,
        init_log_p0=0.0,
        init_log_damping=0.0,
    ):
        super().__init__()
        del init_log_ell_par
        del init_log_ell_perp

        self.dt = dt
        self.num_sites = num_sites
        self.vec_dim = 2
        self.state_dim = num_sites * self.vec_dim
        self.total_steps = max(int(total_steps), 1)
        self.param_window = max(int(param_window), 1)
        self.num_knots = math.ceil(self.total_steps / self.param_window)

        self.log_q_proc_knots = nn.Parameter(torch.full((self.num_knots,), float(init_log_q_proc)))
        self.log_r_obs_knots = nn.Parameter(torch.full((self.num_knots,), float(init_log_r_obs)))
        self.log_p0_knots = nn.Parameter(torch.full((self.num_knots,), float(init_log_p0)))
        self.log_damping_knots = nn.Parameter(torch.full((self.num_knots,), float(init_log_damping)))
        self.init_mean_knots = nn.Parameter(torch.zeros(self.num_knots, self.state_dim))

        selector = torch.zeros(self.vec_dim, self.vec_dim, self.vec_dim * self.vec_dim)
        selector[0, :, 0:2] = torch.eye(self.vec_dim)
        selector[1, :, 2:4] = torch.eye(self.vec_dim)
        self.register_buffer("row_selector", selector)

    @property
    def ell_par(self):
        return self.log_q_proc_knots.new_tensor(1.0)

    @property
    def ell_perp(self):
        return self.log_q_proc_knots.new_tensor(1.0)

    def _expand_scalar_series(self, knots):
        return knots.repeat_interleave(self.param_window, dim=0)[:self.total_steps]

    def _expand_vector_series(self, knots):
        return knots.repeat_interleave(self.param_window, dim=0)[:self.total_steps]

    @property
    def q_proc_series(self):
        return torch.exp(self._expand_scalar_series(self.log_q_proc_knots))

    @property
    def r_obs_series(self):
        return torch.exp(self._expand_scalar_series(self.log_r_obs_knots))

    @property
    def p0_series(self):
        return torch.exp(self._expand_scalar_series(self.log_p0_knots))

    @property
    def damping_series(self):
        return torch.exp(self._expand_scalar_series(self.log_damping_knots))

    @property
    def init_mean_series(self):
        return self._expand_vector_series(self.init_mean_knots)

    @property
    def q_proc(self):
        return self.q_proc_series.mean()

    @property
    def r_obs(self):
        return self.r_obs_series.mean()

    @property
    def p0(self):
        return self.p0_series.mean()

    @property
    def damping(self):
        return self.damping_series.mean()

    def noise_regularization(self):
        return self.q_proc_series.square().mean() + self.r_obs_series.square().mean()

    @torch.no_grad()
    def clamp_parameters_(self, log_min=-4.0, log_max=2.0):
        self.log_q_proc_knots.clamp_(log_min, log_max)
        self.log_r_obs_knots.clamp_(log_min, log_max)
        self.log_p0_knots.clamp_(log_min, log_max)
        self.log_damping_knots.clamp_(log_min, log_max)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        legacy_to_new = {
            "log_q_proc": "log_q_proc_knots",
            "log_r_obs": "log_r_obs_knots",
            "log_p0": "log_p0_knots",
            "log_damping": "log_damping_knots",
        }
        for legacy_key, new_key in legacy_to_new.items():
            legacy_full = f"{prefix}{legacy_key}"
            new_full = f"{prefix}{new_key}"
            if legacy_full in state_dict and new_full not in state_dict:
                legacy_value = state_dict[legacy_full]
                expanded = legacy_value.reshape(1).repeat(self.num_knots)
                state_dict[new_full] = expanded

        legacy_init_mean = f"{prefix}init_mean"
        new_init_mean = f"{prefix}init_mean_knots"
        if legacy_init_mean in state_dict and new_init_mean not in state_dict:
            legacy_value = state_dict[legacy_init_mean].reshape(1, self.state_dim)
            state_dict[new_init_mean] = legacy_value.repeat(self.num_knots, 1)

        for legacy_key in (
            "log_ell_par",
            "log_ell_perp",
            "coupling_raw",
            "init_mean",
            "log_q_proc",
            "log_r_obs",
            "log_p0",
            "log_damping",
        ):
            full_key = f"{prefix}{legacy_key}"
            if full_key in state_dict:
                state_dict.pop(full_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _expand_sites(self, site_lon, site_lat, batch_size):
        if site_lon.ndim == 1:
            site_lon = site_lon.unsqueeze(0).expand(batch_size, -1)
        if site_lat.ndim == 1:
            site_lat = site_lat.unsqueeze(0).expand(batch_size, -1)
        return site_lon, site_lat

    def _flatten_state(self, z_seq):
        if z_seq.ndim == 4:
            batch_size, steps, num_sites, vec_dim = z_seq.shape
            if num_sites != self.num_sites or vec_dim != self.vec_dim:
                raise ValueError(
                    f"Expected z_seq shape [B,T,{self.num_sites},{self.vec_dim}], got {tuple(z_seq.shape)}"
                )
            return z_seq.reshape(batch_size, steps, self.state_dim)

        if z_seq.ndim == 3:
            batch_size, steps, state_dim = z_seq.shape
            if state_dim != self.state_dim:
                raise ValueError(f"Expected z_seq shape [B,T,{self.state_dim}], got {tuple(z_seq.shape)}")
            return z_seq

        raise ValueError(f"Expected z_seq to have 3 or 4 dims, got {tuple(z_seq.shape)}")

    def _default_dynamics(self, batch_size, device, dtype):
        return {
            "mu": torch.zeros(batch_size, 4, device=device, dtype=dtype),
            "sigma": torch.zeros(batch_size, 4, 4, device=device, dtype=dtype),
        }

    def _parse_dynamics(self, dynamics_t, batch_size, device, dtype):
        if dynamics_t is None:
            dynamics_t = self._default_dynamics(batch_size, device, dtype)
        mu = dynamics_t["mu"].to(device=device, dtype=dtype).reshape(batch_size, 4)
        sigma = dynamics_t["sigma"].to(device=device, dtype=dtype).reshape(batch_size, 4, 4)
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        return mu, sigma

    def _make_spd(self, matrix, min_eig=1e-4):
        matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)

        a = matrix[..., 0, 0]
        b = matrix[..., 0, 1]
        c = matrix[..., 1, 1]

        trace = a + c
        disc = torch.sqrt(((a - c) * 0.5) ** 2 + b.square() + 1e-12)
        eig1 = (trace * 0.5 - disc).clamp_min(min_eig)
        eig2 = (trace * 0.5 + disc).clamp_min(min_eig)

        theta = 0.5 * torch.atan2(2.0 * b, a - c + 1e-12)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        spd = matrix.new_empty(matrix.shape)
        spd[..., 0, 0] = cos_t.square() * eig1 + sin_t.square() * eig2
        spd[..., 1, 1] = sin_t.square() * eig1 + cos_t.square() * eig2
        offdiag = cos_t * sin_t * (eig1 - eig2)
        spd[..., 0, 1] = offdiag
        spd[..., 1, 0] = offdiag
        return 0.5 * (spd + spd.transpose(-1, -2))

    def _inv_and_logdet_2x2(self, matrix, min_det=1e-8):
        a = matrix[..., 0, 0]
        b = matrix[..., 0, 1]
        c = matrix[..., 1, 1]
        det = (a * c - b.square()).clamp_min(min_det)

        inv = matrix.new_empty(matrix.shape)
        inv[..., 0, 0] = c / det
        inv[..., 1, 1] = a / det
        inv[..., 0, 1] = -b / det
        inv[..., 1, 0] = -b / det
        logdet = torch.log(det)
        return inv, logdet

    def _selector(self, idx, device, dtype):
        return self.row_selector[idx].to(device=device, dtype=dtype)

    def _pair_selector(self, i, j, device, dtype):
        if i == j:
            return self._selector(i, device, dtype)
        return 0.5 * (self._selector(i, device, dtype) + self._selector(j, device, dtype))

    def _prepare_start_idx(self, start_idx, batch_size, device):
        if start_idx is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(start_idx):
            start_idx = torch.tensor(start_idx, device=device)
        start_idx = start_idx.to(device=device, dtype=torch.long)
        if start_idx.ndim == 0:
            start_idx = start_idx.unsqueeze(0).expand(batch_size)
        return start_idx.reshape(batch_size)

    def _time_indices(self, start_idx, length):
        offsets = torch.arange(length, device=start_idx.device, dtype=torch.long)
        idx = start_idx[:, None] + offsets[None, :]
        return idx.clamp_min(0).clamp_max(self.total_steps - 1)

    def _gather_scalar(self, series, idx):
        flat = idx.reshape(-1)
        gathered = series.index_select(0, flat)
        return gathered.reshape(*idx.shape)

    def _gather_vector(self, series, idx):
        flat = idx.reshape(-1)
        gathered = series.index_select(0, flat)
        return gathered.reshape(*idx.shape, series.shape[-1])

    def _get_time_params(self, idx):
        return {
            "q_proc": self._gather_scalar(self.q_proc_series, idx),
            "r_obs": self._gather_scalar(self.r_obs_series, idx),
            "p0": self._gather_scalar(self.p0_series, idx),
            "damping": self._gather_scalar(self.damping_series, idx),
            "init_mean": self._gather_vector(self.init_mean_series, idx),
        }

    def build_component_kernels(self, site_lon, site_lat, dynamics_t, device, dtype):
        batch_size = site_lon.shape[0] if site_lon.ndim > 1 else dynamics_t["mu"].shape[0]
        site_lon, site_lat = self._expand_sites(site_lon, site_lat, batch_size)
        coords = project_lon_lat(site_lon, site_lat)
        h = coords[:, :, None, :] - coords[:, None, :, :]

        mu_t, sigma_t = self._parse_dynamics(dynamics_t, batch_size, device, dtype)
        eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
        base_D = eye2.expand(batch_size, -1, -1)

        kernels = []
        for target_idx in range(self.vec_dim):
            row_kernels = []
            for source_idx in range(self.vec_dim):
                selector = self._pair_selector(target_idx, source_idx, device, dtype)
                drift_mean = torch.einsum("ab,nb->na", selector, mu_t)
                drift_cov = torch.einsum("ab,nbc,dc->nad", selector, sigma_t, selector)
                drift_cov = 0.5 * (drift_cov + drift_cov.transpose(-1, -2))

                D = base_D + 2.0 * (self.dt ** 2) * drift_cov + 1e-4 * eye2
                D = self._make_spd(D, min_eig=1e-4)
                D_inv, logdet = self._inv_and_logdet_2x2(D, min_det=1e-8)
                drift = self.dt * drift_mean[:, None, None, :]
                diff = h - drift

                q = torch.einsum("bijd,bdf,bijf->bij", diff, D_inv, diff)
                kernel = torch.exp(-0.5 * (q + logdet[:, None, None]))
                kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                row_kernels.append(kernel)
            kernels.append(torch.stack(row_kernels, dim=1))
        return torch.stack(kernels, dim=1)

    def build_transition_matrix(self, site_lon, site_lat, dynamics_t, transition_idx, device, dtype):
        transition_idx = transition_idx.to(device=device, dtype=torch.long).reshape(-1)
        batch_size = transition_idx.shape[0]
        kernels = self.build_component_kernels(
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_t=dynamics_t if dynamics_t is not None else self._default_dynamics(batch_size, device, dtype),
            device=device,
            dtype=dtype,
        )

        operator = kernels.permute(0, 1, 3, 2, 4).reshape(batch_size, self.state_dim, self.state_dim)
        eye = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

        time_params = self._get_time_params(transition_idx[:, None])
        damping_t = time_params["damping"][:, 0].to(dtype=dtype)
        q_proc_t = time_params["q_proc"][:, 0].to(dtype=dtype)

        A = eye + self.dt * (operator - damping_t[:, None, None] * eye)
        Q = q_proc_t[:, None, None].square() * eye
        return A, Q

    def _dynamics_at_t(self, dynamics_seq, t, batch_size, device, dtype):
        if dynamics_seq is None:
            return self._default_dynamics(batch_size, device, dtype)
        return {
            "mu": dynamics_seq["mu"][:, t],
            "sigma": dynamics_seq["sigma"][:, t],
        }

    def _init_filter_state(self, start_idx, device, dtype):
        batch_size = start_idx.shape[0]
        eye = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        start_params = self._get_time_params(start_idx[:, None])
        r0 = start_params["r_obs"][:, 0].to(dtype=dtype)
        p0 = start_params["p0"][:, 0].to(dtype=dtype)
        m = start_params["init_mean"][:, 0].to(dtype=dtype)
        P = p0[:, None, None].square() * eye
        Rm = r0[:, None, None].square() * eye
        return m, P, Rm, eye

    def sequence_nll(self, z_seq, site_lon, site_lat, dynamics_seq=None, start_idx=None):
        z = self._flatten_state(z_seq)
        batch_size, steps, _ = z.shape

        if dynamics_seq is not None and dynamics_seq["mu"].shape[1] != steps - 1:
            raise ValueError(f"Need dynamics length = z length - 1, got z={steps}, dyn={dynamics_seq['mu'].shape[1]}")

        device = z.device
        dtype = z.dtype
        start_idx = self._prepare_start_idx(start_idx, batch_size, device)
        obs_idx = self._time_indices(start_idx, steps)
        trans_idx = self._time_indices(start_idx, max(steps - 1, 1))

        m, P, _, eye = self._init_filter_state(start_idx, device, dtype)
        r_obs_seq = self._gather_scalar(self.r_obs_series, obs_idx).to(dtype=dtype)

        total_nll = z.new_tensor(0.0)
        for t in range(steps):
            z_t = z[:, t]
            Rm_t = r_obs_seq[:, t][:, None, None].square() * eye
            innov = z_t - m
            S_mat = P + Rm_t + 1e-5 * eye

            S_inv = torch.linalg.inv(S_mat)
            _, logdet = torch.linalg.slogdet(S_mat)
            maha = torch.einsum("bi,bij,bj->b", innov, S_inv, innov)
            total_nll = total_nll + 0.5 * (logdet + maha + self.state_dim * math.log(2 * math.pi)).mean()

            K_gain = P @ S_inv
            m = m + torch.einsum("bij,bj->bi", K_gain, innov)
            I_K = eye - K_gain
            P = I_K @ P @ I_K.transpose(-1, -2) + K_gain @ Rm_t @ K_gain.transpose(-1, -2)

            if t == steps - 1:
                continue

            dynamics_t = self._dynamics_at_t(dynamics_seq, t, batch_size, device, dtype)
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, trans_idx[:, t], device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q

        return total_nll / max(steps, 1)

    @torch.no_grad()
    def predict_sequence(self, z_seq, site_lon, site_lat, dynamics_seq=None, start_idx=None):
        z = self._flatten_state(z_seq)
        batch_size, steps, _ = z.shape
        device = z.device
        dtype = z.dtype
        start_idx = self._prepare_start_idx(start_idx, batch_size, device)
        obs_idx = self._time_indices(start_idx, steps)
        trans_idx = self._time_indices(start_idx, max(steps - 1, 1))

        m, P, _, eye = self._init_filter_state(start_idx, device, dtype)
        r_obs_seq = self._gather_scalar(self.r_obs_series, obs_idx).to(dtype=dtype)

        preds = []
        for t in range(steps - 1):
            z_t = z[:, t]
            Rm_t = r_obs_seq[:, t][:, None, None].square() * eye
            innov = z_t - m
            S_mat = P + Rm_t + 1e-5 * eye
            S_inv = torch.linalg.inv(S_mat)

            K_gain = P @ S_inv
            m = m + torch.einsum("bij,bj->bi", K_gain, innov)
            I_K = eye - K_gain
            P = I_K @ P @ I_K.transpose(-1, -2) + K_gain @ Rm_t @ K_gain.transpose(-1, -2)

            dynamics_t = self._dynamics_at_t(dynamics_seq, t, batch_size, device, dtype)
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, trans_idx[:, t], device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q
            preds.append(m.reshape(batch_size, self.num_sites, self.vec_dim))

        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def forecast_next(self, z_hist, site_lon, site_lat, dynamics_seq, start_idx=None):
        z = self._flatten_state(z_hist)
        batch_size, steps, _ = z.shape
        if dynamics_seq["mu"].shape[1] != steps:
            raise ValueError(f"Need dynamics length = history length, got hist={steps}, dyn={dynamics_seq['mu'].shape[1]}")

        device = z.device
        dtype = z.dtype
        start_idx = self._prepare_start_idx(start_idx, batch_size, device)
        obs_idx = self._time_indices(start_idx, steps)
        trans_idx = self._time_indices(start_idx, steps)

        m, P, _, eye = self._init_filter_state(start_idx, device, dtype)
        r_obs_seq = self._gather_scalar(self.r_obs_series, obs_idx).to(dtype=dtype)

        for t in range(steps):
            z_t = z[:, t]
            Rm_t = r_obs_seq[:, t][:, None, None].square() * eye
            innov = z_t - m
            S_mat = P + Rm_t + 1e-5 * eye
            S_inv = torch.linalg.inv(S_mat)

            K_gain = P @ S_inv
            m = m + torch.einsum("bij,bj->bi", K_gain, innov)
            I_K = eye - K_gain
            P = I_K @ P @ I_K.transpose(-1, -2) + K_gain @ Rm_t @ K_gain.transpose(-1, -2)

            dynamics_t = self._dynamics_at_t(dynamics_seq, t, batch_size, device, dtype)
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, trans_idx[:, t], device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q

        return m.reshape(batch_size, self.num_sites, self.vec_dim)

    @torch.no_grad()
    def forecast_multistep(self, z_hist, site_lon, site_lat, dynamics_hist, dynamics_future, start_idx=None):
        z = self._flatten_state(z_hist)
        batch_size, steps, _ = z.shape
        if dynamics_hist["mu"].shape[1] != steps - 1:
            raise ValueError(f"Need hist dynamics length = history transitions, got hist={steps}, dyn={dynamics_hist['mu'].shape[1]}")

        device = z.device
        dtype = z.dtype
        start_idx = self._prepare_start_idx(start_idx, batch_size, device)
        obs_idx = self._time_indices(start_idx, steps)
        hist_trans_idx = self._time_indices(start_idx, max(steps - 1, 1))
        future_start = start_idx + steps - 1
        future_trans_idx = self._time_indices(future_start, dynamics_future["mu"].shape[1])

        m, P, _, eye = self._init_filter_state(start_idx, device, dtype)
        r_obs_seq = self._gather_scalar(self.r_obs_series, obs_idx).to(dtype=dtype)

        for t in range(steps):
            z_t = z[:, t]
            Rm_t = r_obs_seq[:, t][:, None, None].square() * eye
            innov = z_t - m
            S_mat = P + Rm_t + 1e-5 * eye
            S_inv = torch.linalg.inv(S_mat)

            K_gain = P @ S_inv
            m = m + torch.einsum("bij,bj->bi", K_gain, innov)
            I_K = eye - K_gain
            P = I_K @ P @ I_K.transpose(-1, -2) + K_gain @ Rm_t @ K_gain.transpose(-1, -2)

            if t == steps - 1:
                continue

            dynamics_t = self._dynamics_at_t(dynamics_hist, t, batch_size, device, dtype)
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, hist_trans_idx[:, t], device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q

        preds = []
        horizon = dynamics_future["mu"].shape[1]
        for h in range(horizon):
            dynamics_t = {
                "mu": dynamics_future["mu"][:, h],
                "sigma": dynamics_future["sigma"][:, h],
            }
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, future_trans_idx[:, h], device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q + 1e-5 * eye
            preds.append(m.reshape(batch_size, self.num_sites, self.vec_dim))

        return torch.stack(preds, dim=1)
