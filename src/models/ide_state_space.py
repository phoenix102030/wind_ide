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

    The global IDE parameters live in this model:
      - process / observation noise
      - damping
      - static cross-component coupling B

    The ML model supplies the time-varying advection parameters:
      vec(M_t) ~ N_4(mu_t, Sigma_t)

    and Sigma_t enters the propagation shape directly through D_ij.
    """

    def __init__(
        self,
        dt=1.0,
        num_sites=3,
        init_log_ell_par=0.5,
        init_log_ell_perp=0.0,
        init_log_q_proc=-2.0,
        init_log_r_obs=-2.0,
        init_log_p0=0.0,
        init_log_damping=0.0,
    ):
        super().__init__()
        self.dt = dt
        self.num_sites = num_sites
        self.vec_dim = 2
        self.state_dim = num_sites * self.vec_dim

        self.log_q_proc = nn.Parameter(torch.tensor(float(init_log_q_proc)))
        self.log_r_obs = nn.Parameter(torch.tensor(float(init_log_r_obs)))
        self.log_p0 = nn.Parameter(torch.tensor(float(init_log_p0)))
        self.log_damping = nn.Parameter(torch.tensor(float(init_log_damping)))

        self.init_mean = nn.Parameter(torch.zeros(self.state_dim))
        self.coupling_raw = nn.Parameter(torch.zeros(self.vec_dim, self.vec_dim))

        selector = torch.zeros(self.vec_dim, self.vec_dim, self.vec_dim * self.vec_dim)
        selector[0, :, 0:2] = torch.eye(self.vec_dim)
        selector[1, :, 2:4] = torch.eye(self.vec_dim)
        self.register_buffer("row_selector", selector)

    @property
    def ell_par(self):
        return self.log_q_proc.new_tensor(1.0)

    @property
    def ell_perp(self):
        return self.log_q_proc.new_tensor(1.0)

    @property
    def q_proc(self):
        return torch.exp(self.log_q_proc)

    @property
    def r_obs(self):
        return torch.exp(self.log_r_obs)

    @property
    def p0(self):
        return torch.exp(self.log_p0)

    @property
    def damping(self):
        return torch.exp(self.log_damping)

    @property
    def base_coupling(self):
        eye2 = torch.eye(self.vec_dim, device=self.coupling_raw.device, dtype=self.coupling_raw.dtype)
        return eye2 + 0.25 * torch.tanh(self.coupling_raw)

    def noise_regularization(self):
        return self.q_proc.square() + self.r_obs.square()

    @torch.no_grad()
    def clamp_parameters_(self, log_min=-4.0, log_max=2.0):
        self.log_q_proc.clamp_(log_min, log_max)
        self.log_r_obs.clamp_(log_min, log_max)
        self.log_p0.clamp_(log_min, log_max)
        self.log_damping.clamp_(log_min, log_max)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for legacy_key in ("log_ell_par", "log_ell_perp"):
            full_key = f"{prefix}{legacy_key}"
            if full_key in state_dict:
                state_dict.pop(full_key)
        coupling_key = f"{prefix}coupling_raw"
        if coupling_key in state_dict:
            coupling = state_dict[coupling_key]
            if tuple(coupling.shape) == (self.state_dim, self.state_dim):
                blocks = []
                for i in range(self.num_sites):
                    for j in range(self.num_sites):
                        r0 = i * self.vec_dim
                        c0 = j * self.vec_dim
                        blocks.append(coupling[r0:r0 + self.vec_dim, c0:c0 + self.vec_dim])
                state_dict[coupling_key] = torch.stack(blocks, dim=0).mean(dim=0)
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

    def _selector(self, idx, device, dtype):
        return self.row_selector[idx].to(device=device, dtype=dtype)

    def _pair_selector(self, i, j, device, dtype):
        if i == j:
            return self._selector(i, device, dtype)
        return 0.5 * (self._selector(i, device, dtype) + self._selector(j, device, dtype))

    def build_component_kernels(self, site_lon, site_lat, dynamics_t, device, dtype):
        batch_size = site_lon.shape[0] if site_lon.ndim > 1 else dynamics_t["mu"].shape[0]
        site_lon, site_lat = self._expand_sites(site_lon, site_lat, batch_size)
        coords = project_lon_lat(site_lon, site_lat)                         # [B,S,2]
        h = coords[:, :, None, :] - coords[:, None, :, :]                    # [B,S,S,2]

        mu_t, sigma_t = self._parse_dynamics(dynamics_t, batch_size, device, dtype)

        eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
        base_D = eye2.expand(batch_size, -1, -1)

        kernels = []
        for target_idx in range(self.vec_dim):
            row_kernels = []
            for source_idx in range(self.vec_dim):
                selector = self._pair_selector(target_idx, source_idx, device, dtype)      # [2,4]
                drift_mean = torch.einsum("ab,nb->na", selector, mu_t)                      # [B,2]
                drift_cov = torch.einsum("ab,nbc,dc->nad", selector, sigma_t, selector)     # [B,2,2]
                drift_cov = 0.5 * (drift_cov + drift_cov.transpose(-1, -2))

                D = base_D + 2.0 * (self.dt ** 2) * drift_cov + 1e-4 * eye2
                D_inv = torch.linalg.inv(D)
                drift = self.dt * drift_mean[:, None, None, :]
                diff = h - drift

                q = torch.einsum("bijd,bdf,bijf->bij", diff, D_inv, diff)
                _, logdet = torch.linalg.slogdet(D)
                kernel = torch.exp(-0.5 * (q + logdet[:, None, None]))
                kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                row_kernels.append(kernel)
            kernels.append(torch.stack(row_kernels, dim=1))
        return torch.stack(kernels, dim=1)                                   # [B,2,2,S,S]

    def build_transition_matrix(self, site_lon, site_lat, dynamics_t, device, dtype):
        batch_size = site_lon.shape[0] if site_lon.ndim > 1 else 1
        kernels = self.build_component_kernels(
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_t=dynamics_t if dynamics_t is not None else self._default_dynamics(batch_size, device, dtype),
            device=device,
            dtype=dtype,
        )

        coupling = self.base_coupling.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        operator_blocks = kernels * coupling                                 # [B,2,2,S,S]
        operator = operator_blocks.permute(0, 1, 3, 2, 4).reshape(batch_size, self.state_dim, self.state_dim)

        eye = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        damping_t = self.damping.to(device=device, dtype=dtype)
        A = eye + self.dt * (operator - damping_t * eye)
        Q = (self.q_proc.to(device=device, dtype=dtype) ** 2) * eye
        return A, Q

    def _dynamics_at_t(self, dynamics_seq, t, batch_size, device, dtype):
        if dynamics_seq is None:
            return self._default_dynamics(batch_size, device, dtype)
        return {
            "mu": dynamics_seq["mu"][:, t],
            "sigma": dynamics_seq["sigma"][:, t],
        }

    def _init_filter_state(self, batch_size, device, dtype):
        eye = torch.eye(self.state_dim, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        Rm = (self.r_obs.to(device=device, dtype=dtype) ** 2) * eye
        m = self.init_mean.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        P = (self.p0.to(device=device, dtype=dtype) ** 2) * eye
        return m, P, Rm, eye

    def sequence_nll(self, z_seq, site_lon, site_lat, dynamics_seq=None):
        z = self._flatten_state(z_seq)
        batch_size, steps, _ = z.shape

        if dynamics_seq is not None and dynamics_seq["mu"].shape[1] != steps - 1:
            raise ValueError(f"Need dynamics length = z length - 1, got z={steps}, dyn={dynamics_seq['mu'].shape[1]}")

        device = z.device
        dtype = z.dtype
        m, P, Rm, eye = self._init_filter_state(batch_size, device, dtype)

        total_nll = z.new_tensor(0.0)
        for t in range(steps):
            z_t = z[:, t]
            innov = z_t - m
            S_mat = P + Rm + 1e-5 * eye

            S_inv = torch.linalg.inv(S_mat)
            _, logdet = torch.linalg.slogdet(S_mat)
            maha = torch.einsum("bi,bij,bj->b", innov, S_inv, innov)
            total_nll = total_nll + 0.5 * (logdet + maha + self.state_dim * math.log(2 * math.pi)).mean()

            K_gain = P @ S_inv
            m = m + torch.einsum("bij,bj->bi", K_gain, innov)
            I_K = eye - K_gain
            P = I_K @ P @ I_K.transpose(-1, -2) + K_gain @ Rm @ K_gain.transpose(-1, -2)

            if t == steps - 1:
                continue

            dynamics_t = self._dynamics_at_t(dynamics_seq, t, batch_size, device, dtype)
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q

        return total_nll / max(steps, 1)

    @torch.no_grad()
    def predict_sequence(self, z_seq, site_lon, site_lat, dynamics_seq=None):
        z = self._flatten_state(z_seq)
        batch_size, steps, _ = z.shape
        device = z.device
        dtype = z.dtype
        m, P, Rm, eye = self._init_filter_state(batch_size, device, dtype)

        preds = []
        for t in range(steps - 1):
            z_t = z[:, t]
            innov = z_t - m
            S_mat = P + Rm + 1e-5 * eye
            S_inv = torch.linalg.inv(S_mat)

            K_gain = P @ S_inv
            m = m + torch.einsum("bij,bj->bi", K_gain, innov)
            I_K = eye - K_gain
            P = I_K @ P @ I_K.transpose(-1, -2) + K_gain @ Rm @ K_gain.transpose(-1, -2)

            dynamics_t = self._dynamics_at_t(dynamics_seq, t, batch_size, device, dtype)
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q
            preds.append(m.reshape(batch_size, self.num_sites, self.vec_dim))

        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def forecast_next(self, z_hist, site_lon, site_lat, dynamics_seq):
        z = self._flatten_state(z_hist)
        batch_size, steps, _ = z.shape
        if dynamics_seq["mu"].shape[1] != steps:
            raise ValueError(f"Need dynamics length = history length, got hist={steps}, dyn={dynamics_seq['mu'].shape[1]}")

        device = z.device
        dtype = z.dtype
        m, P, Rm, eye = self._init_filter_state(batch_size, device, dtype)

        for t in range(steps):
            z_t = z[:, t]
            innov = z_t - m
            S_mat = P + Rm + 1e-5 * eye
            S_inv = torch.linalg.inv(S_mat)

            K_gain = P @ S_inv
            m = m + torch.einsum("bij,bj->bi", K_gain, innov)
            I_K = eye - K_gain
            P = I_K @ P @ I_K.transpose(-1, -2) + K_gain @ Rm @ K_gain.transpose(-1, -2)

            dynamics_t = self._dynamics_at_t(dynamics_seq, t, batch_size, device, dtype)
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q

        return m.reshape(batch_size, self.num_sites, self.vec_dim)

    @torch.no_grad()
    def forecast_multistep(self, z_hist, site_lon, site_lat, dynamics_hist, dynamics_future):
        z = self._flatten_state(z_hist)
        batch_size, steps, _ = z.shape
        if dynamics_hist["mu"].shape[1] != steps - 1:
            raise ValueError(f"Need hist dynamics length = history transitions, got hist={steps}, dyn={dynamics_hist['mu'].shape[1]}")

        device = z.device
        dtype = z.dtype
        m, P, Rm, eye = self._init_filter_state(batch_size, device, dtype)

        for t in range(steps):
            z_t = z[:, t]
            innov = z_t - m
            S_mat = P + Rm + 1e-5 * eye
            S_inv = torch.linalg.inv(S_mat)

            K_gain = P @ S_inv
            m = m + torch.einsum("bij,bj->bi", K_gain, innov)
            I_K = eye - K_gain
            P = I_K @ P @ I_K.transpose(-1, -2) + K_gain @ Rm @ K_gain.transpose(-1, -2)

            if t == steps - 1:
                continue

            dynamics_t = self._dynamics_at_t(dynamics_hist, t, batch_size, device, dtype)
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q

        preds = []
        horizon = dynamics_future["mu"].shape[1]
        for h in range(horizon):
            dynamics_t = {
                "mu": dynamics_future["mu"][:, h],
                "sigma": dynamics_future["sigma"][:, h],
            }
            A, Q = self.build_transition_matrix(site_lon, site_lat, dynamics_t, device, dtype)
            m = torch.einsum("bij,bj->bi", A, m)
            P = A @ P @ A.transpose(-1, -2) + Q + 1e-5 * eye
            preds.append(m.reshape(batch_size, self.num_sites, self.vec_dim))

        return torch.stack(preds, dim=1)
