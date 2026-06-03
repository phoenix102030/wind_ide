from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .covariance import safe_cholesky, solve_linear_system


def _logit(value: float) -> float:
    value = min(max(value, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


class VectorLagrangianKernel(nn.Module):
    """Build vector-wind transition matrices from advection distributions.

    In the component-advection mode, ``mu`` is ordered as
    ``[u_x, u_y, v_x, v_y]``. The first two entries are the 2D spatial
    displacement of the U-component field and the last two entries are the 2D
    spatial displacement of the V-component field. For within-component blocks,
    the kernel uses the component's two-dimensional time-lag shift. For
    cross-component blocks, the default one-step IDE convention sets the
    source-side time term to zero, so the projection is also the target
    component shift. Setting ``gamma`` above zero activates the full
    cross-component projection ``dt * (E_target - gamma * E_source)``. The same
    projection is used for the mean shift and for projecting the 4x4 advection
    covariance into the 2D spatial dispersion matrix.
    """

    def __init__(
        self,
        n_dim: int = 3,
        dt: float = 1.0,
        gamma: float = 0.0,
        row_normalize: bool = True,
        use_spectral_scaling: bool = False,
        spectral_radius: float = 0.98,
        jitter: float = 1.0e-5,
        ell_init: float = 1.0,
        ell_min: float = 0.05,
        ell_max: float = 10.0,
        learnable_ell: bool = True,
        learnable_gamma: bool = False,
    ) -> None:
        super().__init__()
        if n_dim <= 0:
            raise ValueError("n_dim must be positive")
        if ell_min <= 0 or ell_max <= ell_min:
            raise ValueError("Require 0 < ell_min < ell_max")

        self.n_dim = n_dim
        self.dt = dt
        self.row_normalize = row_normalize
        self.use_spectral_scaling = use_spectral_scaling
        self.spectral_radius = spectral_radius
        self.jitter = jitter
        self.ell_min = ell_min
        self.ell_max = ell_max
        self.learnable_ell = learnable_ell
        self.learnable_gamma = learnable_gamma

        ell_frac = (ell_init - ell_min) / (ell_max - ell_min)
        self.raw_ell = nn.Parameter(torch.full((2, 2), _logit(ell_frac), dtype=torch.float32))
        self.raw_ell.requires_grad_(learnable_ell)
        if learnable_gamma:
            self.raw_gamma = nn.Parameter(torch.tensor(_logit(gamma), dtype=torch.float32))
        else:
            self.register_buffer("fixed_gamma", torch.tensor(float(gamma), dtype=torch.float32))

    def get_ell(self) -> Tensor:
        return self.ell_min + (self.ell_max - self.ell_min) * torch.sigmoid(self.raw_ell)

    def gamma_value(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.learnable_gamma:
            return torch.sigmoid(self.raw_gamma).to(device=device, dtype=dtype)
        return self.fixed_gamma.to(device=device, dtype=dtype)

    def component_projection(
        self,
        target_idx: int,
        source_idx: int,
        selectors: tuple[Tensor, Tensor],
        gamma: Tensor,
    ) -> Tensor:
        """Project stacked component advection to a 2D kernel displacement.

        Diagonal blocks use the stationary same-component time-lag projection
        ``dt * E_target``. Off-diagonal blocks use the target-component
        projection when ``gamma=0`` and the full cross-component projection
        ``dt * (E_target - gamma * E_source)`` when ``gamma`` is positive.
        """
        if target_idx == source_idx:
            return self.dt * selectors[target_idx]
        return self.dt * (selectors[target_idx] - gamma * selectors[source_idx])

    @staticmethod
    def selectors(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        E_u = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        E_v = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        return E_u, E_v

    def forward_single(self, S: Tensor, mu: Tensor, Sigma: Tensor, alpha_weights: Tensor) -> Tensor:
        """Return one transition matrix with shape ``[2*n_dim, 2*n_dim]``."""
        if S.shape != (self.n_dim, 2):
            raise ValueError(f"Expected S shape {(self.n_dim, 2)}, got {tuple(S.shape)}")
        if mu.shape[-1] != 4:
            raise ValueError("mu must have trailing size 4")
        if Sigma.shape[-2:] != (4, 4):
            raise ValueError("Sigma must have trailing shape [4, 4]")
        if alpha_weights.shape[-2:] != (2, 2):
            raise ValueError("alpha_weights must have trailing shape [2, 2]")

        S = S.to(device=mu.device, dtype=mu.dtype)
        Es = self.selectors(mu.device, mu.dtype)
        ell = self.get_ell().to(device=mu.device, dtype=mu.dtype)
        gamma = self.gamma_value(mu.device, mu.dtype)
        eye2 = torch.eye(2, device=mu.device, dtype=mu.dtype)
        H = S[:, None, :] - S[None, :, :]

        row_blocks = []
        for i in range(2):
            col_blocks = []
            for j in range(2):
                projection = self.component_projection(i, j, Es, gamma)
                shift = projection @ mu
                D = ell[i, j].pow(2) * eye2 + 2.0 * projection @ Sigma @ projection.T
                D = D + self.jitter * eye2
                L = safe_cholesky(D)

                H_prime = H - shift.view(1, 1, 2)
                flat = H_prime.reshape(-1, 2)
                solved = solve_linear_system(D, flat.T).T
                maha = (flat * solved).sum(dim=-1).reshape(self.n_dim, self.n_dim)
                logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
                K = torch.exp(-maha - 0.5 * logdet)
                col_blocks.append(alpha_weights[i, j] * K)
            row_blocks.append(torch.cat(col_blocks, dim=1))

        M = torch.cat(row_blocks, dim=0)
        if self.row_normalize:
            M = M / M.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        if self.use_spectral_scaling:
            eigvals = torch.linalg.eigvals(M)
            rho = eigvals.abs().max().real
            scale = torch.clamp(rho / self.spectral_radius, min=1.0)
            M = M / scale
        return M

    def forward(self, S: Tensor, mu_seq: Tensor, Sigma_seq: Tensor, alpha_seq: Tensor) -> Tensor:
        single = mu_seq.ndim == 1
        if single:
            mu_seq = mu_seq.unsqueeze(0)
            Sigma_seq = Sigma_seq.unsqueeze(0)
            alpha_seq = alpha_seq.unsqueeze(0)

        if S.shape != (self.n_dim, 2):
            raise ValueError(f"Expected S shape {(self.n_dim, 2)}, got {tuple(S.shape)}")
        if mu_seq.ndim != 2 or mu_seq.shape[-1] != 4:
            raise ValueError("mu_seq must have shape [T,4]")
        if Sigma_seq.ndim != 3 or Sigma_seq.shape[-2:] != (4, 4):
            raise ValueError("Sigma_seq must have shape [T,4,4]")
        if alpha_seq.ndim != 3 or alpha_seq.shape[-2:] != (2, 2):
            raise ValueError("alpha_seq must have shape [T,2,2]")
        if not (mu_seq.shape[0] == Sigma_seq.shape[0] == alpha_seq.shape[0]):
            raise ValueError("mu_seq, Sigma_seq, and alpha_seq must have matching time dimension")

        S = S.to(device=mu_seq.device, dtype=mu_seq.dtype)
        Es = self.selectors(mu_seq.device, mu_seq.dtype)
        ell = self.get_ell().to(device=mu_seq.device, dtype=mu_seq.dtype)
        gamma = self.gamma_value(mu_seq.device, mu_seq.dtype)
        eye2 = torch.eye(2, device=mu_seq.device, dtype=mu_seq.dtype)
        H = S[:, None, :] - S[None, :, :]
        n_pairs = self.n_dim * self.n_dim

        row_blocks = []
        for i in range(2):
            col_blocks = []
            for j in range(2):
                projection = self.component_projection(i, j, Es, gamma)
                shift = mu_seq @ projection.T
                projected_sigma = torch.matmul(projection.unsqueeze(0), Sigma_seq)
                D = ell[i, j].pow(2) * eye2 + 2.0 * torch.matmul(projected_sigma, projection.T)
                D = D + self.jitter * eye2
                L = safe_cholesky(D)

                H_prime = H.unsqueeze(0) - shift[:, None, None, :]
                flat = H_prime.reshape(mu_seq.shape[0], n_pairs, 2)
                solved = solve_linear_system(D, flat.transpose(-1, -2)).transpose(-1, -2)
                maha = (flat * solved).sum(dim=-1).reshape(mu_seq.shape[0], self.n_dim, self.n_dim)
                logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
                K = torch.exp(-maha - 0.5 * logdet[:, None, None])
                col_blocks.append(alpha_seq[:, i, j, None, None] * K)
            row_blocks.append(torch.cat(col_blocks, dim=2))

        out = torch.cat(row_blocks, dim=1)
        if self.row_normalize:
            out = out / out.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        if self.use_spectral_scaling:
            eigvals = torch.linalg.eigvals(out)
            rho = eigvals.abs().amax(dim=-1).real
            scale = torch.clamp(rho / self.spectral_radius, min=1.0)
            out = out / scale[:, None, None]
        return out[0] if single else out

    def forward_shared_flow(
        self,
        S: Tensor,
        flow_mu_seq: Tensor,
        flow_Sigma_seq: Tensor,
        B_seq: Tensor,
    ) -> Tensor:
        """Transition matrix from a shared 2D flow and signed component deformation.

        The spatial kernel is shared by U and V. Component-specific behavior is
        represented by ``B_seq`` in block form:
        ``[[B_UU K, B_UV K], [B_VU K, B_VV K]]``.
        """
        single = flow_mu_seq.ndim == 1
        if single:
            flow_mu_seq = flow_mu_seq.unsqueeze(0)
            flow_Sigma_seq = flow_Sigma_seq.unsqueeze(0)
            B_seq = B_seq.unsqueeze(0)

        if S.shape != (self.n_dim, 2):
            raise ValueError(f"Expected S shape {(self.n_dim, 2)}, got {tuple(S.shape)}")
        if flow_mu_seq.ndim != 2 or flow_mu_seq.shape[-1] != 2:
            raise ValueError("flow_mu_seq must have shape [T,2]")
        if flow_Sigma_seq.ndim != 3 or flow_Sigma_seq.shape[-2:] != (2, 2):
            raise ValueError("flow_Sigma_seq must have shape [T,2,2]")
        if B_seq.ndim != 3 or B_seq.shape[-2:] != (2, 2):
            raise ValueError("B_seq must have shape [T,2,2]")
        if not (flow_mu_seq.shape[0] == flow_Sigma_seq.shape[0] == B_seq.shape[0]):
            raise ValueError("flow_mu_seq, flow_Sigma_seq, and B_seq must have matching time dimension")

        S = S.to(device=flow_mu_seq.device, dtype=flow_mu_seq.dtype)
        eye2 = torch.eye(2, device=flow_mu_seq.device, dtype=flow_mu_seq.dtype)
        ell = self.get_ell().to(device=flow_mu_seq.device, dtype=flow_mu_seq.dtype).mean()
        H = S[:, None, :] - S[None, :, :]
        n_pairs = self.n_dim * self.n_dim

        shift = self.dt * flow_mu_seq
        D = ell.pow(2) * eye2 + 2.0 * flow_Sigma_seq
        D = D + self.jitter * eye2
        L = safe_cholesky(D)

        H_prime = H.unsqueeze(0) - shift[:, None, None, :]
        flat = H_prime.reshape(flow_mu_seq.shape[0], n_pairs, 2)
        solved = solve_linear_system(D, flat.transpose(-1, -2)).transpose(-1, -2)
        maha = (flat * solved).sum(dim=-1).reshape(flow_mu_seq.shape[0], self.n_dim, self.n_dim)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
        K = torch.exp(-maha - 0.5 * logdet[:, None, None])
        if self.row_normalize:
            K = K / K.sum(dim=2, keepdim=True).clamp_min(1.0e-8)

        row_blocks = []
        for i in range(2):
            col_blocks = []
            for j in range(2):
                col_blocks.append(B_seq[:, i, j, None, None] * K)
            row_blocks.append(torch.cat(col_blocks, dim=2))
        out = torch.cat(row_blocks, dim=1)

        if self.use_spectral_scaling:
            eigvals = torch.linalg.eigvals(out)
            rho = eigvals.abs().amax(dim=-1).real
            scale = torch.clamp(rho / self.spectral_radius, min=1.0)
            out = out / scale[:, None, None]
        return out[0] if single else out

    def forward_pairwise_flow(
        self,
        S: Tensor,
        pair_flow_mu_seq: Tensor,
        pair_flow_Sigma_seq: Tensor,
    ) -> Tensor:
        """Transition matrix from four separately evaluated component-pair kernels.

        ``pair_flow_mu_seq[:, i, j]`` and ``pair_flow_Sigma_seq[:, i, j]``
        define the spatial kernel for source component ``j`` contributing to
        target component ``i``. Unlike ``forward_shared_flow``, the component
        matrix is not multiplied onto the finished kernel weights; it has
        already shaped the flow moments before the kernels are evaluated.
        """
        single = pair_flow_mu_seq.ndim == 3
        if single:
            pair_flow_mu_seq = pair_flow_mu_seq.unsqueeze(0)
            pair_flow_Sigma_seq = pair_flow_Sigma_seq.unsqueeze(0)

        if S.shape != (self.n_dim, 2):
            raise ValueError(f"Expected S shape {(self.n_dim, 2)}, got {tuple(S.shape)}")
        if pair_flow_mu_seq.ndim != 4 or pair_flow_mu_seq.shape[-3:] != (2, 2, 2):
            raise ValueError("pair_flow_mu_seq must have shape [T,2,2,2]")
        if pair_flow_Sigma_seq.ndim != 5 or pair_flow_Sigma_seq.shape[-4:] != (2, 2, 2, 2):
            raise ValueError("pair_flow_Sigma_seq must have shape [T,2,2,2,2]")
        if pair_flow_mu_seq.shape[0] != pair_flow_Sigma_seq.shape[0]:
            raise ValueError("pair_flow_mu_seq and pair_flow_Sigma_seq must have matching time dimension")

        S = S.to(device=pair_flow_mu_seq.device, dtype=pair_flow_mu_seq.dtype)
        eye2 = torch.eye(2, device=pair_flow_mu_seq.device, dtype=pair_flow_mu_seq.dtype)
        ell = self.get_ell().to(device=pair_flow_mu_seq.device, dtype=pair_flow_mu_seq.dtype)
        H = S[:, None, :] - S[None, :, :]
        n_pairs = self.n_dim * self.n_dim

        row_blocks = []
        for i in range(2):
            col_blocks = []
            for j in range(2):
                shift = self.dt * pair_flow_mu_seq[:, i, j, :]
                D = ell[i, j].pow(2) * eye2 + 2.0 * pair_flow_Sigma_seq[:, i, j, :, :]
                D = D + self.jitter * eye2
                L = safe_cholesky(D)

                H_prime = H.unsqueeze(0) - shift[:, None, None, :]
                flat = H_prime.reshape(pair_flow_mu_seq.shape[0], n_pairs, 2)
                solved = solve_linear_system(D, flat.transpose(-1, -2)).transpose(-1, -2)
                maha = (flat * solved).sum(dim=-1).reshape(pair_flow_mu_seq.shape[0], self.n_dim, self.n_dim)
                logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
                K = torch.exp(-maha - 0.5 * logdet[:, None, None])
                col_blocks.append(K)
            row_blocks.append(torch.cat(col_blocks, dim=2))
        out = torch.cat(row_blocks, dim=1)

        if self.row_normalize:
            out = out / out.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        if self.use_spectral_scaling:
            eigvals = torch.linalg.eigvals(out)
            rho = eigvals.abs().amax(dim=-1).real
            scale = torch.clamp(rho / self.spectral_radius, min=1.0)
            out = out / scale[:, None, None]
        return out[0] if single else out
