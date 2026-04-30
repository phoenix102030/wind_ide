from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .covariance import safe_cholesky, solve_linear_system


def _logit(value: float) -> float:
    value = min(max(value, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


class VectorLagrangianKernel(nn.Module):
    """Build vector-wind transition matrices from advection distributions."""

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
        self.learnable_gamma = learnable_gamma

        ell_frac = (ell_init - ell_min) / (ell_max - ell_min)
        self.raw_ell = nn.Parameter(torch.full((2, 2), _logit(ell_frac), dtype=torch.float32))
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

    def forward_single(self, S: Tensor, mu: Tensor, Sigma: Tensor, A: Tensor) -> Tensor:
        """Return one transition matrix with shape ``[2*n_dim, 2*n_dim]``."""
        if S.shape != (self.n_dim, 2):
            raise ValueError(f"Expected S shape {(self.n_dim, 2)}, got {tuple(S.shape)}")
        if mu.shape[-1] != 4:
            raise ValueError("mu must have trailing size 4")
        if Sigma.shape[-2:] != (4, 4):
            raise ValueError("Sigma must have trailing shape [4, 4]")
        if A.shape[-2:] != (2, 2):
            raise ValueError("A must have trailing shape [2, 2]")

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
                B = self.dt * (Es[i] - gamma * Es[j])
                shift = B @ mu
                D = ell[i, j].pow(2) * eye2 + 2.0 * B @ Sigma @ B.T
                D = D + self.jitter * eye2
                L = safe_cholesky(D)

                H_prime = H - shift.view(1, 1, 2)
                flat = H_prime.reshape(-1, 2)
                alpha = solve_linear_system(D, flat.T).T
                maha = (flat * alpha).sum(dim=-1).reshape(self.n_dim, self.n_dim)
                logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
                K = torch.exp(-maha - 0.5 * logdet)
                col_blocks.append(A[i, j] * K)
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

    def forward(self, S: Tensor, mu_seq: Tensor, Sigma_seq: Tensor, A_seq: Tensor) -> Tensor:
        single = mu_seq.ndim == 1
        if single:
            mu_seq = mu_seq.unsqueeze(0)
            Sigma_seq = Sigma_seq.unsqueeze(0)
            A_seq = A_seq.unsqueeze(0)

        Ms = [
            self.forward_single(S, mu_seq[t], Sigma_seq[t], A_seq[t])
            for t in range(mu_seq.shape[0])
        ]
        out = torch.stack(Ms, dim=0)
        return out[0] if single else out
