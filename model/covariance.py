from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def inverse_softplus(value: float) -> float:
    """Stable inverse of softplus for positive scalar initialization."""
    if value <= 0:
        raise ValueError("inverse_softplus expects a positive value")
    return math.log(math.expm1(value)) if value < 20.0 else value


def build_lower_cholesky(raw: Tensor, dim: int, jitter: float = 1.0e-4) -> Tensor:
    """Build a lower-triangular Cholesky factor with positive diagonal.

    Args:
        raw: Tensor with trailing size ``dim * (dim + 1) // 2``.
        dim: Matrix dimension.
        jitter: Positive diagonal floor.

    Returns:
        Tensor with shape ``raw.shape[:-1] + (dim, dim)``.
    """
    expected = dim * (dim + 1) // 2
    if raw.shape[-1] != expected:
        raise ValueError(f"Expected trailing size {expected}, got {raw.shape[-1]}")

    out_shape = raw.shape[:-1] + (dim, dim)
    L = raw.new_zeros(out_shape)
    tril_i, tril_j = torch.tril_indices(dim, dim, device=raw.device)
    is_diag = (tril_i == tril_j).to(device=raw.device)
    values = torch.where(is_diag, F.softplus(raw) + jitter, raw)
    L[..., tril_i, tril_j] = values
    return L


def covariance_from_cholesky_raw(raw: Tensor, dim: int, jitter: float = 1.0e-4) -> tuple[Tensor, Tensor]:
    """Return ``(L, L @ L.T)`` from unconstrained lower-triangular params."""
    L = build_lower_cholesky(raw, dim=dim, jitter=jitter)
    cov = L @ L.transpose(-1, -2)
    return L, cov


def add_jitter(matrix: Tensor, jitter: float) -> Tensor:
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    return matrix + jitter * eye


def solve_linear_system(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Solve ``matrix @ x = rhs`` with a CPU fallback for backend gaps."""
    try:
        return torch.linalg.solve(matrix, rhs)
    except NotImplementedError:
        result = torch.linalg.solve(matrix.cpu(), rhs.cpu())
        return result.to(device=matrix.device, dtype=matrix.dtype)


def cholesky_logdet_quad(cov: Tensor, residual: Tensor, jitter: float = 1.0e-5) -> tuple[Tensor, Tensor]:
    """Compute logdet(cov) and residual.T @ cov^{-1} @ residual.

    Supports batched covariance and residual tensors. ``residual`` must have
    trailing size equal to ``cov.shape[-1]``.
    """
    cov = add_jitter(cov, jitter)
    L = torch.linalg.cholesky(cov)
    rhs = residual.unsqueeze(-1)
    alpha = solve_linear_system(cov, rhs).squeeze(-1)
    quad = (residual * alpha).sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    return logdet, quad


def gaussian_nll_from_cov(
    residual: Tensor,
    cov: Tensor,
    jitter: float = 1.0e-5,
    include_constant: bool = True,
) -> Tensor:
    """Gaussian negative log likelihood for batched residuals."""
    logdet, quad = cholesky_logdet_quad(cov, residual, jitter=jitter)
    nll = logdet + quad
    if include_constant:
        nll = nll + residual.shape[-1] * math.log(2.0 * math.pi)
    return 0.5 * nll


def advection_nll_loss(
    v_star: Tensor,
    mu: Tensor,
    sigma: Tensor,
    jitter: float = 1.0e-5,
    include_constant: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Supervised advection loss under ``N(mu, sigma)``.

    Rows containing NaN or inf in ``v_star`` are ignored. This makes the loss
    usable when pseudo labels are unavailable for a subset of times.
    """
    if v_star is None:
        return mu.new_tensor(0.0)
    valid = torch.isfinite(v_star).all(dim=-1)
    if valid.sum() == 0:
        return mu.new_tensor(0.0)
    residual = v_star[valid] - mu[valid]
    nll = gaussian_nll_from_cov(
        residual,
        sigma[valid],
        jitter=jitter,
        include_constant=include_constant,
    )
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"Unknown reduction: {reduction}")


def smoothness_loss(mu: Tensor, A: Tensor) -> Tensor:
    if mu.shape[0] <= 1:
        return mu.new_tensor(0.0)
    mu_loss = (mu[1:] - mu[:-1]).pow(2).mean()
    A_loss = (A[1:] - A[:-1]).pow(2).mean()
    return mu_loss + A_loss


def l2_regularization(parameters: Iterable[Tensor]) -> Tensor:
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return torch.tensor(0.0)
    total = params[0].new_tensor(0.0)
    for param in params:
        total = total + param.pow(2).mean()
    return total


class ComponentSiteCovariance(nn.Module):
    """Positive-definite 2-component covariance expanded across sites.

    With state order ``[U(s1..sn), V(s1..sn)]``, this module returns
    ``Sigma_component kron I_n``.
    """

    def __init__(
        self,
        n_sites: int = 3,
        diag_init: float = 0.2,
        offdiag_init: float = 0.0,
        jitter: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.jitter = jitter
        raw_diag = inverse_softplus(max(diag_init, jitter) - jitter)
        raw = torch.tensor([raw_diag, offdiag_init, raw_diag], dtype=torch.float32)
        self.raw_chol = nn.Parameter(raw)

    def component_cholesky(self) -> Tensor:
        return build_lower_cholesky(self.raw_chol, dim=2, jitter=self.jitter)

    def component_covariance(self) -> Tensor:
        L = self.component_cholesky()
        return L @ L.T

    def forward(self) -> Tensor:
        component = self.component_covariance()
        eye_sites = torch.eye(
            self.n_sites,
            device=component.device,
            dtype=component.dtype,
        )
        return torch.kron(component, eye_sites)


class QRParameters(nn.Module):
    """Learnable process and observation covariance modules."""

    def __init__(
        self,
        n_sites: int = 3,
        q_init: float = 0.2,
        r_init: float = 0.2,
        jitter: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.q_cov = ComponentSiteCovariance(n_sites=n_sites, diag_init=q_init, jitter=jitter)
        self.r_cov = ComponentSiteCovariance(n_sites=n_sites, diag_init=r_init, jitter=jitter)

    def process(self) -> Tensor:
        return self.q_cov()

    def observation(self) -> Tensor:
        return self.r_cov()
