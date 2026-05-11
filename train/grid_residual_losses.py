from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if mask is not None:
        valid = valid & (mask.to(device=pred.device, dtype=torch.bool))
    if not valid.any():
        return pred.new_tensor(0.0)
    diff = pred - torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    return diff[valid].pow(2).mean()


def spatial_smoothness_loss(pred: Tensor) -> Tensor:
    if pred.ndim != 5:
        raise ValueError(f"Expected pred [B,T,1,H,W], got {tuple(pred.shape)}")
    dx = pred[..., :, 1:] - pred[..., :, :-1]
    dy = pred[..., 1:, :] - pred[..., :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


def temporal_smoothness_loss(pred: Tensor) -> Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    return (pred[:, 1:] - pred[:, :-1]).pow(2).mean()


def prior_loss(pred: Tensor) -> Tensor:
    return pred.pow(2).mean()


def distance_weighted_prior_loss(pred: Tensor, distance_features: Tensor) -> Tensor:
    nearest = distance_features.to(device=pred.device, dtype=pred.dtype).min(dim=0).values
    nearest = nearest / nearest.max().clamp_min(1.0e-6)
    weight = nearest.view(1, 1, 1, *nearest.shape)
    return (weight * pred.pow(2)).mean()


def _mean_grid_jacobian(grid_x: Tensor, grid_y: Tensor) -> Tensor:
    drow_x = (grid_x[1:, :] - grid_x[:-1, :]).mean()
    dcol_x = (grid_x[:, 1:] - grid_x[:, :-1]).mean()
    drow_y = (grid_y[1:, :] - grid_y[:-1, :]).mean()
    dcol_y = (grid_y[:, 1:] - grid_y[:, :-1]).mean()
    return torch.stack(
        [
            torch.stack([drow_x, dcol_x]),
            torch.stack([drow_y, dcol_y]),
        ]
    )


def semi_lagrangian_advect(
    field: Tensor,
    u: Tensor,
    v: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    dt_seconds: float,
) -> Tensor:
    """Backtrace ``field`` one step with wind ``u/v`` on a near-regular grid.

    ``u`` and ``v`` are interpreted as east/north m/s. ``grid_x``/``grid_y`` are
    local projected km coordinates. The local grid is represented by its mean
    Jacobian, which is adequate for this small, almost affine NWP grid.
    """
    if field.ndim != 5 or field.shape[2] != 1:
        raise ValueError("field must have shape [B,T,1,H,W]")
    if u.shape != v.shape or u.shape != field[:, :, 0].shape:
        raise ValueError("u/v must have shape [B,T,H,W] matching field")
    bsz, steps, _, height, width = field.shape
    if steps == 0:
        return field

    dtype = field.dtype
    device = field.device
    grid_x = grid_x.to(device=device, dtype=dtype)
    grid_y = grid_y.to(device=device, dtype=dtype)
    jac = _mean_grid_jacobian(grid_x, grid_y)
    inv_jac = torch.linalg.pinv(jac)

    dx_km = u.to(device=device, dtype=dtype) * (float(dt_seconds) / 1000.0)
    dy_km = v.to(device=device, dtype=dtype) * (float(dt_seconds) / 1000.0)
    row_delta = inv_jac[0, 0] * dx_km + inv_jac[0, 1] * dy_km
    col_delta = inv_jac[1, 0] * dx_km + inv_jac[1, 1] * dy_km

    base_y, base_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    sample_y = base_y.view(1, 1, height, width) - row_delta
    sample_x = base_x.view(1, 1, height, width) - col_delta
    norm_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    norm_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).reshape(bsz * steps, height, width, 2)
    flat = field.reshape(bsz * steps, 1, height, width)
    advected = F.grid_sample(
        flat,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return advected.reshape(bsz, steps, 1, height, width)


def advection_consistency_loss(
    pred: Tensor,
    u: Tensor,
    v: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    dt_seconds: float,
) -> Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_next_from_adv = semi_lagrangian_advect(
        pred[:, :-1],
        u[:, :-1],
        v[:, :-1],
        grid_x=grid_x,
        grid_y=grid_y,
        dt_seconds=dt_seconds,
    )
    return (pred[:, 1:] - pred_next_from_adv).pow(2).mean()
