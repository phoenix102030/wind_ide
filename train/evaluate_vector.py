from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.vector_data_utils import load_vector_dataset
from train.train_vector_offline import build_model, print_device_info, resolve_device


STATE_NAMES = [
    "U_E05_140m",
    "U_E06_140m",
    "U_ASOW6_140m",
    "V_E05_140m",
    "V_E06_140m",
    "V_ASOW6_140m",
]
ADV_NAMES = ["mu_Uu", "mu_Uv", "mu_Vu", "mu_Vv"]
FLOW_NAMES = ["flow_x", "flow_y"]
A_NAMES = ["A_UU", "A_UV", "A_VU", "A_VV"]
B_NAMES = ["B_UU", "B_UV", "B_VU", "B_VV"]
PROCESSED_FLOW_NAMES = ["U_prime", "V_prime"]
BASE_BLOCK_VECTOR_NAMES = ["UU_block", "UV_block", "VU_block", "VV_block"]
COMPONENT_FLOW_BLOCK_NAMES = ["UU_flow", "UV_flow", "VU_flow", "VV_flow"]
TARGET_COMPONENT_FLOW_NAMES = ["target_U_flow", "target_V_flow"]
ROW_SCALED_B_FLOW_NAMES = ["flow_x_B_UU", "flow_x_B_UV", "flow_y_B_VU", "flow_y_B_VV"]


def matrix_block_vectors(M: np.ndarray, n_sites: int = 3) -> np.ndarray:
    """Aggregate 2x2 component blocks of a [T,2N,2N] transition into row vectors.

    Each scalar is the mean target-row sum of one spatial block. For a row-normalized
    shared kernel this matches the corresponding B entry; otherwise it also reflects
    the total spatial kernel mass in that block.
    """
    if M.ndim != 3 or M.shape[1] != 2 * n_sites or M.shape[2] != 2 * n_sites:
        return np.empty((0, 2, 2), dtype=np.float32)
    out = np.full((M.shape[0], 2, 2), np.nan, dtype=np.float32)
    for i in range(2):
        row_slice = slice(i * n_sites, (i + 1) * n_sites)
        for j in range(2):
            col_slice = slice(j * n_sites, (j + 1) * n_sites)
            block = M[:, row_slice, col_slice]
            out[:, i, j] = np.nanmean(np.nansum(block, axis=2), axis=1)
    return out


def component_spatial_flows(flow_mu: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine shared spatial flow with component coupling for visualization.

    Returns:
      block_flows[t, i, j, :] = B[t, i, j] * flow_mu[t]
      target_flows[t, i, :] = sum_j block_flows[t, i, j, :]
      block_strength[t, i, j] = signed scalar B[t, i, j] * ||flow_mu[t]||
    """
    if (
        flow_mu.ndim != 2
        or flow_mu.shape[-1] != 2
        or B.ndim != 3
        or B.shape[-2:] != (2, 2)
        or B.shape[0] != flow_mu.shape[0]
    ):
        return (
            np.empty((0, 2, 2, 2), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
        )
    block_flows = B[:, :, :, None] * flow_mu[:, None, None, :]
    target_flows = np.nansum(block_flows, axis=2)
    flow_norm = np.linalg.norm(flow_mu, axis=1)
    block_strength = B * flow_norm[:, None, None]
    return (
        block_flows.astype(np.float32),
        target_flows.astype(np.float32),
        block_strength.astype(np.float32),
    )


def row_scaled_B_flows(flow_mu: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Visualization requested as [flow_x * B_U row, flow_y * B_V row]."""
    if (
        flow_mu.ndim != 2
        or flow_mu.shape[-1] != 2
        or B.ndim != 3
        or B.shape[-2:] != (2, 2)
        or B.shape[0] != flow_mu.shape[0]
    ):
        return np.empty((0, 2, 2), dtype=np.float32)
    out = np.empty((flow_mu.shape[0], 2, 2), dtype=np.float32)
    out[:, 0, :] = flow_mu[:, 0, None] * B[:, 0, :]
    out[:, 1, :] = flow_mu[:, 1, None] * B[:, 1, :]
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tensor_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    skip_initial: int = 1,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if skip_initial > 0:
        pred = pred[skip_initial:]
        target = target[skip_initial:]
        if valid_mask is not None:
            valid_mask = valid_mask[skip_initial:]

    mask = torch.isfinite(target) & torch.isfinite(pred)
    if valid_mask is not None:
        mask = mask & valid_mask
    err = pred - target
    safe_err = torch.where(mask, err, torch.zeros_like(err))
    count = mask.sum(dim=0).clamp_min(1)

    mae_by_dim = safe_err.abs().sum(dim=0) / count
    rmse_by_dim = torch.sqrt(safe_err.pow(2).sum(dim=0) / count)

    all_count = mask.sum().clamp_min(1)
    mae = safe_err.abs().sum() / all_count
    rmse = torch.sqrt(safe_err.pow(2).sum() / all_count)

    u_mask = mask[:, :3]
    v_mask = mask[:, 3:]
    u_err = safe_err[:, :3]
    v_err = safe_err[:, 3:]
    u_rmse = torch.sqrt(u_err.pow(2).sum() / u_mask.sum().clamp_min(1))
    v_rmse = torch.sqrt(v_err.pow(2).sum() / v_mask.sum().clamp_min(1))
    u_mae = u_err.abs().sum() / u_mask.sum().clamp_min(1)
    v_mae = v_err.abs().sum() / v_mask.sum().clamp_min(1)

    return {
        "rmse": float(rmse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "rmse_u": float(u_rmse.detach().cpu()),
        "mae_u": float(u_mae.detach().cpu()),
        "rmse_v": float(v_rmse.detach().cpu()),
        "mae_v": float(v_mae.detach().cpu()),
        "rmse_by_dim": {
            name: float(value.detach().cpu())
            for name, value in zip(STATE_NAMES, rmse_by_dim)
        },
        "mae_by_dim": {
            name: float(value.detach().cpu())
            for name, value in zip(STATE_NAMES, mae_by_dim)
        },
        "observed_values": int(mask.sum().detach().cpu()),
    }


def transition_diagnostics(
    M: np.ndarray,
    A: np.ndarray,
    ell: np.ndarray,
    coords: np.ndarray,
    kernel_weight: np.ndarray | None = None,
    residual_decay: np.ndarray | None = None,
    transition_control: np.ndarray | None = None,
    B: np.ndarray | None = None,
) -> dict[str, Any]:
    eye = np.eye(M.shape[-1], dtype=np.float32)
    finite = np.isfinite(M).all(axis=(1, 2))
    M_finite = M[finite]
    if M_finite.size == 0:
        return {}

    offdiag_mask = ~np.eye(M.shape[-1], dtype=bool)
    diag_vals = np.diagonal(M_finite, axis1=1, axis2=2)
    offdiag_vals = M_finite[:, offdiag_mask]
    residual = M_finite - eye
    site_dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    diagnostics = {
        "transition_diag_mean": float(np.nanmean(diag_vals)),
        "transition_diag_min": float(np.nanmin(diag_vals)),
        "transition_diag_max": float(np.nanmax(diag_vals)),
        "transition_offdiag_mean": float(np.nanmean(offdiag_vals)),
        "transition_offdiag_max": float(np.nanmax(offdiag_vals)),
        "transition_mean_abs_M_minus_I": float(np.nanmean(np.abs(residual))),
        "transition_max_abs_M_minus_I": float(np.nanmax(np.abs(residual))),
        "transition_rowsum_min": float(np.nanmin(np.nansum(M_finite, axis=2))),
        "transition_rowsum_max": float(np.nanmax(np.nansum(M_finite, axis=2))),
        "A_mean": {
            name: float(value)
            for name, value in zip(A_NAMES, np.nanmean(A.reshape(A.shape[0], 4), axis=0))
        },
        "A_min": {
            name: float(value)
            for name, value in zip(A_NAMES, np.nanmin(A.reshape(A.shape[0], 4), axis=0))
        },
        "A_max": {
            name: float(value)
            for name, value in zip(A_NAMES, np.nanmax(A.reshape(A.shape[0], 4), axis=0))
        },
        "ell": ell.tolist(),
        "station_distance_km": site_dist.tolist(),
        "station_distance_nonzero_min_km": float(site_dist[site_dist > 0].min()),
        "station_distance_nonzero_max_km": float(site_dist.max()),
    }
    if kernel_weight is not None and kernel_weight.size:
        diagnostics.update(
            {
                "kernel_weight_mean": float(np.nanmean(kernel_weight)),
                "kernel_weight_min": float(np.nanmin(kernel_weight)),
                "kernel_weight_max": float(np.nanmax(kernel_weight)),
            }
        )
    if residual_decay is not None and residual_decay.size:
        diagnostics.update(
            {
                "residual_decay_mean": float(np.nanmean(residual_decay)),
                "residual_decay_min": float(np.nanmin(residual_decay)),
                "residual_decay_max": float(np.nanmax(residual_decay)),
            }
        )
    if transition_control is not None and transition_control.size:
        diagnostics.update(
            {
                "transition_control_abs_mean": float(np.nanmean(np.abs(transition_control))),
                "transition_control_abs_max": float(np.nanmax(np.abs(transition_control))),
            }
        )
    if B is not None and B.size:
        diagnostics.update(
            {
                "B_mean": {
                    name: float(value)
                    for name, value in zip(B_NAMES, np.nanmean(B.reshape(B.shape[0], 4), axis=0))
                },
                "B_min": {
                    name: float(value)
                    for name, value in zip(B_NAMES, np.nanmin(B.reshape(B.shape[0], 4), axis=0))
                },
                "B_max": {
                    name: float(value)
                    for name, value in zip(B_NAMES, np.nanmax(B.reshape(B.shape[0], 4), axis=0))
                },
            }
        )
    return diagnostics


def persistence_forecast(z: torch.Tensor, horizon: int = 1) -> torch.Tensor:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    pred = torch.full_like(z, torch.nan)
    if horizon < z.shape[0]:
        pred[horizon:] = z[:-horizon]
    return pred


def paired_forecast_metrics(
    pred: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    skip_initial: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common_mask = torch.isfinite(pred) & torch.isfinite(baseline) & torch.isfinite(target)
    return (
        tensor_metrics(pred, target, skip_initial=skip_initial, valid_mask=common_mask),
        tensor_metrics(baseline, target, skip_initial=skip_initial, valid_mask=common_mask),
    )


def metric_improvement(model_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "rmse_percent": 100.0
        * (baseline_metrics["rmse"] - model_metrics["rmse"])
        / max(baseline_metrics["rmse"], 1.0e-12),
        "mae_percent": 100.0
        * (baseline_metrics["mae"] - model_metrics["mae"])
        / max(baseline_metrics["mae"], 1.0e-12),
    }


def multi_horizon_forecasts(
    model: torch.nn.Module,
    filter_means: torch.Tensor,
    filter_covs: torch.Tensor,
    M_seq: torch.Tensor,
    max_horizon: int,
    control_seq: torch.Tensor | None = None,
) -> torch.Tensor:
    if max_horizon < 1:
        raise ValueError("max_horizon must be >= 1")

    T, state_dim = filter_means.shape
    predictions = filter_means.new_full((max_horizon, T, state_dim), torch.nan)
    for origin in range(T - 1):
        horizon = min(max_horizon, T - origin - 1)
        if horizon <= 0:
            continue
        future = model.dstm.torch_multi_step_forecast(
            filter_mean=filter_means[origin],
            filter_cov=filter_covs[origin],
            future_M_seq=M_seq[origin + 1 : origin + horizon + 1],
            future_control_seq=None if control_seq is None else control_seq[origin + 1 : origin + horizon + 1],
        )
        for step in range(horizon):
            predictions[step, origin + step + 1] = future["means"][step]
    return predictions


def rolling_history_forecasts(
    model: torch.nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    coords: torch.Tensor,
    max_horizon: int,
    history_window_size: int,
) -> torch.Tensor:
    """Forecast each origin from a fixed-length recent observation history.

    For origin t, the model filters only z[t-history_window_size+1:t] and then
    rolls forward with NWP-driven transition matrices for t+1 ... t+h.
    """
    if max_horizon < 1:
        raise ValueError("max_horizon must be >= 1")
    if history_window_size < 1:
        raise ValueError("history_window_size must be >= 1")

    T, state_dim = z.shape
    predictions = z.new_full((max_horizon, T, state_dim), torch.nan)
    for origin in range(history_window_size - 1, T - 1):
        horizon = min(max_horizon, T - origin - 1)
        if horizon <= 0:
            continue
        history_start = origin - history_window_size + 1
        chunk_end = origin + horizon + 1
        outputs = model(x[history_start:chunk_end], coords)
        history_len = origin - history_start + 1
        control_seq = outputs.get("transition_control")
        kf = model.dstm.kalman_filter(
            z=z[history_start : origin + 1],
            M_seq=outputs["M"][:history_len],
            control_seq=None if control_seq is None else control_seq[:history_len],
            return_history=True,
        )
        future = model.dstm.torch_multi_step_forecast(
            filter_mean=kf["filter_means"][-1],
            filter_cov=kf["filter_covs"][-1],
            future_M_seq=outputs["M"][history_len : history_len + horizon],
            future_control_seq=(
                None
                if control_seq is None
                else control_seq[history_len : history_len + horizon]
            ),
        )
        for step in range(horizon):
            predictions[step, origin + step + 1] = future["means"][step]
    return predictions


def inverse_model_target(values: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    standardizer = data.get("z_standardizer")
    if standardizer is None:
        return values.astype(np.float32, copy=False)
    return standardizer.inverse_transform(values).astype(np.float32, copy=False)


def measurement_from_model_target(values: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    raw_values = inverse_model_target(values, data)
    target_mode = str(data.get("target_mode", "measurement")).lower()
    if target_mode in {"residual_nwp", "nwp_residual", "residual"}:
        baseline = data["nwp_baseline"].astype(np.float32, copy=False)
        if raw_values.ndim == 3:
            baseline = baseline[None, :, :]
        return (baseline + raw_values).astype(np.float32, copy=False)
    return raw_values


def evaluate(
    model: torch.nn.Module,
    data: dict[str, Any],
    device: torch.device,
    eval_window_size: int,
    eval_stride: int | None = None,
    forecast_horizon: int = 1,
    history_window_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    coords = torch.from_numpy(data["coords"]).to(device)
    T = int(data["X"].shape[0])
    if eval_stride is None:
        eval_stride = eval_window_size

    M_sum = np.zeros((T, 6, 6), dtype=np.float64)
    M_base_sum = np.zeros((T, 6, 6), dtype=np.float64)
    mu_sum = np.zeros((T, 4), dtype=np.float64)
    sigma_sum = np.zeros((T, 4, 4), dtype=np.float64)
    A_sum = np.zeros((T, 2, 2), dtype=np.float64)
    param_counts = np.zeros((T, 1), dtype=np.float64)
    flow_mu_sum: np.ndarray | None = None
    flow_sigma_sum: np.ndarray | None = None
    B_sum: np.ndarray | None = None
    control_sum: np.ndarray | None = None
    kernel_weight_sum: np.ndarray | None = None
    residual_decay_sum: np.ndarray | None = None

    model.eval()
    with torch.no_grad():
        starts = list(range(0, T, eval_stride))
        for start in starts:
            end = min(start + eval_window_size, T)
            if end <= start:
                continue
            x_chunk = torch.from_numpy(data["X"][start:end]).to(device)
            outputs = model(x_chunk, coords)
            M_sum[start:end] += outputs["M"].detach().cpu().numpy()
            M_base_sum[start:end] += outputs.get("M_base", outputs["M"]).detach().cpu().numpy()
            mu_sum[start:end] += outputs["mu"].detach().cpu().numpy()
            sigma_sum[start:end] += outputs["Sigma"].detach().cpu().numpy()
            A_sum[start:end] += outputs["A"].detach().cpu().numpy()
            if "flow_mu" in outputs:
                if flow_mu_sum is None:
                    flow_mu_sum = np.zeros((T, outputs["flow_mu"].shape[-1]), dtype=np.float64)
                flow_mu_sum[start:end] += outputs["flow_mu"].detach().cpu().numpy()
            if "flow_Sigma" in outputs:
                if flow_sigma_sum is None:
                    flow_sigma_sum = np.zeros((T, 2, 2), dtype=np.float64)
                flow_sigma_sum[start:end] += outputs["flow_Sigma"].detach().cpu().numpy()
            if "B" in outputs:
                if B_sum is None:
                    B_sum = np.zeros((T, 2, 2), dtype=np.float64)
                B_sum[start:end] += outputs["B"].detach().cpu().numpy()
            if "transition_control" in outputs:
                if control_sum is None:
                    control_sum = np.zeros((T, outputs["transition_control"].shape[-1]), dtype=np.float64)
                control_sum[start:end] += outputs["transition_control"].detach().cpu().numpy()
            if "kernel_weight" in outputs:
                if kernel_weight_sum is None:
                    kernel_weight_sum = np.zeros((T, outputs["kernel_weight"].shape[-1]), dtype=np.float64)
                kernel_weight_sum[start:end] += outputs["kernel_weight"].detach().cpu().numpy()
            if "residual_decay" in outputs:
                if residual_decay_sum is None:
                    residual_decay_sum = np.zeros((T, outputs["residual_decay"].shape[-1]), dtype=np.float64)
                residual_decay_sum[start:end] += outputs["residual_decay"].detach().cpu().numpy()
            param_counts[start:end] += 1.0

    valid_params = param_counts[:, 0] > 0.0
    M_np = np.full((T, 6, 6), np.nan, dtype=np.float32)
    M_base_np = np.full((T, 6, 6), np.nan, dtype=np.float32)
    mu_np = np.full((T, 4), np.nan, dtype=np.float32)
    sigma_np = np.full((T, 4, 4), np.nan, dtype=np.float32)
    A_np = np.full((T, 2, 2), np.nan, dtype=np.float32)
    counts = param_counts[valid_params]
    M_np[valid_params] = (M_sum[valid_params] / counts[:, :, None]).astype(np.float32)
    M_base_np[valid_params] = (M_base_sum[valid_params] / counts[:, :, None]).astype(np.float32)
    mu_np[valid_params] = (mu_sum[valid_params] / counts).astype(np.float32)
    sigma_np[valid_params] = (sigma_sum[valid_params] / counts[:, :, None]).astype(np.float32)
    A_np[valid_params] = (A_sum[valid_params] / counts[:, :, None]).astype(np.float32)
    flow_mu_np = None
    flow_sigma_np = None
    B_np = None
    if flow_mu_sum is not None:
        flow_mu_np = np.full(flow_mu_sum.shape, np.nan, dtype=np.float32)
        flow_mu_np[valid_params] = (flow_mu_sum[valid_params] / counts).astype(np.float32)
    if flow_sigma_sum is not None:
        flow_sigma_np = np.full(flow_sigma_sum.shape, np.nan, dtype=np.float32)
        flow_sigma_np[valid_params] = (flow_sigma_sum[valid_params] / counts[:, :, None]).astype(np.float32)
    if B_sum is not None:
        B_np = np.full(B_sum.shape, np.nan, dtype=np.float32)
        B_np[valid_params] = (B_sum[valid_params] / counts[:, :, None]).astype(np.float32)
    processed_flow_mu_np = None
    if flow_mu_np is not None and B_np is not None:
        processed_flow_mu_np = np.einsum("tij,tj->ti", B_np, flow_mu_np).astype(np.float32)
    component_flow_blocks_np = np.empty((0, 2, 2, 2), dtype=np.float32)
    target_component_flow_np = np.empty((0, 2, 2), dtype=np.float32)
    component_flow_strength_np = np.empty((0, 2, 2), dtype=np.float32)
    row_scaled_B_flow_np = np.empty((0, 2, 2), dtype=np.float32)
    if flow_mu_np is not None and B_np is not None:
        component_flow_blocks_np, target_component_flow_np, component_flow_strength_np = component_spatial_flows(
            flow_mu_np,
            B_np,
        )
        row_scaled_B_flow_np = row_scaled_B_flows(flow_mu_np, B_np)
    control_np = None
    kernel_weight_np = None
    residual_decay_np = None
    if control_sum is not None:
        control_np = np.full(control_sum.shape, np.nan, dtype=np.float32)
        control_np[valid_params] = (control_sum[valid_params] / counts).astype(np.float32)
    if kernel_weight_sum is not None:
        kernel_weight_np = np.full(kernel_weight_sum.shape, np.nan, dtype=np.float32)
        kernel_weight_np[valid_params] = (kernel_weight_sum[valid_params] / counts).astype(np.float32)
    if residual_decay_sum is not None:
        residual_decay_np = np.full(residual_decay_sum.shape, np.nan, dtype=np.float32)
        residual_decay_np[valid_params] = (residual_decay_sum[valid_params] / counts).astype(np.float32)

    if not np.isfinite(M_np).all():
        missing = int((~np.isfinite(M_np).all(axis=(1, 2))).sum())
        raise ValueError(f"Transition matrices are missing for {missing} time steps; reduce eval stride.")

    target_np = data.get("Y", data["Z"]).astype(np.float32, copy=False)
    nwp_baseline_np = data.get("nwp_baseline", np.full_like(target_np, np.nan)).astype(np.float32, copy=False)
    model_target_np = inverse_model_target(data["Z"].astype(np.float32, copy=False), data)

    z = torch.from_numpy(data["Z"]).to(device)
    M_torch = torch.from_numpy(M_np).to(device)
    control_torch = torch.from_numpy(control_np).to(device) if control_np is not None else None
    with torch.no_grad():
        kf = model.dstm.kalman_filter(
            z=z,
            M_seq=M_torch,
            control_seq=control_torch,
            reduction="sum",
            return_history=True,
        )
        pred = kf["pred_means"]
        pred_model_target_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
        multi_pred = multi_horizon_forecasts(
            model=model,
            filter_means=kf["filter_means"],
            filter_covs=kf["filter_covs"],
            M_seq=M_torch,
            max_horizon=forecast_horizon,
            control_seq=control_torch,
        )
        if history_window_size is not None:
            x_all = torch.from_numpy(data["X"]).to(device)
            multi_pred = rolling_history_forecasts(
                model=model,
                x=x_all,
                z=z,
                coords=coords,
                max_horizon=forecast_horizon,
                history_window_size=history_window_size,
            )
            pred_model_target_np = multi_pred[0].detach().cpu().numpy().astype(np.float32, copy=False)

    pred_np = measurement_from_model_target(pred_model_target_np, data)
    multi_pred_model_target_np = multi_pred.detach().cpu().numpy().astype(np.float32, copy=False)
    multi_pred_np = measurement_from_model_target(multi_pred_model_target_np, data)

    target = torch.from_numpy(target_np).to(device)
    prediction = torch.from_numpy(pred_np).to(device)
    nwp_baseline = torch.from_numpy(nwp_baseline_np).to(device)
    baseline = persistence_forecast(target, horizon=1)
    residual_persistence = nwp_baseline + persistence_forecast(
        torch.from_numpy(target_np - nwp_baseline_np).to(device),
        horizon=1,
    )
    model_metrics, baseline_metrics = paired_forecast_metrics(
        prediction,
        baseline,
        target,
        skip_initial=1,
    )
    nwp_model_metrics, nwp_metrics = paired_forecast_metrics(
        prediction,
        nwp_baseline,
        target,
        skip_initial=1,
    )
    residual_persistence_model_metrics, residual_persistence_metrics = paired_forecast_metrics(
        prediction,
        residual_persistence,
        target,
        skip_initial=1,
    )
    improvement = metric_improvement(model_metrics, baseline_metrics)

    multi_step: dict[str, Any] = {}
    multi_baselines = []
    multi_nwp_baselines = []
    multi_residual_persistence = []
    curve_columns = ["model", "persistence", "nwp", "nwp_residual_persistence"]
    rmse_curve = np.full((forecast_horizon, len(curve_columns)), np.nan, dtype=np.float32)
    mae_curve = np.full((forecast_horizon, len(curve_columns)), np.nan, dtype=np.float32)
    for horizon in range(1, forecast_horizon + 1):
        horizon_pred = torch.from_numpy(multi_pred_np[horizon - 1]).to(device)
        horizon_baseline = persistence_forecast(target, horizon=horizon)
        horizon_nwp = nwp_baseline
        horizon_residual_persistence = nwp_baseline + persistence_forecast(
            torch.from_numpy(target_np - nwp_baseline_np).to(device),
            horizon=horizon,
        )
        horizon_model_metrics, horizon_baseline_metrics = paired_forecast_metrics(
            horizon_pred,
            horizon_baseline,
            target,
            skip_initial=horizon,
        )
        _, horizon_nwp_metrics = paired_forecast_metrics(
            horizon_pred,
            horizon_nwp,
            target,
            skip_initial=horizon,
        )
        _, horizon_residual_persistence_metrics = paired_forecast_metrics(
            horizon_pred,
            horizon_residual_persistence,
            target,
            skip_initial=horizon,
        )
        multi_baselines.append(horizon_baseline.detach().cpu().numpy().astype(np.float32, copy=False))
        multi_nwp_baselines.append(horizon_nwp.detach().cpu().numpy().astype(np.float32, copy=False))
        multi_residual_persistence.append(
            horizon_residual_persistence.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        rmse_curve[horizon - 1] = [
            horizon_model_metrics["rmse"],
            horizon_baseline_metrics["rmse"],
            horizon_nwp_metrics["rmse"],
            horizon_residual_persistence_metrics["rmse"],
        ]
        mae_curve[horizon - 1] = [
            horizon_model_metrics["mae"],
            horizon_baseline_metrics["mae"],
            horizon_nwp_metrics["mae"],
            horizon_residual_persistence_metrics["mae"],
        ]
        multi_step[str(horizon)] = {
            "model": horizon_model_metrics,
            "persistence_baseline": horizon_baseline_metrics,
            "nwp_baseline": horizon_nwp_metrics,
            "nwp_residual_persistence_baseline": horizon_residual_persistence_metrics,
            "model_vs_persistence_improvement": metric_improvement(
                horizon_model_metrics,
                horizon_baseline_metrics,
            ),
            "model_vs_nwp_improvement": metric_improvement(
                horizon_model_metrics,
                horizon_nwp_metrics,
            ),
            "model_vs_nwp_residual_persistence_improvement": metric_improvement(
                horizon_model_metrics,
                horizon_residual_persistence_metrics,
            ),
        }
    multi_baseline_np = np.stack(multi_baselines, axis=0)
    multi_nwp_baseline_np = np.stack(multi_nwp_baselines, axis=0)
    multi_residual_persistence_np = np.stack(multi_residual_persistence, axis=0)

    with torch.no_grad():
        ell = model.kernel.get_ell().detach().cpu().numpy()
        gamma = np.asarray(float(model.kernel.gamma_value(device, torch.float32).detach().cpu()))
        Q = model.dstm.process_covariance().detach().cpu().numpy()
        R = model.dstm.observation_covariance().detach().cpu().numpy()

    results = {
        "kalman_nll_per_observation": float(kf["nll_sum"].detach().cpu())
        / max(float(kf["obs_count"].detach().cpu()), 1.0),
        "model": model_metrics,
        "persistence_baseline": baseline_metrics,
        "nwp_baseline": nwp_metrics,
        "nwp_residual_persistence_baseline": residual_persistence_metrics,
        "model_vs_persistence_improvement": improvement,
        "model_vs_nwp_improvement": metric_improvement(nwp_model_metrics, nwp_metrics),
        "model_vs_nwp_residual_persistence_improvement": metric_improvement(
            residual_persistence_model_metrics,
            residual_persistence_metrics,
        ),
        "target_mode": data.get("target_mode", "measurement"),
        "forecast_horizon": forecast_horizon,
        "history_window_size": history_window_size,
        "multi_step": multi_step,
        "diagnostics": transition_diagnostics(
            M_np,
            A_np,
            ell,
            data["coords"],
            kernel_weight=kernel_weight_np,
            residual_decay=residual_decay_np,
            transition_control=control_np,
            B=B_np,
        ),
        "eval_window_size": eval_window_size,
        "eval_stride": eval_stride,
    }
    artifacts = {
        "target": target_np,
        "measurement_target": target_np,
        "model_target": data["Z"].astype(np.float32, copy=False),
        "model_target_raw": model_target_np,
        "nwp_baseline": nwp_baseline_np,
        "prediction": pred_np,
        "model_target_prediction": pred_model_target_np,
        "persistence_prediction": baseline.detach().cpu().numpy().astype(np.float32, copy=False),
        "nwp_baseline_prediction": nwp_baseline_np,
        "nwp_residual_persistence_prediction": residual_persistence.detach().cpu().numpy().astype(np.float32, copy=False),
        "multi_step_prediction": multi_pred_np,
        "multi_step_model_target_prediction": multi_pred_model_target_np,
        "multi_step_persistence_prediction": multi_baseline_np,
        "multi_step_nwp_baseline_prediction": multi_nwp_baseline_np,
        "multi_step_nwp_residual_persistence_prediction": multi_residual_persistence_np,
        "multi_step_rmse_curve": rmse_curve,
        "multi_step_mae_curve": mae_curve,
        "multi_step_curve_columns": np.asarray(curve_columns),
        "horizons": np.arange(1, forecast_horizon + 1, dtype=np.int32),
        "transition_matrices": M_np,
        "transition_base_matrices": M_base_np,
        "transition_base_block_vectors": matrix_block_vectors(M_base_np, n_sites=int(data["coords"].shape[0])),
        "mu": mu_np,
        "Sigma": sigma_np,
        "Sigma_diag": np.diagonal(sigma_np, axis1=1, axis2=2),
        "A": A_np,
        "flow_mu": (
            flow_mu_np
            if flow_mu_np is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
        "flow_Sigma": (
            flow_sigma_np
            if flow_sigma_np is not None
            else np.empty((0, 2, 2), dtype=np.float32)
        ),
        "flow_Sigma_diag": (
            np.diagonal(flow_sigma_np, axis1=1, axis2=2)
            if flow_sigma_np is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
        "B": (
            B_np
            if B_np is not None
            else np.empty((0, 2, 2), dtype=np.float32)
        ),
        "processed_flow_mu": (
            processed_flow_mu_np
            if processed_flow_mu_np is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
        "component_flow_blocks": component_flow_blocks_np,
        "target_component_flow": target_component_flow_np,
        "component_flow_strength": component_flow_strength_np,
        "row_scaled_B_flow": row_scaled_B_flow_np,
        "kernel_weight": (
            kernel_weight_np
            if kernel_weight_np is not None
            else np.empty((0, 1), dtype=np.float32)
        ),
        "residual_decay": (
            residual_decay_np
            if residual_decay_np is not None
            else np.empty((0, 1), dtype=np.float32)
        ),
        "transition_control": (
            control_np
            if control_np is not None
            else np.empty((0, 6), dtype=np.float32)
        ),
        "ell": ell.astype(np.float32, copy=False),
        "gamma": gamma.astype(np.float32, copy=False),
        "Q": Q.astype(np.float32, copy=False),
        "R": R.astype(np.float32, copy=False),
        "coords": data["coords"].astype(np.float32, copy=False),
        "baseline_grid_indices": data.get("baseline_grid_indices", np.empty((0, 2), dtype=np.int64)),
    }
    return results, artifacts


def default_output_dir(checkpoint_path: str | Path, split: str) -> Path:
    stem = Path(checkpoint_path).stem
    return Path("outputs") / "evaluation" / f"{stem}_{split}"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_artifact_arrays(output_dir: Path, artifacts: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "forecasts.npz",
        target=artifacts["target"],
        model_target=artifacts["model_target"],
        model_target_raw=artifacts["model_target_raw"],
        nwp_baseline=artifacts["nwp_baseline"],
        prediction=artifacts["prediction"],
        model_target_prediction=artifacts["model_target_prediction"],
        persistence_prediction=artifacts["persistence_prediction"],
        nwp_baseline_prediction=artifacts["nwp_baseline_prediction"],
        nwp_residual_persistence_prediction=artifacts["nwp_residual_persistence_prediction"],
        multi_step_prediction=artifacts["multi_step_prediction"],
        multi_step_model_target_prediction=artifacts["multi_step_model_target_prediction"],
        multi_step_persistence_prediction=artifacts["multi_step_persistence_prediction"],
        multi_step_nwp_baseline_prediction=artifacts["multi_step_nwp_baseline_prediction"],
        multi_step_nwp_residual_persistence_prediction=artifacts[
            "multi_step_nwp_residual_persistence_prediction"
        ],
        horizons=artifacts["horizons"],
        state_names=np.asarray(STATE_NAMES),
    )
    np.savez_compressed(
        output_dir / "multi_step_metrics.npz",
        horizons=artifacts["horizons"],
        rmse_curve=artifacts["multi_step_rmse_curve"],
        mae_curve=artifacts["multi_step_mae_curve"],
        curve_columns=artifacts["multi_step_curve_columns"],
    )
    np.savez_compressed(
        output_dir / "transition_matrices.npz",
        M=artifacts["transition_matrices"],
        M_mean=np.nanmean(artifacts["transition_matrices"], axis=0),
        M_row_sums=np.nansum(artifacts["transition_matrices"], axis=2),
        M_base=artifacts["transition_base_matrices"],
        M_base_mean=np.nanmean(artifacts["transition_base_matrices"], axis=0),
        M_base_row_sums=np.nansum(artifacts["transition_base_matrices"], axis=2),
        M_base_block_vectors=artifacts["transition_base_block_vectors"],
        state_names=np.asarray(STATE_NAMES),
        base_block_vector_names=np.asarray(BASE_BLOCK_VECTOR_NAMES),
    )
    np.savez_compressed(
        output_dir / "advection_parameters.npz",
        mu=artifacts["mu"],
        Sigma=artifacts["Sigma"],
        Sigma_diag=artifacts["Sigma_diag"],
        A=artifacts["A"],
        flow_mu=artifacts["flow_mu"],
        flow_Sigma=artifacts["flow_Sigma"],
        flow_Sigma_diag=artifacts["flow_Sigma_diag"],
        B=artifacts["B"],
        processed_flow_mu=artifacts["processed_flow_mu"],
        component_flow_blocks=artifacts["component_flow_blocks"],
        target_component_flow=artifacts["target_component_flow"],
        component_flow_strength=artifacts["component_flow_strength"],
        row_scaled_B_flow=artifacts["row_scaled_B_flow"],
        kernel_weight=artifacts["kernel_weight"],
        residual_decay=artifacts["residual_decay"],
        transition_control=artifacts["transition_control"],
        ell=artifacts["ell"],
        gamma=artifacts["gamma"],
        Q=artifacts["Q"],
        R=artifacts["R"],
        coords=artifacts["coords"],
        baseline_grid_indices=artifacts["baseline_grid_indices"],
        advection_names=np.asarray(ADV_NAMES),
        flow_names=np.asarray(FLOW_NAMES),
        B_names=np.asarray(B_NAMES),
        processed_flow_names=np.asarray(PROCESSED_FLOW_NAMES),
        component_flow_block_names=np.asarray(COMPONENT_FLOW_BLOCK_NAMES),
        target_component_flow_names=np.asarray(TARGET_COMPONENT_FLOW_NAMES),
        row_scaled_B_flow_names=np.asarray(ROW_SCALED_B_FLOW_NAMES),
        state_names=np.asarray(STATE_NAMES),
    )

    param_csv = np.column_stack(
        [
            np.arange(artifacts["mu"].shape[0]),
            artifacts["mu"],
            artifacts["Sigma_diag"],
            artifacts["A"].reshape(artifacts["A"].shape[0], 4),
        ]
    )
    header = ",".join(["time_index", *ADV_NAMES, "var_Ux", "var_Uy", "var_Vx", "var_Vy", *A_NAMES])
    if artifacts["flow_mu"].shape[0] == artifacts["mu"].shape[0]:
        param_csv = np.column_stack([param_csv, artifacts["flow_mu"], artifacts["flow_Sigma_diag"]])
        header += "," + ",".join([*FLOW_NAMES, "flow_var_x", "flow_var_y"])
    if artifacts["B"].shape[0] == artifacts["mu"].shape[0]:
        param_csv = np.column_stack([param_csv, artifacts["B"].reshape(artifacts["B"].shape[0], 4)])
        header += "," + ",".join(B_NAMES)
    if artifacts["processed_flow_mu"].shape[0] == artifacts["mu"].shape[0]:
        param_csv = np.column_stack([param_csv, artifacts["processed_flow_mu"]])
        header += "," + ",".join(PROCESSED_FLOW_NAMES)
    if artifacts["kernel_weight"].shape[0] == artifacts["mu"].shape[0]:
        param_csv = np.column_stack([param_csv, artifacts["kernel_weight"]])
        header += ",kernel_weight"
    if artifacts["residual_decay"].shape[0] == artifacts["mu"].shape[0]:
        param_csv = np.column_stack([param_csv, artifacts["residual_decay"]])
        header += ",residual_decay"
    if artifacts["transition_control"].shape[0] == artifacts["mu"].shape[0]:
        param_csv = np.column_stack([param_csv, artifacts["transition_control"]])
        header += "," + ",".join(f"control_{name}" for name in STATE_NAMES)
    np.savetxt(output_dir / "time_parameters.csv", param_csv, delimiter=",", header=header, comments="")


def _downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, max_points).round().astype(int)


def _line_plot(path: Path, values: np.ndarray, labels: list[str], title: str, ylabel: str, max_points: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = _downsample_indices(values.shape[0], max_points)
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label in enumerate(labels):
        ax.plot(idx, values[idx, col], linewidth=1.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("time index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=min(len(labels), 4), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _heatmap(
    path: Path,
    matrix: np.ndarray,
    title: str,
    xlabels: list[str],
    ylabels: list[str],
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    kwargs = {}
    if center_zero:
        limit = float(np.nanmax(np.abs(matrix)))
        kwargs = {"vmin": -limit, "vmax": limit}
    im = ax.imshow(matrix, cmap=cmap, **kwargs)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ylabels, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _horizon_plot(
    path: Path,
    horizons: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label in enumerate(labels.tolist()):
        ax.plot(horizons, values[:, col], marker="o", linewidth=1.8, label=str(label))
    ax.set_title(title)
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _bubble_matrix(
    path: Path,
    matrix: np.ndarray,
    title: str,
    xlabels: list[str],
    ylabels: list[str],
    cmap: str = "viridis",
    center_zero: bool = False,
    vmax_abs: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = np.isfinite(matrix)
    if not finite.any():
        return
    if center_zero:
        limit = float(vmax_abs if vmax_abs is not None else np.nanmax(np.abs(matrix)))
        limit = max(limit, 1.0e-8)
        vmin, vmax = -limit, limit
        size_den = limit
        size_values = np.abs(matrix)
    else:
        vmin = 0.0
        vmax = float(vmax_abs if vmax_abs is not None else np.nanmax(matrix))
        vmax = max(vmax, 1.0e-8)
        size_den = vmax
        size_values = np.clip(matrix, 0.0, None)

    yy, xx = np.indices(matrix.shape)
    sizes = 25.0 + 950.0 * np.clip(size_values / size_den, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        s=sizes.ravel(),
        c=matrix.ravel(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.35,
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="lightgray", linewidth=0.5, alpha=0.6)
    fig.colorbar(scatter, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _bubble_matrix_frame(
    matrix: np.ndarray,
    title: str,
    xlabels: list[str],
    ylabels: list[str],
    cmap: str,
    center_zero: bool,
    vmax_abs: float,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if center_zero:
        vmin, vmax = -vmax_abs, vmax_abs
        size_values = np.abs(matrix)
    else:
        vmin, vmax = 0.0, vmax_abs
        size_values = np.clip(matrix, 0.0, None)

    yy, xx = np.indices(matrix.shape)
    size_den = max(vmax_abs, 1.0e-8)
    sizes = 25.0 + 950.0 * np.clip(size_values / size_den, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    scatter = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        s=sizes.ravel(),
        c=matrix.ravel(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.35,
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="lightgray", linewidth=0.5, alpha=0.6)
    fig.colorbar(scatter, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return frame


def _advection_mu_frame(mu_t: np.ndarray, title: str, axis_limit: float) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(6, 6))
    for offset, label, color in [(0, "U", "tab:blue"), (2, "V", "tab:orange")]:
        mean = mu_t[offset : offset + 2]
        ax.arrow(
            0.0,
            0.0,
            mean[0],
            mean[1],
            color=color,
            width=0.012 * axis_limit,
            head_width=0.075 * axis_limit,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.scatter([mean[0]], [mean[1]], color=color, s=34, label=f"{label} mean")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x displacement / step")
    ax.set_ylabel("y displacement / step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return frame


def _shared_flow_frame(flow_t: np.ndarray, title: str, axis_limit: float) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.arrow(
        0.0,
        0.0,
        flow_t[0],
        flow_t[1],
        color="tab:green",
        width=0.012 * axis_limit,
        head_width=0.075 * axis_limit,
        length_includes_head=True,
        alpha=0.95,
    )
    ax.scatter([flow_t[0]], [flow_t[1]], color="tab:green", s=40, label="shared flow")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x displacement / step")
    ax.set_ylabel("y displacement / step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return frame


def _processed_flow_frame(
    flow_t: np.ndarray,
    processed_flow_t: np.ndarray,
    title: str,
    axis_limit: float,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(6, 6))
    for vector, label, color in [
        (flow_t, "shared flow", "tab:green"),
        (processed_flow_t, "B @ flow", "tab:red"),
    ]:
        ax.arrow(
            0.0,
            0.0,
            vector[0],
            vector[1],
            color=color,
            width=0.012 * axis_limit,
            head_width=0.075 * axis_limit,
            length_includes_head=True,
            alpha=0.9,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=40, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x displacement / step")
    ax.set_ylabel("y displacement / step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return frame


def _B_row_vector_plot(path: Path, B: np.ndarray, title: str, axis_limit: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if axis_limit is None:
        axis_limit = max(float(np.nanmax(np.abs(B))) + 0.25, 1.25)
    fig, ax = plt.subplots(figsize=(6, 6))
    for vector, label, color in [
        (np.asarray([1.0, 0.0], dtype=np.float32), "identity U", "lightsteelblue"),
        (np.asarray([0.0, 1.0], dtype=np.float32), "identity V", "navajowhite"),
        (B[0], "target U row", "tab:blue"),
        (B[1], "target V row", "tab:orange"),
    ]:
        alpha = 0.35 if label.startswith("identity") else 0.95
        linestyle = "--" if label.startswith("identity") else "-"
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.01 * axis_limit,
            head_width=0.06 * axis_limit,
            length_includes_head=True,
            alpha=alpha,
            linestyle=linestyle,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=32, alpha=alpha, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("source U coefficient")
    ax.set_ylabel("source V coefficient")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _B_row_vector_frame(B: np.ndarray, title: str, axis_limit: float) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(6, 6))
    for vector, label, color in [
        (np.asarray([1.0, 0.0], dtype=np.float32), "identity U", "lightsteelblue"),
        (np.asarray([0.0, 1.0], dtype=np.float32), "identity V", "navajowhite"),
        (B[0], "target U row", "tab:blue"),
        (B[1], "target V row", "tab:orange"),
    ]:
        alpha = 0.35 if label.startswith("identity") else 0.95
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.01 * axis_limit,
            head_width=0.06 * axis_limit,
            length_includes_head=True,
            alpha=alpha,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=32, alpha=alpha, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("source U coefficient")
    ax.set_ylabel("source V coefficient")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return frame


def _component_block_vector_plot(
    path: Path,
    vectors: np.ndarray,
    title: str,
    axis_limit: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if axis_limit is None:
        axis_limit = max(float(np.nanmax(np.abs(vectors))) + 0.25, 1.25)
    fig, ax = plt.subplots(figsize=(6, 6))
    for vector, label, color in [
        (vectors[0], "target U blocks", "tab:blue"),
        (vectors[1], "target V blocks", "tab:orange"),
    ]:
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.01 * axis_limit,
            head_width=0.06 * axis_limit,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=34, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("source U total block weight")
    ax.set_ylabel("source V total block weight")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _component_block_vector_frame(vectors: np.ndarray, title: str, axis_limit: float) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(6, 6))
    for vector, label, color in [
        (vectors[0], "target U blocks", "tab:blue"),
        (vectors[1], "target V blocks", "tab:orange"),
    ]:
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.01 * axis_limit,
            head_width=0.06 * axis_limit,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=34, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("source U total block weight")
    ax.set_ylabel("source V total block weight")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return frame


def _component_spatial_flow_plot(
    path: Path,
    shared_flow: np.ndarray,
    block_flows: np.ndarray,
    target_flows: np.ndarray,
    title: str,
    axis_limit: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.concatenate(
        [
            shared_flow.reshape(1, 2),
            block_flows.reshape(-1, 2),
            target_flows.reshape(-1, 2),
        ],
        axis=0,
    )
    if axis_limit is None:
        axis_limit = max(float(np.nanmax(np.abs(values))) + 0.5, 1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.arrow(
        0.0,
        0.0,
        float(shared_flow[0]),
        float(shared_flow[1]),
        color="0.35",
        width=0.008 * axis_limit,
        head_width=0.055 * axis_limit,
        length_includes_head=True,
        alpha=0.55,
    )
    ax.scatter([shared_flow[0]], [shared_flow[1]], color="0.35", s=30, alpha=0.55, label="shared flow")

    for idx, (i, j, label, color) in enumerate(
        [
            (0, 0, "UU block flow", "tab:blue"),
            (0, 1, "UV block flow", "cornflowerblue"),
            (1, 0, "VU block flow", "darkorange"),
            (1, 1, "VV block flow", "tab:orange"),
        ]
    ):
        vector = block_flows[i, j]
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.005 * axis_limit,
            head_width=0.04 * axis_limit,
            length_includes_head=True,
            alpha=0.45,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=24, alpha=0.45, label=label)

    for vector, label, color in [
        (target_flows[0], "target U net flow", "tab:blue"),
        (target_flows[1], "target V net flow", "tab:orange"),
    ]:
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.012 * axis_limit,
            head_width=0.075 * axis_limit,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=46, alpha=0.95, label=label)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x displacement / step")
    ax.set_ylabel("y displacement / step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _component_spatial_flow_frame(
    shared_flow: np.ndarray,
    block_flows: np.ndarray,
    target_flows: np.ndarray,
    title: str,
    axis_limit: float,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.arrow(
        0.0,
        0.0,
        float(shared_flow[0]),
        float(shared_flow[1]),
        color="0.35",
        width=0.008 * axis_limit,
        head_width=0.055 * axis_limit,
        length_includes_head=True,
        alpha=0.55,
    )
    ax.scatter([shared_flow[0]], [shared_flow[1]], color="0.35", s=30, alpha=0.55, label="shared flow")

    for i, j, label, color in [
        (0, 0, "UU block flow", "tab:blue"),
        (0, 1, "UV block flow", "cornflowerblue"),
        (1, 0, "VU block flow", "darkorange"),
        (1, 1, "VV block flow", "tab:orange"),
    ]:
        vector = block_flows[i, j]
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.005 * axis_limit,
            head_width=0.04 * axis_limit,
            length_includes_head=True,
            alpha=0.45,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=24, alpha=0.45, label=label)

    for vector, label, color in [
        (target_flows[0], "target U net flow", "tab:blue"),
        (target_flows[1], "target V net flow", "tab:orange"),
    ]:
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.012 * axis_limit,
            head_width=0.075 * axis_limit,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=46, alpha=0.95, label=label)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x displacement / step")
    ax.set_ylabel("y displacement / step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return frame


def _row_scaled_B_flow_plot(
    path: Path,
    vectors: np.ndarray,
    title: str,
    axis_limit: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if axis_limit is None:
        axis_limit = max(float(np.nanmax(np.abs(vectors))) + 0.5, 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for vector, label, color in [
        (vectors[0], "flow_x * [B_UU, B_UV]", "tab:blue"),
        (vectors[1], "flow_y * [B_VU, B_VV]", "tab:orange"),
    ]:
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.01 * axis_limit,
            head_width=0.06 * axis_limit,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=42, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("first scaled B component")
    ax.set_ylabel("second scaled B component")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _row_scaled_B_flow_frame(vectors: np.ndarray, title: str, axis_limit: float) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for vector, label, color in [
        (vectors[0], "flow_x * [B_UU, B_UV]", "tab:blue"),
        (vectors[1], "flow_y * [B_VU, B_VV]", "tab:orange"),
    ]:
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.01 * axis_limit,
            head_width=0.06 * axis_limit,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=42, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("first scaled B component")
    ax.set_ylabel("second scaled B component")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return frame


def save_plots(output_dir: Path, artifacts: dict[str, np.ndarray], max_points: int, max_gif_frames: int) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:
        (plots_dir / "PLOTS_NOT_CREATED.txt").write_text(
            f"Install matplotlib and pillow to create plots/GIFs: {exc}\n",
            encoding="utf-8",
        )
        return

    _line_plot(plots_dir / "advection_mu.png", artifacts["mu"], ADV_NAMES, "Advection mean mu", "coordinate units / step", max_points)
    _line_plot(plots_dir / "advection_sigma_diag.png", artifacts["Sigma_diag"], ["var_Uu", "var_Uv", "var_Vu", "var_Vv"], "Advection covariance diagonal", "variance", max_points)
    _line_plot(plots_dir / "mixing_A.png", artifacts["A"].reshape(artifacts["A"].shape[0], 4), A_NAMES, "Component mixing matrix A", "weight", max_points)
    if artifacts["flow_mu"].shape[0] == artifacts["mu"].shape[0]:
        _line_plot(plots_dir / "shared_flow_mu.png", artifacts["flow_mu"], FLOW_NAMES, "Shared advection flow", "coordinate units / step", max_points)
        _line_plot(plots_dir / "shared_flow_sigma_diag.png", artifacts["flow_Sigma_diag"], ["flow_var_x", "flow_var_y"], "Shared flow covariance diagonal", "variance", max_points)
    if artifacts["B"].shape[0] == artifacts["mu"].shape[0]:
        B_values = artifacts["B"].reshape(artifacts["B"].shape[0], 4)
        _line_plot(plots_dir / "deformation_B.png", B_values, B_NAMES, "Component deformation matrix B", "signed coefficient", max_points)
        B_minus_I = artifacts["B"] - np.eye(2, dtype=artifacts["B"].dtype)
        _line_plot(plots_dir / "deformation_B_minus_identity.png", B_minus_I.reshape(B_minus_I.shape[0], 4), B_NAMES, "Component deformation B - I", "signed coefficient", max_points)
    if artifacts["processed_flow_mu"].shape[0] == artifacts["mu"].shape[0]:
        _line_plot(
            plots_dir / "processed_flow_mu.png",
            artifacts["processed_flow_mu"],
            PROCESSED_FLOW_NAMES,
            "B-processed advection flow",
            "coordinate units / step",
            max_points,
        )
    if artifacts["target_component_flow"].shape[0] == artifacts["mu"].shape[0]:
        target_component_flow = artifacts["target_component_flow"]
        _line_plot(
            plots_dir / "target_component_flow_mu.png",
            target_component_flow.reshape(target_component_flow.shape[0], 4),
            ["target_U_x", "target_U_y", "target_V_x", "target_V_y"],
            "Target component spatial flows from B and shared flow",
            "coordinate units / step",
            max_points,
        )
    if artifacts["row_scaled_B_flow"].shape[0] == artifacts["mu"].shape[0]:
        row_scaled_B_flow = artifacts["row_scaled_B_flow"]
        _line_plot(
            plots_dir / "row_scaled_B_flow.png",
            row_scaled_B_flow.reshape(row_scaled_B_flow.shape[0], 4),
            ROW_SCALED_B_FLOW_NAMES,
            "Row-scaled B flow: [flow_x*B_U_row, flow_y*B_V_row]",
            "scaled coefficient",
            max_points,
        )
    if artifacts["kernel_weight"].shape[0] == artifacts["mu"].shape[0]:
        _line_plot(plots_dir / "kernel_weight.png", artifacts["kernel_weight"], ["kernel_weight"], "Kernel mixing weight", "weight", max_points)
    if artifacts["residual_decay"].shape[0] == artifacts["mu"].shape[0]:
        _line_plot(plots_dir / "residual_decay.png", artifacts["residual_decay"], ["residual_decay"], "Residual decay", "decay", max_points)
    if artifacts["transition_control"].shape[0] == artifacts["mu"].shape[0]:
        _line_plot(plots_dir / "transition_control.png", artifacts["transition_control"], STATE_NAMES, "Transition control", "state units / step", max_points)
    _line_plot(plots_dir / "transition_row_sums.png", np.nansum(artifacts["transition_matrices"], axis=2), STATE_NAMES, "Transition matrix row sums", "row sum", max_points)
    _horizon_plot(
        plots_dir / "multi_step_rmse.png",
        artifacts["horizons"],
        artifacts["multi_step_rmse_curve"],
        artifacts["multi_step_curve_columns"],
        "Multi-step RMSE",
        "RMSE",
    )
    _horizon_plot(
        plots_dir / "multi_step_mae.png",
        artifacts["horizons"],
        artifacts["multi_step_mae_curve"],
        artifacts["multi_step_curve_columns"],
        "Multi-step MAE",
        "MAE",
    )

    M_mean = np.nanmean(artifacts["transition_matrices"], axis=0)
    identity = np.eye(M_mean.shape[0], dtype=M_mean.dtype)
    M_minus_I = M_mean - identity
    signed_transition = bool(np.nanmin(artifacts["transition_matrices"]) < 0.0)
    _heatmap(
        plots_dir / "transition_matrix_mean.png",
        M_mean,
        "Mean transition matrix M",
        STATE_NAMES,
        STATE_NAMES,
        cmap="coolwarm" if signed_transition else "viridis",
        center_zero=signed_transition,
    )
    _heatmap(plots_dir / "transition_matrix_mean_minus_identity.png", M_minus_I, "Mean transition matrix M - I", STATE_NAMES, STATE_NAMES, cmap="coolwarm", center_zero=True)
    if not signed_transition:
        M_log_mean = np.nanmean(np.log10(np.clip(artifacts["transition_matrices"], 1.0e-12, None)), axis=0)
        _heatmap(plots_dir / "transition_matrix_log10_mean.png", M_log_mean, "Mean log10 transition matrix", STATE_NAMES, STATE_NAMES)
    _bubble_matrix(
        plots_dir / "transition_matrix_mean_bubble.png",
        M_mean,
        "Mean transition matrix M",
        STATE_NAMES,
        STATE_NAMES,
        cmap="coolwarm" if signed_transition else "viridis",
        center_zero=signed_transition,
    )
    _bubble_matrix(
        plots_dir / "transition_matrix_mean_minus_identity_bubble.png",
        M_minus_I,
        "Mean transition matrix M - I",
        STATE_NAMES,
        STATE_NAMES,
        cmap="coolwarm",
        center_zero=True,
    )
    M_base = artifacts.get("transition_base_matrices", np.empty((0, 6, 6), dtype=np.float32))
    if M_base.shape == artifacts["transition_matrices"].shape:
        M_base_mean = np.nanmean(M_base, axis=0)
        signed_base = bool(np.nanmin(M_base) < 0.0)
        _heatmap(
            plots_dir / "transition_base_matrix_mean.png",
            M_base_mean,
            "Mean base transition matrix M_base",
            STATE_NAMES,
            STATE_NAMES,
            cmap="coolwarm" if signed_base else "viridis",
            center_zero=signed_base,
        )
        _bubble_matrix(
            plots_dir / "transition_base_matrix_mean_bubble.png",
            M_base_mean,
            "Mean base transition matrix M_base",
            STATE_NAMES,
            STATE_NAMES,
            cmap="coolwarm" if signed_base else "viridis",
            center_zero=signed_base,
        )
        block_vectors = artifacts.get("transition_base_block_vectors", np.empty((0, 2, 2), dtype=np.float32))
        if block_vectors.shape[0] == M_base.shape[0]:
            _component_block_vector_plot(
                plots_dir / "transition_base_block_vectors_mean.png",
                np.nanmean(block_vectors, axis=0),
                "Mean M_base component block vectors",
            )
    _heatmap(plots_dir / "kernel_lengthscale_ell.png", artifacts["ell"], "Kernel lengthscales ell", ["U source", "V source"], ["U target", "V target"])
    if artifacts["B"].shape[0] == artifacts["mu"].shape[0]:
        B_mean = np.nanmean(artifacts["B"], axis=0)
        _heatmap(plots_dir / "deformation_B_mean.png", B_mean, "Mean component deformation B", ["U source", "V source"], ["U target", "V target"], cmap="coolwarm", center_zero=True)
        _bubble_matrix(plots_dir / "deformation_B_mean_bubble.png", B_mean, "Mean component deformation B", ["U source", "V source"], ["U target", "V target"], cmap="coolwarm", center_zero=True)
        _B_row_vector_plot(
            plots_dir / "deformation_B_row_vectors_mean.png",
            B_mean,
            "Mean component deformation B row vectors",
        )
    component_flow_blocks = artifacts.get("component_flow_blocks", np.empty((0, 2, 2, 2), dtype=np.float32))
    target_component_flow = artifacts.get("target_component_flow", np.empty((0, 2, 2), dtype=np.float32))
    component_flow_strength = artifacts.get("component_flow_strength", np.empty((0, 2, 2), dtype=np.float32))
    if (
        artifacts["flow_mu"].shape[0] == artifacts["mu"].shape[0]
        and component_flow_blocks.shape[0] == artifacts["mu"].shape[0]
        and target_component_flow.shape[0] == artifacts["mu"].shape[0]
    ):
        _component_spatial_flow_plot(
            plots_dir / "component_spatial_flow_vectors_mean.png",
            np.nanmean(artifacts["flow_mu"], axis=0),
            np.nanmean(component_flow_blocks, axis=0),
            np.nanmean(target_component_flow, axis=0),
            "Mean component spatial flows from B and shared flow",
        )
    if component_flow_strength.shape[0] == artifacts["mu"].shape[0]:
        strength_mean = np.nanmean(component_flow_strength, axis=0)
        _bubble_matrix(
            plots_dir / "component_spatial_flow_strength_mean_bubble.png",
            strength_mean,
            "Mean signed component spatial flow strength",
            ["U source", "V source"],
            ["U target", "V target"],
            cmap="coolwarm",
            center_zero=True,
        )
    row_scaled_B_flow = artifacts.get("row_scaled_B_flow", np.empty((0, 2, 2), dtype=np.float32))
    if row_scaled_B_flow.shape[0] == artifacts["mu"].shape[0]:
        _row_scaled_B_flow_plot(
            plots_dir / "row_scaled_B_flow_vectors_mean.png",
            np.nanmean(row_scaled_B_flow, axis=0),
            "Mean row-scaled B flow",
        )
    _heatmap(plots_dir / "process_covariance_Q.png", artifacts["Q"], "Process covariance Q", STATE_NAMES, STATE_NAMES)
    _heatmap(plots_dir / "observation_covariance_R.png", artifacts["R"], "Observation covariance R", STATE_NAMES, STATE_NAMES)

    mu = artifacts["mu"]
    Sigma = artifacts["Sigma"]
    finite_adv_times = np.where(np.isfinite(mu).all(axis=1) & np.isfinite(Sigma).all(axis=(1, 2)))[0]
    if finite_adv_times.size:
        adv_frame_indices = finite_adv_times[_downsample_indices(finite_adv_times.size, max_gif_frames)]
        mu_axis_limit = float(np.nanpercentile(np.abs(mu[adv_frame_indices]), 99))
        mu_axis_limit = max(mu_axis_limit, 1.0)
        frames = []
        for t in adv_frame_indices:
            frames.append(_advection_mu_frame(mu[t], f"Advection mean mu, t={int(t)}", mu_axis_limit))
        if frames:
            frames[0].save(
                plots_dir / "advection_mu_vectors.gif",
                save_all=True,
                append_images=frames[1:],
                duration=140,
                loop=0,
            )

        sigma_vmax = float(np.nanpercentile(np.abs(Sigma[adv_frame_indices]), 99))
        sigma_vmax = max(sigma_vmax, 1.0e-8)
        frames = []
        for t in adv_frame_indices:
            frames.append(
                _bubble_matrix_frame(
                    Sigma[t],
                    f"Advection covariance Sigma, t={int(t)}",
                    ADV_NAMES,
                    ADV_NAMES,
                    cmap="coolwarm",
                    center_zero=True,
                    vmax_abs=sigma_vmax,
                )
            )
        if frames:
            frames[0].save(
                plots_dir / "advection_sigma_matrix.gif",
                save_all=True,
                append_images=frames[1:],
                duration=140,
                loop=0,
            )

    flow_mu = artifacts["flow_mu"]
    flow_sigma = artifacts["flow_Sigma"]
    processed_flow_mu = artifacts["processed_flow_mu"]
    component_flow_blocks = artifacts.get("component_flow_blocks", np.empty((0, 2, 2, 2), dtype=np.float32))
    target_component_flow = artifacts.get("target_component_flow", np.empty((0, 2, 2), dtype=np.float32))
    component_flow_strength = artifacts.get("component_flow_strength", np.empty((0, 2, 2), dtype=np.float32))
    row_scaled_B_flow = artifacts.get("row_scaled_B_flow", np.empty((0, 2, 2), dtype=np.float32))
    if flow_mu.shape[0] == artifacts["mu"].shape[0]:
        finite_flow_times = np.where(np.isfinite(flow_mu).all(axis=1))[0]
        if finite_flow_times.size:
            flow_frame_indices = finite_flow_times[_downsample_indices(finite_flow_times.size, max_gif_frames)]
            flow_axis_limit = float(np.nanpercentile(np.abs(flow_mu[flow_frame_indices]), 99))
            if processed_flow_mu.shape[0] == flow_mu.shape[0]:
                flow_axis_limit = max(
                    flow_axis_limit,
                    float(np.nanpercentile(np.abs(processed_flow_mu[flow_frame_indices]), 99)),
                )
            flow_axis_limit = max(flow_axis_limit, 1.0)
            frames = []
            for t in flow_frame_indices:
                frames.append(_shared_flow_frame(flow_mu[t], f"Shared advection flow, t={int(t)}", flow_axis_limit))
            if frames:
                frames[0].save(
                    plots_dir / "shared_flow_vectors.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=140,
                    loop=0,
                )

            if (
                component_flow_blocks.shape[0] == flow_mu.shape[0]
                and target_component_flow.shape[0] == flow_mu.shape[0]
            ):
                finite_component = (
                    np.isfinite(component_flow_blocks[flow_frame_indices]).all(axis=(1, 2, 3))
                    & np.isfinite(target_component_flow[flow_frame_indices]).all(axis=(1, 2))
                )
                component_indices = flow_frame_indices[finite_component]
                if component_indices.size:
                    component_axis_limit = float(
                        np.nanpercentile(
                            np.abs(
                                np.concatenate(
                                    [
                                        flow_mu[component_indices].reshape(-1, 2),
                                        component_flow_blocks[component_indices].reshape(-1, 2),
                                        target_component_flow[component_indices].reshape(-1, 2),
                                    ],
                                    axis=0,
                                )
                            ),
                            99,
                        )
                    )
                    component_axis_limit = max(component_axis_limit, 1.0)
                    frames = []
                    for t in component_indices:
                        frames.append(
                            _component_spatial_flow_frame(
                                flow_mu[t],
                                component_flow_blocks[t],
                                target_component_flow[t],
                                f"Component spatial flows, t={int(t)}",
                                component_axis_limit,
                            )
                        )
                    if frames:
                        frames[0].save(
                            plots_dir / "component_spatial_flow_vectors.gif",
                            save_all=True,
                            append_images=frames[1:],
                            duration=140,
                            loop=0,
                        )

            if component_flow_strength.shape[0] == flow_mu.shape[0]:
                finite_strength = np.isfinite(component_flow_strength[flow_frame_indices]).all(axis=(1, 2))
                strength_indices = flow_frame_indices[finite_strength]
                if strength_indices.size:
                    strength_vmax = float(np.nanpercentile(np.abs(component_flow_strength[strength_indices]), 99))
                    strength_vmax = max(strength_vmax, 1.0e-8)
                    frames = []
                    for t in strength_indices:
                        frames.append(
                            _bubble_matrix_frame(
                                component_flow_strength[t],
                                f"Signed component spatial flow strength, t={int(t)}",
                                ["U source", "V source"],
                                ["U target", "V target"],
                                cmap="coolwarm",
                                center_zero=True,
                                vmax_abs=strength_vmax,
                            )
                        )
                    if frames:
                        frames[0].save(
                            plots_dir / "component_spatial_flow_strength_bubble.gif",
                            save_all=True,
                            append_images=frames[1:],
                            duration=140,
                            loop=0,
                        )

            if row_scaled_B_flow.shape[0] == flow_mu.shape[0]:
                finite_row_scaled = np.isfinite(row_scaled_B_flow[flow_frame_indices]).all(axis=(1, 2))
                row_scaled_indices = flow_frame_indices[finite_row_scaled]
                if row_scaled_indices.size:
                    row_scaled_axis_limit = float(
                        np.nanpercentile(np.abs(row_scaled_B_flow[row_scaled_indices]), 99)
                    )
                    row_scaled_axis_limit = max(row_scaled_axis_limit, 1.0)
                    frames = []
                    for t in row_scaled_indices:
                        frames.append(
                            _row_scaled_B_flow_frame(
                                row_scaled_B_flow[t],
                                f"Row-scaled B flow, t={int(t)}",
                                row_scaled_axis_limit,
                            )
                        )
                    if frames:
                        frames[0].save(
                            plots_dir / "row_scaled_B_flow_vectors.gif",
                            save_all=True,
                            append_images=frames[1:],
                            duration=140,
                            loop=0,
                        )

            if processed_flow_mu.shape[0] == flow_mu.shape[0]:
                finite_processed = np.isfinite(processed_flow_mu[flow_frame_indices]).all(axis=1)
                frames = []
                for t in flow_frame_indices[finite_processed]:
                    frames.append(
                        _processed_flow_frame(
                            flow_mu[t],
                            processed_flow_mu[t],
                            f"Shared flow and B-processed flow, t={int(t)}",
                            flow_axis_limit,
                        )
                    )
                if frames:
                    frames[0].save(
                        plots_dir / "processed_flow_vectors.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=140,
                        loop=0,
                    )

            if flow_sigma.shape[0] == flow_mu.shape[0]:
                flow_sigma_vmax = float(np.nanpercentile(np.abs(flow_sigma[flow_frame_indices]), 99))
                flow_sigma_vmax = max(flow_sigma_vmax, 1.0e-8)
                frames = []
                for t in flow_frame_indices:
                    frames.append(
                        _bubble_matrix_frame(
                            flow_sigma[t],
                            f"Shared flow covariance, t={int(t)}",
                            FLOW_NAMES,
                            FLOW_NAMES,
                            cmap="coolwarm",
                            center_zero=True,
                            vmax_abs=flow_sigma_vmax,
                        )
                    )
                if frames:
                    frames[0].save(
                        plots_dir / "shared_flow_sigma_matrix.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=140,
                        loop=0,
                    )

    B = artifacts["B"]
    if B.shape[0] == artifacts["mu"].shape[0]:
        finite_B_times = np.where(np.isfinite(B).all(axis=(1, 2)))[0]
        if finite_B_times.size:
            B_frame_indices = finite_B_times[_downsample_indices(finite_B_times.size, max_gif_frames)]
            B_vmax = float(np.nanpercentile(np.abs(B[B_frame_indices]), 99))
            B_vmax = max(B_vmax, 1.0e-8)
            frames = []
            for t in B_frame_indices:
                frames.append(
                    _bubble_matrix_frame(
                        B[t],
                        f"Component deformation B, t={int(t)}",
                        ["U source", "V source"],
                        ["U target", "V target"],
                        cmap="coolwarm",
                        center_zero=True,
                        vmax_abs=B_vmax,
                    )
                )
            if frames:
                frames[0].save(
                    plots_dir / "deformation_B_matrix.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=140,
                    loop=0,
                )

            B_axis_limit = float(np.nanpercentile(np.abs(B[B_frame_indices]), 99)) + 0.25
            B_axis_limit = max(B_axis_limit, 1.25)
            frames = []
            for t in B_frame_indices:
                frames.append(
                    _B_row_vector_frame(
                        B[t],
                        f"Component deformation B row vectors, t={int(t)}",
                        B_axis_limit,
                    )
                )
            if frames:
                frames[0].save(
                    plots_dir / "deformation_B_row_vectors.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=140,
                    loop=0,
                )

            B_residual = B - np.eye(2, dtype=B.dtype)
            B_resid_vmax = float(np.nanpercentile(np.abs(B_residual[B_frame_indices]), 99))
            B_resid_vmax = max(B_resid_vmax, 1.0e-8)
            frames = []
            for t in B_frame_indices:
                frames.append(
                    _bubble_matrix_frame(
                        B_residual[t],
                        f"Component deformation B - I, t={int(t)}",
                        ["U source", "V source"],
                        ["U target", "V target"],
                        cmap="coolwarm",
                        center_zero=True,
                        vmax_abs=B_resid_vmax,
                    )
                )
            if frames:
                frames[0].save(
                    plots_dir / "deformation_B_minus_identity.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=140,
                    loop=0,
                )

    M = artifacts["transition_matrices"]
    finite_times = np.where(np.isfinite(M).all(axis=(1, 2)))[0]
    if finite_times.size == 0:
        return
    frame_indices = finite_times[_downsample_indices(finite_times.size, max_gif_frames)]
    signed_transition = bool(np.nanmin(M[frame_indices]) < 0.0)
    if signed_transition:
        vmax_abs_M = float(np.nanpercentile(np.abs(M[frame_indices]), 99))
        vmax_abs_M = max(vmax_abs_M, 1.0e-8)
        vmin, vmax = -vmax_abs_M, vmax_abs_M
        transition_cmap = "coolwarm"
    else:
        vmin = float(np.nanpercentile(M[frame_indices], 1))
        vmax = float(np.nanpercentile(M[frame_indices], 99))
        transition_cmap = "viridis"
    frames = []
    for t in frame_indices:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(M[t], cmap=transition_cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Transition matrix M, t={int(t)}")
        ax.set_xticks(np.arange(len(STATE_NAMES)))
        ax.set_yticks(np.arange(len(STATE_NAMES)))
        ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(STATE_NAMES, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    if frames:
        frames[0].save(
            plots_dir / "transition_matrix.gif",
            save_all=True,
            append_images=frames[1:],
            duration=140,
            loop=0,
        )

    frames = []
    bubble_vmax = float(np.nanpercentile(np.abs(M[frame_indices]) if signed_transition else M[frame_indices], 99))
    bubble_vmax = max(bubble_vmax, 1.0e-8)
    for t in frame_indices:
        frames.append(
            _bubble_matrix_frame(
                M[t],
                f"Transition matrix M, t={int(t)}",
                STATE_NAMES,
                STATE_NAMES,
                cmap=transition_cmap,
                center_zero=signed_transition,
                vmax_abs=bubble_vmax,
            )
        )
    if frames:
        frames[0].save(
            plots_dir / "transition_matrix_bubble.gif",
            save_all=True,
            append_images=frames[1:],
            duration=140,
            loop=0,
        )

    M_base = artifacts.get("transition_base_matrices", np.empty((0, 6, 6), dtype=np.float32))
    if M_base.shape == M.shape:
        signed_base = bool(np.nanmin(M_base[frame_indices]) < 0.0)
        if signed_base:
            vmax_abs_base = float(np.nanpercentile(np.abs(M_base[frame_indices]), 99))
            vmax_abs_base = max(vmax_abs_base, 1.0e-8)
            base_vmin, base_vmax = -vmax_abs_base, vmax_abs_base
            base_cmap = "coolwarm"
        else:
            base_vmin = float(np.nanpercentile(M_base[frame_indices], 1))
            base_vmax = float(np.nanpercentile(M_base[frame_indices], 99))
            base_cmap = "viridis"

        frames = []
        for t in frame_indices:
            fig, ax = plt.subplots(figsize=(6, 5.5))
            im = ax.imshow(M_base[t], cmap=base_cmap, vmin=base_vmin, vmax=base_vmax)
            ax.set_title(f"Base transition matrix M_base, t={int(t)}")
            ax.set_xticks(np.arange(len(STATE_NAMES)))
            ax.set_yticks(np.arange(len(STATE_NAMES)))
            ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(STATE_NAMES, fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
            plt.close(fig)
        if frames:
            frames[0].save(
                plots_dir / "transition_base_matrix.gif",
                save_all=True,
                append_images=frames[1:],
                duration=140,
                loop=0,
            )

        frames = []
        base_bubble_vmax = float(
            np.nanpercentile(np.abs(M_base[frame_indices]) if signed_base else M_base[frame_indices], 99)
        )
        base_bubble_vmax = max(base_bubble_vmax, 1.0e-8)
        for t in frame_indices:
            frames.append(
                _bubble_matrix_frame(
                    M_base[t],
                    f"Base transition matrix M_base, t={int(t)}",
                    STATE_NAMES,
                    STATE_NAMES,
                    cmap=base_cmap,
                    center_zero=signed_base,
                    vmax_abs=base_bubble_vmax,
                )
            )
        if frames:
            frames[0].save(
                plots_dir / "transition_base_matrix_bubble.gif",
                save_all=True,
                append_images=frames[1:],
                duration=140,
                loop=0,
            )

        block_vectors = artifacts.get("transition_base_block_vectors", np.empty((0, 2, 2), dtype=np.float32))
        if block_vectors.shape[0] == M_base.shape[0]:
            vector_indices = frame_indices[np.isfinite(block_vectors[frame_indices]).all(axis=(1, 2))]
            if vector_indices.size:
                vector_axis_limit = float(np.nanpercentile(np.abs(block_vectors[vector_indices]), 99)) + 0.25
                vector_axis_limit = max(vector_axis_limit, 1.25)
                frames = []
                for t in vector_indices:
                    frames.append(
                        _component_block_vector_frame(
                            block_vectors[t],
                            f"M_base component block vectors, t={int(t)}",
                            vector_axis_limit,
                        )
                    )
                if frames:
                    frames[0].save(
                        plots_dir / "transition_base_block_vectors.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=140,
                        loop=0,
                    )

    frames = []
    residual = M - np.eye(M.shape[-1], dtype=M.dtype)
    vmax_resid = float(np.nanpercentile(np.abs(residual[frame_indices]), 99))
    vmax_resid = max(vmax_resid, 1.0e-8)
    for t in frame_indices:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(residual[t], cmap="coolwarm", vmin=-vmax_resid, vmax=vmax_resid)
        ax.set_title(f"Transition residual M - I, t={int(t)}")
        ax.set_xticks(np.arange(len(STATE_NAMES)))
        ax.set_yticks(np.arange(len(STATE_NAMES)))
        ax.set_xticklabels(STATE_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(STATE_NAMES, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    if frames:
        frames[0].save(
            plots_dir / "transition_matrix_minus_identity.gif",
            save_all=True,
            append_images=frames[1:],
            duration=140,
            loop=0,
        )

    frames = []
    for t in frame_indices:
        frames.append(
            _bubble_matrix_frame(
                residual[t],
                f"Transition residual M - I, t={int(t)}",
                STATE_NAMES,
                STATE_NAMES,
                cmap="coolwarm",
                center_zero=True,
                vmax_abs=vmax_resid,
            )
        )
    if frames:
        frames[0].save(
            plots_dir / "transition_matrix_minus_identity_bubble.gif",
            save_all=True,
            append_images=frames[1:],
            duration=140,
            loop=0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VectorMIDE checkpoint.")
    parser.add_argument("--config", default="yml_files/VectorMIDE.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["offline", "online"], default="online")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-window-size", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--forecast-horizon", type=int, default=None, help="Evaluate forecasts from 1 to this horizon.")
    parser.add_argument(
        "--history-window-size",
        type=int,
        default=None,
        help="If set, forecast metrics use only this many recent observed steps before each origin.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for JSON, arrays, plots, and GIF.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG/GIF plot generation.")
    parser.add_argument("--max-plot-points", type=int, default=2500)
    parser.add_argument("--max-gif-frames", type=int, default=120)
    args = parser.parse_args()

    config = load_config(args.config)
    device_name = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(
        device_name,
        allow_fallback=bool(config.get("allow_device_fallback", True)),
    )
    print_device_info(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = Path(config.get("checkpoint_dir", "checkpoints")) / config.get(
            "offline_checkpoint_name",
            "vector_mide_offline.pt",
        )
    checkpoint = torch.load(ckpt_path, map_location=device)

    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:
        print(f"Initialized model parameters not found in checkpoint: {missing}")
    if unexpected:
        print(f"Ignored checkpoint parameters not used by this config: {unexpected}")

    eval_window_size = args.eval_window_size
    if eval_window_size is None:
        eval_window_size = int(
            min(
                model_config.get("window_size", config.get("window_size", 1008)),
                model_config.get("transformer_max_len", config.get("transformer_max_len", 4096)),
            )
        )
    data = load_vector_dataset(config, split=args.split, time_limit=args.limit)
    forecast_horizon = args.forecast_horizon
    if forecast_horizon is None:
        forecast_horizon = int(config.get("forecast_horizon", 1))
    if forecast_horizon < 1:
        raise ValueError("--forecast-horizon must be >= 1")
    if args.history_window_size is not None and args.history_window_size < 1:
        raise ValueError("--history-window-size must be >= 1")
    results, artifacts = evaluate(
        model,
        data,
        device,
        eval_window_size=eval_window_size,
        eval_stride=args.eval_stride,
        forecast_horizon=forecast_horizon,
        history_window_size=args.history_window_size,
    )
    results["split"] = args.split
    results["checkpoint"] = str(ckpt_path)
    results["n_time"] = int(data["X"].shape[0])
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(ckpt_path, args.split)
    results["output_dir"] = str(output_dir)

    text = json.dumps(results, indent=2)
    print(text)
    save_json(output_dir / "results.json", results)
    save_artifact_arrays(output_dir, artifacts)
    if not args.no_plots:
        save_plots(
            output_dir,
            artifacts,
            max_points=int(args.max_plot_points),
            max_gif_frames=int(args.max_gif_frames),
        )
    print(f"Saved evaluation artifacts to {output_dir}")
    if args.output:
        save_json(Path(args.output), results)


if __name__ == "__main__":
    main()
