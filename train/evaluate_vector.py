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
from train.train_vector_offline import build_covariance_proxy_arrays, build_model, print_device_info, resolve_device


STATE_NAMES = ["U1", "U2", "U3", "V1", "V2", "V3"]
ADV_NAMES = ["A_u_x", "A_u_y", "A_v_x", "A_v_y"]
SIGMA_DIAG_NAMES = ["var_A_u_x", "var_A_u_y", "var_A_v_x", "var_A_v_y"]
ALPHA_NAMES = ["alpha_uu", "alpha_uv", "alpha_vu", "alpha_vv"]
CURVE_COLUMNS = ["model", "persistence", "nwp", "nwp_residual_persistence"]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_compatible_state_dict(model: torch.nn.Module, checkpoint: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    state = checkpoint["model_state"]
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), list(unexpected), skipped


def default_output_dir(checkpoint_path: str | Path, split: str) -> Path:
    stem = Path(checkpoint_path).stem
    return Path("outputs") / "evaluation" / f"{stem}_{split}"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def target_mode_is_residual(data: dict[str, Any]) -> bool:
    return str(data.get("target_mode", "measurement")).lower() in {
        "residual_nwp",
        "nwp_residual",
        "residual",
    }


def measurement_from_model_target(values: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    """Convert model-target predictions to measurement-space predictions."""
    values = np.asarray(values, dtype=np.float32)
    if not target_mode_is_residual(data):
        return values
    baseline = np.asarray(data["nwp_baseline"], dtype=np.float32)
    if values.ndim == 2:
        return values + baseline
    if values.ndim == 3:
        return values + baseline[None, :, :]
    raise ValueError(f"Expected 2D or 3D predictions, got {values.shape}")


def persistence_forecast(values: torch.Tensor, horizon: int) -> torch.Tensor:
    pred = torch.full_like(values, torch.nan)
    if horizon < values.shape[0]:
        pred[horizon:] = values[:-horizon]
    return pred


def residual_persistence_forecast(
    target: torch.Tensor,
    nwp_baseline: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    residual = target - nwp_baseline
    return nwp_baseline + persistence_forecast(residual, horizon=horizon)


def finite_metrics(prediction: torch.Tensor, target: torch.Tensor, skip_initial: int = 0) -> dict[str, float]:
    pred = prediction[skip_initial:]
    truth = target[skip_initial:]
    mask = torch.isfinite(pred) & torch.isfinite(truth)
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "count": 0.0}
    err = pred[mask] - truth[mask]
    return {
        "rmse": float(torch.sqrt(err.pow(2).mean()).detach().cpu()),
        "mae": float(err.abs().mean().detach().cpu()),
        "bias": float(err.mean().detach().cpu()),
        "count": float(mask.sum().detach().cpu()),
    }


def improvement(model_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("rmse", "mae"):
        base = float(baseline_metrics.get(key, float("nan")))
        val = float(model_metrics.get(key, float("nan")))
        if np.isfinite(base) and abs(base) > 1.0e-12 and np.isfinite(val):
            out[f"{key}_reduction_fraction"] = (base - val) / base
        else:
            out[f"{key}_reduction_fraction"] = float("nan")
    return out


def vector_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=-1)


def angle_diff_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed angle from vector ``b`` to vector ``a`` in degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cross = b[..., 0] * a[..., 1] - b[..., 1] * a[..., 0]
    dot = b[..., 0] * a[..., 0] + b[..., 1] * a[..., 1]
    return np.rad2deg(np.arctan2(cross, dot)).astype(np.float32)


def corrcoef_finite(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    sx = float(np.nanstd(x[mask]))
    sy = float(np.nanstd(y[mask]))
    if sx <= 1.0e-12 or sy <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, max_points).round().astype(np.int64)


def collect_model_outputs(
    model: torch.nn.Module,
    data: dict[str, Any],
    coords: torch.Tensor,
    device: torch.device,
    eval_window_size: int,
    eval_stride: int,
) -> dict[str, np.ndarray]:
    T = int(data["X"].shape[0])
    state_dim = int(data["Z"].shape[1])
    counts = np.zeros((T, 1), dtype=np.float64)
    sums = {
        "M": np.zeros((T, state_dim, state_dim), dtype=np.float64),
        "M_base": np.zeros((T, state_dim, state_dim), dtype=np.float64),
        "mu": np.zeros((T, 4), dtype=np.float64),
        "Sigma": np.zeros((T, 4, 4), dtype=np.float64),
        "alpha": np.zeros((T, 2, 2), dtype=np.float64),
    }
    static_outputs: dict[str, np.ndarray] = {}

    model.eval()
    with torch.no_grad():
        for start in range(0, T, eval_stride):
            end = min(start + eval_window_size, T)
            if end <= start:
                continue
            x = torch.from_numpy(data["X"][start:end]).to(device)
            anchor_np = data.get("A_anchor")
            advection_anchor = torch.from_numpy(anchor_np[start:end]).to(device) if anchor_np is not None else None
            outputs = model(x, coords, advection_anchor=advection_anchor)
            sums["M"][start:end] += outputs["M"].detach().cpu().numpy()
            sums["M_base"][start:end] += outputs.get("M_base", outputs["M"]).detach().cpu().numpy()
            sums["mu"][start:end] += outputs["mu"].detach().cpu().numpy()
            sums["Sigma"][start:end] += outputs["Sigma"].detach().cpu().numpy()
            sums["alpha"][start:end] += outputs["alpha"].detach().cpu().numpy()
            for key in (
                "covariance_regime_logits",
                "covariance_regime_probs",
                "covariance_scale",
                "covariance_block_Sigma",
            ):
                if key not in outputs:
                    continue
                value = outputs[key].detach().cpu().numpy()
                if key not in sums:
                    sums[key] = np.zeros((T,) + value.shape[1:], dtype=np.float64)
                sums[key][start:end] += value
            for key in ("covariance_regime_Sigma",):
                if key in outputs:
                    static_outputs[key] = outputs[key].detach().cpu().numpy().astype(np.float32, copy=False)
            counts[start:end] += 1.0

    valid = counts[:, 0] > 0
    if not valid.all():
        missing = int((~valid).sum())
        raise ValueError(f"Model outputs are missing for {missing} time steps; reduce --eval-stride.")

    averaged: dict[str, np.ndarray] = {}
    for key, value in sums.items():
        denom = counts.reshape((T,) + (1,) * (value.ndim - 1))
        averaged[key] = (value / denom).astype(np.float32)
    averaged.update(static_outputs)
    return averaged


def multi_horizon_forecasts(
    model: torch.nn.Module,
    filter_means: torch.Tensor,
    filter_covs: torch.Tensor,
    M_seq: torch.Tensor,
    max_horizon: int,
) -> torch.Tensor:
    T, state_dim = filter_means.shape
    predictions = filter_means.new_full((max_horizon, T, state_dim), torch.nan)
    for origin in range(T - 1):
        horizon = min(max_horizon, T - origin - 1)
        if horizon <= 0:
            continue
        mean = filter_means[origin]
        cov = filter_covs[origin]
        for step in range(1, horizon + 1):
            mean, cov = model.dstm.get_forecast_dist(
                mean,
                cov,
                M_seq[origin + step],
                future_control=None,
            )
            predictions[step - 1, origin + step] = mean
    return predictions


def advection_summary(mu: np.ndarray, Sigma: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    au = mu[:, :2]
    av = mu[:, 2:]
    diff = au - av
    sigma_diag = np.diagonal(Sigma, axis1=1, axis2=2)
    return {
        "mu_mean": {name: float(value) for name, value in zip(ADV_NAMES, np.nanmean(mu, axis=0))},
        "mu_std": {name: float(value) for name, value in zip(ADV_NAMES, np.nanstd(mu, axis=0))},
        "Au_speed_mean": float(np.nanmean(np.linalg.norm(au, axis=1))),
        "Av_speed_mean": float(np.nanmean(np.linalg.norm(av, axis=1))),
        "Au_minus_Av_speed_mean": float(np.nanmean(np.linalg.norm(diff, axis=1))),
        "Sigma_diag_mean": {
            name: float(value) for name, value in zip(SIGMA_DIAG_NAMES, np.nanmean(sigma_diag, axis=0))
        },
        "alpha_mean": {
            name: float(value) for name, value in zip(ALPHA_NAMES, np.nanmean(alpha.reshape(alpha.shape[0], 4), axis=0))
        },
    }


def component_projection_np(target_idx: int, source_idx: int, gamma: float) -> np.ndarray:
    e_u = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    e_v = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    selectors = (e_u, e_v)
    if target_idx == source_idx:
        return selectors[target_idx]
    return selectors[target_idx] - float(gamma) * selectors[source_idx]


def covariance_kernel_diagnostics(
    Sigma: np.ndarray,
    ell: np.ndarray,
    gamma: float,
    sigma_scale: float,
) -> dict[str, np.ndarray]:
    sigma = np.asarray(Sigma, dtype=np.float64)
    ell_arr = np.asarray(ell, dtype=np.float64)
    T = sigma.shape[0]
    eye2 = np.eye(2, dtype=np.float64)
    projected_sigma = np.zeros((T, 2, 2, 2, 2), dtype=np.float32)
    effective_D = np.zeros((T, 2, 2, 2, 2), dtype=np.float32)
    projected_trace = np.zeros((T, 2, 2), dtype=np.float32)
    base_trace = np.zeros((2, 2), dtype=np.float32)
    ratio = np.zeros((T, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            projection = component_projection_np(i, j, float(gamma))
            projected = np.einsum("ab,tbc,dc->tad", projection, sigma, projection)
            scaled_projected = float(sigma_scale) * 2.0 * projected
            base = float(ell_arr[i, j]) ** 2 * eye2
            D = base[None, :, :] + scaled_projected
            projected_sigma[:, i, j] = scaled_projected.astype(np.float32, copy=False)
            effective_D[:, i, j] = D.astype(np.float32, copy=False)
            projected_trace[:, i, j] = np.trace(scaled_projected, axis1=1, axis2=2).astype(np.float32, copy=False)
            base_trace[i, j] = np.float32(2.0 * float(ell_arr[i, j]) ** 2)
            ratio[:, i, j] = projected_trace[:, i, j] / max(float(base_trace[i, j]), 1.0e-12)
    return {
        "projected_sigma": projected_sigma,
        "effective_diffusion_D": effective_D,
        "projected_sigma_trace": projected_trace,
        "effective_diffusion_trace": np.trace(effective_D, axis1=-2, axis2=-1).astype(np.float32, copy=False),
        "base_diffusion_trace": base_trace,
        "projected_sigma_ratio": ratio,
        "projected_sigma_ratio_mean": np.nanmean(ratio.reshape(T, 4), axis=1).astype(np.float32, copy=False),
    }


def covariance_diagnostics_summary(artifacts: dict[str, np.ndarray]) -> dict[str, Any]:
    ratio = artifacts.get("projected_sigma_ratio")
    out: dict[str, Any] = {}
    if ratio is not None:
        flat = np.asarray(ratio, dtype=np.float64).reshape(ratio.shape[0], -1)
        out["projected_sigma_ratio_mean"] = float(np.nanmean(flat))
        out["projected_sigma_ratio_quantiles"] = {
            str(q): float(v)
            for q, v in zip(
                [0.0, 0.1, 0.5, 0.9, 0.99, 1.0],
                np.nanquantile(flat, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]),
            )
        }
    regime_probs = artifacts.get("covariance_regime_probs")
    if regime_probs is not None:
        regime_probs64 = np.asarray(regime_probs, dtype=np.float64)
        out["covariance_regime_prob_mean"] = {
            f"pi_{i + 1}": float(v) for i, v in enumerate(np.nanmean(regime_probs64, axis=0))
        }
        out["covariance_regime_prob_std"] = {
            f"pi_{i + 1}": float(v) for i, v in enumerate(np.nanstd(regime_probs64, axis=0))
        }
    regime_sigma = artifacts.get("covariance_regime_Sigma")
    if regime_sigma is not None:
        traces = np.trace(np.asarray(regime_sigma, dtype=np.float64), axis1=1, axis2=2)
        out["covariance_regime_trace"] = {
            f"Sigma_{i + 1}": float(v) for i, v in enumerate(traces)
        }
    proxy = artifacts.get("covariance_proxy")
    ratio_mean = artifacts.get("projected_sigma_ratio_mean")
    if proxy is not None and ratio_mean is not None:
        out["proxy_ratio_correlation"] = corrcoef_finite(
            np.asarray(proxy, dtype=np.float64),
            np.asarray(ratio_mean, dtype=np.float64),
        )
    covariance_scale = artifacts.get("covariance_scale")
    if covariance_scale is not None:
        scale = np.asarray(covariance_scale, dtype=np.float64).reshape(-1)
        out["covariance_scale_mean"] = float(np.nanmean(scale))
        out["covariance_scale_std"] = float(np.nanstd(scale))
        out["covariance_scale_quantiles"] = {
            str(q): float(v)
            for q, v in zip(
                [0.0, 0.1, 0.5, 0.9, 0.99, 1.0],
                np.nanquantile(scale, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]),
            )
        }
    block_sigma = artifacts.get("covariance_block_Sigma")
    if block_sigma is not None:
        blocks = np.asarray(block_sigma, dtype=np.float64)
        traces = np.trace(blocks, axis1=-2, axis2=-1)
        out["covariance_block_trace_mean"] = {
            "u": float(np.nanmean(traces[:, 0])),
            "v": float(np.nanmean(traces[:, 1])),
        }
        out["covariance_block_offdiag_mean"] = {
            "u_xy": float(np.nanmean(blocks[:, 0, 0, 1])),
            "v_xy": float(np.nanmean(blocks[:, 1, 0, 1])),
        }
    return out


def advection_validation_summary(
    Au: np.ndarray,
    Av: np.ndarray,
    nwp_displacement: np.ndarray,
    optical_flow: np.ndarray,
) -> dict[str, Any]:
    optical_u = optical_flow[:, :2] if optical_flow.shape[0] == Au.shape[0] else np.empty((0, 2), dtype=np.float32)
    optical_v = optical_flow[:, 2:] if optical_flow.shape[0] == Av.shape[0] else np.empty((0, 2), dtype=np.float32)
    nwp_speed = vector_norm(nwp_displacement)
    summary: dict[str, Any] = {
        "Au_speed_vs_nwp_speed_corr": corrcoef_finite(vector_norm(Au), nwp_speed),
        "Av_speed_vs_nwp_speed_corr": corrcoef_finite(vector_norm(Av), nwp_speed),
        "Au_angle_minus_nwp_mean_deg": float(np.nanmean(np.abs(angle_diff_degrees(Au, nwp_displacement)))),
        "Av_angle_minus_nwp_mean_deg": float(np.nanmean(np.abs(angle_diff_degrees(Av, nwp_displacement)))),
    }
    if optical_u.shape[0] == Au.shape[0]:
        summary.update(
            {
                "Au_speed_vs_optical_flow_u_corr": corrcoef_finite(vector_norm(Au), vector_norm(optical_u)),
                "Av_speed_vs_optical_flow_v_corr": corrcoef_finite(vector_norm(Av), vector_norm(optical_v)),
                "Au_angle_minus_optical_flow_u_mean_deg": float(np.nanmean(np.abs(angle_diff_degrees(Au, optical_u)))),
                "Av_angle_minus_optical_flow_v_mean_deg": float(np.nanmean(np.abs(angle_diff_degrees(Av, optical_v)))),
            }
        )
    return summary


def transition_summary(M: np.ndarray, M_base: np.ndarray, ell: np.ndarray) -> dict[str, Any]:
    M_mean = np.nanmean(M, axis=0)
    row_sums = np.nansum(M, axis=2)
    eye = np.eye(M.shape[1], dtype=bool)
    diag_values = np.diagonal(M, axis1=1, axis2=2)
    offdiag_values = M[:, ~eye]
    same_site_mask = np.zeros((M.shape[1], M.shape[2]), dtype=bool)
    n = M.shape[1] // 2
    for row in range(M.shape[1]):
        site = row % n
        same_site_mask[row, site] = True
        same_site_mask[row, site + n] = True
    same_site_mass = M[:, same_site_mask].reshape(M.shape[0], M.shape[1], 2).sum(axis=2)
    block_mass = np.zeros((M.shape[0], 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            block = M[:, i * n : (i + 1) * n, j * n : (j + 1) * n]
            block_mass[:, i, j] = np.nanmean(np.nansum(block, axis=2), axis=1)
    return {
        "row_sum_mean": float(np.nanmean(row_sums)),
        "row_sum_min": float(np.nanmin(row_sums)),
        "row_sum_max": float(np.nanmax(row_sums)),
        "transition_mean_min": float(np.nanmin(M_mean)),
        "transition_mean_max": float(np.nanmax(M_mean)),
        "diagonal_mass_mean": float(np.nanmean(diag_values)),
        "diagonal_mass_min": float(np.nanmin(diag_values)),
        "offdiagonal_mass_mean": float(np.nanmean(offdiag_values)),
        "same_site_mass_mean": float(np.nanmean(same_site_mass)),
        "temporal_std_mean": float(np.nanmean(np.nanstd(M, axis=0))),
        "consecutive_change_mean_abs": float(np.nanmean(np.abs(np.diff(M, axis=0)))) if M.shape[0] > 1 else 0.0,
        "block_mass_mean": {
            name: float(value) for name, value in zip(ALPHA_NAMES, np.nanmean(block_mass.reshape(block_mass.shape[0], 4), axis=0))
        },
        "ell": ell.astype(float).tolist(),
        "M_base_diff_mean_abs": float(np.nanmean(np.abs(M - M_base))),
    }


def evaluate(
    model: torch.nn.Module,
    data: dict[str, Any],
    device: torch.device,
    eval_window_size: int,
    eval_stride: int,
    forecast_horizon: int,
    dt_seconds: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    coords = torch.from_numpy(data["coords"]).to(device)
    outputs = collect_model_outputs(
        model=model,
        data=data,
        coords=coords,
        device=device,
        eval_window_size=eval_window_size,
        eval_stride=eval_stride,
    )

    z_model = torch.from_numpy(data["Z"].astype(np.float32, copy=False)).to(device)
    M_torch = torch.from_numpy(outputs["M"]).to(device)
    with torch.no_grad():
        kf = model.dstm.kalman_filter(
            z=z_model,
            M_seq=M_torch,
            reduction="sum",
            return_history=True,
        )
        one_step_model = kf["pred_means"].detach().cpu().numpy().astype(np.float32, copy=False)
        multi_model = multi_horizon_forecasts(
            model=model,
            filter_means=kf["filter_means"],
            filter_covs=kf["filter_covs"],
            M_seq=M_torch,
            max_horizon=forecast_horizon,
        ).detach().cpu().numpy().astype(np.float32, copy=False)

    target_np = data.get("Y", data["Z"]).astype(np.float32, copy=False)
    nwp_baseline_np = data.get("nwp_baseline", np.full_like(target_np, np.nan)).astype(np.float32, copy=False)
    n_sites = nwp_baseline_np.shape[1] // 2
    nwp_u = np.nanmean(nwp_baseline_np[:, :n_sites], axis=1)
    nwp_v = np.nanmean(nwp_baseline_np[:, n_sites:], axis=1)
    nwp_displacement_np = np.stack([nwp_u, nwp_v], axis=-1).astype(np.float32) * float(dt_seconds) / 1000.0
    training_advection_target_np = data.get("V_star")
    if training_advection_target_np is None:
        training_advection_target_np = np.full_like(outputs["mu"], np.nan, dtype=np.float32)
    else:
        training_advection_target_np = np.asarray(training_advection_target_np, dtype=np.float32)
    optical_flow_np = data.get("V_optical_star")
    if optical_flow_np is None:
        optical_flow_np = training_advection_target_np
    else:
        optical_flow_np = np.asarray(optical_flow_np, dtype=np.float32)
    station_optical_flow_np = data.get("V_station_optical_star")
    if station_optical_flow_np is None:
        station_optical_flow_np = np.full_like(outputs["mu"], np.nan, dtype=np.float32)
    else:
        station_optical_flow_np = np.asarray(station_optical_flow_np, dtype=np.float32)
    nwp_advection_target_np = data.get("V_nwp_star")
    if nwp_advection_target_np is None:
        nwp_advection_target_np = np.full_like(outputs["mu"], np.nan, dtype=np.float32)
    else:
        nwp_advection_target_np = np.asarray(nwp_advection_target_np, dtype=np.float32)
    advection_anchor_np = data.get("A_anchor")
    if advection_anchor_np is None:
        advection_anchor_np = np.full_like(outputs["mu"], np.nan, dtype=np.float32)
    else:
        advection_anchor_np = np.asarray(advection_anchor_np, dtype=np.float32)
    one_step_pred_np = measurement_from_model_target(one_step_model, data)
    multi_pred_np = measurement_from_model_target(multi_model, data)

    target = torch.from_numpy(target_np).to(device)
    one_step_pred = torch.from_numpy(one_step_pred_np).to(device)
    nwp_baseline = torch.from_numpy(nwp_baseline_np).to(device)
    persistence = persistence_forecast(target, horizon=1)
    residual_persistence = residual_persistence_forecast(target, nwp_baseline, horizon=1)

    model_metrics = finite_metrics(one_step_pred, target, skip_initial=1)
    persistence_metrics = finite_metrics(persistence, target, skip_initial=1)
    nwp_metrics = finite_metrics(nwp_baseline, target, skip_initial=1)
    residual_persistence_metrics = finite_metrics(residual_persistence, target, skip_initial=1)

    rmse_curve = np.full((forecast_horizon, len(CURVE_COLUMNS)), np.nan, dtype=np.float32)
    mae_curve = np.full((forecast_horizon, len(CURVE_COLUMNS)), np.nan, dtype=np.float32)
    multi_step: dict[str, Any] = {}
    multi_persistence = []
    multi_nwp = []
    multi_residual_persistence = []
    for horizon in range(1, forecast_horizon + 1):
        pred_h = torch.from_numpy(multi_pred_np[horizon - 1]).to(device)
        persistence_h = persistence_forecast(target, horizon=horizon)
        nwp_h = nwp_baseline
        residual_h = residual_persistence_forecast(target, nwp_baseline, horizon=horizon)

        metrics_model = finite_metrics(pred_h, target, skip_initial=horizon)
        metrics_persistence = finite_metrics(persistence_h, target, skip_initial=horizon)
        metrics_nwp = finite_metrics(nwp_h, target, skip_initial=horizon)
        metrics_residual = finite_metrics(residual_h, target, skip_initial=horizon)

        rmse_curve[horizon - 1] = [
            metrics_model["rmse"],
            metrics_persistence["rmse"],
            metrics_nwp["rmse"],
            metrics_residual["rmse"],
        ]
        mae_curve[horizon - 1] = [
            metrics_model["mae"],
            metrics_persistence["mae"],
            metrics_nwp["mae"],
            metrics_residual["mae"],
        ]
        multi_step[str(horizon)] = {
            "model": metrics_model,
            "persistence_baseline": metrics_persistence,
            "nwp_baseline": metrics_nwp,
            "nwp_residual_persistence_baseline": metrics_residual,
            "model_vs_persistence_improvement": improvement(metrics_model, metrics_persistence),
            "model_vs_nwp_improvement": improvement(metrics_model, metrics_nwp),
            "model_vs_nwp_residual_persistence_improvement": improvement(metrics_model, metrics_residual),
        }
        multi_persistence.append(persistence_h.detach().cpu().numpy().astype(np.float32, copy=False))
        multi_nwp.append(nwp_h.detach().cpu().numpy().astype(np.float32, copy=False))
        multi_residual_persistence.append(residual_h.detach().cpu().numpy().astype(np.float32, copy=False))

    with torch.no_grad():
        ell = model.kernel.get_ell().detach().cpu().numpy().astype(np.float32, copy=False)
        gamma = np.asarray(float(model.kernel.gamma_value(device, torch.float32).detach().cpu()), dtype=np.float32)
        kernel_sigma_scale = np.asarray(
            float(model.kernel.sigma_scale_value(device, torch.float32).detach().cpu()),
            dtype=np.float32,
        )
        kernel_dt = np.asarray(float(model.kernel.dt), dtype=np.float32)
        Q = model.dstm.process_covariance().detach().cpu().numpy().astype(np.float32, copy=False)
        R = model.dstm.observation_covariance().detach().cpu().numpy().astype(np.float32, copy=False)

    artifacts = {
        "target": target_np,
        "model_target": data["Z"].astype(np.float32, copy=False),
        "nwp_baseline": nwp_baseline_np,
        "prediction": one_step_pred_np,
        "model_target_prediction": one_step_model,
        "persistence_prediction": persistence.detach().cpu().numpy().astype(np.float32, copy=False),
        "nwp_baseline_prediction": nwp_baseline_np,
        "nwp_residual_persistence_prediction": residual_persistence.detach().cpu().numpy().astype(np.float32, copy=False),
        "multi_step_prediction": multi_pred_np,
        "multi_step_model_target_prediction": multi_model,
        "multi_step_persistence_prediction": np.stack(multi_persistence, axis=0),
        "multi_step_nwp_baseline_prediction": np.stack(multi_nwp, axis=0),
        "multi_step_nwp_residual_persistence_prediction": np.stack(multi_residual_persistence, axis=0),
        "multi_step_rmse_curve": rmse_curve,
        "multi_step_mae_curve": mae_curve,
        "multi_step_curve_columns": np.asarray(CURVE_COLUMNS),
        "horizons": np.arange(1, forecast_horizon + 1, dtype=np.int32),
        "transition_matrices": outputs["M"],
        "transition_base_matrices": outputs["M_base"],
        "mu": outputs["mu"],
        "Au": outputs["mu"][:, :2],
        "Av": outputs["mu"][:, 2:],
        "Sigma": outputs["Sigma"],
        "Sigma_diag": np.diagonal(outputs["Sigma"], axis1=1, axis2=2),
        "alpha": outputs["alpha"],
        "advection_training_target": training_advection_target_np,
        "nwp_wind_displacement": nwp_displacement_np,
        "nwp_advection_target": nwp_advection_target_np,
        "optical_flow_advection": optical_flow_np,
        "station_optical_flow_advection": station_optical_flow_np,
        "advection_anchor": advection_anchor_np,
        "ell": ell,
        "gamma": gamma,
        "kernel_sigma_scale": kernel_sigma_scale,
        "kernel_dt": kernel_dt,
        "Q": Q,
        "R": R,
        "coords": data["coords"].astype(np.float32, copy=False),
        "baseline_grid_indices": data.get("baseline_grid_indices", np.empty((0, 2), dtype=np.int64)),
    }
    for key in ("covariance_regime_logits", "covariance_regime_probs", "covariance_regime_Sigma"):
        if key in outputs:
            artifacts[key] = outputs[key]
    if "covariance_scale" in outputs:
        artifacts["covariance_scale"] = outputs["covariance_scale"]
    if "covariance_block_Sigma" in outputs:
        artifacts["covariance_block_Sigma"] = outputs["covariance_block_Sigma"]
    if "covariance_cross_Sigma" in outputs:
        artifacts["covariance_cross_Sigma"] = outputs["covariance_cross_Sigma"]
    if "covariance_cross_correlation" in outputs:
        artifacts["covariance_cross_correlation"] = outputs["covariance_cross_correlation"]
    if data.get("covariance_proxy") is not None:
        artifacts["covariance_proxy"] = np.asarray(data["covariance_proxy"], dtype=np.float32)
    artifacts.update(
        covariance_kernel_diagnostics(
            Sigma=outputs["Sigma"],
            ell=ell,
            gamma=float(gamma),
            sigma_scale=float(kernel_sigma_scale),
        )
    )

    results = {
        "kalman_nll_per_observation": float(kf["nll_sum"].detach().cpu())
        / max(float(kf["obs_count"].detach().cpu()), 1.0),
        "model": model_metrics,
        "persistence_baseline": persistence_metrics,
        "nwp_baseline": nwp_metrics,
        "nwp_residual_persistence_baseline": residual_persistence_metrics,
        "model_vs_persistence_improvement": improvement(model_metrics, persistence_metrics),
        "model_vs_nwp_improvement": improvement(model_metrics, nwp_metrics),
        "model_vs_nwp_residual_persistence_improvement": improvement(model_metrics, residual_persistence_metrics),
        "target_mode": data.get("target_mode", "measurement"),
        "forecast_horizon": forecast_horizon,
        "multi_step": multi_step,
        "advection": advection_summary(outputs["mu"], outputs["Sigma"], outputs["alpha"]),
        "advection_validation": advection_validation_summary(
            outputs["mu"][:, :2],
            outputs["mu"][:, 2:],
            nwp_displacement_np,
            optical_flow_np,
        ),
        "transition": transition_summary(outputs["M"], outputs["M_base"], ell),
        "covariance_diagnostics": covariance_diagnostics_summary(artifacts),
        "eval_window_size": eval_window_size,
        "eval_stride": eval_stride,
    }
    return results, artifacts


def save_artifact_arrays(output_dir: Path, artifacts: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "forecasts.npz",
        target=artifacts["target"],
        model_target=artifacts["model_target"],
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
        multi_step_nwp_residual_persistence_prediction=artifacts["multi_step_nwp_residual_persistence_prediction"],
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
        M_base=artifacts["transition_base_matrices"],
        M_base_mean=np.nanmean(artifacts["transition_base_matrices"], axis=0),
        M_row_sums=np.nansum(artifacts["transition_matrices"], axis=2),
        state_names=np.asarray(STATE_NAMES),
    )
    advection_payload = {
        "mu": artifacts["mu"],
        "Au": artifacts["Au"],
        "Av": artifacts["Av"],
        "Sigma": artifacts["Sigma"],
        "Sigma_diag": artifacts["Sigma_diag"],
        "alpha": artifacts["alpha"],
        "advection_training_target": artifacts["advection_training_target"],
        "advection_anchor": artifacts["advection_anchor"],
        "nwp_wind_displacement": artifacts["nwp_wind_displacement"],
        "nwp_advection_target": artifacts["nwp_advection_target"],
        "optical_flow_advection": artifacts["optical_flow_advection"],
        "station_optical_flow_advection": artifacts["station_optical_flow_advection"],
        "ell": artifacts["ell"],
        "gamma": artifacts["gamma"],
        "kernel_sigma_scale": artifacts["kernel_sigma_scale"],
        "kernel_dt": artifacts["kernel_dt"],
        "Q": artifacts["Q"],
        "R": artifacts["R"],
        "coords": artifacts["coords"],
        "baseline_grid_indices": artifacts["baseline_grid_indices"],
        "advection_names": np.asarray(ADV_NAMES),
        "sigma_diag_names": np.asarray(SIGMA_DIAG_NAMES),
        "alpha_names": np.asarray(ALPHA_NAMES),
        "state_names": np.asarray(STATE_NAMES),
    }
    for key in (
        "covariance_regime_logits",
        "covariance_regime_probs",
        "covariance_regime_Sigma",
        "covariance_scale",
        "covariance_block_Sigma",
        "covariance_cross_Sigma",
        "covariance_cross_correlation",
        "covariance_proxy",
        "projected_sigma",
        "effective_diffusion_D",
        "projected_sigma_trace",
        "effective_diffusion_trace",
        "base_diffusion_trace",
        "projected_sigma_ratio",
        "projected_sigma_ratio_mean",
    ):
        if key in artifacts:
            advection_payload[key] = artifacts[key]
    np.savez_compressed(output_dir / "advection_parameters.npz", **advection_payload)

    sigma = artifacts["Sigma"]
    cov_u_xy = sigma[:, 0, 1]
    cov_v_xy = sigma[:, 2, 3]
    cov_uv_xx = sigma[:, 0, 2]
    cov_uv_xy = sigma[:, 0, 3]
    cov_uv_yx = sigma[:, 1, 2]
    cov_uv_yy = sigma[:, 1, 3]
    corr_u_xy = cov_u_xy / np.sqrt(np.maximum(sigma[:, 0, 0] * sigma[:, 1, 1], 1.0e-12))
    corr_v_xy = cov_v_xy / np.sqrt(np.maximum(sigma[:, 2, 2] * sigma[:, 3, 3], 1.0e-12))
    corr_uv_xx = cov_uv_xx / np.sqrt(np.maximum(sigma[:, 0, 0] * sigma[:, 2, 2], 1.0e-12))
    corr_uv_xy = cov_uv_xy / np.sqrt(np.maximum(sigma[:, 0, 0] * sigma[:, 3, 3], 1.0e-12))
    corr_uv_yx = cov_uv_yx / np.sqrt(np.maximum(sigma[:, 1, 1] * sigma[:, 2, 2], 1.0e-12))
    corr_uv_yy = cov_uv_yy / np.sqrt(np.maximum(sigma[:, 1, 1] * sigma[:, 3, 3], 1.0e-12))
    param_columns = [
        np.arange(artifacts["mu"].shape[0])[:, None],
        artifacts["mu"],
        artifacts["Sigma_diag"],
        np.column_stack(
            [
                cov_u_xy,
                cov_v_xy,
                cov_uv_xx,
                cov_uv_xy,
                cov_uv_yx,
                cov_uv_yy,
                corr_u_xy,
                corr_v_xy,
                corr_uv_xx,
                corr_uv_xy,
                corr_uv_yx,
                corr_uv_yy,
            ]
        ),
        artifacts["alpha"].reshape(artifacts["alpha"].shape[0], 4),
    ]
    header_names = [
        "time_index",
        *ADV_NAMES,
        *SIGMA_DIAG_NAMES,
        "cov_u_xy",
        "cov_v_xy",
        "cov_Au_x_Av_x",
        "cov_Au_x_Av_y",
        "cov_Au_y_Av_x",
        "cov_Au_y_Av_y",
        "corr_u_xy",
        "corr_v_xy",
        "corr_Au_x_Av_x",
        "corr_Au_x_Av_y",
        "corr_Au_y_Av_x",
        "corr_Au_y_Av_y",
        *ALPHA_NAMES,
    ]
    if "projected_sigma_ratio_mean" in artifacts:
        param_columns.append(artifacts["projected_sigma_ratio_mean"][:, None])
        header_names.append("projected_sigma_ratio_mean")
    if "projected_sigma_ratio" in artifacts:
        ratio = artifacts["projected_sigma_ratio"].reshape(artifacts["projected_sigma_ratio"].shape[0], 4)
        param_columns.append(ratio)
        header_names.extend(["ratio_uu", "ratio_uv", "ratio_vu", "ratio_vv"])
    if "covariance_scale" in artifacts:
        param_columns.append(artifacts["covariance_scale"].reshape(-1, 1))
        header_names.append("covariance_scale")
    if "covariance_proxy" in artifacts:
        param_columns.append(artifacts["covariance_proxy"].reshape(-1, 1))
        header_names.append("covariance_proxy")
    if "covariance_regime_probs" in artifacts:
        probs = artifacts["covariance_regime_probs"]
        param_columns.append(probs)
        header_names.extend([f"pi_{idx + 1}" for idx in range(probs.shape[1])])
    param_csv = np.column_stack(param_columns)
    header = ",".join(header_names)
    np.savetxt(output_dir / "time_parameters.csv", param_csv, delimiter=",", header=header, comments="")


def line_plot(path: Path, values: np.ndarray, labels: list[str], title: str, ylabel: str, max_points: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = downsample_indices(values.shape[0], max_points)
    fig, ax = plt.subplots(figsize=(12, 4.8))
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


def heatmap(path: Path, matrix: np.ndarray, title: str, xlabels: list[str], ylabels: list[str], center_zero: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    kwargs: dict[str, Any] = {}
    cmap = "coolwarm" if center_zero else "viridis"
    if center_zero:
        lim = max(float(np.nanmax(np.abs(matrix))), 1.0e-8)
        kwargs = {"vmin": -lim, "vmax": lim}
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


def horizon_plot(path: Path, horizons: np.ndarray, values: np.ndarray, labels: np.ndarray, title: str, ylabel: str) -> None:
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
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def advection_vector_plot(
    path: Path,
    Au: np.ndarray,
    Av: np.ndarray,
    max_points: int,
    *,
    kernel_dt: float = 1.0,
) -> None:
    """Visualize the learned U-field and V-field advection vectors.

    Each point is one time step's learned two-dimensional displacement. The
    arrows show the temporal mean of the two component-field advections.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = downsample_indices(Au.shape[0], max_points)
    au = Au[idx] * float(kernel_dt)
    av = Av[idx] * float(kernel_dt)
    values = np.concatenate([au, av], axis=0)
    lim = max(float(np.nanmax(np.abs(values))) * 1.1, 1.0e-6)

    fig, ax = plt.subplots(figsize=(7, 7))
    color = np.linspace(0.0, 1.0, idx.size)
    sc_u = ax.scatter(au[:, 0], au[:, 1], c=color, cmap="Blues", s=16, alpha=0.55, label="A_u(t)")
    ax.scatter(av[:, 0], av[:, 1], c=color, cmap="Oranges", s=16, alpha=0.55, label="A_v(t)")
    mean_u = np.nanmean(Au, axis=0) * float(kernel_dt)
    mean_v = np.nanmean(Av, axis=0) * float(kernel_dt)
    for vector, label, color_name in [(mean_u, "mean A_u", "tab:blue"), (mean_v, "mean A_v", "tab:orange")]:
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color_name,
            width=0.012 * lim,
            head_width=0.07 * lim,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.scatter([vector[0]], [vector[1]], color=color_name, s=50, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Effective component kernel shifts")
    ax.set_xlabel("x kernel shift (km)")
    ax.set_ylabel("y kernel shift (km)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(sc_u, ax=ax, shrink=0.75)
    cbar.set_label("normalized time index")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def advection_vs_nwp_plot(
    path: Path,
    Au: np.ndarray,
    Av: np.ndarray,
    nwp_displacement: np.ndarray,
    max_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = downsample_indices(Au.shape[0], max_points)
    nwp = nwp_displacement[idx]
    au = Au[idx]
    av = Av[idx]
    nwp_speed = vector_norm(nwp)
    au_speed = vector_norm(au)
    av_speed = vector_norm(av)
    au_angle = angle_diff_degrees(au, nwp)
    av_angle = angle_diff_degrees(av, nwp)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].scatter(nwp_speed, au_speed, s=12, alpha=0.45, color="tab:blue", label="A_u")
    axes[0, 0].scatter(nwp_speed, av_speed, s=12, alpha=0.45, color="tab:orange", label="A_v")
    axes[0, 0].set_xlabel("|NWP wind displacement|")
    axes[0, 0].set_ylabel("|learned A|")
    axes[0, 0].set_title("Speed alignment")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(au_angle[np.isfinite(au_angle)], bins=40, alpha=0.55, color="tab:blue", label="A_u - NWP")
    axes[0, 1].hist(av_angle[np.isfinite(av_angle)], bins=40, alpha=0.55, color="tab:orange", label="A_v - NWP")
    axes[0, 1].set_xlabel("angle difference (degrees)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("Direction alignment")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].scatter(nwp[:, 0], au[:, 0], s=12, alpha=0.45, color="tab:blue", label="A_u,x")
    axes[1, 0].scatter(nwp[:, 0], av[:, 0], s=12, alpha=0.45, color="tab:orange", label="A_v,x")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1, 0].axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1, 0].set_xlabel("NWP x displacement")
    axes[1, 0].set_ylabel("learned x displacement")
    axes[1, 0].set_title("x-component comparison")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].scatter(nwp[:, 1], au[:, 1], s=12, alpha=0.45, color="tab:blue", label="A_u,y")
    axes[1, 1].scatter(nwp[:, 1], av[:, 1], s=12, alpha=0.45, color="tab:orange", label="A_v,y")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1, 1].set_xlabel("NWP y displacement")
    axes[1, 1].set_ylabel("learned y displacement")
    axes[1, 1].set_title("y-component comparison")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Learned advection vs NWP 140m wind displacement")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def advection_vs_optical_flow_plot(
    path: Path,
    Au: np.ndarray,
    Av: np.ndarray,
    optical_flow: np.ndarray,
    max_points: int,
    label_name: str = "optical-flow",
) -> None:
    if optical_flow.shape[0] != Au.shape[0] or optical_flow.shape[1] < 4:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = downsample_indices(Au.shape[0], max_points)
    au = Au[idx]
    av = Av[idx]
    of_u = optical_flow[idx, :2]
    of_v = optical_flow[idx, 2:4]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].scatter(vector_norm(of_u), vector_norm(au), s=12, alpha=0.45, color="tab:blue")
    axes[0, 0].set_xlabel(f"|{label_name} U target|")
    axes[0, 0].set_ylabel("|A_u|")
    axes[0, 0].set_title("U-field speed")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].scatter(vector_norm(of_v), vector_norm(av), s=12, alpha=0.45, color="tab:orange")
    axes[0, 1].set_xlabel(f"|{label_name} V target|")
    axes[0, 1].set_ylabel("|A_v|")
    axes[0, 1].set_title("V-field speed")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].hist(angle_diff_degrees(au, of_u), bins=40, alpha=0.65, color="tab:blue")
    axes[1, 0].set_xlabel(f"A_u - {label_name} U angle (degrees)")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].set_title("U-field direction")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].hist(angle_diff_degrees(av, of_v), bins=40, alpha=0.65, color="tab:orange")
    axes[1, 1].set_xlabel(f"A_v - {label_name} V angle (degrees)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("V-field direction")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"Learned advection vs component {label_name} labels")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def kernel_source_center_map(
    path: Path,
    coords: np.ndarray,
    Au: np.ndarray,
    Av: np.ndarray,
    *,
    kernel_dt: float = 1.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_au = np.nanmean(Au, axis=0) * float(kernel_dt)
    mean_av = np.nanmean(Av, axis=0) * float(kernel_dt)
    all_points = np.concatenate([coords, coords - mean_au[None, :], coords - mean_av[None, :]], axis=0)
    pad = max(float(np.nanmax(np.ptp(all_points, axis=0))) * 0.15, 0.5)
    xmin, ymin = np.nanmin(all_points, axis=0) - pad
    xmax, ymax = np.nanmax(all_points, axis=0) + pad

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(coords[:, 0], coords[:, 1], color="black", s=42, label="target sites")
    for idx, site in enumerate(coords):
        src_u = site - mean_au
        src_v = site - mean_av
        ax.scatter([src_u[0]], [src_u[1]], color="tab:blue", s=28, alpha=0.75)
        ax.scatter([src_v[0]], [src_v[1]], color="tab:orange", s=28, alpha=0.75)
        ax.arrow(
            src_u[0],
            src_u[1],
            mean_au[0],
            mean_au[1],
            color="tab:blue",
            width=0.008,
            head_width=0.055,
            length_includes_head=True,
            alpha=0.75,
        )
        ax.arrow(
            src_v[0],
            src_v[1],
            mean_av[0],
            mean_av[1],
            color="tab:orange",
            width=0.008,
            head_width=0.055,
            length_includes_head=True,
            alpha=0.75,
        )
        ax.text(site[0], site[1], f" s{idx + 1}", fontsize=8, va="bottom")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Kernel source-center map: source center -> target site")
    ax.set_xlabel("x coordinate (km)")
    ax.set_ylabel("y coordinate (km)")
    ax.grid(True, alpha=0.25)
    ax.plot([], [], color="tab:blue", label="U source center -> target")
    ax.plot([], [], color="tab:orange", label="V source center -> target")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def figure_to_gif_frame(fig: Any) -> Any:
    from PIL import Image

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)


def save_gif(path: Path, frames: list[Any], duration: int = 140) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def advection_vector_frame(
    au_t: np.ndarray,
    av_t: np.ndarray,
    title: str,
    axis_limit: float,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for vector, label, color in [
        (au_t, "A_u(t)", "tab:blue"),
        (av_t, "A_v(t)", "tab:orange"),
    ]:
        ax.arrow(
            0.0,
            0.0,
            float(vector[0]),
            float(vector[1]),
            color=color,
            width=0.012 * axis_limit,
            head_width=0.07 * axis_limit,
            length_includes_head=True,
            alpha=0.9,
        )
        ax.scatter([vector[0]], [vector[1]], color=color, s=42, label=label)
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
    frame = figure_to_gif_frame(fig)
    plt.close(fig)
    return frame


def matrix_frame(
    matrix: np.ndarray,
    title: str,
    xlabels: list[str],
    ylabels: list[str],
    vmin: float,
    vmax: float,
    cmap: str,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ylabels, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.78)
    fig.tight_layout()
    frame = figure_to_gif_frame(fig)
    plt.close(fig)
    return frame


def save_time_varying_gifs(output_dir: Path, artifacts: dict[str, np.ndarray], max_gif_frames: int) -> None:
    if max_gif_frames <= 0:
        return
    plots_dir = output_dir / "plots"
    idx = downsample_indices(artifacts["mu"].shape[0], max_gif_frames)

    adv_values = np.concatenate(
        [
            artifacts["Au"],
            artifacts["Av"],
        ],
        axis=0,
    )
    axis_limit = max(float(np.nanmax(np.abs(adv_values))) * 1.15, 1.0e-6)
    frames = [
        advection_vector_frame(
            artifacts["Au"][t],
            artifacts["Av"][t],
            f"Component advection vectors, t={int(t)}",
            axis_limit,
        )
        for t in idx
    ]
    save_gif(plots_dir / "advection_component_vectors.gif", frames)

    sigma = artifacts["Sigma"]
    sigma_limit = max(float(np.nanmax(np.abs(sigma))), 1.0e-8)
    frames = [
        matrix_frame(
            sigma[t],
            f"Advection covariance Sigma_A, t={int(t)}",
            ADV_NAMES,
            ADV_NAMES,
            -sigma_limit,
            sigma_limit,
            "coolwarm",
        )
        for t in idx
    ]
    save_gif(plots_dir / "advection_sigma.gif", frames)

    alpha = artifacts["alpha"]
    frames = [
        matrix_frame(
            alpha[t],
            f"Component mixing alpha, t={int(t)}",
            ["U source", "V source"],
            ["U target", "V target"],
            0.0,
            max(float(np.nanmax(alpha)), 1.0e-8),
            "viridis",
        )
        for t in idx
    ]
    save_gif(plots_dir / "mixing_alpha.gif", frames)

    transition = artifacts["transition_matrices"]
    signed_transition = bool(np.nanmin(transition) < 0.0)
    if signed_transition:
        limit = max(float(np.nanmax(np.abs(transition))), 1.0e-8)
        vmin, vmax, cmap = -limit, limit, "coolwarm"
    else:
        vmin, vmax, cmap = 0.0, max(float(np.nanmax(transition)), 1.0e-8), "viridis"
    frames = [
        matrix_frame(
            transition[t],
            f"Transition matrix M_t, t={int(t)}",
            STATE_NAMES,
            STATE_NAMES,
            vmin,
            vmax,
            cmap,
        )
        for t in idx
    ]
    save_gif(plots_dir / "transition_matrix.gif", frames)


def save_plots(output_dir: Path, artifacts: dict[str, np.ndarray], max_points: int, max_gif_frames: int) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as exc:
        (plots_dir / "PLOTS_NOT_CREATED.txt").write_text(
            f"Install matplotlib and pillow to create plots: {exc}\n",
            encoding="utf-8",
        )
        return

    line_plot(plots_dir / "advection_mu.png", artifacts["mu"], ADV_NAMES, "Advection mean", "km / step", max_points)
    line_plot(plots_dir / "advection_sigma_diag.png", artifacts["Sigma_diag"], SIGMA_DIAG_NAMES, "Advection covariance diagonal", "variance", max_points)
    if "covariance_regime_probs" in artifacts:
        probs = artifacts["covariance_regime_probs"]
        line_plot(
            plots_dir / "covariance_regime_probs.png",
            probs,
            [f"pi_{idx + 1}" for idx in range(probs.shape[1])],
            "Covariance regime probabilities",
            "probability",
            max_points,
        )
    if "projected_sigma_ratio" in artifacts:
        ratio = artifacts["projected_sigma_ratio"].reshape(artifacts["projected_sigma_ratio"].shape[0], 4)
        values = np.column_stack([artifacts["projected_sigma_ratio_mean"], ratio])
        line_plot(
            plots_dir / "projected_sigma_ratio.png",
            values,
            ["mean", "uu", "uv", "vu", "vv"],
            "Projected covariance contribution ratio",
            r"trace(lambda * 2P Sigma P^T) / trace(ell^2 I)",
            max_points,
        )
    if "covariance_scale" in artifacts:
        line_plot(
            plots_dir / "covariance_scale.png",
            artifacts["covariance_scale"].reshape(-1, 1),
            ["covariance_scale"],
            "Dynamic covariance scale",
            "scale",
            max_points,
        )
    if "covariance_proxy" in artifacts:
        line_plot(
            plots_dir / "covariance_proxy.png",
            artifacts["covariance_proxy"].reshape(-1, 1),
            ["covariance_proxy"],
            "Covariance proxy",
            "proxy",
            max_points,
        )
    if "effective_diffusion_trace" in artifacts:
        diffusion_trace = artifacts["effective_diffusion_trace"].reshape(
            artifacts["effective_diffusion_trace"].shape[0],
            4,
        )
        line_plot(
            plots_dir / "effective_diffusion_trace.png",
            diffusion_trace,
            ["D_uu", "D_uv", "D_vu", "D_vv"],
            "Effective kernel diffusion trace",
            r"trace(D_t)",
            max_points,
        )
    if "effective_diffusion_D" in artifacts:
        mean_blocks = np.nanmean(artifacts["effective_diffusion_D"], axis=0)
        mean_D = np.block([[mean_blocks[0, 0], mean_blocks[0, 1]], [mean_blocks[1, 0], mean_blocks[1, 1]]])
        heatmap(
            plots_dir / "effective_diffusion_D_mean.png",
            mean_D,
            "Mean effective diffusion D",
            ["Ux src", "Uy src", "Vx src", "Vy src"],
            ["Ux tgt", "Uy tgt", "Vx tgt", "Vy tgt"],
            center_zero=False,
        )
    if "covariance_regime_Sigma" in artifacts:
        traces = np.trace(artifacts["covariance_regime_Sigma"], axis1=1, axis2=2).reshape(-1, 1)
        heatmap(
            plots_dir / "covariance_regime_trace.png",
            traces,
            "Covariance regime trace",
            ["trace"],
            [f"Sigma_{idx + 1}" for idx in range(traces.shape[0])],
        )
    line_plot(
        plots_dir / "advection_speed.png",
        np.column_stack(
            [
                np.linalg.norm(artifacts["Au"], axis=1),
                np.linalg.norm(artifacts["Av"], axis=1),
            ]
        ),
        ["|A_u|", "|A_v|"],
        "Component advection speeds",
        "km / step",
        max_points,
    )
    advection_vector_plot(
        plots_dir / "advection_component_vectors.png",
        artifacts["Au"],
        artifacts["Av"],
        max_points,
        kernel_dt=float(artifacts["kernel_dt"]),
    )
    advection_vs_nwp_plot(
        plots_dir / "advection_vs_nwp_wind.png",
        artifacts["Au"],
        artifacts["Av"],
        artifacts["nwp_wind_displacement"],
        max_points,
    )
    advection_vs_optical_flow_plot(
        plots_dir / "advection_vs_optical_flow.png",
        artifacts["Au"],
        artifacts["Av"],
        artifacts["optical_flow_advection"],
        max_points,
        label_name="optical-flow",
    )
    advection_vs_optical_flow_plot(
        plots_dir / "advection_vs_anchor.png",
        artifacts["Au"],
        artifacts["Av"],
        artifacts["advection_anchor"],
        max_points,
        label_name="anchor",
    )
    kernel_source_center_map(
        plots_dir / "kernel_source_center_map.png",
        artifacts["coords"],
        artifacts["Au"],
        artifacts["Av"],
        kernel_dt=float(artifacts["kernel_dt"]),
    )
    heatmap(
        plots_dir / "advection_sigma_mean.png",
        np.nanmean(artifacts["Sigma"], axis=0),
        "Mean advection covariance Sigma_A",
        ADV_NAMES,
        ADV_NAMES,
        center_zero=True,
    )
    line_plot(
        plots_dir / "mixing_alpha.png",
        artifacts["alpha"].reshape(artifacts["alpha"].shape[0], 4),
        ALPHA_NAMES,
        "Component mixing weights",
        "weight",
        max_points,
    )
    M_mean = np.nanmean(artifacts["transition_matrices"], axis=0)
    heatmap(plots_dir / "transition_matrix_mean.png", M_mean, "Mean transition matrix M", STATE_NAMES, STATE_NAMES)
    heatmap(
        plots_dir / "transition_matrix_mean_minus_identity.png",
        M_mean - np.eye(M_mean.shape[0], dtype=M_mean.dtype),
        "Mean transition matrix M - I",
        STATE_NAMES,
        STATE_NAMES,
        center_zero=True,
    )
    line_plot(
        plots_dir / "transition_row_sums.png",
        np.nansum(artifacts["transition_matrices"], axis=2),
        STATE_NAMES,
        "Transition matrix row sums",
        "row sum",
        max_points,
    )
    heatmap(
        plots_dir / "kernel_lengthscale_ell.png",
        artifacts["ell"],
        "Kernel lengthscales ell",
        ["U source", "V source"],
        ["U target", "V target"],
    )
    horizon_plot(
        plots_dir / "multi_step_rmse.png",
        artifacts["horizons"],
        artifacts["multi_step_rmse_curve"],
        artifacts["multi_step_curve_columns"],
        "Multi-step RMSE",
        "RMSE",
    )
    horizon_plot(
        plots_dir / "multi_step_mae.png",
        artifacts["horizons"],
        artifacts["multi_step_mae_curve"],
        artifacts["multi_step_curve_columns"],
        "Multi-step MAE",
        "MAE",
    )
    save_time_varying_gifs(output_dir, artifacts, max_gif_frames=max_gif_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VectorMIDE checkpoint.")
    parser.add_argument("--config", default="yml_files/VectorMIDE.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["offline", "online"], default="online")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-window-size", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--forecast-horizon", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-plots", action="store_true")
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
    missing, unexpected, skipped = load_compatible_state_dict(model, checkpoint)
    if missing:
        print(f"Initialized model parameters not found in checkpoint: {missing}")
    if unexpected:
        print(f"Ignored checkpoint parameters not used by this config: {unexpected}")
    if skipped:
        print(f"Skipped checkpoint parameters with incompatible shapes: {skipped}")

    eval_window_size = args.eval_window_size
    if eval_window_size is None:
        eval_window_size = int(
            min(
                model_config.get("window_size", config.get("window_size", 1008)),
                model_config.get("transformer_max_len", config.get("transformer_max_len", 4096)),
            )
        )
    eval_stride = int(args.eval_stride or eval_window_size)
    forecast_horizon = int(args.forecast_horizon or config.get("forecast_horizon", 1))
    data = load_vector_dataset(config, split=args.split, time_limit=args.limit)
    data.update(build_covariance_proxy_arrays(data, model_config))

    results, artifacts = evaluate(
        model=model,
        data=data,
        device=device,
        eval_window_size=eval_window_size,
        eval_stride=eval_stride,
        forecast_horizon=forecast_horizon,
        dt_seconds=float(config.get("dt_seconds", model_config.get("dt_seconds", 600.0))),
    )
    results["split"] = args.split
    results["checkpoint"] = str(ckpt_path)
    results["n_time"] = int(data["X"].shape[0])
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(ckpt_path, args.split)
    results["output_dir"] = str(output_dir)

    text = json.dumps(results, indent=2)
    print(text)
    save_json(output_dir / "results.json", results)
    if args.output:
        save_json(Path(args.output), results)
    save_artifact_arrays(output_dir, artifacts)
    if not args.no_plots:
        save_plots(
            output_dir,
            artifacts,
            max_points=int(args.max_plot_points),
            max_gif_frames=int(args.max_gif_frames),
        )
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
