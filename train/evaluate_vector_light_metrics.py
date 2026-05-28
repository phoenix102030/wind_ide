from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.vector_data_utils import load_vector_dataset
from train.evaluate_vector import (
    STATE_NAMES,
    load_config,
)
from train.train_vector_offline import build_model, print_device_info, resolve_device


CURVE_COLUMNS = ["model", "persistence", "nwp", "nwp_residual_persistence"]
STATION_NAMES = ["E05", "E06", "ASOW6"]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def target_mode_is_residual(data: dict[str, Any]) -> bool:
    return str(data.get("target_mode", "measurement")).lower() in {
        "residual_nwp",
        "nwp_residual",
        "residual",
    }


def model_target_scale(data: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    standardizer = data.get("z_standardizer")
    if standardizer is None:
        return torch.ones(len(STATE_NAMES), device=device, dtype=dtype)
    std = np.asarray(standardizer.std, dtype=np.float32).reshape(-1)
    return torch.from_numpy(std).to(device=device, dtype=dtype)


def model_target_mean_offset(data: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    standardizer = data.get("z_standardizer")
    if standardizer is None:
        return torch.zeros(len(STATE_NAMES), device=device, dtype=dtype)
    mean = np.asarray(standardizer.mean, dtype=np.float32).reshape(-1)
    return torch.from_numpy(mean).to(device=device, dtype=dtype)


def model_target_to_measurement(
    mean: torch.Tensor,
    cov: torch.Tensor,
    data: dict[str, Any],
    target_indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = model_target_scale(data, device, mean.dtype)
    offset = model_target_mean_offset(data, device, mean.dtype)
    mean = mean * scale + offset
    var = torch.diagonal(cov, dim1=-2, dim2=-1) * scale.pow(2)
    if target_mode_is_residual(data):
        nwp = torch.from_numpy(data["nwp_baseline"]).to(device=device, dtype=mean.dtype)
        mean = mean + nwp[target_indices]
    return mean, var.clamp_min(1.0e-12)


def model_cov_to_measurement_cov(
    cov: torch.Tensor,
    data: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scale = model_target_scale(data, device, dtype)
    cov = cov * scale.view(1, -1, 1) * scale.view(1, 1, -1)
    return 0.5 * (cov + cov.transpose(1, 2))


def gaussian_crps(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(var.clamp_min(1.0e-12))
    z = (target - mean) / std
    phi = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = std * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return torch.where(std > 1.0e-6, crps, torch.abs(target - mean))


def metric_block(
    pred: torch.Tensor,
    target: torch.Tensor,
    crps: torch.Tensor | None = None,
) -> dict[str, Any]:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    err = pred - target
    safe_err = torch.where(mask, err, torch.zeros_like(err))
    count_by_dim = mask.sum(dim=0).clamp_min(1)
    all_count = mask.sum().clamp_min(1)

    rmse_by_dim = torch.sqrt(safe_err.pow(2).sum(dim=0) / count_by_dim)
    mae_by_dim = safe_err.abs().sum(dim=0) / count_by_dim
    rmse = torch.sqrt(safe_err.pow(2).sum() / all_count)
    mae = safe_err.abs().sum() / all_count

    u_mask = mask[:, :3]
    v_mask = mask[:, 3:]
    u_err = safe_err[:, :3]
    v_err = safe_err[:, 3:]
    out: dict[str, Any] = {
        "rmse": float(rmse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "rmse_u": float(torch.sqrt(u_err.pow(2).sum() / u_mask.sum().clamp_min(1)).detach().cpu()),
        "mae_u": float((u_err.abs().sum() / u_mask.sum().clamp_min(1)).detach().cpu()),
        "rmse_v": float(torch.sqrt(v_err.pow(2).sum() / v_mask.sum().clamp_min(1)).detach().cpu()),
        "mae_v": float((v_err.abs().sum() / v_mask.sum().clamp_min(1)).detach().cpu()),
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
    if crps is not None:
        crps_mask = mask & torch.isfinite(crps)
        safe_crps = torch.where(crps_mask, crps, torch.zeros_like(crps))
        crps_count_by_dim = crps_mask.sum(dim=0).clamp_min(1)
        crps_count = crps_mask.sum().clamp_min(1)
        crps_by_dim = safe_crps.sum(dim=0) / crps_count_by_dim
        out["crps"] = float((safe_crps.sum() / crps_count).detach().cpu())
        out["crps_u"] = float((safe_crps[:, :3].sum() / crps_mask[:, :3].sum().clamp_min(1)).detach().cpu())
        out["crps_v"] = float((safe_crps[:, 3:].sum() / crps_mask[:, 3:].sum().clamp_min(1)).detach().cpu())
        out["crps_by_dim"] = {
            name: float(value.detach().cpu())
            for name, value in zip(STATE_NAMES, crps_by_dim)
        }
    return out


def metric_improvement(model_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "rmse_percent": 100.0
        * (baseline_metrics["rmse"] - model_metrics["rmse"])
        / max(float(baseline_metrics["rmse"]), 1.0e-12),
        "mae_percent": 100.0
        * (baseline_metrics["mae"] - model_metrics["mae"])
        / max(float(baseline_metrics["mae"]), 1.0e-12),
    }


def compute_transition_sequences(
    model: torch.nn.Module,
    data: dict[str, Any],
    coords: torch.Tensor,
    device: torch.device,
    eval_window_size: int,
    eval_stride: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None, np.ndarray | None]:
    T = int(data["X"].shape[0])
    if eval_stride is None:
        eval_stride = eval_window_size
    M_sum = np.zeros((T, len(STATE_NAMES), len(STATE_NAMES)), dtype=np.float64)
    control_sum: np.ndarray | None = None
    flow_sum: np.ndarray | None = None
    counts = np.zeros((T, 1), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for start in range(0, T, eval_stride):
            end = min(start + eval_window_size, T)
            if end <= start:
                continue
            x_chunk = torch.from_numpy(data["X"][start:end]).to(device)
            outputs = model(x_chunk, coords)
            M_sum[start:end] += outputs["M"].detach().cpu().numpy()
            if "transition_control" in outputs:
                values = outputs["transition_control"].detach().cpu().numpy()
                if control_sum is None:
                    control_sum = np.zeros((T, values.shape[-1]), dtype=np.float64)
                control_sum[start:end] += values
            if "flow_mu" in outputs:
                values = outputs["flow_mu"].detach().cpu().numpy()
                if flow_sum is None:
                    flow_sum = np.zeros((T, values.shape[-1]), dtype=np.float64)
                flow_sum[start:end] += values
            counts[start:end] += 1.0

    valid = counts[:, 0] > 0.0
    if not valid.all():
        missing = int((~valid).sum())
        raise ValueError(f"Missing transition parameters for {missing} time steps; reduce --eval-stride.")
    M = (M_sum / counts[:, :, None]).astype(np.float32)
    control = None
    if control_sum is not None:
        control = (control_sum / counts).astype(np.float32)
    flow = None
    if flow_sum is not None:
        flow = (flow_sum / counts).astype(np.float32)
    M_torch = torch.from_numpy(M).to(device)
    control_torch = torch.from_numpy(control).to(device) if control is not None else None
    return M_torch, control_torch, flow


def station_speed_np(values: np.ndarray) -> np.ndarray:
    u = values[..., :3]
    v = values[..., 3:6]
    return np.sqrt(np.maximum(u * u + v * v, 0.0))


def speed_sigma_from_uv_cov_np(prediction: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    speed = station_speed_np(prediction)
    sigma = np.full_like(speed, np.nan, dtype=np.float32)
    for station_idx in range(3):
        u = prediction[:, station_idx]
        v = prediction[:, station_idx + 3]
        s = np.maximum(speed[:, station_idx], 1.0e-6)
        grad = np.stack([u / s, v / s], axis=1)
        uv_idx = [station_idx, station_idx + 3]
        uv_cov = covariances[:, uv_idx][:, :, uv_idx]
        var = np.einsum("ni,nij,nj->n", grad, uv_cov, grad)
        sigma[:, station_idx] = np.sqrt(np.maximum(var, 0.0))
    return sigma


def choose_interval_window(
    target_speed: np.ndarray,
    time_index: np.ndarray,
    flow: np.ndarray | None,
    window_size: int,
) -> tuple[int, int]:
    n = int(target_speed.shape[0])
    if n <= window_size:
        return 0, n
    if flow is not None and flow.shape[0] > int(time_index[-1]):
        score_source = np.linalg.norm(flow[time_index], axis=1)
    else:
        diffs = np.abs(np.diff(target_speed, axis=0, prepend=target_speed[:1]))
        score_source = np.nanmean(diffs, axis=1)
    score_source = np.nan_to_num(score_source, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.ones(window_size, dtype=np.float64)
    scores = np.convolve(score_source.astype(np.float64), kernel, mode="valid")
    start = int(np.argmax(scores)) if scores.size else 0
    return start, min(start + window_size, n)


def plot_station_speed_forecast_interval(
    interval_payload: dict[str, np.ndarray],
    output_dir: Path,
    window_size: int = 720,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping station interval plot because matplotlib is unavailable: {exc}")
        return {}

    prediction = np.asarray(interval_payload["prediction"], dtype=np.float32)
    target = np.asarray(interval_payload["target"], dtype=np.float32)
    nwp = np.asarray(interval_payload["nwp"], dtype=np.float32)
    covariances = np.asarray(interval_payload["covariance"], dtype=np.float32)
    time_index = np.asarray(interval_payload["time_index"], dtype=np.int64)
    flow = interval_payload.get("flow")

    pred_speed = station_speed_np(prediction)
    target_speed = station_speed_np(target)
    nwp_speed = station_speed_np(nwp)
    speed_sigma = speed_sigma_from_uv_cov_np(prediction, covariances)
    lower = pred_speed - 1.96 * speed_sigma
    upper = pred_speed + 1.96 * speed_sigma

    start, end = choose_interval_window(target_speed, time_index, flow, window_size)
    sl = slice(start, end)
    x = time_index[sl]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "station_speed_forecast_intervals.npz",
        time_index=time_index,
        prediction_speed=pred_speed,
        target_speed=target_speed,
        nwp_speed=nwp_speed,
        lower_95=lower,
        upper_95=upper,
        speed_sigma=speed_sigma,
        station_names=np.asarray(STATION_NAMES),
        plotted_start_index=int(x[0]) if x.size else 0,
        plotted_end_index=int(x[-1]) if x.size else 0,
    )

    fig, axes = plt.subplots(3, 1, figsize=(16.5, 10.0), sharex=True, constrained_layout=True)
    panel_labels = ["a", "b", "c"]
    blue = "#1f77b4"
    orange = "#ef7f62"
    interval_color = "#b7d4ee"
    for station_idx, ax in enumerate(axes):
        ax.fill_between(
            x,
            lower[sl, station_idx],
            upper[sl, station_idx],
            color=interval_color,
            alpha=0.45,
            label="VectorMIDE 95% interval",
        )
        ax.plot(x, pred_speed[sl, station_idx], color=blue, linewidth=3.0, label="VectorMIDE mean")
        ax.scatter(
            x,
            target_speed[sl, station_idx],
            s=16,
            color="#222222",
            alpha=0.78,
            label="measurement",
            zorder=3,
        )
        ax.plot(x, nwp_speed[sl, station_idx], color=orange, linewidth=2.7, label="NWP")
        ax.set_title(f"{STATION_NAMES[station_idx]} wind speed forecast interval", fontsize=22, fontweight="bold")
        ax.set_ylabel("Speed", fontsize=18, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=17, width=1.5, length=6)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.text(
            0.012,
            0.20,
            panel_labels[station_idx],
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "#18365f", "linewidth": 1.8, "pad": 5.0},
        )
        if station_idx == 0:
            legend = ax.legend(loc="upper left", ncol=4, fontsize=13, frameon=True)
            for text in legend.get_texts():
                text.set_fontweight("bold")
    axes[-1].set_xlabel("Time index", fontsize=18, fontweight="bold")
    fig.savefig(output_dir / "station_speed_forecast_95_interval.png", dpi=220)
    plt.close(fig)

    return {
        "station_speed_forecast_95_interval": str(output_dir / "station_speed_forecast_95_interval.png"),
        "station_speed_forecast_intervals": str(output_dir / "station_speed_forecast_intervals.npz"),
        "plotted_time_index_start": int(x[0]) if x.size else None,
        "plotted_time_index_end": int(x[-1]) if x.size else None,
        "interval_unit": "m/s",
    }


def run_light_eval(
    config_path: Path,
    checkpoint_path: Path | None,
    split: str,
    device_name: str | None,
    limit: int | None,
    eval_window_size: int | None,
    eval_stride: int | None,
    forecast_horizon: int | None,
    interval_window_size: int = 720,
    make_interval_plot: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    device = resolve_device(
        device_name if device_name is not None else config.get("device", "auto"),
        allow_fallback=bool(config.get("allow_device_fallback", True)),
    )
    print_device_info(device)

    ckpt_path = checkpoint_path
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

    data = load_vector_dataset(config, split=split, time_limit=limit)
    T = int(data["X"].shape[0])
    if eval_window_size is None:
        eval_window_size = int(
            min(
                model_config.get("window_size", config.get("window_size", 1008)),
                model_config.get("transformer_max_len", config.get("transformer_max_len", 4096)),
            )
        )
    max_horizon = int(forecast_horizon or config.get("forecast_horizon", 1))
    max_horizon = min(max_horizon, max(T - 1, 1))
    coords = torch.from_numpy(data["coords"]).to(device)

    M_seq, control_seq, flow = compute_transition_sequences(
        model=model,
        data=data,
        coords=coords,
        device=device,
        eval_window_size=eval_window_size,
        eval_stride=eval_stride,
    )
    z = torch.from_numpy(data["Z"]).to(device)
    target = torch.from_numpy(data.get("Y", data["Z"]).astype(np.float32, copy=False)).to(device)
    nwp = torch.from_numpy(data.get("nwp_baseline", np.full_like(data["Z"], np.nan))).to(device)
    Q = model.dstm.process_covariance().to(device=device, dtype=z.dtype)

    with torch.no_grad():
        kf = model.dstm.kalman_filter(
            z=z,
            M_seq=M_seq,
            control_seq=control_seq,
            reduction="sum",
            return_history=True,
        )

        prev_mean = kf["filter_means"][:-1]
        prev_cov = kf["filter_covs"][:-1]
        multi_step: dict[str, Any] = {}
        rmse_curve = np.full((max_horizon, len(CURVE_COLUMNS)), np.nan, dtype=np.float32)
        mae_curve = np.full((max_horizon, len(CURVE_COLUMNS)), np.nan, dtype=np.float32)
        crps_curve = np.full((max_horizon, len(CURVE_COLUMNS)), np.nan, dtype=np.float32)
        interval_payload: dict[str, np.ndarray] | None = None

        for horizon in range(1, max_horizon + 1):
            n = T - horizon
            if n <= 0:
                break
            prev_mean = prev_mean[:n]
            prev_cov = prev_cov[:n]
            M_h = M_seq[horizon:T]
            pred_mean_model = torch.bmm(M_h, prev_mean.unsqueeze(-1)).squeeze(-1)
            pred_cov_model = torch.bmm(torch.bmm(M_h, prev_cov), M_h.transpose(1, 2)) + Q
            pred_cov_model = 0.5 * (pred_cov_model + pred_cov_model.transpose(1, 2))
            if control_seq is not None:
                pred_mean_model = pred_mean_model + control_seq[horizon:T]

            target_indices = torch.arange(horizon, T, device=device)
            pred_measurement, pred_var = model_target_to_measurement(
                pred_mean_model,
                pred_cov_model,
                data,
                target_indices,
                device,
            )
            target_h = target[horizon:T]
            crps_h = gaussian_crps(pred_measurement, pred_var, target_h)
            model_metrics = metric_block(pred_measurement, target_h, crps=crps_h)

            if horizon == 1:
                pred_cov_measurement = model_cov_to_measurement_cov(
                    pred_cov_model,
                    data,
                    device,
                    pred_measurement.dtype,
                )
                interval_payload = {
                    "prediction": pred_measurement.detach().cpu().numpy(),
                    "target": target_h.detach().cpu().numpy(),
                    "nwp": nwp[horizon:T].detach().cpu().numpy(),
                    "covariance": pred_cov_measurement.detach().cpu().numpy(),
                    "time_index": np.arange(horizon, T, dtype=np.int64),
                }
                if flow is not None:
                    interval_payload["flow"] = flow

            persistence_pred = target[:n]
            nwp_pred = nwp[horizon:T]
            residual_persistence_pred = nwp[horizon:T] + (target[:n] - nwp[:n])
            persistence_metrics = metric_block(persistence_pred, target_h, crps=torch.abs(persistence_pred - target_h))
            nwp_metrics = metric_block(nwp_pred, target_h, crps=torch.abs(nwp_pred - target_h))
            residual_persistence_metrics = metric_block(
                residual_persistence_pred,
                target_h,
                crps=torch.abs(residual_persistence_pred - target_h),
            )

            rmse_curve[horizon - 1] = [
                model_metrics["rmse"],
                persistence_metrics["rmse"],
                nwp_metrics["rmse"],
                residual_persistence_metrics["rmse"],
            ]
            mae_curve[horizon - 1] = [
                model_metrics["mae"],
                persistence_metrics["mae"],
                nwp_metrics["mae"],
                residual_persistence_metrics["mae"],
            ]
            crps_curve[horizon - 1] = [
                model_metrics["crps"],
                persistence_metrics["crps"],
                nwp_metrics["crps"],
                residual_persistence_metrics["crps"],
            ]
            multi_step[str(horizon)] = {
                "model": model_metrics,
                "persistence_baseline": persistence_metrics,
                "nwp_baseline": nwp_metrics,
                "nwp_residual_persistence_baseline": residual_persistence_metrics,
                "model_vs_persistence_improvement": metric_improvement(model_metrics, persistence_metrics),
                "model_vs_nwp_improvement": metric_improvement(model_metrics, nwp_metrics),
                "model_vs_nwp_residual_persistence_improvement": metric_improvement(
                    model_metrics,
                    residual_persistence_metrics,
                ),
            }

            prev_mean = pred_mean_model
            prev_cov = pred_cov_model

    results = {
        "kalman_nll_per_observation": float(kf["nll_sum"].detach().cpu())
        / max(float(kf["obs_count"].detach().cpu()), 1.0),
        "model": multi_step["1"]["model"],
        "persistence_baseline": multi_step["1"]["persistence_baseline"],
        "nwp_baseline": multi_step["1"]["nwp_baseline"],
        "nwp_residual_persistence_baseline": multi_step["1"]["nwp_residual_persistence_baseline"],
        "model_vs_persistence_improvement": multi_step["1"]["model_vs_persistence_improvement"],
        "model_vs_nwp_improvement": multi_step["1"]["model_vs_nwp_improvement"],
        "model_vs_nwp_residual_persistence_improvement": multi_step["1"][
            "model_vs_nwp_residual_persistence_improvement"
        ],
        "target_mode": data.get("target_mode", "measurement"),
        "forecast_horizon": max_horizon,
        "multi_step": multi_step,
        "split": split,
        "checkpoint": str(ckpt_path),
        "n_time": T,
        "eval_window_size": eval_window_size,
        "eval_stride": eval_stride if eval_stride is not None else eval_window_size,
        "metric_note": (
            "Model CRPS is univariate Gaussian CRPS per U/V component from Kalman forecast covariance. "
            "Baseline CRPS treats persistence/NWP baselines as deterministic forecasts, so it equals MAE."
        ),
        "_interval_payload": interval_payload,
        "_interval_window_size": int(interval_window_size),
        "_make_interval_plot": bool(make_interval_plot),
        "_curves": {
            "horizons": list(range(1, max_horizon + 1)),
            "rmse_curve": rmse_curve.tolist(),
            "mae_curve": mae_curve.tolist(),
            "crps_curve": crps_curve.tolist(),
            "curve_columns": CURVE_COLUMNS,
        },
    }
    return results


def curves_from_existing_eval(eval_dir: Path) -> dict[str, Any]:
    results_path = eval_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    curves: dict[str, Any] = {}
    metrics_path = eval_dir / "multi_step_metrics.npz"
    if metrics_path.exists():
        with np.load(metrics_path, allow_pickle=True) as data:
            curves = {
                "horizons": data["horizons"].astype(int).tolist(),
                "rmse_curve": data["rmse_curve"].astype(float).tolist(),
                "mae_curve": data["mae_curve"].astype(float).tolist(),
                "curve_columns": [str(x) for x in data["curve_columns"].tolist()],
            }
        if "multi_step" in results:
            crps_curve = []
            for h in curves["horizons"]:
                row = results["multi_step"].get(str(h), {})
                model_row = row.get("model", {})
                persistence_row = row.get("persistence_baseline", {})
                nwp_row = row.get("nwp_baseline", {})
                residual_row = row.get("nwp_residual_persistence_baseline", {})
                crps_curve.append(
                    [
                        model_row.get("crps", float("nan")),
                        persistence_row.get("crps", persistence_row.get("mae", float("nan"))),
                        nwp_row.get("crps", nwp_row.get("mae", float("nan"))),
                        residual_row.get("crps", residual_row.get("mae", float("nan"))),
                    ]
                )
            curves["crps_curve"] = crps_curve
    elif "multi_step" in results:
        horizons = sorted(int(h) for h in results["multi_step"])
        rmse_curve = []
        mae_curve = []
        crps_curve = []
        for h in horizons:
            row = results["multi_step"][str(h)]
            rmse_curve.append(
                [
                    row["model"]["rmse"],
                    row["persistence_baseline"]["rmse"],
                    row["nwp_baseline"]["rmse"],
                    row["nwp_residual_persistence_baseline"]["rmse"],
                ]
            )
            mae_curve.append(
                [
                    row["model"]["mae"],
                    row["persistence_baseline"]["mae"],
                    row["nwp_baseline"]["mae"],
                    row["nwp_residual_persistence_baseline"]["mae"],
                ]
            )
            crps_curve.append(
                [
                    row["model"].get("crps", float("nan")),
                    row["persistence_baseline"].get("crps", row["persistence_baseline"]["mae"]),
                    row["nwp_baseline"].get("crps", row["nwp_baseline"]["mae"]),
                    row["nwp_residual_persistence_baseline"].get(
                        "crps",
                        row["nwp_residual_persistence_baseline"]["mae"],
                    ),
                ]
            )
        curves = {
            "horizons": horizons,
            "rmse_curve": rmse_curve,
            "mae_curve": mae_curve,
            "crps_curve": crps_curve,
            "curve_columns": CURVE_COLUMNS,
        }
    results["_curves"] = curves
    results["crps_available"] = any(
        "crps" in results.get("multi_step", {}).get(str(h), {}).get("model", {})
        for h in curves.get("horizons", [])
    )
    if not results["crps_available"]:
        results["crps_note"] = (
            "Existing full eval artifacts contain RMSE/MAE but not forecast covariance, "
            "so Gaussian CRPS cannot be reconstructed exactly from this directory. "
            "Run this script with --config and --checkpoint to compute model CRPS without plots/artifacts. "
            "Baseline CRPS is still plotted as deterministic CRPS, equal to MAE."
        )
    return results


def write_horizon_csv(results: dict[str, Any], path: Path, index_name: str = "horizon") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    curves = results.get("_curves", {})
    horizons = curves.get("horizons", [])
    columns = curves.get("curve_columns", CURVE_COLUMNS)
    rmse = np.asarray(curves.get("rmse_curve", []), dtype=float)
    mae = np.asarray(curves.get("mae_curve", []), dtype=float)
    crps = np.asarray(curves.get("crps_curve", []), dtype=float)
    if rmse.ndim == 1 and rmse.size:
        rmse = rmse[:, None]
    if mae.ndim == 1 and mae.size:
        mae = mae[:, None]
    if crps.ndim == 1 and crps.size:
        crps = crps[:, None]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [index_name]
        for name in columns:
            header.extend([f"{name}_rmse", f"{name}_mae", f"{name}_crps"])
        writer.writerow(header)
        for i, horizon in enumerate(horizons):
            row: list[Any] = [horizon]
            for j in range(len(columns)):
                row.append(float(rmse[i, j]) if rmse.size else "")
                row.append(float(mae[i, j]) if mae.size else "")
                row.append(
                    float(crps[i, j])
                    if crps.ndim == 2 and i < crps.shape[0] and j < crps.shape[1] and np.isfinite(crps[i, j])
                    else ""
                )
            writer.writerow(row)


def hourly_mean_curves(results: dict[str, Any], group_size: int = 6) -> dict[str, Any]:
    curves = results.get("_curves", {})
    horizons = np.asarray(curves.get("horizons", []), dtype=int)
    if horizons.size == 0:
        return {}
    columns = curves.get("curve_columns", CURVE_COLUMNS)
    hour_ids = ((horizons - 1) // group_size) + 1
    hours = np.unique(hour_ids)
    grouped: dict[str, Any] = {"horizons": hours.astype(int).tolist(), "curve_columns": columns}
    for key in ("rmse_curve", "mae_curve", "crps_curve"):
        values = np.asarray(curves.get(key, []), dtype=float)
        if values.size == 0:
            continue
        if values.ndim == 1:
            values = values[:, None]
        out = np.full((len(hours), values.shape[1]), np.nan, dtype=float)
        for i, hour in enumerate(hours):
            with np.errstate(invalid="ignore"):
                out[i] = np.nanmean(values[hour_ids == hour], axis=0)
        grouped[key] = out.tolist()
    return grouped


def write_hourly_csv(results: dict[str, Any], path: Path) -> None:
    hourly_results = {"_curves": hourly_mean_curves(results)}
    write_horizon_csv(hourly_results, path, index_name="forecast_hour")


def curve_records(curves: dict[str, Any], index_name: str) -> list[dict[str, Any]]:
    indices = curves.get("horizons", [])
    columns = [str(x) for x in curves.get("curve_columns", CURVE_COLUMNS)]
    rmse = np.asarray(curves.get("rmse_curve", []), dtype=float)
    mae = np.asarray(curves.get("mae_curve", []), dtype=float)
    crps = np.asarray(curves.get("crps_curve", []), dtype=float)
    if rmse.ndim == 1 and rmse.size:
        rmse = rmse[:, None]
    if mae.ndim == 1 and mae.size:
        mae = mae[:, None]
    if crps.ndim == 1 and crps.size:
        crps = crps[:, None]

    records: list[dict[str, Any]] = []
    for i, value in enumerate(indices):
        record: dict[str, Any] = {index_name: int(value)}
        for j, name in enumerate(columns):
            method: dict[str, float | None] = {}
            method["rmse"] = (
                float(rmse[i, j])
                if rmse.ndim == 2 and i < rmse.shape[0] and j < rmse.shape[1] and np.isfinite(rmse[i, j])
                else None
            )
            method["mae"] = (
                float(mae[i, j])
                if mae.ndim == 2 and i < mae.shape[0] and j < mae.shape[1] and np.isfinite(mae[i, j])
                else None
            )
            method["crps"] = (
                float(crps[i, j])
                if crps.ndim == 2 and i < crps.shape[0] and j < crps.shape[1] and np.isfinite(crps[i, j])
                else None
            )
            record[name] = method
        records.append(record)
    return records


def add_public_metric_curves(results: dict[str, Any]) -> None:
    horizon_curves = results.get("_curves", {})
    hour_curves = hourly_mean_curves(results)
    results["metric_curves_per_horizon"] = curve_records(horizon_curves, "horizon")
    results["metric_curves_per_hour"] = curve_records(hour_curves, "forecast_hour")


def plot_curves(results: dict[str, Any], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    labels = {
        "model": "model",
        "persistence": "persistence",
        "nwp": "nwp",
        "nwp_residual_persistence": "nwp residual persistence",
    }
    markers = ["o", "s", "^", "D", "v", "P"]
    title_font = {"fontsize": 16, "fontweight": "bold"}
    label_font = {"fontsize": 14, "fontweight": "bold"}
    legend_font = {"size": 10, "weight": "bold"}

    output_dir.mkdir(parents=True, exist_ok=True)

    def normalize_curve(values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1 and arr.size:
            arr = arr[:, None]
        return arr

    def style_axis(ax: Any, xlabel: str) -> None:
        ax.set_xlabel(xlabel, **label_font)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=13, width=1.4, length=5.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.legend(prop=legend_font)

    def plot_rmse_mae(curves: dict[str, Any], xlabel: str, title_prefix: str, filename: str) -> None:
        x_values = np.asarray(curves.get("horizons", []), dtype=float)
        if x_values.size == 0:
            return
        columns = [str(x) for x in curves.get("curve_columns", CURVE_COLUMNS)]
        rmse = normalize_curve(curves.get("rmse_curve", []))
        mae = normalize_curve(curves.get("mae_curve", []))
        if not (rmse.size or mae.size):
            return
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
        for j, name in enumerate(columns):
            if rmse.size and j < rmse.shape[1]:
                axes[0].plot(
                    x_values,
                    rmse[:, j],
                    marker=markers[j % len(markers)],
                    markersize=4.8,
                    linewidth=2.1,
                    label=labels.get(name, name),
                )
            if mae.size and j < mae.shape[1]:
                axes[1].plot(
                    x_values,
                    mae[:, j],
                    marker=markers[j % len(markers)],
                    markersize=4.8,
                    linewidth=2.1,
                    label=labels.get(name, name),
                )
        axes[0].set_title(f"{title_prefix} RMSE", **title_font)
        axes[1].set_title(f"{title_prefix} MAE", **title_font)
        axes[0].set_ylabel("RMSE", **label_font)
        axes[1].set_ylabel("MAE", **label_font)
        for ax in axes:
            style_axis(ax, xlabel)
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)

    def plot_crps(curves: dict[str, Any], xlabel: str, title: str, filename: str) -> None:
        x_values = np.asarray(curves.get("horizons", []), dtype=float)
        if x_values.size == 0:
            return
        columns = [str(x) for x in curves.get("curve_columns", CURVE_COLUMNS)]
        crps = normalize_curve(curves.get("crps_curve", []))
        if not (crps.size and np.isfinite(crps).any()):
            return
        fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
        for j, name in enumerate(columns):
            if j >= crps.shape[1] or not np.isfinite(crps[:, j]).any():
                continue
            ax.plot(
                x_values,
                crps[:, j],
                marker=markers[j % len(markers)],
                markersize=5.2,
                linewidth=2.2,
                label=labels.get(name, name),
            )
        ax.set_title(title, **title_font)
        ax.set_ylabel("CRPS", **label_font)
        style_axis(ax, xlabel)
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)

    horizon_curves = results.get("_curves", {})
    hour_curves = hourly_mean_curves(results)
    plot_rmse_mae(
        horizon_curves,
        xlabel="forecast horizon (10 min steps)",
        title_prefix="Per-Horizon",
        filename="rmse_mae_curves_by_horizon.png",
    )
    plot_crps(
        horizon_curves,
        xlabel="forecast horizon (10 min steps)",
        title="Per-Horizon CRPS",
        filename="crps_curve_by_horizon.png",
    )
    plot_rmse_mae(
        hour_curves,
        xlabel="forecast hour",
        title_prefix="Hourly Mean",
        filename="rmse_mae_curves_by_hour.png",
    )
    plot_crps(
        hour_curves,
        xlabel="forecast hour",
        title="Hourly Mean CRPS",
        filename="crps_curve_by_hour.png",
    )
    plot_rmse_mae(hour_curves, xlabel="forecast hour", title_prefix="Hourly Mean", filename="rmse_mae_curves.png")
    plot_crps(hour_curves, xlabel="forecast hour", title="Hourly Mean CRPS", filename="crps_curve.png")


def plot_extra_artifacts(results: dict[str, Any], output_dir: Path) -> None:
    payload = results.get("_interval_payload")
    if not results.get("_make_interval_plot", False) or payload is None:
        return
    summary = plot_station_speed_forecast_interval(
        payload,
        output_dir,
        window_size=int(results.get("_interval_window_size", 720)),
    )
    if summary:
        results["station_speed_interval_plot"] = summary


def strip_internal_fields(results: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in results.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight VectorMIDE RMSE/MAE/CRPS evaluation and summary."
    )
    parser.add_argument("--eval-dir", type=Path, default=None, help="Reuse an existing full eval directory.")
    parser.add_argument("--config", type=Path, default=Path("yml_files/VectorMIDE.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["offline", "online"], default="online")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-window-size", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--forecast-horizon", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-interval-plot", action="store_true")
    parser.add_argument("--interval-window-size", type=int, default=720)
    args = parser.parse_args()

    if args.eval_dir is not None:
        results = curves_from_existing_eval(args.eval_dir)
        output_dir = args.output_dir or (args.eval_dir / "light_metrics")
    else:
        results = run_light_eval(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            split=args.split,
            device_name=args.device,
            limit=args.limit,
            eval_window_size=args.eval_window_size,
            eval_stride=args.eval_stride,
            forecast_horizon=args.forecast_horizon,
            interval_window_size=args.interval_window_size,
            make_interval_plot=not args.no_interval_plot,
        )
        output_dir = args.output_dir or Path("outputs") / "light_metrics" / Path(results["checkpoint"]).stem
        results["output_dir"] = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_horizon_csv(results, output_dir / "metrics_by_horizon.csv")
    write_hourly_csv(results, output_dir / "metrics_by_hour.csv")
    add_public_metric_curves(results)
    if not args.no_plots:
        plot_curves(results, output_dir)
        plot_extra_artifacts(results, output_dir)
    clean_results = strip_internal_fields(results)
    save_json(output_dir / "metrics_summary.json", clean_results)
    print(json.dumps(clean_results, indent=2))
    print(f"Saved lightweight metrics to {output_dir}")


if __name__ == "__main__":
    main()
