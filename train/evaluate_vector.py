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
ADV_NAMES = ["mu_Ux", "mu_Uy", "mu_Vx", "mu_Vy"]
A_NAMES = ["A_UU", "A_UV", "A_VU", "A_VV"]


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


def transition_diagnostics(M: np.ndarray, A: np.ndarray, ell: np.ndarray, coords: np.ndarray) -> dict[str, Any]:
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

    return {
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
        )
        for step in range(horizon):
            predictions[step, origin + step + 1] = future["means"][step]
    return predictions


def evaluate(
    model: torch.nn.Module,
    data: dict[str, Any],
    device: torch.device,
    eval_window_size: int,
    eval_stride: int | None = None,
    forecast_horizon: int = 1,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    coords = torch.from_numpy(data["coords"]).to(device)
    T = int(data["X"].shape[0])
    if eval_stride is None:
        eval_stride = eval_window_size

    M_sum = np.zeros((T, 6, 6), dtype=np.float64)
    mu_sum = np.zeros((T, 4), dtype=np.float64)
    sigma_sum = np.zeros((T, 4, 4), dtype=np.float64)
    A_sum = np.zeros((T, 2, 2), dtype=np.float64)
    param_counts = np.zeros((T, 1), dtype=np.float64)

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
            mu_sum[start:end] += outputs["mu"].detach().cpu().numpy()
            sigma_sum[start:end] += outputs["Sigma"].detach().cpu().numpy()
            A_sum[start:end] += outputs["A"].detach().cpu().numpy()
            param_counts[start:end] += 1.0

    valid_params = param_counts[:, 0] > 0.0
    M_np = np.full((T, 6, 6), np.nan, dtype=np.float32)
    mu_np = np.full((T, 4), np.nan, dtype=np.float32)
    sigma_np = np.full((T, 4, 4), np.nan, dtype=np.float32)
    A_np = np.full((T, 2, 2), np.nan, dtype=np.float32)
    counts = param_counts[valid_params]
    M_np[valid_params] = (M_sum[valid_params] / counts[:, :, None]).astype(np.float32)
    mu_np[valid_params] = (mu_sum[valid_params] / counts).astype(np.float32)
    sigma_np[valid_params] = (sigma_sum[valid_params] / counts[:, :, None]).astype(np.float32)
    A_np[valid_params] = (A_sum[valid_params] / counts[:, :, None]).astype(np.float32)

    if not np.isfinite(M_np).all():
        missing = int((~np.isfinite(M_np).all(axis=(1, 2))).sum())
        raise ValueError(f"Transition matrices are missing for {missing} time steps; reduce eval stride.")

    z = torch.from_numpy(data["Z"]).to(device)
    M_torch = torch.from_numpy(M_np).to(device)
    with torch.no_grad():
        kf = model.dstm.kalman_filter(
            z=z,
            M_seq=M_torch,
            reduction="sum",
            return_history=True,
        )
        pred = kf["pred_means"]
        pred_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
        multi_pred = multi_horizon_forecasts(
            model=model,
            filter_means=kf["filter_means"],
            filter_covs=kf["filter_covs"],
            M_seq=M_torch,
            max_horizon=forecast_horizon,
        )

    baseline = persistence_forecast(z, horizon=1)
    model_metrics, baseline_metrics = paired_forecast_metrics(
        pred,
        baseline,
        z,
        skip_initial=1,
    )
    improvement = metric_improvement(model_metrics, baseline_metrics)

    multi_step: dict[str, Any] = {}
    multi_baselines = []
    multi_pred_np = multi_pred.detach().cpu().numpy().astype(np.float32, copy=False)
    rmse_curve = np.full((forecast_horizon, 2), np.nan, dtype=np.float32)
    mae_curve = np.full((forecast_horizon, 2), np.nan, dtype=np.float32)
    for horizon in range(1, forecast_horizon + 1):
        horizon_pred = multi_pred[horizon - 1]
        horizon_baseline = persistence_forecast(z, horizon=horizon)
        horizon_model_metrics, horizon_baseline_metrics = paired_forecast_metrics(
            horizon_pred,
            horizon_baseline,
            z,
            skip_initial=horizon,
        )
        multi_baselines.append(horizon_baseline.detach().cpu().numpy().astype(np.float32, copy=False))
        rmse_curve[horizon - 1] = [
            horizon_model_metrics["rmse"],
            horizon_baseline_metrics["rmse"],
        ]
        mae_curve[horizon - 1] = [
            horizon_model_metrics["mae"],
            horizon_baseline_metrics["mae"],
        ]
        multi_step[str(horizon)] = {
            "model": horizon_model_metrics,
            "persistence_baseline": horizon_baseline_metrics,
            "model_vs_persistence_improvement": metric_improvement(
                horizon_model_metrics,
                horizon_baseline_metrics,
            ),
        }
    multi_baseline_np = np.stack(multi_baselines, axis=0)

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
        "model_vs_persistence_improvement": improvement,
        "forecast_horizon": forecast_horizon,
        "multi_step": multi_step,
        "diagnostics": transition_diagnostics(M_np, A_np, ell, data["coords"]),
        "eval_window_size": eval_window_size,
        "eval_stride": eval_stride,
    }
    artifacts = {
        "target": data["Z"].astype(np.float32, copy=False),
        "prediction": pred_np,
        "persistence_prediction": baseline.detach().cpu().numpy().astype(np.float32, copy=False),
        "multi_step_prediction": multi_pred_np,
        "multi_step_persistence_prediction": multi_baseline_np,
        "multi_step_rmse_curve": rmse_curve,
        "multi_step_mae_curve": mae_curve,
        "horizons": np.arange(1, forecast_horizon + 1, dtype=np.int32),
        "transition_matrices": M_np,
        "mu": mu_np,
        "Sigma": sigma_np,
        "Sigma_diag": np.diagonal(sigma_np, axis1=1, axis2=2),
        "A": A_np,
        "ell": ell.astype(np.float32, copy=False),
        "gamma": gamma.astype(np.float32, copy=False),
        "Q": Q.astype(np.float32, copy=False),
        "R": R.astype(np.float32, copy=False),
        "coords": data["coords"].astype(np.float32, copy=False),
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
        prediction=artifacts["prediction"],
        persistence_prediction=artifacts["persistence_prediction"],
        multi_step_prediction=artifacts["multi_step_prediction"],
        multi_step_persistence_prediction=artifacts["multi_step_persistence_prediction"],
        horizons=artifacts["horizons"],
        state_names=np.asarray(STATE_NAMES),
    )
    np.savez_compressed(
        output_dir / "multi_step_metrics.npz",
        horizons=artifacts["horizons"],
        rmse_curve=artifacts["multi_step_rmse_curve"],
        mae_curve=artifacts["multi_step_mae_curve"],
        curve_columns=np.asarray(["model", "persistence"]),
    )
    np.savez_compressed(
        output_dir / "transition_matrices.npz",
        M=artifacts["transition_matrices"],
        M_mean=np.nanmean(artifacts["transition_matrices"], axis=0),
        M_row_sums=np.nansum(artifacts["transition_matrices"], axis=2),
        state_names=np.asarray(STATE_NAMES),
    )
    np.savez_compressed(
        output_dir / "advection_parameters.npz",
        mu=artifacts["mu"],
        Sigma=artifacts["Sigma"],
        Sigma_diag=artifacts["Sigma_diag"],
        A=artifacts["A"],
        ell=artifacts["ell"],
        gamma=artifacts["gamma"],
        Q=artifacts["Q"],
        R=artifacts["R"],
        coords=artifacts["coords"],
        advection_names=np.asarray(ADV_NAMES),
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


def _horizon_plot(path: Path, horizons: np.ndarray, values: np.ndarray, title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(horizons, values[:, 0], marker="o", linewidth=1.8, label="model")
    ax.plot(horizons, values[:, 1], marker="o", linewidth=1.8, label="persistence")
    ax.set_title(title)
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
    _line_plot(plots_dir / "advection_sigma_diag.png", artifacts["Sigma_diag"], ["var_Ux", "var_Uy", "var_Vx", "var_Vy"], "Advection covariance diagonal", "variance", max_points)
    _line_plot(plots_dir / "mixing_A.png", artifacts["A"].reshape(artifacts["A"].shape[0], 4), A_NAMES, "Component mixing matrix A", "weight", max_points)
    _line_plot(plots_dir / "transition_row_sums.png", np.nansum(artifacts["transition_matrices"], axis=2), STATE_NAMES, "Transition matrix row sums", "row sum", max_points)
    _horizon_plot(plots_dir / "multi_step_rmse.png", artifacts["horizons"], artifacts["multi_step_rmse_curve"], "Multi-step RMSE", "RMSE")
    _horizon_plot(plots_dir / "multi_step_mae.png", artifacts["horizons"], artifacts["multi_step_mae_curve"], "Multi-step MAE", "MAE")

    M_mean = np.nanmean(artifacts["transition_matrices"], axis=0)
    identity = np.eye(M_mean.shape[0], dtype=M_mean.dtype)
    M_minus_I = M_mean - identity
    M_log_mean = np.nanmean(np.log10(np.clip(artifacts["transition_matrices"], 1.0e-12, None)), axis=0)
    _heatmap(plots_dir / "transition_matrix_mean.png", M_mean, "Mean transition matrix M", STATE_NAMES, STATE_NAMES)
    _heatmap(plots_dir / "transition_matrix_mean_minus_identity.png", M_minus_I, "Mean transition matrix M - I", STATE_NAMES, STATE_NAMES, cmap="coolwarm", center_zero=True)
    _heatmap(plots_dir / "transition_matrix_log10_mean.png", M_log_mean, "Mean log10 transition matrix", STATE_NAMES, STATE_NAMES)
    _heatmap(plots_dir / "kernel_lengthscale_ell.png", artifacts["ell"], "Kernel lengthscales ell", ["U source", "V source"], ["U target", "V target"])
    _heatmap(plots_dir / "process_covariance_Q.png", artifacts["Q"], "Process covariance Q", STATE_NAMES, STATE_NAMES)
    _heatmap(plots_dir / "observation_covariance_R.png", artifacts["R"], "Observation covariance R", STATE_NAMES, STATE_NAMES)

    M = artifacts["transition_matrices"]
    finite_times = np.where(np.isfinite(M).all(axis=(1, 2)))[0]
    if finite_times.size == 0:
        return
    frame_indices = finite_times[_downsample_indices(finite_times.size, max_gif_frames)]
    vmin = float(np.nanpercentile(M[frame_indices], 1))
    vmax = float(np.nanpercentile(M[frame_indices], 99))
    frames = []
    for t in frame_indices:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(M[t], cmap="viridis", vmin=vmin, vmax=vmax)
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
    model.load_state_dict(checkpoint["model_state"])

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
    results, artifacts = evaluate(
        model,
        data,
        device,
        eval_window_size=eval_window_size,
        eval_stride=args.eval_stride,
        forecast_horizon=forecast_horizon,
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
