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


STATE_NAMES = ["U1", "U2", "U3", "V1", "V2", "V3"]
ADV_NAMES = ["A_u_x", "A_u_y", "A_v_x", "A_v_y"]
SIGMA_DIAG_NAMES = ["var_A_u_x", "var_A_u_y", "var_A_v_x", "var_A_v_y"]
ALPHA_NAMES = ["alpha_uu", "alpha_uv", "alpha_vu", "alpha_vv"]
CURVE_COLUMNS = ["model", "persistence", "nwp", "nwp_residual_persistence"]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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

    model.eval()
    with torch.no_grad():
        for start in range(0, T, eval_stride):
            end = min(start + eval_window_size, T)
            if end <= start:
                continue
            x = torch.from_numpy(data["X"][start:end]).to(device)
            outputs = model(x, coords)
            sums["M"][start:end] += outputs["M"].detach().cpu().numpy()
            sums["M_base"][start:end] += outputs.get("M_base", outputs["M"]).detach().cpu().numpy()
            sums["mu"][start:end] += outputs["mu"].detach().cpu().numpy()
            sums["Sigma"][start:end] += outputs["Sigma"].detach().cpu().numpy()
            sums["alpha"][start:end] += outputs["alpha"].detach().cpu().numpy()
            counts[start:end] += 1.0

    valid = counts[:, 0] > 0
    if not valid.all():
        missing = int((~valid).sum())
        raise ValueError(f"Model outputs are missing for {missing} time steps; reduce --eval-stride.")

    averaged: dict[str, np.ndarray] = {}
    for key, value in sums.items():
        if value.ndim == 3:
            averaged[key] = (value / counts[:, :, None]).astype(np.float32)
        elif value.ndim == 4:
            averaged[key] = (value / counts[:, :, None, None]).astype(np.float32)
        else:
            averaged[key] = (value / counts).astype(np.float32)
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


def transition_summary(M: np.ndarray, M_base: np.ndarray, ell: np.ndarray) -> dict[str, Any]:
    M_mean = np.nanmean(M, axis=0)
    row_sums = np.nansum(M, axis=2)
    block_mass = np.zeros((M.shape[0], 2, 2), dtype=np.float32)
    n = M.shape[1] // 2
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
        "ell": ell,
        "gamma": gamma,
        "Q": Q,
        "R": R,
        "coords": data["coords"].astype(np.float32, copy=False),
        "baseline_grid_indices": data.get("baseline_grid_indices", np.empty((0, 2), dtype=np.int64)),
    }

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
        "transition": transition_summary(outputs["M"], outputs["M_base"], ell),
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
    np.savez_compressed(
        output_dir / "advection_parameters.npz",
        mu=artifacts["mu"],
        Au=artifacts["Au"],
        Av=artifacts["Av"],
        Sigma=artifacts["Sigma"],
        Sigma_diag=artifacts["Sigma_diag"],
        alpha=artifacts["alpha"],
        ell=artifacts["ell"],
        gamma=artifacts["gamma"],
        Q=artifacts["Q"],
        R=artifacts["R"],
        coords=artifacts["coords"],
        baseline_grid_indices=artifacts["baseline_grid_indices"],
        advection_names=np.asarray(ADV_NAMES),
        sigma_diag_names=np.asarray(SIGMA_DIAG_NAMES),
        alpha_names=np.asarray(ALPHA_NAMES),
        state_names=np.asarray(STATE_NAMES),
    )

    param_csv = np.column_stack(
        [
            np.arange(artifacts["mu"].shape[0]),
            artifacts["mu"],
            artifacts["Sigma_diag"],
            artifacts["alpha"].reshape(artifacts["alpha"].shape[0], 4),
        ]
    )
    header = ",".join(["time_index", *ADV_NAMES, *SIGMA_DIAG_NAMES, *ALPHA_NAMES])
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


def advection_vector_plot(path: Path, Au: np.ndarray, Av: np.ndarray, max_points: int) -> None:
    """Visualize the learned U-field and V-field advection vectors.

    Each point is one time step's learned two-dimensional displacement. The
    arrows show the temporal mean of the two component-field advections.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = downsample_indices(Au.shape[0], max_points)
    au = Au[idx]
    av = Av[idx]
    values = np.concatenate([au, av], axis=0)
    lim = max(float(np.nanmax(np.abs(values))) * 1.1, 1.0e-6)

    fig, ax = plt.subplots(figsize=(7, 7))
    color = np.linspace(0.0, 1.0, idx.size)
    sc_u = ax.scatter(au[:, 0], au[:, 1], c=color, cmap="Blues", s=16, alpha=0.55, label="A_u(t)")
    ax.scatter(av[:, 0], av[:, 1], c=color, cmap="Oranges", s=16, alpha=0.55, label="A_v(t)")
    mean_u = np.nanmean(Au, axis=0)
    mean_v = np.nanmean(Av, axis=0)
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
    ax.set_title("Component advection vectors")
    ax.set_xlabel("x displacement / step")
    ax.set_ylabel("y displacement / step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(sc_u, ax=ax, shrink=0.75)
    cbar.set_label("relative time")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plots(output_dir: Path, artifacts: dict[str, np.ndarray], max_points: int) -> None:
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

    line_plot(plots_dir / "advection_mu.png", artifacts["mu"], ADV_NAMES, "Advection mean", "coordinate units / step", max_points)
    line_plot(plots_dir / "advection_sigma_diag.png", artifacts["Sigma_diag"], SIGMA_DIAG_NAMES, "Advection covariance diagonal", "variance", max_points)
    line_plot(
        plots_dir / "advection_speed.png",
        np.column_stack(
            [
                np.linalg.norm(artifacts["Au"], axis=1),
                np.linalg.norm(artifacts["Av"], axis=1),
                np.linalg.norm(artifacts["Au"] - artifacts["Av"], axis=1),
            ]
        ),
        ["|A_u|", "|A_v|", "|A_u-A_v|"],
        "Component advection speeds",
        "coordinate units / step",
        max_points,
    )
    advection_vector_plot(plots_dir / "advection_component_vectors.png", artifacts["Au"], artifacts["Av"], max_points)
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
    eval_stride = int(args.eval_stride or eval_window_size)
    forecast_horizon = int(args.forecast_horizon or config.get("forecast_horizon", 1))
    data = load_vector_dataset(config, split=args.split, time_limit=args.limit)

    results, artifacts = evaluate(
        model=model,
        data=data,
        device=device,
        eval_window_size=eval_window_size,
        eval_stride=eval_stride,
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
    if args.output:
        save_json(Path(args.output), results)
    save_artifact_arrays(output_dir, artifacts)
    if not args.no_plots:
        save_plots(output_dir, artifacts, max_points=int(args.max_plot_points))
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
