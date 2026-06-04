from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.vector_data_utils import load_vector_dataset
from train.evaluate_vector import load_config
from train.evaluate_vector_light_metrics import (
    add_public_metric_curves,
    direction_error_deg,
    hourly_mean_curves,
    metric_block,
    plot_curves,
    save_json,
    strip_internal_fields,
    write_horizon_csv,
    write_hourly_csv,
)


STATE_DIM = 6
DEFAULT_COLUMNS = ["arma", "persistence", "nwp"]


@dataclass
class ArmaModel:
    p: int
    q: int
    d: int
    intercept: float
    ar: np.ndarray
    ma: np.ndarray
    residuals: np.ndarray
    sigma2: float
    mean: float
    std: float


def finite_fill(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D series, got {arr.shape}")
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    idx = np.arange(arr.size)
    if finite.any():
        arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    else:
        arr[:] = 0.0
    return arr


def difference(values: np.ndarray, order: int) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    for _ in range(order):
        out = np.diff(out)
    return out


def inverse_difference_forecast(history: np.ndarray, diff_forecast: np.ndarray, order: int) -> np.ndarray:
    if order == 0:
        return diff_forecast.astype(np.float64, copy=False)
    levels = [np.asarray(history, dtype=np.float64)]
    for k in range(1, order):
        levels.append(np.diff(levels[k - 1]))
    last_values = [float(level[-1]) for level in levels]
    out: list[float] = []
    for value in diff_forecast:
        carry = float(value)
        for k in range(order - 1, -1, -1):
            last_values[k] += carry
            carry = last_values[k]
        out.append(last_values[0])
    return np.asarray(out, dtype=np.float64)


def design_matrix(series: np.ndarray, residuals: np.ndarray, p: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    max_lag = max(p, q, 1)
    rows: list[list[float]] = []
    target: list[float] = []
    for t in range(max_lag, series.size):
        row = [1.0]
        row.extend(float(series[t - lag]) for lag in range(1, p + 1))
        row.extend(float(residuals[t - lag]) for lag in range(1, q + 1))
        rows.append(row)
        target.append(float(series[t]))
    if not rows:
        return np.empty((0, 1 + p + q), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64), np.asarray(target, dtype=np.float64)


def one_step_residuals(series: np.ndarray, intercept: float, ar: np.ndarray, ma: np.ndarray) -> np.ndarray:
    p = int(ar.size)
    q = int(ma.size)
    max_lag = max(p, q, 1)
    residuals = np.zeros(series.size, dtype=np.float64)
    for t in range(max_lag, series.size):
        pred = float(intercept)
        for lag in range(1, p + 1):
            pred += float(ar[lag - 1]) * float(series[t - lag])
        for lag in range(1, q + 1):
            pred += float(ma[lag - 1]) * float(residuals[t - lag])
        residuals[t] = float(series[t]) - pred
    return residuals


def fit_arma_1d(
    values: np.ndarray,
    p: int,
    q: int,
    d: int = 0,
    max_iter: int = 6,
    ridge: float = 1.0e-6,
) -> ArmaModel:
    raw = finite_fill(values)
    if d < 0:
        raise ValueError("ARIMA differencing order d must be non-negative.")
    if raw.size <= d + max(p, q, 1) + 2:
        raise ValueError(f"Series is too short for order ({p},{d},{q}); length={raw.size}.")
    series = difference(raw, d)
    mean = float(np.nanmean(series))
    std = float(np.nanstd(series))
    if not np.isfinite(std) or std < 1.0e-8:
        std = 1.0
    y = (series - mean) / std
    residuals = np.zeros_like(y)
    coeff = np.zeros(1 + p + q, dtype=np.float64)
    for _ in range(max(1, max_iter)):
        X, target = design_matrix(y, residuals, p, q)
        if X.size == 0:
            break
        lhs = X.T @ X + ridge * np.eye(X.shape[1], dtype=np.float64)
        rhs = X.T @ target
        coeff = np.linalg.solve(lhs, rhs)
        residuals = one_step_residuals(y, coeff[0], coeff[1 : 1 + p], coeff[1 + p :])
    ar = coeff[1 : 1 + p].copy()
    ma = coeff[1 + p :].copy()
    residuals = one_step_residuals(y, coeff[0], ar, ma)
    valid_start = max(p, q, 1)
    sigma2 = float(np.nanmean(residuals[valid_start:] ** 2)) if residuals.size > valid_start else 0.0
    return ArmaModel(
        p=p,
        q=q,
        d=d,
        intercept=float(coeff[0]),
        ar=ar,
        ma=ma,
        residuals=residuals,
        sigma2=max(sigma2, 1.0e-12),
        mean=mean,
        std=std,
    )


def normalized_series(values: np.ndarray, model: ArmaModel) -> np.ndarray:
    return (difference(finite_fill(values), model.d) - model.mean) / model.std


def predict_arma_transformed(model: ArmaModel, series: np.ndarray, residuals: np.ndarray, origin_idx: int, horizon: int) -> np.ndarray:
    if origin_idx < 0:
        return np.full(horizon, np.nan, dtype=np.float64)
    y_hist = [float(x) for x in series[: origin_idx + 1]]
    e_hist = [float(x) for x in residuals[: origin_idx + 1]]
    out: list[float] = []
    for _ in range(horizon):
        pred = float(model.intercept)
        for lag in range(1, model.p + 1):
            pred += float(model.ar[lag - 1]) * y_hist[-lag]
        for lag in range(1, model.q + 1):
            pred += float(model.ma[lag - 1]) * e_hist[-lag]
        y_hist.append(pred)
        e_hist.append(0.0)
        out.append(pred * model.std + model.mean)
    return np.asarray(out, dtype=np.float64)


def predict_arma_original(model: ArmaModel, observed_history: np.ndarray, residual_source: np.ndarray, origin_idx: int, horizon: int) -> np.ndarray:
    transformed = normalized_series(observed_history, model)
    residuals = one_step_residuals(transformed, model.intercept, model.ar, model.ma)
    train_resid_len = min(model.residuals.size, residuals.size, residual_source.size)
    if train_resid_len > 0:
        residuals[:train_resid_len] = model.residuals[:train_resid_len]
    transformed_origin = origin_idx - model.d
    diff_pred = predict_arma_transformed(model, transformed, residuals, transformed_origin, horizon)
    return inverse_difference_forecast(observed_history[: origin_idx + 1], diff_pred, model.d)


def fit_models(train_y: np.ndarray, p: int, q: int, d: int, max_iter: int, ridge: float) -> list[ArmaModel]:
    return [
        fit_arma_1d(train_y[:, dim], p=p, q=q, d=d, max_iter=max_iter, ridge=ridge)
        for dim in range(train_y.shape[1])
    ]


def rolling_forecasts(
    models: list[ArmaModel],
    combined_y: np.ndarray,
    train_len: int,
    eval_len: int,
    max_horizon: int,
    clip_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    preds = np.full((max_horizon, eval_len, STATE_DIM), np.nan, dtype=np.float32)
    lower = upper = None
    if clip_bounds is not None:
        lower, upper = clip_bounds
    for origin_local in range(eval_len):
        origin = train_len + origin_local
        for dim, model in enumerate(models):
            max_available_h = min(max_horizon, eval_len - origin_local - 1)
            if max_available_h <= 0:
                continue
            values = predict_arma_original(
                model=model,
                observed_history=combined_y[:, dim],
                residual_source=combined_y[:, dim],
                origin_idx=origin,
                horizon=max_available_h,
            )
            if lower is not None and upper is not None:
                values = np.clip(values, float(lower[dim]), float(upper[dim]))
            preds[:max_available_h, origin_local, dim] = values[:max_available_h].astype(np.float32)
    return preds


def deterministic_direction_crps(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direction_err, direction_mask = direction_error_deg(pred, target)
    return torch.where(direction_mask, direction_err.abs(), torch.full_like(direction_err, float("nan")))


def deterministic_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred_t = torch.from_numpy(pred.astype(np.float32, copy=False))
    target_t = torch.from_numpy(target.astype(np.float32, copy=False))
    return metric_block(
        pred_t,
        target_t,
        crps=torch.abs(pred_t - target_t),
        direction_crps=deterministic_direction_crps(pred_t, target_t),
    )


def improvement(model_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "rmse_percent": 100.0
        * (float(baseline_metrics["rmse"]) - float(model_metrics["rmse"]))
        / max(float(baseline_metrics["rmse"]), 1.0e-12),
        "mae_percent": 100.0
        * (float(baseline_metrics["mae"]) - float(model_metrics["mae"]))
        / max(float(baseline_metrics["mae"]), 1.0e-12),
    }


def build_train_eval_arrays(
    config: dict[str, Any],
    train_split: str,
    eval_split: str,
    train_fraction: float,
    train_limit: int | None,
    eval_limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    if train_split == eval_split:
        data = load_vector_dataset(config, split=train_split, time_limit=eval_limit)
        y_all = np.asarray(data["Y"], dtype=np.float32)
        nwp_all = np.asarray(data["nwp_baseline"], dtype=np.float32)
        train_len = int(round(y_all.shape[0] * train_fraction))
        train_len = min(max(train_len, 10), y_all.shape[0] - 2)
        if train_limit is not None:
            train_len = min(train_len, int(train_limit))
        return y_all[:train_len], y_all[train_len:], nwp_all[train_len:], train_len, data
    train_data = load_vector_dataset(config, split=train_split, time_limit=train_limit)
    eval_data = load_vector_dataset(config, split=eval_split, time_limit=eval_limit)
    train_y = np.asarray(train_data["Y"], dtype=np.float32)
    eval_y = np.asarray(eval_data["Y"], dtype=np.float32)
    eval_nwp = np.asarray(eval_data["nwp_baseline"], dtype=np.float32)
    return train_y, eval_y, eval_nwp, train_y.shape[0], eval_data


def run_benchmark(
    config_path: Path,
    output_dir: Path,
    train_split: str,
    eval_split: str,
    train_fraction: float,
    p: int,
    q: int,
    d: int,
    forecast_horizon: int,
    train_limit: int | None,
    eval_limit: int | None,
    max_iter: int,
    ridge: float,
    clip_quantile: float | None,
    no_plots: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    train_y, eval_y, eval_nwp, train_len, data = build_train_eval_arrays(
        config=config,
        train_split=train_split,
        eval_split=eval_split,
        train_fraction=train_fraction,
        train_limit=train_limit,
        eval_limit=eval_limit,
    )
    if eval_y.shape[0] <= 1:
        raise ValueError("Evaluation segment must contain at least two time steps.")
    max_horizon = min(int(forecast_horizon), eval_y.shape[0] - 1)
    models = fit_models(train_y, p=p, q=q, d=d, max_iter=max_iter, ridge=ridge)
    combined_y = np.concatenate([train_y, eval_y], axis=0).astype(np.float32, copy=False)
    clip_bounds = None
    if clip_quantile is not None and clip_quantile > 0.0:
        qlo = min(max(float(clip_quantile), 0.0), 0.49)
        qhi = 1.0 - qlo
        lower = np.nanquantile(train_y, qlo, axis=0)
        upper = np.nanquantile(train_y, qhi, axis=0)
        spread = np.nanstd(train_y, axis=0)
        lower = lower - spread
        upper = upper + spread
        clip_bounds = (lower.astype(np.float64), upper.astype(np.float64))
    arma_preds = rolling_forecasts(
        models,
        combined_y,
        train_len=train_len,
        eval_len=eval_y.shape[0],
        max_horizon=max_horizon,
        clip_bounds=clip_bounds,
    )

    curves = {
        "horizons": list(range(1, max_horizon + 1)),
        "curve_columns": DEFAULT_COLUMNS,
        "rmse_curve": [],
        "mae_curve": [],
        "crps_curve": [],
        "direction_rmse_curve": [],
        "direction_mae_curve": [],
        "direction_crps_curve": [],
    }
    multi_step: dict[str, Any] = {}
    for horizon in range(1, max_horizon + 1):
        n = eval_y.shape[0] - horizon
        target_h = eval_y[horizon:]
        arma_h = arma_preds[horizon - 1, :n]
        persistence_h = eval_y[:n]
        nwp_h = eval_nwp[horizon:]
        arma_metrics = deterministic_metrics(arma_h, target_h)
        persistence_metrics = deterministic_metrics(persistence_h, target_h)
        nwp_metrics = deterministic_metrics(nwp_h, target_h)
        curves["rmse_curve"].append([arma_metrics["rmse"], persistence_metrics["rmse"], nwp_metrics["rmse"]])
        curves["mae_curve"].append([arma_metrics["mae"], persistence_metrics["mae"], nwp_metrics["mae"]])
        curves["crps_curve"].append([arma_metrics["crps"], persistence_metrics["crps"], nwp_metrics["crps"]])
        curves["direction_rmse_curve"].append(
            [
                arma_metrics["direction_rmse_deg"],
                persistence_metrics["direction_rmse_deg"],
                nwp_metrics["direction_rmse_deg"],
            ]
        )
        curves["direction_mae_curve"].append(
            [
                arma_metrics["direction_mae_deg"],
                persistence_metrics["direction_mae_deg"],
                nwp_metrics["direction_mae_deg"],
            ]
        )
        curves["direction_crps_curve"].append(
            [
                arma_metrics["direction_crps_deg"],
                persistence_metrics["direction_crps_deg"],
                nwp_metrics["direction_crps_deg"],
            ]
        )
        multi_step[str(horizon)] = {
            "arma": arma_metrics,
            "persistence_baseline": persistence_metrics,
            "nwp_baseline": nwp_metrics,
            "arma_vs_persistence_improvement": improvement(arma_metrics, persistence_metrics),
            "arma_vs_nwp_improvement": improvement(arma_metrics, nwp_metrics),
        }

    model_name = "ARIMA" if d > 0 else "ARMA"
    results: dict[str, Any] = {
        "model_name": model_name,
        "order": {"p": p, "d": d, "q": q},
        "model": multi_step["1"]["arma"],
        "persistence_baseline": multi_step["1"]["persistence_baseline"],
        "nwp_baseline": multi_step["1"]["nwp_baseline"],
        "model_vs_persistence_improvement": multi_step["1"]["arma_vs_persistence_improvement"],
        "model_vs_nwp_improvement": multi_step["1"]["arma_vs_nwp_improvement"],
        "target_mode": "measurement",
        "train_split": train_split,
        "eval_split": eval_split,
        "train_length": int(train_y.shape[0]),
        "eval_length": int(eval_y.shape[0]),
        "forecast_horizon": max_horizon,
        "prediction_clip_quantile": clip_quantile,
        "multi_step": multi_step,
        "metric_note": (
            f"{model_name} is fit independently for each U/V state dimension with conditional least squares. "
            "CRPS is deterministic CRPS, equal to absolute error, because this benchmark does not estimate calibrated predictive distributions."
        ),
        "_curves": curves,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_horizon_csv(results, output_dir / "metrics_by_horizon.csv")
    write_hourly_csv(results, output_dir / "metrics_by_hour.csv")
    add_public_metric_curves(results)
    if not no_plots:
        plot_curves(results, output_dir)
    clean = strip_internal_fields(results)
    save_json(output_dir / "metrics_summary.json", clean)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARMA/ARIMA benchmarks on VectorMIDE station U/V data.")
    parser.add_argument("--config", type=Path, default=Path("yml_files/VectorMIDE_cuda.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/arma_benchmark"))
    parser.add_argument("--train-split", choices=["offline", "online"], default="offline")
    parser.add_argument("--eval-split", choices=["offline", "online"], default="online")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Used only when train and eval split are the same.")
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--d", type=int, default=0, help="0 gives ARMA; 1 or higher gives ARIMA.")
    parser.add_argument("--forecast-horizon", type=int, default=72)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.001,
        help="Clip predictions to [q,1-q] training quantiles plus one training std per dimension. Use 0 to disable.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    clean = run_benchmark(
        config_path=args.config,
        output_dir=args.output_dir,
        train_split=args.train_split,
        eval_split=args.eval_split,
        train_fraction=args.train_fraction,
        p=args.p,
        q=args.q,
        d=args.d,
        forecast_horizon=args.forecast_horizon,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        max_iter=args.max_iter,
        ridge=args.ridge,
        clip_quantile=args.clip_quantile,
        no_plots=args.no_plots,
    )
    print(json.dumps(clean, indent=2))
    print(f"Saved {clean['model_name']} benchmark metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
