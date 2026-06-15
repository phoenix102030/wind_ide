from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.vector_data_utils import load_vector_dataset
from train.evaluate_arma_benchmark import (
    difference,
    finite_fill,
    inverse_difference_forecast,
    least_squares_coeff,
)
from train.evaluate_vector import load_config
from train.train_vector_offline import print_device_info, resolve_device


STATE_DIM = 6
STATION_COUNT = 3
DEFAULT_BENCHMARKS = ["arimax", "deepar", "stgp", "convlstm", "gaussian_ide"]


def state_speed(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(values[..., :3] ** 2 + values[..., 3:] ** 2, 0.0))


def nan_json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(type(obj).__name__)


def save_forecasts(
    output_dir: Path,
    model_name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    nwp: np.ndarray,
    horizons: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "benchmark_forecasts.npz",
        prediction=prediction.astype(np.float32, copy=False),
        target=target.astype(np.float32, copy=False),
        persistence=target.astype(np.float32, copy=False),
        nwp=nwp.astype(np.float32, copy=False),
        model_name=np.asarray(model_name),
        curve_columns=np.asarray([model_name.lower(), "persistence", "nwp"]),
        horizons=horizons.astype(np.int64, copy=False),
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=nan_json_default) + "\n")


def build_train_eval_data(
    config: dict[str, Any],
    train_split: str,
    eval_split: str,
    train_limit: int | None,
    eval_limit: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_data = load_vector_dataset(config, split=train_split, time_limit=train_limit)
    eval_data = load_vector_dataset(config, split=eval_split, time_limit=eval_limit)
    return train_data, eval_data


def standardize_train(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1.0e-6), std, 1.0).astype(np.float32)
    return ((values - mean) / std).astype(np.float32), mean, std


def apply_standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def inverse_standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values * std + mean).astype(np.float32)


def robust_clip_bounds(values: np.ndarray, margin: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    lower = np.nanpercentile(values, 0.5, axis=0)
    upper = np.nanpercentile(values, 99.5, axis=0)
    span = np.maximum(upper - lower, 1.0)
    return (lower - margin * span).astype(np.float32), (upper + margin * span).astype(np.float32)


# ---------------------------------------------------------------------------
# ARIMAX


@dataclass
class ARIMAX1D:
    p: int
    q: int
    d: int
    intercept: float
    ar: np.ndarray
    ma: np.ndarray
    beta: np.ndarray
    residuals: np.ndarray
    y_mean: float
    y_std: float
    x_mean: np.ndarray
    x_std: np.ndarray


def arimax_exog(raw_y: np.ndarray, raw_x: np.ndarray, d: int) -> np.ndarray:
    raw_y = finite_fill(raw_y)
    raw_x = finite_fill(raw_x)
    if d > 0:
        x_delta = difference(raw_x, d)
        x_level = raw_x[d:]
    else:
        x_delta = np.zeros_like(raw_x)
        x_level = raw_x
    return np.stack([x_level, x_delta], axis=1).astype(np.float64)


def arimax_design(
    y: np.ndarray,
    exog: np.ndarray,
    residuals: np.ndarray,
    p: int,
    q: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_lag = max(p, q, 1)
    rows: list[list[float]] = []
    target: list[float] = []
    for t in range(max_lag, y.size):
        row = [1.0]
        row.extend(float(y[t - lag]) for lag in range(1, p + 1))
        row.extend(float(residuals[t - lag]) for lag in range(1, q + 1))
        row.extend(float(v) for v in exog[t])
        rows.append(row)
        target.append(float(y[t]))
    if not rows:
        return np.empty((0, 1 + p + q + exog.shape[1]), dtype=np.float64), np.empty((0,), dtype=np.float64)
    X = np.asarray(rows, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(z) & np.isfinite(X).all(axis=1)
    return X[finite], z[finite]


def arimax_residuals(y: np.ndarray, exog: np.ndarray, model: ARIMAX1D) -> np.ndarray:
    max_lag = max(model.p, model.q, 1)
    residuals = np.zeros(y.size, dtype=np.float64)
    for t in range(max_lag, y.size):
        pred = float(model.intercept)
        for lag in range(1, model.p + 1):
            pred += float(model.ar[lag - 1]) * float(y[t - lag])
        for lag in range(1, model.q + 1):
            pred += float(model.ma[lag - 1]) * float(residuals[t - lag])
        pred += float(np.dot(model.beta, exog[t]))
        residuals[t] = float(y[t]) - pred
    return residuals


def fit_arimax_1d(
    y_values: np.ndarray,
    x_values: np.ndarray,
    p: int,
    q: int,
    d: int,
    max_iter: int,
    ridge: float,
) -> ARIMAX1D:
    raw_y = finite_fill(y_values)
    y_diff = difference(raw_y, d)
    y_mean = float(np.nanmean(y_diff))
    y_std = float(np.nanstd(y_diff))
    if not np.isfinite(y_std) or y_std < 1.0e-8:
        y_std = 1.0
    y = (y_diff - y_mean) / y_std
    raw_exog = arimax_exog(raw_y, x_values, d)
    x_mean = np.nanmean(raw_exog, axis=0)
    x_std = np.nanstd(raw_exog, axis=0)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1.0e-8), x_std, 1.0)
    exog = (raw_exog - x_mean) / x_std
    residuals = np.zeros_like(y)
    coeff = np.zeros(1 + p + q + exog.shape[1], dtype=np.float64)
    for _ in range(max(1, max_iter)):
        X, target = arimax_design(y, exog, residuals, p, q)
        if X.size == 0:
            break
        coeff = least_squares_coeff(X, target, ridge)
        temp = ARIMAX1D(
            p,
            q,
            d,
            float(coeff[0]),
            coeff[1 : 1 + p].copy(),
            coeff[1 + p : 1 + p + q].copy(),
            coeff[1 + p + q :].copy(),
            residuals,
            y_mean,
            y_std,
            x_mean,
            x_std,
        )
        residuals = arimax_residuals(y, exog, temp)
        if not np.isfinite(residuals).all() or np.nanmax(np.abs(residuals)) > 1.0e6:
            break
    model = ARIMAX1D(
        p,
        q,
        d,
        float(coeff[0]) if np.isfinite(coeff[0]) else 0.0,
        np.nan_to_num(coeff[1 : 1 + p].copy()),
        np.nan_to_num(coeff[1 + p : 1 + p + q].copy()),
        np.nan_to_num(coeff[1 + p + q :].copy()),
        residuals,
        y_mean,
        y_std,
        x_mean,
        x_std,
    )
    model.residuals = arimax_residuals(y, exog, model)
    return model


def arimax_predict_1d(
    model: ARIMAX1D,
    observed_y: np.ndarray,
    observed_x: np.ndarray,
    origin_idx: int,
    horizon: int,
) -> np.ndarray:
    raw_y = finite_fill(observed_y)
    y_diff = difference(raw_y, model.d)
    y = (y_diff - model.y_mean) / model.y_std
    raw_exog = arimax_exog(raw_y, observed_x, model.d)
    exog = (raw_exog - model.x_mean) / model.x_std
    residuals = arimax_residuals(y, exog, model)
    diff_origin = origin_idx - model.d
    if diff_origin < 0:
        return np.full(horizon, raw_y[origin_idx], dtype=np.float64)

    y_hist = [float(v) for v in y[: diff_origin + 1]]
    e_hist = [float(v) for v in residuals[: diff_origin + 1]]
    diff_pred: list[float] = []
    for step in range(1, horizon + 1):
        target_idx = origin_idx + step
        if model.d > 0:
            level = observed_x[target_idx]
            delta = observed_x[target_idx] - observed_x[target_idx - model.d]
        else:
            level = observed_x[target_idx]
            delta = 0.0
        x_vec = np.asarray([level, delta], dtype=np.float64)
        x_vec = (x_vec - model.x_mean) / model.x_std
        pred = float(model.intercept)
        for lag in range(1, model.p + 1):
            pred += float(model.ar[lag - 1]) * y_hist[-lag]
        for lag in range(1, model.q + 1):
            pred += float(model.ma[lag - 1]) * e_hist[-lag]
        pred += float(np.dot(model.beta, x_vec))
        y_hist.append(pred)
        e_hist.append(0.0)
        diff_pred.append(pred * model.y_std + model.y_mean)
    return inverse_difference_forecast(raw_y[: origin_idx + 1], np.asarray(diff_pred), model.d)


def run_arimax(
    train_data: dict[str, Any],
    eval_data: dict[str, Any],
    horizons: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    train_y = train_data["Y"].astype(np.float32)
    train_nwp = train_data["nwp_baseline"].astype(np.float32)
    eval_y = eval_data["Y"].astype(np.float32)
    eval_nwp = eval_data["nwp_baseline"].astype(np.float32)
    models = [
        fit_arimax_1d(
            train_y[:, dim],
            train_nwp[:, dim],
            p=args.arimax_p,
            q=args.arimax_q,
            d=args.arimax_d,
            max_iter=args.arimax_iter,
            ridge=args.ridge,
        )
        for dim in range(STATE_DIM)
    ]
    combined_y = np.concatenate([train_y, eval_y], axis=0)
    combined_nwp = np.concatenate([train_nwp, eval_nwp], axis=0)
    train_len = train_y.shape[0]
    clip_lo, clip_hi = robust_clip_bounds(train_y)
    pred = np.full((len(horizons), eval_y.shape[0], STATE_DIM), np.nan, dtype=np.float32)
    for origin in range(eval_y.shape[0]):
        global_origin = train_len + origin
        max_h = min(int(horizons[-1]), eval_y.shape[0] - origin - 1)
        if max_h <= 0:
            continue
        for dim, model in enumerate(models):
            values = arimax_predict_1d(model, combined_y[:, dim], combined_nwp[:, dim], global_origin, max_h)
            values = np.nan_to_num(values[:max_h], nan=float(train_y[:, dim].mean()), posinf=float(clip_hi[dim]), neginf=float(clip_lo[dim]))
            pred[:max_h, origin, dim] = np.clip(values, clip_lo[dim], clip_hi[dim])
    save_forecasts(
        output_dir,
        "ARIMAX",
        pred,
        eval_y,
        eval_nwp,
        horizons,
        {
            "benchmark": "ARIMAX",
            "order": {"p": args.arimax_p, "d": args.arimax_d, "q": args.arimax_q},
            "exogenous": "station nearest-grid NWP U/V component, using level and differenced exogenous terms",
            "prediction_clip": {"lower": clip_lo, "upper": clip_hi, "source": "train 0.5/99.5 percentiles with 50% margin"},
        },
    )


# ---------------------------------------------------------------------------
# STGP residual smoother


def gaussian_spatial_kernel(coords: np.ndarray, ell: float) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(delta * delta, axis=-1)
    K = np.exp(-0.5 * d2 / max(float(ell) ** 2, 1.0e-8))
    return K / np.maximum(K.sum(axis=1, keepdims=True), 1.0e-12)


def stgp_predict_residuals(
    residual: np.ndarray,
    origin: int,
    horizon: int,
    coords: np.ndarray,
    ell_space: float,
    tau_time: float,
    history: int,
) -> np.ndarray:
    K = gaussian_spatial_kernel(coords, ell_space)
    out = np.zeros(STATE_DIM, dtype=np.float64)
    time_weights = []
    source_values = []
    for lag in range(history):
        idx = origin - lag
        if idx < 0:
            break
        dt = horizon + lag
        w = math.exp(-0.5 * (dt * dt) / max(float(tau_time) ** 2, 1.0e-8))
        time_weights.append(w)
        source_values.append(residual[idx])
    if not source_values:
        return out
    weights = np.asarray(time_weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1.0e-12)
    src = np.stack(source_values, axis=0)
    for comp_offset in (0, 3):
        comp = np.zeros(STATION_COUNT, dtype=np.float64)
        for lag_idx, w in enumerate(weights):
            comp += w * (K @ src[lag_idx, comp_offset : comp_offset + STATION_COUNT])
        out[comp_offset : comp_offset + STATION_COUNT] = comp
    return out


def stgp_forecasts(
    y: np.ndarray,
    nwp: np.ndarray,
    coords: np.ndarray,
    horizons: np.ndarray,
    ell_space: float,
    tau_time: float,
    history: int,
) -> np.ndarray:
    residual = y - nwp
    pred = np.full((len(horizons), y.shape[0], STATE_DIM), np.nan, dtype=np.float32)
    for origin in range(y.shape[0]):
        max_h = min(int(horizons[-1]), y.shape[0] - origin - 1)
        for h in range(1, max_h + 1):
            pred[h - 1, origin] = nwp[origin + h] + stgp_predict_residuals(
                residual, origin, h, coords, ell_space, tau_time, history
            )
    return pred


def choose_stgp_hyperparams(train_y: np.ndarray, train_nwp: np.ndarray, coords: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    n = train_y.shape[0]
    val_size = min(args.stgp_val_size, max(n // 5, 10))
    if n <= val_size + 3:
        return float(args.stgp_ell_grid[0]), float(args.stgp_tau_grid[0])
    train_end = n - val_size
    origins = np.arange(train_end, n - 1)
    if origins.size > args.stgp_max_val_origins:
        origins = origins[np.linspace(0, origins.size - 1, args.stgp_max_val_origins).astype(int)]
    residual = train_y - train_nwp
    best = (float("inf"), float(args.stgp_ell_grid[0]), float(args.stgp_tau_grid[0]))
    val_horizons = [h for h in args.stgp_val_horizons if h < val_size]
    if not val_horizons:
        val_horizons = [1]
    for ell in args.stgp_ell_grid:
        for tau in args.stgp_tau_grid:
            sqerr = []
            for origin in origins:
                for h in val_horizons:
                    if origin + h >= n:
                        continue
                    pred_res = stgp_predict_residuals(residual, int(origin), int(h), coords, ell, tau, args.stgp_history)
                    pred = train_nwp[origin + h] + pred_res
                    sqerr.append(np.nanmean((pred - train_y[origin + h]) ** 2))
            score = float(np.nanmean(sqerr)) if sqerr else float("inf")
            if score < best[0]:
                best = (score, float(ell), float(tau))
    return best[1], best[2]


def run_stgp(train_data: dict[str, Any], eval_data: dict[str, Any], horizons: np.ndarray, args: argparse.Namespace, output_dir: Path) -> None:
    train_y = train_data["Y"].astype(np.float32)
    train_nwp = train_data["nwp_baseline"].astype(np.float32)
    eval_y = eval_data["Y"].astype(np.float32)
    eval_nwp = eval_data["nwp_baseline"].astype(np.float32)
    coords = train_data["coords"].astype(np.float64)
    ell, tau = choose_stgp_hyperparams(train_y, train_nwp, coords, args)
    combined_y = np.concatenate([train_y, eval_y], axis=0)
    combined_nwp = np.concatenate([train_nwp, eval_nwp], axis=0)
    combined_pred = stgp_forecasts(combined_y, combined_nwp, coords, horizons, ell, tau, args.stgp_history)
    pred = combined_pred[:, train_y.shape[0] : train_y.shape[0] + eval_y.shape[0]]
    save_forecasts(
        output_dir,
        "STGP",
        pred,
        eval_y,
        eval_nwp,
        horizons,
        {
            "benchmark": "STGP",
            "description": "Gaussian spatio-temporal residual smoother using station distance and temporal lag kernels.",
            "selected_ell_space_km": ell,
            "selected_tau_steps": tau,
            "history": args.stgp_history,
        },
    )


# ---------------------------------------------------------------------------
# Gaussian IDE with only ell learnable


class SimpleGaussianIDE(nn.Module):
    def __init__(self, coords: np.ndarray, ell_init: float, ell_min: float, ell_max: float) -> None:
        super().__init__()
        self.register_buffer("coords", torch.as_tensor(coords, dtype=torch.float32))
        self.ell_min = float(ell_min)
        self.ell_max = float(ell_max)
        frac = (float(ell_init) - self.ell_min) / max(self.ell_max - self.ell_min, 1.0e-6)
        frac = min(max(frac, 1.0e-5), 1.0 - 1.0e-5)
        self.raw_ell = nn.Parameter(torch.tensor(math.log(frac / (1.0 - frac)), dtype=torch.float32))

    def ell(self) -> torch.Tensor:
        return self.ell_min + (self.ell_max - self.ell_min) * torch.sigmoid(self.raw_ell)

    def matrix(self) -> torch.Tensor:
        delta = self.coords[:, None, :] - self.coords[None, :, :]
        d2 = (delta * delta).sum(dim=-1)
        K = torch.exp(-0.5 * d2 / self.ell().pow(2).clamp_min(1.0e-8))
        K = K / K.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        zero = torch.zeros_like(K)
        return torch.cat([torch.cat([K, zero], dim=1), torch.cat([zero, K], dim=1)], dim=0)


def train_gaussian_ide(train_y: np.ndarray, train_nwp: np.ndarray, coords: np.ndarray, args: argparse.Namespace, device: torch.device) -> SimpleGaussianIDE:
    model = SimpleGaussianIDE(coords, args.ide_ell_init, args.ide_ell_min, args.ide_ell_max).to(device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    nwp = torch.as_tensor(train_nwp, dtype=torch.float32, device=device)
    residual = y - nwp
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ide_lr)
    horizons = [h for h in args.ide_train_horizons if h < train_y.shape[0]]
    if not horizons:
        horizons = [1]
    origins = torch.arange(0, train_y.shape[0] - max(horizons), device=device)
    if origins.numel() > args.ide_max_origins:
        idx = torch.linspace(0, origins.numel() - 1, args.ide_max_origins, device=device).long()
        origins = origins[idx]
    for _ in range(args.ide_steps):
        A = model.matrix()
        loss_terms = []
        for h in horizons:
            Ah = torch.linalg.matrix_power(A, int(h))
            pred_res = residual[origins] @ Ah.T
            pred = nwp[origins + int(h)] + pred_res
            loss_terms.append(F.mse_loss(pred, y[origins + int(h)]))
        loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def run_gaussian_ide(
    train_data: dict[str, Any],
    eval_data: dict[str, Any],
    horizons: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> None:
    train_y = train_data["Y"].astype(np.float32)
    train_nwp = train_data["nwp_baseline"].astype(np.float32)
    eval_y = eval_data["Y"].astype(np.float32)
    eval_nwp = eval_data["nwp_baseline"].astype(np.float32)
    coords = train_data["coords"].astype(np.float32)
    model = train_gaussian_ide(train_y, train_nwp, coords, args, device)
    with torch.no_grad():
        A = model.matrix().detach().cpu().numpy()
    residual = eval_y - eval_nwp
    pred = np.full((len(horizons), eval_y.shape[0], STATE_DIM), np.nan, dtype=np.float32)
    for h in horizons:
        Ah = np.linalg.matrix_power(A, int(h))
        n = eval_y.shape[0] - int(h)
        if n <= 0:
            continue
        pred[int(h) - 1, :n] = eval_nwp[int(h) :] + residual[:n] @ Ah.T
    save_forecasts(
        output_dir,
        "GaussianIDE",
        pred,
        eval_y,
        eval_nwp,
        horizons,
        {
            "benchmark": "GaussianIDE",
            "description": "Residual IDE using only a stationary Gaussian station kernel; only ell is learned.",
            "learned_ell_km": float(model.ell().detach().cpu()),
            "train_horizons": horizons.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# DeepAR


class DeepARNet(nn.Module):
    def __init__(self, input_dim: int = 12, state_dim: int = STATE_DIM, hidden_dim: int = 64, layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.mean_head = nn.Linear(hidden_dim, state_dim)
        self.log_scale_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, hidden = self.rnn(x, hidden)
        mean = self.mean_head(out)
        log_scale = self.log_scale_head(out).clamp(-6.0, 3.0)
        return mean, log_scale, hidden


def gaussian_nll(mean: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = torch.exp(log_scale).clamp_min(1.0e-5)
    return (0.5 * ((target - mean) / scale).pow(2) + log_scale).mean()


def train_deepar(train_y: np.ndarray, train_nwp: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[DeepARNet, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_std, y_mean, y_scale = standardize_train(train_y)
    x_std, x_mean, x_scale = standardize_train(train_nwp)
    model = DeepARNet(hidden_dim=args.deepar_hidden, layers=args.deepar_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.deepar_lr, weight_decay=args.weight_decay)
    y = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    x = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    max_start = train_y.shape[0] - args.context_length - 1
    if max_start <= 0:
        raise ValueError("Training series is too short for DeepAR context_length.")
    for epoch in range(args.deepar_epochs):
        model.train()
        losses = []
        steps = max(1, args.deepar_steps_per_epoch)
        for _ in range(steps):
            starts = torch.randint(0, max_start, (args.batch_size,), device=device)
            seq_inputs = []
            seq_targets = []
            for start in starts.tolist():
                prev_y = y[start : start + args.context_length]
                future_x = x[start + 1 : start + args.context_length + 1]
                seq_inputs.append(torch.cat([prev_y, future_x], dim=1))
                seq_targets.append(y[start + 1 : start + args.context_length + 1])
            inp = torch.stack(seq_inputs, dim=0)
            target = torch.stack(seq_targets, dim=0)
            mean, log_scale, _ = model(inp)
            loss = gaussian_nll(mean, log_scale, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"DeepAR epoch {epoch + 1}/{args.deepar_epochs}: loss={np.mean(losses):.6g}")
    return model, y_mean, y_scale, x_mean, x_scale


def deepar_forecast(
    model: DeepARNet,
    eval_y: np.ndarray,
    eval_nwp: np.ndarray,
    horizons: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    y = apply_standardize(eval_y, y_mean, y_scale)
    x = apply_standardize(eval_nwp, x_mean, x_scale)
    pred = np.full((len(horizons), eval_y.shape[0], STATE_DIM), np.nan, dtype=np.float32)
    max_h = int(horizons[-1])
    with torch.no_grad():
        for start in range(0, eval_y.shape[0], args.eval_batch_size):
            end = min(start + args.eval_batch_size, eval_y.shape[0])
            origins = np.arange(start, end)
            context_inputs = []
            valid_origins = []
            for origin in origins:
                if origin + 1 >= eval_y.shape[0]:
                    continue
                ctx_idx = np.arange(origin - args.context_length + 1, origin + 1)
                ctx_idx = np.clip(ctx_idx, 0, eval_y.shape[0] - 1)
                context_inputs.append(np.concatenate([y[ctx_idx], x[ctx_idx]], axis=1))
                valid_origins.append(origin)
            if not context_inputs:
                continue
            inp = torch.as_tensor(np.stack(context_inputs), dtype=torch.float32, device=device)
            _, _, hidden = model(inp)
            prev = torch.as_tensor(y[valid_origins], dtype=torch.float32, device=device)
            for h in range(1, max_h + 1):
                active = np.asarray([origin + h < eval_y.shape[0] for origin in valid_origins], dtype=bool)
                if not active.any():
                    break
                exog = np.stack([x[min(origin + h, eval_y.shape[0] - 1)] for origin in valid_origins])
                step_in = torch.as_tensor(np.concatenate([prev.detach().cpu().numpy(), exog], axis=1), dtype=torch.float32, device=device).unsqueeze(1)
                mean, _, hidden = model(step_in, hidden)
                prev = mean[:, -1]
                out = inverse_standardize(prev.detach().cpu().numpy(), y_mean, y_scale)
                for row_idx, origin in enumerate(valid_origins):
                    if origin + h < eval_y.shape[0]:
                        pred[h - 1, origin] = out[row_idx]
    return pred


def run_deepar(train_data: dict[str, Any], eval_data: dict[str, Any], horizons: np.ndarray, args: argparse.Namespace, output_dir: Path, device: torch.device) -> None:
    train_y = train_data["Y"].astype(np.float32)
    train_nwp = train_data["nwp_baseline"].astype(np.float32)
    eval_y = eval_data["Y"].astype(np.float32)
    eval_nwp = eval_data["nwp_baseline"].astype(np.float32)
    model, y_mean, y_scale, x_mean, x_scale = train_deepar(train_y, train_nwp, args, device)
    pred = deepar_forecast(model, eval_y, eval_nwp, horizons, y_mean, y_scale, x_mean, x_scale, args, device)
    torch.save(
        {
            "model_state": model.state_dict(),
            "y_mean": y_mean,
            "y_scale": y_scale,
            "x_mean": x_mean,
            "x_scale": x_scale,
            "args": vars(args),
        },
        output_dir / "deepar.pt",
    )
    save_forecasts(
        output_dir,
        "DeepAR",
        pred,
        eval_y,
        eval_nwp,
        horizons,
        {"benchmark": "DeepAR", "context_length": args.context_length, "epochs": args.deepar_epochs},
    )


# ---------------------------------------------------------------------------
# ConvLSTM


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            h = x.new_zeros((x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]))
            c = x.new_zeros((x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]))
        else:
            h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMStationNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, station_indices: np.ndarray) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels)
        self.register_buffer("station_indices", torch.as_tensor(station_indices, dtype=torch.long))
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * STATION_COUNT + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, STATE_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = None
        for t in range(x.shape[1]):
            state = self.cell(x[:, t], state)
        h, _ = state
        pooled = h.mean(dim=(2, 3))
        station_feats = []
        for idx in self.station_indices:
            station_feats.append(h[:, :, idx[0], idx[1]])
        feats = torch.cat([*station_feats, pooled], dim=1)
        return self.head(feats)


def window_batch(X: np.ndarray, indices: np.ndarray, context: int) -> np.ndarray:
    windows = []
    for target_idx in indices:
        idx = np.arange(int(target_idx) - context + 1, int(target_idx) + 1)
        idx = np.clip(idx, 0, X.shape[0] - 1)
        windows.append(X[idx])
    return np.stack(windows, axis=0).astype(np.float32)


def train_convlstm(train_data: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[ConvLSTMStationNet, np.ndarray, np.ndarray]:
    X = train_data["X"].astype(np.float32)
    y_raw = train_data["Y"].astype(np.float32)
    y, y_mean, y_scale = standardize_train(y_raw)
    model = ConvLSTMStationNet(X.shape[1], args.convlstm_hidden, train_data["baseline_grid_indices"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.convlstm_lr, weight_decay=args.weight_decay)
    max_idx = X.shape[0] - 1
    for epoch in range(args.convlstm_epochs):
        model.train()
        losses = []
        for _ in range(args.convlstm_steps_per_epoch):
            idx = np.random.randint(0, max_idx + 1, size=args.batch_size)
            xb = torch.as_tensor(window_batch(X, idx, args.context_length), dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"ConvLSTM epoch {epoch + 1}/{args.convlstm_epochs}: loss={np.mean(losses):.6g}")
    return model, y_mean, y_scale


def convlstm_predict_by_time(model: ConvLSTMStationNet, X: np.ndarray, y_mean: np.ndarray, y_scale: np.ndarray, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    pred = np.full((X.shape[0], STATE_DIM), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], args.eval_batch_size):
            end = min(start + args.eval_batch_size, X.shape[0])
            idx = np.arange(start, end)
            xb = torch.as_tensor(window_batch(X, idx, args.context_length), dtype=torch.float32, device=device)
            out = model(xb).detach().cpu().numpy()
            pred[start:end] = inverse_standardize(out, y_mean, y_scale)
    return pred


def run_convlstm(train_data: dict[str, Any], eval_data: dict[str, Any], horizons: np.ndarray, args: argparse.Namespace, output_dir: Path, device: torch.device) -> None:
    eval_y = eval_data["Y"].astype(np.float32)
    eval_nwp = eval_data["nwp_baseline"].astype(np.float32)
    model, y_mean, y_scale = train_convlstm(train_data, args, device)
    pred_by_time = convlstm_predict_by_time(model, eval_data["X"].astype(np.float32), y_mean, y_scale, args, device)
    pred = np.full((len(horizons), eval_y.shape[0], STATE_DIM), np.nan, dtype=np.float32)
    for h in horizons:
        n = eval_y.shape[0] - int(h)
        if n > 0:
            pred[int(h) - 1, :n] = pred_by_time[int(h) :]
    torch.save({"model_state": model.state_dict(), "y_mean": y_mean, "y_scale": y_scale, "args": vars(args)}, output_dir / "convlstm.pt")
    save_forecasts(
        output_dir,
        "ConvLSTM",
        pred,
        eval_y,
        eval_nwp,
        horizons,
        {"benchmark": "ConvLSTM", "context_length": args.context_length, "epochs": args.convlstm_epochs},
    )


# ---------------------------------------------------------------------------
# CLI


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def benchmark_output_dir(root: Path, name: str) -> Path:
    return root / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run additional VectorMIDE benchmark models.")
    parser.add_argument("--config", type=Path, default=Path("yml_files/VectorMIDE_cuda.yaml"))
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS, help=f"Subset of {DEFAULT_BENCHMARKS} or 'all'.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/additional_benchmarks"))
    parser.add_argument("--train-split", choices=["offline", "online"], default="offline")
    parser.add_argument("--eval-split", choices=["offline", "online"], default="online")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--forecast-horizon", type=int, default=72)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=36)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ridge", type=float, default=1.0e-5)

    parser.add_argument("--arimax-p", type=int, default=6)
    parser.add_argument("--arimax-d", type=int, default=1)
    parser.add_argument("--arimax-q", type=int, default=1)
    parser.add_argument("--arimax-iter", type=int, default=6)

    parser.add_argument("--stgp-history", type=int, default=72)
    parser.add_argument("--stgp-ell-grid", type=parse_float_list, default=parse_float_list("5,10,25,50,100,200"))
    parser.add_argument("--stgp-tau-grid", type=parse_float_list, default=parse_float_list("3,6,12,24,48,96"))
    parser.add_argument("--stgp-val-horizons", type=parse_int_list, default=parse_int_list("1,6,12,24,36,72"))
    parser.add_argument("--stgp-val-size", type=int, default=2000)
    parser.add_argument("--stgp-max-val-origins", type=int, default=256)

    parser.add_argument("--ide-ell-init", type=float, default=50.0)
    parser.add_argument("--ide-ell-min", type=float, default=1.0)
    parser.add_argument("--ide-ell-max", type=float, default=500.0)
    parser.add_argument("--ide-lr", type=float, default=0.05)
    parser.add_argument("--ide-steps", type=int, default=300)
    parser.add_argument("--ide-max-origins", type=int, default=4096)
    parser.add_argument("--ide-train-horizons", type=parse_int_list, default=parse_int_list("1,6,12,24,36,72"))

    parser.add_argument("--deepar-hidden", type=int, default=64)
    parser.add_argument("--deepar-layers", type=int, default=1)
    parser.add_argument("--deepar-lr", type=float, default=1.0e-3)
    parser.add_argument("--deepar-epochs", type=int, default=20)
    parser.add_argument("--deepar-steps-per-epoch", type=int, default=200)

    parser.add_argument("--convlstm-hidden", type=int, default=16)
    parser.add_argument("--convlstm-lr", type=float, default=1.0e-3)
    parser.add_argument("--convlstm-epochs", type=int, default=10)
    parser.add_argument("--convlstm-steps-per-epoch", type=int, default=200)
    args = parser.parse_args()

    requested = [x.lower() for x in args.benchmarks]
    if "all" in requested:
        requested = DEFAULT_BENCHMARKS
    unknown = sorted(set(requested) - set(DEFAULT_BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {unknown}; expected {DEFAULT_BENCHMARKS}")

    config = load_config(args.config)
    device = resolve_device(args.device, allow_fallback=True)
    print_device_info(device)
    train_data, eval_data = build_train_eval_data(config, args.train_split, args.eval_split, args.train_limit, args.eval_limit)
    max_h = min(int(args.forecast_horizon), eval_data["Y"].shape[0] - 1)
    horizons = np.arange(1, max_h + 1, dtype=np.int64)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "config": str(args.config),
        "benchmarks": requested,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_time_steps": int(train_data["Y"].shape[0]),
        "eval_time_steps": int(eval_data["Y"].shape[0]),
        "forecast_horizon": int(max_h),
    }
    (args.output_root / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")

    for name in requested:
        out = benchmark_output_dir(args.output_root, name)
        out.mkdir(parents=True, exist_ok=True)
        print(f"Running benchmark {name} -> {out}")
        if name == "arimax":
            run_arimax(train_data, eval_data, horizons, args, out)
        elif name == "stgp":
            run_stgp(train_data, eval_data, horizons, args, out)
        elif name == "gaussian_ide":
            run_gaussian_ide(train_data, eval_data, horizons, args, out, device)
        elif name == "deepar":
            run_deepar(train_data, eval_data, horizons, args, out, device)
        elif name == "convlstm":
            run_convlstm(train_data, eval_data, horizons, args, out, device)
        print(f"Saved {name} benchmark to {out / 'benchmark_forecasts.npz'}")


if __name__ == "__main__":
    main()
