from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


def station_speed(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.maximum(values[..., :3] ** 2 + values[..., 3:6] ** 2, 0.0))


def wind_direction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.arctan2(values[..., 3:6], values[..., :3])


def angle_diff_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = np.arctan2(np.sin(pred - target), np.cos(pred - target))
    return np.rad2deg(diff)


def uv_metrics_by_hour(prediction: np.ndarray, target: np.ndarray, horizons: np.ndarray, group_size: int = 6) -> dict[str, np.ndarray]:
    rows = []
    for h_idx, horizon in enumerate(horizons):
        h = int(horizon)
        n = min(target.shape[0] - h, prediction.shape[1] - h_idx + h_idx)
        if n <= 0 or h_idx >= prediction.shape[0]:
            continue
        pred_h = prediction[h_idx, :n]
        target_h = target[h:]
        mask = np.isfinite(pred_h) & np.isfinite(target_h)
        err = np.where(mask, pred_h - target_h, np.nan)
        rows.append(
            {
                "horizon": h,
                "hour": (h - 1) // group_size + 1,
                "rmse": float(np.sqrt(np.nanmean(err**2))),
                "mae": float(np.nanmean(np.abs(err))),
                "crps": float(np.nanmean(np.abs(err))),
            }
        )
    hours = sorted({row["hour"] for row in rows})
    out = {"forecast_hour": np.asarray(hours, dtype=int)}
    for metric in ("rmse", "mae", "crps"):
        out[metric] = np.asarray(
            [np.nanmean([row[metric] for row in rows if row["hour"] == hour]) for hour in hours],
            dtype=float,
        )
    return out


def direction_metrics_by_hour(prediction: np.ndarray, target: np.ndarray, horizons: np.ndarray, group_size: int = 6) -> dict[str, np.ndarray]:
    pred_dir = wind_direction(prediction)
    target_dir = wind_direction(target)
    rows = []
    for h_idx, horizon in enumerate(horizons):
        h = int(horizon)
        n = target.shape[0] - h
        if n <= 0 or h_idx >= pred_dir.shape[0]:
            continue
        diff = angle_diff_deg(pred_dir[h_idx, :n], target_dir[h:])
        mask = np.isfinite(diff)
        safe = np.where(mask, diff, np.nan)
        rows.append(
            {
                "horizon": h,
                "hour": (h - 1) // group_size + 1,
                "rmse": float(np.sqrt(np.nanmean(safe**2))),
                "mae": float(np.nanmean(np.abs(safe))),
                "crps": float(np.nanmean(np.abs(safe))),
                "cpe": float(np.nanmean(np.abs(np.sin(np.deg2rad(safe) / 2.0))) * 100.0),
            }
        )
    hours = sorted({row["hour"] for row in rows})
    out = {"forecast_hour": np.asarray(hours, dtype=int)}
    for metric in ("rmse", "mae", "crps", "cpe"):
        out[metric] = np.asarray(
            [np.nanmean([row[metric] for row in rows if row["hour"] == hour]) for hour in hours],
            dtype=float,
        )
    return out


def speed_mape_by_hour(prediction: np.ndarray, target: np.ndarray, horizons: np.ndarray, mape_eps: float, group_size: int = 6) -> np.ndarray:
    pred_speed = station_speed(prediction)
    target_speed = station_speed(target)
    rows = []
    for h_idx, horizon in enumerate(horizons):
        h = int(horizon)
        n = target.shape[0] - h
        if n <= 0 or h_idx >= pred_speed.shape[0]:
            continue
        pred_h = pred_speed[h_idx, :n]
        target_h = target_speed[h:]
        mask = np.isfinite(pred_h) & np.isfinite(target_h)
        denom = np.maximum(np.abs(target_h), mape_eps)
        mape = np.nanmean(np.where(mask, np.abs(pred_h - target_h) / denom, np.nan)) * 100.0
        rows.append({"hour": (h - 1) // group_size + 1, "mape": float(mape)})
    hours = sorted({row["hour"] for row in rows})
    return np.asarray([np.nanmean([row["mape"] for row in rows if row["hour"] == hour]) for hour in hours], dtype=float)


def read_existing_hourly_metrics(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hours = np.asarray([int(row["forecast_hour"]) for row in rows], dtype=int)
    methods = ["model", "persistence", "nwp"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        out[method] = {
            "forecast_hour": hours,
            "rmse": np.asarray([float(row[f"{method}_rmse"]) for row in rows], dtype=float),
            "mae": np.asarray([float(row[f"{method}_mae"]) for row in rows], dtype=float),
            "crps": np.asarray([float(row[f"{method}_crps"]) for row in rows], dtype=float),
        }
    return out


def read_existing_direction_hourly_metrics(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hours = np.asarray([int(row["forecast_hour"]) for row in rows], dtype=int)
    methods = ["model", "persistence", "nwp"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        out[method] = {
            "forecast_hour": hours,
            "rmse": np.asarray([float(row[f"{method}_direction_rmse_deg"]) for row in rows], dtype=float),
            "mae": np.asarray([float(row[f"{method}_direction_mae_deg"]) for row in rows], dtype=float),
            "crps": np.asarray([float(row[f"{method}_direction_crps_deg"]) for row in rows], dtype=float),
        }
    return out


def add_direction_cpe_from_vector_forecasts(metrics: dict[str, dict[str, np.ndarray]], vector_forecasts_path: Path) -> None:
    with np.load(vector_forecasts_path, allow_pickle=True) as data:
        target = np.asarray(data["target"], dtype=np.float64)
        horizons = np.asarray(data["horizons"], dtype=int)
        mapping = {
            "model": "multi_step_prediction",
            "persistence": "multi_step_persistence_prediction",
            "nwp": "multi_step_nwp_baseline_prediction",
        }
        for method, key in mapping.items():
            metrics[method]["cpe"] = direction_metrics_by_hour(
                np.asarray(data[key], dtype=np.float64),
                target,
                horizons,
            )["cpe"]


def add_mape_from_vector_forecasts(metrics: dict[str, dict[str, np.ndarray]], vector_forecasts_path: Path, mape_eps: float) -> None:
    with np.load(vector_forecasts_path, allow_pickle=True) as data:
        target = np.asarray(data["target"], dtype=np.float64)
        horizons = np.asarray(data["horizons"], dtype=int)
        mapping = {
            "model": "multi_step_prediction",
            "persistence": "multi_step_persistence_prediction",
            "nwp": "multi_step_nwp_baseline_prediction",
        }
        for method, key in mapping.items():
            metrics[method]["mape"] = speed_mape_by_hour(
                np.asarray(data[key], dtype=np.float64),
                target,
                horizons,
                mape_eps=mape_eps,
            )


def add_arima(metrics: dict[str, dict[str, np.ndarray]], arima_forecasts_path: Path, mape_eps: float) -> None:
    with np.load(arima_forecasts_path, allow_pickle=True) as data:
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        horizons = np.asarray(data["horizons"], dtype=int)
        name = str(data["model_name"].item()) if "model_name" in data else "ARIMA"
    arima = uv_metrics_by_hour(prediction, target, horizons)
    arima["mape"] = speed_mape_by_hour(prediction, target, horizons, mape_eps=mape_eps)
    metrics[name.upper()] = arima


def add_arima_direction(metrics: dict[str, dict[str, np.ndarray]], arima_forecasts_path: Path) -> None:
    with np.load(arima_forecasts_path, allow_pickle=True) as data:
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        horizons = np.asarray(data["horizons"], dtype=int)
        name = str(data["model_name"].item()) if "model_name" in data else "ARIMA"
    metrics[name.upper()] = direction_metrics_by_hour(prediction, target, horizons)


def write_csv(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "forecast_hour", "rmse", "mae", "crps", "mape_percent"])
        for model, values in metrics.items():
            hours = values["forecast_hour"]
            for i, hour in enumerate(hours):
                writer.writerow(
                    [
                        model,
                        int(hour),
                        float(values["rmse"][i]),
                        float(values["mae"][i]),
                        float(values["crps"][i]),
                        float(values["mape"][i]) if "mape" in values and i < len(values["mape"]) else "",
                    ]
                )


def write_direction_csv(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "forecast_hour",
                "direction_rmse_deg",
                "direction_mae_deg",
                "direction_crps_deg",
                "direction_cpe_percent",
            ]
        )
        for model, values in metrics.items():
            hours = values["forecast_hour"]
            for i, hour in enumerate(hours):
                writer.writerow(
                    [
                        model,
                        int(hour),
                        float(values["rmse"][i]),
                        float(values["mae"][i]),
                        float(values["crps"][i]),
                        float(values["cpe"][i]) if "cpe" in values and i < len(values["cpe"]) else "",
                    ]
                )


def plot(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"model": "VectorMIDE", "persistence": "persistence", "nwp": "NWP", "ARIMA": "ARIMA"}
    colors = {"model": "#1f77b4", "persistence": "#ff7f0e", "nwp": "#2ca02c", "ARIMA": "#9467bd"}
    markers = {"model": "o", "persistence": "s", "nwp": "^", "ARIMA": "D"}
    specs = [
        ("rmse", "Hourly Mean Speed RMSE", "Wind speed RMSE (m/s)"),
        ("mae", "Hourly Mean Speed MAE", "Wind speed MAE (m/s)"),
        ("crps", "Hourly Mean Speed CRPS", "Wind speed CRPS (m/s)"),
        ("mape", "Hourly Mean Speed MAPE", "Wind speed MAPE (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 10.2), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        for model, values in metrics.items():
            if metric not in values:
                continue
            x = values["forecast_hour"]
            y = values[metric]
            ax.plot(
                x,
                y,
                marker=markers.get(model, "o"),
                markersize=5.2,
                linewidth=2.4,
                color=colors.get(model),
                label=labels.get(model, model),
            )
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_xlabel("forecast hour", fontsize=16, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=14, width=1.4, length=5.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        legend = ax.legend(fontsize=12, frameon=True)
        for text in legend.get_texts():
            text.set_fontweight("bold")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_direction(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"model": "VectorMIDE", "persistence": "persistence", "nwp": "NWP", "ARIMA": "ARIMA"}
    colors = {"model": "#1f77b4", "persistence": "#ff7f0e", "nwp": "#2ca02c", "ARIMA": "#9467bd"}
    markers = {"model": "o", "persistence": "s", "nwp": "^", "ARIMA": "D"}
    specs = [
        ("rmse", "Hourly Mean Direction RMSE", "Wind direction RMSE (deg)"),
        ("mae", "Hourly Mean Direction MAE", "Wind direction MAE (deg)"),
        ("crps", "Hourly Mean Direction CRPS", "Wind direction CRPS (deg)"),
        ("cpe", "Hourly Mean Direction CPE", "Wind direction CPE (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 10.2), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        for model, values in metrics.items():
            if metric not in values:
                continue
            x = values["forecast_hour"]
            y = values[metric]
            ax.plot(
                x,
                y,
                marker=markers.get(model, "o"),
                markersize=5.2,
                linewidth=2.4,
                color=colors.get(model),
                label=labels.get(model, model),
            )
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("forecast hour", fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=13, width=1.4, length=5.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        legend = ax.legend(fontsize=11, frameon=True)
        for text in legend.get_texts():
            text.set_fontweight("bold")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine VectorMIDE hourly metrics with ARIMA and add MAPE.")
    parser.add_argument("--hourly-metrics", type=Path, required=True)
    parser.add_argument("--direction-hourly-metrics", type=Path, default=None)
    parser.add_argument("--vector-forecasts", type=Path, required=True)
    parser.add_argument("--arima-forecasts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mape-eps", type=float, default=1.0)
    args = parser.parse_args()

    metrics = read_existing_hourly_metrics(args.hourly_metrics)
    add_mape_from_vector_forecasts(metrics, args.vector_forecasts, mape_eps=args.mape_eps)
    add_arima(metrics, args.arima_forecasts, mape_eps=args.mape_eps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "hourly_metrics_with_arima_mape.csv", metrics)
    plot(args.output_dir / "hourly_metrics_with_arima_mape.png", metrics)
    if args.direction_hourly_metrics is not None:
        direction_metrics = read_existing_direction_hourly_metrics(args.direction_hourly_metrics)
        add_direction_cpe_from_vector_forecasts(direction_metrics, args.vector_forecasts)
        add_arima_direction(direction_metrics, args.arima_forecasts)
        write_direction_csv(args.output_dir / "hourly_direction_metrics_with_arima.csv", direction_metrics)
        plot_direction(args.output_dir / "hourly_direction_metrics_with_arima.png", direction_metrics)
    print(f"Wrote {args.output_dir / 'hourly_metrics_with_arima_mape.csv'}")
    print(f"Wrote {args.output_dir / 'hourly_metrics_with_arima_mape.png'}")
    if args.direction_hourly_metrics is not None:
        print(f"Wrote {args.output_dir / 'hourly_direction_metrics_with_arima.csv'}")
        print(f"Wrote {args.output_dir / 'hourly_direction_metrics_with_arima.png'}")


if __name__ == "__main__":
    main()
