from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


def station_speed(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.maximum(values[..., :3] ** 2 + values[..., 3:6] ** 2, 0.0))


def load_forecast(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        model_name = str(data["model_name"].item()) if "model_name" in data else path.stem
        return {
            "path": path,
            "model_name": model_name,
            "prediction": np.asarray(data["prediction"], dtype=np.float64),
            "target": np.asarray(data["target"], dtype=np.float64),
            "horizons": np.asarray(data["horizons"], dtype=int)
            if "horizons" in data
            else np.arange(1, np.asarray(data["prediction"]).shape[0] + 1, dtype=int),
        }


def speed_metrics(prediction: np.ndarray, target: np.ndarray, horizons: np.ndarray, mape_eps: float) -> list[dict[str, float]]:
    pred_speed = station_speed(prediction)
    target_speed = station_speed(target)
    rows: list[dict[str, float]] = []
    for h_idx, horizon in enumerate(horizons):
        n = target.shape[0] - int(horizon)
        if n <= 0 or h_idx >= pred_speed.shape[0]:
            continue
        pred_h = pred_speed[h_idx, :n]
        target_h = target_speed[int(horizon) :]
        mask = np.isfinite(pred_h) & np.isfinite(target_h)
        err = pred_h - target_h
        denom = np.maximum(np.abs(target_h), mape_eps)
        rows.append(
            {
                "horizon": int(horizon),
                "rmse": float(np.sqrt(np.nanmean(np.where(mask, err**2, np.nan)))),
                "mae": float(np.nanmean(np.where(mask, np.abs(err), np.nan))),
                "crps": float(np.nanmean(np.where(mask, np.abs(err), np.nan))),
                "mape_percent": float(np.nanmean(np.where(mask, np.abs(err) / denom, np.nan)) * 100.0),
            }
        )
    return rows


def write_csv(path: Path, all_rows: dict[str, list[dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "horizon", "rmse", "mae", "crps", "mape_percent"])
        for model, rows in all_rows.items():
            for row in rows:
                writer.writerow(
                    [
                        model,
                        row["horizon"],
                        row["rmse"],
                        row["mae"],
                        row["crps"],
                        row["mape_percent"],
                    ]
                )


def plot_metrics(path: Path, all_rows: dict[str, list[dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_specs = [
        ("rmse", "Speed RMSE", "Wind speed RMSE (m/s)"),
        ("mae", "Speed MAE", "Wind speed MAE (m/s)"),
        ("crps", "Speed CRPS", "Wind speed CRPS (m/s)"),
        ("mape_percent", "Speed MAPE", "Wind speed MAPE (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), constrained_layout=True)
    axes_flat = axes.ravel()
    palette = ["#1f77b4", "#d55e00", "#009e73", "#cc79a7", "#0072b2", "#e69f00"]
    markers = ["o", "s", "^", "D", "P", "v"]
    title_font = {"fontsize": 20, "fontweight": "bold"}
    label_font = {"fontsize": 16, "fontweight": "bold"}
    legend_font = {"size": 13, "weight": "bold"}
    for ax, (metric, title, ylabel) in zip(axes_flat, metric_specs):
        for idx, (model, rows) in enumerate(all_rows.items()):
            x = np.asarray([row["horizon"] for row in rows], dtype=float)
            y = np.asarray([row[metric] for row in rows], dtype=float)
            ax.plot(
                x,
                y,
                marker=markers[idx % len(markers)],
                markersize=4.6,
                linewidth=2.4,
                color=palette[idx % len(palette)],
                label=model,
            )
        ax.set_title(title, **title_font)
        ax.set_xlabel("forecast horizon (10 min steps)", **label_font)
        ax.set_ylabel(ylabel, **label_font)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="both", labelsize=14, width=1.4, length=5.5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.legend(prop=legend_font)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark speed RMSE/MAE/CRPS/MAPE without NWP/persistence baselines.")
    parser.add_argument("forecasts", nargs="+", type=Path, help="One or more benchmark_forecasts.npz files.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mape-eps", type=float, default=1.0, help="Minimum target speed denominator in m/s.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.forecasts[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, float]]] = {}
    for forecast_path in args.forecasts:
        loaded = load_forecast(forecast_path)
        rows = speed_metrics(
            loaded["prediction"],
            loaded["target"],
            loaded["horizons"],
            mape_eps=float(args.mape_eps),
        )
        all_rows[loaded["model_name"]] = rows
    csv_path = output_dir / "benchmark_speed_rmse_mae_crps_mape.csv"
    plot_path = output_dir / "benchmark_speed_rmse_mae_crps_mape.png"
    write_csv(csv_path, all_rows)
    plot_metrics(plot_path, all_rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
