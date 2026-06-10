from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        model = row["model"]
        if model not in metrics:
            metrics[model] = {
                "forecast_hour": [],
                "rmse": [],
                "mae": [],
                "crps": [],
                "cpe": [],
            }
        metrics[model]["forecast_hour"].append(float(row["forecast_hour"]))
        metrics[model]["rmse"].append(float(row["direction_rmse_deg"]))
        metrics[model]["mae"].append(float(row["direction_mae_deg"]))
        metrics[model]["crps"].append(float(row["direction_crps_deg"]))
        metrics[model]["cpe"].append(float(row["direction_cpe_percent"]))
    return {
        model: {key: np.asarray(values, dtype=float) for key, values in model_rows.items()}
        for model, model_rows in metrics.items()
    }


def plot(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
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
            ax.plot(
                values["forecast_hour"],
                values[metric],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot hourly direction metrics from CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.csv_path.with_suffix(".png")
    plot(output, load_rows(args.csv_path))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
