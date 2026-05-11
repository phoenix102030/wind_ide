from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u * u + v * v)
    direction = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    return speed, direction


def representative_points(height: int, width: int, count: int) -> list[tuple[int, int]]:
    count = max(1, int(count))
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    row_values = np.linspace(0, height - 1, rows).round().astype(int)
    col_values = np.linspace(0, width - 1, cols).round().astype(int)
    return [(int(row), int(col)) for row in row_values for col in col_values][:count]


def nearest_grid_indices(grid_coords: np.ndarray, station_coords: np.ndarray, height: int, width: int) -> np.ndarray:
    indices = []
    for coord in station_coords:
        flat = int(np.argmin(((grid_coords - coord[None, :]) ** 2).sum(axis=1)))
        indices.append(np.unravel_index(flat, (height, width)))
    return np.asarray(indices, dtype=np.int64)


def plot_maps(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    corr_u: np.ndarray,
    corr_v: np.ndarray,
    station_indices: np.ndarray,
    time_index: int,
    out_path: Path,
    quiver_stride: int,
) -> None:
    raw_speed, raw_dir = speed_direction(raw_u[time_index], raw_v[time_index])
    corr_speed, corr_dir = speed_direction(corr_u[time_index], corr_v[time_index])
    delta = corr_speed - raw_speed
    resid_speed, _ = speed_direction(corr_u[time_index] - raw_u[time_index], corr_v[time_index] - raw_v[time_index])
    panels = [
        (raw_speed, "Raw NWP speed"),
        (corr_speed, "NWP + VectorMIDE residual speed"),
        (delta, "Corrected - raw speed"),
        (raw_dir, "Raw NWP direction"),
        (corr_dir, "Corrected direction"),
        (resid_speed, "Estimated residual vector magnitude"),
    ]
    yy, xx = np.mgrid[0 : raw_speed.shape[0], 0 : raw_speed.shape[1]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (field, title) in zip(axes.ravel(), panels):
        image = ax.imshow(field, origin="lower", cmap="viridis")
        ax.scatter(station_indices[:, 1], station_indices[:, 0], c="white", edgecolors="black", s=55, zorder=3)
        ax.quiver(
            xx[::quiver_stride, ::quiver_stride],
            yy[::quiver_stride, ::quiver_stride],
            corr_u[time_index, ::quiver_stride, ::quiver_stride],
            corr_v[time_index, ::quiver_stride, ::quiver_stride],
            color="black",
            alpha=0.55,
            scale=350,
            width=0.002,
        )
        ax.set_title(title)
        ax.set_xlabel("grid col")
        ax.set_ylabel("grid row")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(f"VectorMIDE grid residual extension at time index {time_index}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grid_lines(raw_speed: np.ndarray, corr_speed: np.ndarray, points: list[tuple[int, int]], out_path: Path) -> None:
    cols = 5
    rows = int(np.ceil(len(points) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, max(2.2 * rows, 5)), sharex=True, constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    time = np.arange(raw_speed.shape[0])
    for ax, (row, col) in zip(axes_flat, points):
        ax.plot(time, raw_speed[:, row, col], color="#4c78a8", linewidth=1.0, label="NWP")
        ax.plot(time, corr_speed[:, row, col], color="#f58518", linewidth=1.0, label="NWP + residual")
        ax.set_title(f"({row}, {col})", fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(points) :]:
        ax.axis("off")
    axes_flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Wind-speed time series at {len(points)} representative grid points")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_station_lines(
    measurement: np.ndarray,
    station_nwp: np.ndarray,
    station_prediction: np.ndarray,
    nearest_corrected: np.ndarray,
    out_path: Path,
) -> None:
    obs_u, obs_v = measurement[:, :3], measurement[:, 3:]
    nwp_u, nwp_v = station_nwp[:, :3], station_nwp[:, 3:]
    pred_u = station_nwp[:, :3] + station_prediction[:, :3]
    pred_v = station_nwp[:, 3:] + station_prediction[:, 3:]
    obs_speed, _ = speed_direction(obs_u, obs_v)
    nwp_speed, _ = speed_direction(nwp_u, nwp_v)
    pred_speed, _ = speed_direction(pred_u, pred_v)
    nearest_speed, _ = speed_direction(nearest_corrected[:, :3], nearest_corrected[:, 3:])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    time = np.arange(obs_speed.shape[0])
    for idx, ax in enumerate(axes):
        ax.plot(time, obs_speed[:, idx], color="#111111", linewidth=1.2, label="measurement")
        ax.plot(time, nwp_speed[:, idx], color="#4c78a8", linewidth=1.0, label="station NWP")
        ax.plot(time, pred_speed[:, idx], color="#f58518", linewidth=1.0, label="station VectorMIDE prediction")
        ax.plot(
            time,
            nearest_speed[:, idx],
            color="#54a24b",
            linewidth=0.9,
            linestyle="--",
            label="nearest-grid NWP + residual",
        )
        ax.set_title(f"Measurement station {idx}")
        ax.set_ylabel("wind speed")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("time index")
    fig.suptitle("Station observations vs original VectorMIDE and nearest grid extension")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VectorMIDE full-grid residual extension artifacts.")
    parser.add_argument("--npz", required=True, help="Path to grid_residual_extension.npz.")
    parser.add_argument("--state", choices=["prediction", "analysis"], default="prediction")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--num-points", type=int, default=40)
    parser.add_argument("--quiver-stride", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    payload = np.load(args.npz)
    raw_u = payload["raw_u"]
    raw_v = payload["raw_v"]
    corr_u = payload[f"corrected_{args.state}_u"]
    corr_v = payload[f"corrected_{args.state}_v"]
    raw_speed, _ = speed_direction(raw_u, raw_v)
    corr_speed, _ = speed_direction(corr_u, corr_v)
    n_time, height, width = raw_u.shape
    time_index = args.time_index if args.time_index >= 0 else n_time + args.time_index
    time_index = int(np.clip(time_index, 0, n_time - 1))

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.npz).parent / f"plots_{args.state}"
    out_dir.mkdir(parents=True, exist_ok=True)
    station_indices = nearest_grid_indices(payload["grid_coords"], payload["coords"], height, width)
    points = representative_points(height, width, args.num_points)

    plot_maps(
        raw_u,
        raw_v,
        corr_u,
        corr_v,
        station_indices=station_indices,
        time_index=time_index,
        out_path=out_dir / f"wind_maps_t{time_index}.png",
        quiver_stride=max(1, int(args.quiver_stride)),
    )
    plot_grid_lines(raw_speed, corr_speed, points, out_path=out_dir / f"{len(points)}_grid_point_speed_lines.png")

    nearest = np.concatenate(
        [
            np.stack([corr_u[:, row, col] for row, col in station_indices], axis=1),
            np.stack([corr_v[:, row, col] for row, col in station_indices], axis=1),
        ],
        axis=1,
    )
    station_residual_key = f"station_{args.state}_residual"
    plot_station_lines(
        measurement=payload["measurement_target"],
        station_nwp=payload["station_nwp_baseline"],
        station_prediction=payload[station_residual_key],
        nearest_corrected=nearest,
        out_path=out_dir / "station_measurement_vs_vector_grid_extension.png",
    )

    point_rows = []
    for row, col in points:
        point_rows.append(
            np.stack(
                [
                    np.full(n_time, row),
                    np.full(n_time, col),
                    np.arange(n_time),
                    raw_speed[:, row, col],
                    corr_speed[:, row, col],
                    corr_speed[:, row, col] - raw_speed[:, row, col],
                ],
                axis=1,
            )
        )
    np.savetxt(
        out_dir / "representative_grid_point_speed.csv",
        np.concatenate(point_rows, axis=0),
        delimiter=",",
        header="row,col,time_index,nwp_speed,corrected_speed,speed_delta",
        comments="",
    )
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": str(args.npz),
                "state": args.state,
                "time_index": time_index,
                "points": points,
                "station_nearest_indices": station_indices.tolist(),
            },
            handle,
            indent=2,
        )
    print(f"Saved VectorMIDE grid-extension plots to {out_dir}")


if __name__ == "__main__":
    main()
