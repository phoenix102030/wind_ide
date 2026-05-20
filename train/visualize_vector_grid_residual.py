from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
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


def plot_summary_maps(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    corr_u: np.ndarray,
    corr_v: np.ndarray,
    station_indices: np.ndarray,
    out_path: Path,
) -> None:
    raw_speed, _ = speed_direction(raw_u, raw_v)
    corr_speed, _ = speed_direction(corr_u, corr_v)
    delta = corr_speed - raw_speed
    residual_speed, _ = speed_direction(corr_u - raw_u, corr_v - raw_v)
    fields = [
        (np.nanmean(raw_speed, axis=0), "Mean raw NWP speed"),
        (np.nanmean(corr_speed, axis=0), "Mean corrected speed"),
        (np.nanmean(delta, axis=0), "Mean speed correction"),
        (np.sqrt(np.nanmean(delta * delta, axis=0)), "RMS speed correction"),
        (np.nanpercentile(np.abs(delta), 95, axis=0), "P95 |speed correction|"),
        (np.nanmean(residual_speed, axis=0), "Mean residual vector magnitude"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (field, title) in zip(axes.ravel(), fields):
        image = ax.imshow(field, origin="lower", cmap="viridis")
        ax.scatter(station_indices[:, 1], station_indices[:, 0], c="white", edgecolors="black", s=55, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("grid col")
        ax.set_ylabel("grid row")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Time-aggregated VectorMIDE grid residual extension")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_wind_gif(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    corr_u: np.ndarray,
    corr_v: np.ndarray,
    station_indices: np.ndarray,
    frame_indices: np.ndarray,
    out_path: Path,
    fps: int,
    quiver_stride: int,
) -> None:
    raw_speed, _ = speed_direction(raw_u, raw_v)
    corr_speed, _ = speed_direction(corr_u, corr_v)
    delta = corr_speed - raw_speed
    residual_speed, _ = speed_direction(corr_u - raw_u, corr_v - raw_v)

    speed_min, speed_max = np.nanpercentile(np.concatenate([raw_speed.ravel(), corr_speed.ravel()]), [2, 98])
    delta_lim = float(np.nanpercentile(np.abs(delta), 98))
    residual_min, residual_max = np.nanpercentile(residual_speed, [2, 98])
    yy, xx = np.mgrid[0 : raw_speed.shape[1], 0 : raw_speed.shape[2]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes_flat = axes.ravel()
    images = [
        axes_flat[0].imshow(raw_speed[0], origin="lower", cmap="viridis", vmin=speed_min, vmax=speed_max),
        axes_flat[1].imshow(corr_speed[0], origin="lower", cmap="viridis", vmin=speed_min, vmax=speed_max),
        axes_flat[2].imshow(delta[0], origin="lower", cmap="coolwarm", vmin=-delta_lim, vmax=delta_lim),
        axes_flat[3].imshow(residual_speed[0], origin="lower", cmap="magma", vmin=residual_min, vmax=residual_max),
    ]
    titles = [
        "Raw NWP speed",
        "Corrected speed",
        "Speed correction",
        "Residual vector magnitude",
    ]
    quivers = []
    for ax, image, title in zip(axes_flat, images, titles):
        ax.scatter(station_indices[:, 1], station_indices[:, 0], c="white", edgecolors="black", s=45, zorder=3)
        quiver = ax.quiver(
            xx[::quiver_stride, ::quiver_stride],
            yy[::quiver_stride, ::quiver_stride],
            corr_u[0, ::quiver_stride, ::quiver_stride],
            corr_v[0, ::quiver_stride, ::quiver_stride],
            color="black",
            alpha=0.55,
            scale=350,
            width=0.0025,
        )
        quivers.append(quiver)
        ax.set_title(title)
        ax.set_xlabel("grid col")
        ax.set_ylabel("grid row")
        fig.colorbar(image, ax=ax, shrink=0.8)

    writer = PillowWriter(fps=max(1, int(fps)))
    with writer.saving(fig, str(out_path), dpi=120):
        for idx in frame_indices:
            images[0].set_data(raw_speed[idx])
            images[1].set_data(corr_speed[idx])
            images[2].set_data(delta[idx])
            images[3].set_data(residual_speed[idx])
            for quiver in quivers:
                quiver.set_UVC(
                    corr_u[idx, ::quiver_stride, ::quiver_stride],
                    corr_v[idx, ::quiver_stride, ::quiver_stride],
                )
            fig.suptitle(f"VectorMIDE grid residual extension, time index {int(idx)}")
            writer.grab_frame()
    plt.close(fig)


def make_corrected_wind_gif(
    corr_u: np.ndarray,
    corr_v: np.ndarray,
    station_indices: np.ndarray,
    frame_indices: np.ndarray,
    out_path: Path,
    fps: int,
    quiver_stride: int,
) -> None:
    corr_speed, _ = speed_direction(corr_u, corr_v)
    speed_min, speed_max = np.nanpercentile(corr_speed, [2, 98])
    yy, xx = np.mgrid[0 : corr_speed.shape[1], 0 : corr_speed.shape[2]]

    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    image = ax.imshow(
        corr_speed[0],
        origin="lower",
        cmap="viridis",
        vmin=speed_min,
        vmax=speed_max,
    )
    quiver = ax.quiver(
        xx[::quiver_stride, ::quiver_stride],
        yy[::quiver_stride, ::quiver_stride],
        corr_u[0, ::quiver_stride, ::quiver_stride],
        corr_v[0, ::quiver_stride, ::quiver_stride],
        color="white",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.9,
        scale=350,
        width=0.003,
    )
    ax.scatter(
        station_indices[:, 1],
        station_indices[:, 0],
        marker="*",
        s=180,
        c="gold",
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
        label="measurement station",
    )
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("corrected wind speed")

    writer = PillowWriter(fps=max(1, int(fps)))
    with writer.saving(fig, str(out_path), dpi=130):
        for idx in frame_indices:
            image.set_data(corr_speed[idx])
            quiver.set_UVC(
                corr_u[idx, ::quiver_stride, ::quiver_stride],
                corr_v[idx, ::quiver_stride, ::quiver_stride],
            )
            ax.set_title(f"Corrected wind field, time index {int(idx)}")
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VectorMIDE full-grid residual extension artifacts.")
    parser.add_argument("--npz", required=True, help="Path to grid_residual_extension.npz.")
    parser.add_argument("--state", choices=["prediction", "analysis"], default="prediction")
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--num-points", type=int, default=40)
    parser.add_argument("--quiver-stride", type=int, default=4)
    parser.add_argument("--gif", action="store_true", help="Also write an animated GIF over sampled time indices.")
    parser.add_argument(
        "--corrected-gif",
        action="store_true",
        help="Write a single-panel GIF: corrected wind-speed background plus vector arrows.",
    )
    parser.add_argument("--gif-frames", type=int, default=120)
    parser.add_argument("--gif-fps", type=int, default=8)
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
    plot_summary_maps(raw_u, raw_v, corr_u, corr_v, station_indices, out_path=out_dir / "summary_correction_maps.png")

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
    gif_name = None
    if args.gif:
        frame_count = max(1, min(int(args.gif_frames), n_time))
        frame_indices = np.linspace(0, n_time - 1, frame_count).round().astype(int)
        gif_name = f"wind_correction_{args.state}.gif"
        make_wind_gif(
            raw_u,
            raw_v,
            corr_u,
            corr_v,
            station_indices=station_indices,
            frame_indices=frame_indices,
            out_path=out_dir / gif_name,
            fps=int(args.gif_fps),
            quiver_stride=max(1, int(args.quiver_stride)),
        )
    corrected_gif_name = None
    if args.corrected_gif:
        frame_count = max(1, min(int(args.gif_frames), n_time))
        frame_indices = np.linspace(0, n_time - 1, frame_count).round().astype(int)
        corrected_gif_name = f"corrected_wind_field_{args.state}.gif"
        make_corrected_wind_gif(
            corr_u,
            corr_v,
            station_indices=station_indices,
            frame_indices=frame_indices,
            out_path=out_dir / corrected_gif_name,
            fps=int(args.gif_fps),
            quiver_stride=max(1, int(args.quiver_stride)),
        )
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": str(args.npz),
                "state": args.state,
                "time_index": time_index,
                "points": points,
                "station_nearest_indices": station_indices.tolist(),
                "gif": gif_name,
                "corrected_gif": corrected_gif_name,
            },
            handle,
            indent=2,
        )
    print(f"Saved VectorMIDE grid-extension plots to {out_dir}")


if __name__ == "__main__":
    main()
