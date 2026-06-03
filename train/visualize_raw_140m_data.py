from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.vector_data_utils import build_z_from_measurements, load_mat_variable, load_nwp_uv140


STATION_NAMES = ("E05", "E06", "ASOW6")
DEFAULT_PATHS = {
    "offline": {
        "measurement": Path("data/measurement/wv_h100_180_offline_imputed.mat"),
        "nwp": Path("data/nwp/data_grid_offline.mat"),
    },
    "online": {
        "measurement": Path("data/measurement/wv_h100_180_online_imputed.mat"),
        "nwp": Path("data/nwp/data_grid_online.mat"),
    },
}


def speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u * u + v * v)


def nearest_grid_indices(lat_grid: np.ndarray, lon_grid: np.ndarray, station_lat: np.ndarray, station_lon: np.ndarray) -> np.ndarray:
    grid_coords = np.stack([lat_grid.reshape(-1), lon_grid.reshape(-1)], axis=1)
    station_coords = np.stack([station_lat.reshape(-1), station_lon.reshape(-1)], axis=1)
    indices = []
    for coord in station_coords:
        flat = int(np.argmin(np.sum((grid_coords - coord[None, :]) ** 2, axis=1)))
        indices.append(np.unravel_index(flat, lat_grid.shape))
    return np.asarray(indices, dtype=np.int64)


def sample_nearest(field: np.ndarray, station_yx: np.ndarray) -> np.ndarray:
    return np.stack([field[:, row, col] for row, col in station_yx], axis=1)


def select_balanced_uv_start(raw_u: np.ndarray, raw_v: np.ndarray, frame_count: int, frame_step: int) -> int:
    window = max(1, int(frame_count) * max(1, int(frame_step)))
    n_time = raw_u.shape[0]
    if n_time <= window:
        return 0
    mean_abs_u = np.nanmean(np.abs(raw_u), axis=(1, 2))
    mean_abs_v = np.nanmean(np.abs(raw_v), axis=(1, 2))
    mean_speed = np.nanmean(speed(raw_u, raw_v), axis=(1, 2))
    balance = np.minimum(mean_abs_u, mean_abs_v) * mean_speed
    kernel = np.ones(window, dtype=np.float64) / float(window)
    window_score = np.convolve(balance, kernel, mode="valid")
    return int(np.nanargmax(window_score))


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_nwp_map(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    meas_u: np.ndarray,
    meas_v: np.ndarray,
    station_yx: np.ndarray,
    time_index: int,
    out_path: Path,
    quiver_stride: int,
) -> None:
    plt = setup_matplotlib()
    nwp_speed = speed(raw_u[time_index], raw_v[time_index])
    fields = [
        (raw_u[time_index], "NWP U 140m", "coolwarm", raw_u[time_index], np.zeros_like(raw_v[time_index])),
        (raw_v[time_index], "NWP V 140m", "coolwarm", np.zeros_like(raw_u[time_index]), raw_v[time_index]),
        (nwp_speed, "NWP wind speed 140m", "viridis", raw_u[time_index], raw_v[time_index]),
    ]
    yy, xx = np.mgrid[0 : raw_u.shape[1], 0 : raw_u.shape[2]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for panel_idx, (ax, (field, title, cmap, arrow_u, arrow_v)) in enumerate(zip(axes, fields)):
        if cmap == "coolwarm":
            lim = float(np.nanpercentile(np.abs(field), 98))
            image = ax.imshow(field, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
        else:
            image = ax.imshow(field, origin="lower", cmap=cmap)
        ax.quiver(
            xx[::quiver_stride, ::quiver_stride],
            yy[::quiver_stride, ::quiver_stride],
            arrow_u[::quiver_stride, ::quiver_stride],
            arrow_v[::quiver_stride, ::quiver_stride],
            color="black",
            alpha=0.55,
            scale=350,
            width=0.0025,
        )
        measurement_arrow_u = meas_u[time_index] if panel_idx != 1 else np.zeros_like(meas_u[time_index])
        measurement_arrow_v = meas_v[time_index] if panel_idx != 0 else np.zeros_like(meas_v[time_index])
        ax.quiver(
            station_yx[:, 1],
            station_yx[:, 0],
            measurement_arrow_u,
            measurement_arrow_v,
            color="white",
            edgecolor="black",
            linewidth=0.4,
            scale=180,
            width=0.006,
            zorder=4,
        )
        ax.scatter(station_yx[:, 1], station_yx[:, 0], c="gold", edgecolors="black", s=75, zorder=5)
        for idx, name in enumerate(STATION_NAMES):
            ax.text(station_yx[idx, 1] + 0.4, station_yx[idx, 0] + 0.4, name, fontsize=8, weight="bold")
        ax.set_title(title)
        ax.set_xlabel("grid col")
        ax.set_ylabel("grid row")
        fig.colorbar(image, ax=ax, shrink=0.84)
    fig.suptitle(f"Raw 140m NWP grid and measurement vectors, time index {time_index}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_station_timeseries(
    meas_u: np.ndarray,
    meas_v: np.ndarray,
    nwp_u: np.ndarray,
    nwp_v: np.ndarray,
    out_path: Path,
    max_points: int,
) -> None:
    plt = setup_matplotlib()
    meas_speed = speed(meas_u, meas_v)
    nwp_speed = speed(nwp_u, nwp_v)
    n_time = meas_u.shape[0]
    if max_points and n_time > max_points:
        idx = np.linspace(0, n_time - 1, max_points).round().astype(int)
    else:
        idx = np.arange(n_time)

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex=True, constrained_layout=True)
    panels = [
        (meas_u, nwp_u, "U"),
        (meas_v, nwp_v, "V"),
        (meas_speed, nwp_speed, "wind speed"),
    ]
    for station_idx, station_name in enumerate(STATION_NAMES):
        for col_idx, (obs, nwp, label) in enumerate(panels):
            ax = axes[station_idx, col_idx]
            ax.plot(idx, obs[idx, station_idx], color="#111111", linewidth=1.0, label="measurement")
            ax.plot(idx, nwp[idx, station_idx], color="#4c78a8", linewidth=1.0, label="nearest NWP")
            ax.set_title(f"{station_name} {label}")
            ax.grid(alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("m/s")
    axes[0, 0].legend(loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time index")
    fig.suptitle("140m measurement vs nearest-grid NWP")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_nwp_summary(raw_u: np.ndarray, raw_v: np.ndarray, station_yx: np.ndarray, out_path: Path) -> None:
    plt = setup_matplotlib()
    raw_speed = speed(raw_u, raw_v)
    fields = [
        (np.nanmean(raw_u, axis=0), "Mean NWP U 140m", "coolwarm"),
        (np.nanmean(raw_v, axis=0), "Mean NWP V 140m", "coolwarm"),
        (np.nanmean(raw_speed, axis=0), "Mean NWP wind speed 140m", "viridis"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (field, title, cmap) in zip(axes, fields):
        if cmap == "coolwarm":
            lim = float(np.nanpercentile(np.abs(field), 98))
            image = ax.imshow(field, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
        else:
            image = ax.imshow(field, origin="lower", cmap=cmap)
        ax.scatter(station_yx[:, 1], station_yx[:, 0], c="gold", edgecolors="black", s=75, zorder=4)
        for idx, name in enumerate(STATION_NAMES):
            ax.text(station_yx[idx, 1] + 0.4, station_yx[idx, 0] + 0.4, name, fontsize=8, weight="bold")
        ax.set_title(title)
        ax.set_xlabel("grid col")
        ax.set_ylabel("grid row")
        fig.colorbar(image, ax=ax, shrink=0.84)
    fig.suptitle("Time-mean raw 140m NWP fields")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_wind_gif(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    meas_u: np.ndarray,
    meas_v: np.ndarray,
    station_yx: np.ndarray,
    out_path: Path,
    start_index: int,
    frame_count: int,
    frame_step: int,
    fps: int,
    quiver_stride: int,
) -> None:
    plt = setup_matplotlib()
    from matplotlib.animation import PillowWriter

    n_time = raw_u.shape[0]
    start_index = int(np.clip(start_index, 0, n_time - 1))
    frame_count = max(1, min(int(frame_count), n_time))
    frame_step = max(1, int(frame_step))
    stop_index = min(n_time, start_index + frame_count * frame_step)
    frame_indices = np.arange(start_index, stop_index, frame_step, dtype=np.int64)
    if frame_indices.size == 0:
        frame_indices = np.asarray([start_index], dtype=np.int64)
    raw_speed = speed(raw_u, raw_v)
    speed_min, speed_max = np.nanpercentile(raw_speed[frame_indices], [2, 98])
    yy, xx = np.mgrid[0 : raw_u.shape[1], 0 : raw_u.shape[2]]

    fig, ax = plt.subplots(figsize=(7.8, 7.2), constrained_layout=True)
    first = int(frame_indices[0])
    image = ax.imshow(raw_speed[first], origin="lower", cmap="viridis", vmin=speed_min, vmax=speed_max)
    u_quiver = ax.quiver(
        xx[::quiver_stride, ::quiver_stride],
        yy[::quiver_stride, ::quiver_stride],
        raw_u[first, ::quiver_stride, ::quiver_stride],
        np.zeros_like(raw_v[first, ::quiver_stride, ::quiver_stride]),
        color="#2563eb",
        alpha=0.72,
        scale=350,
        width=0.003,
        label="NWP U component",
    )
    v_quiver = ax.quiver(
        xx[::quiver_stride, ::quiver_stride],
        yy[::quiver_stride, ::quiver_stride],
        np.zeros_like(raw_u[first, ::quiver_stride, ::quiver_stride]),
        raw_v[first, ::quiver_stride, ::quiver_stride],
        color="#dc2626",
        alpha=0.72,
        scale=350,
        width=0.003,
        label="NWP V component",
    )
    wind_quiver = ax.quiver(
        xx[::quiver_stride, ::quiver_stride],
        yy[::quiver_stride, ::quiver_stride],
        raw_u[first, ::quiver_stride, ::quiver_stride],
        raw_v[first, ::quiver_stride, ::quiver_stride],
        color="white",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.9,
        scale=350,
        width=0.0035,
        label="NWP wind vector",
    )
    measurement_u_quiver = ax.quiver(
        station_yx[:, 1],
        station_yx[:, 0],
        meas_u[first],
        np.zeros_like(meas_v[first]),
        color="#0f172a",
        edgecolor="white",
        linewidth=0.6,
        scale=180,
        width=0.009,
        zorder=5,
        label="measurement U",
    )
    measurement_v_quiver = ax.quiver(
        station_yx[:, 1],
        station_yx[:, 0],
        np.zeros_like(meas_u[first]),
        meas_v[first],
        color="#f59e0b",
        edgecolor="black",
        linewidth=0.6,
        scale=180,
        width=0.009,
        zorder=5,
        label="measurement V",
    )
    measurement_wind_quiver = ax.quiver(
        station_yx[:, 1],
        station_yx[:, 0],
        meas_u[first],
        meas_v[first],
        color="#fef3c7",
        edgecolor="black",
        linewidth=0.75,
        scale=180,
        width=0.01,
        zorder=6,
        label="measurement wind",
    )
    ax.scatter(
        station_yx[:, 1],
        station_yx[:, 0],
        marker="o",
        s=95,
        c="gold",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )
    for idx, name in enumerate(STATION_NAMES):
        ax.text(station_yx[idx, 1] + 0.4, station_yx[idx, 0] + 0.4, name, fontsize=8, weight="bold")
    ax.set_xlim(-0.5, raw_u.shape[2] - 0.5)
    ax.set_ylim(-0.5, raw_u.shape[1] - 0.5)
    ax.set_aspect("equal")
    ax.grid(color="#d4d4d4", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    ax.legend(loc="upper right", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("NWP wind speed 140m")

    writer = PillowWriter(fps=max(1, int(fps)))
    with writer.saving(fig, str(out_path), dpi=130):
        for idx in frame_indices:
            idx = int(idx)
            image.set_data(raw_speed[idx])
            u_quiver.set_UVC(
                raw_u[idx, ::quiver_stride, ::quiver_stride],
                np.zeros_like(raw_v[idx, ::quiver_stride, ::quiver_stride]),
            )
            v_quiver.set_UVC(
                np.zeros_like(raw_u[idx, ::quiver_stride, ::quiver_stride]),
                raw_v[idx, ::quiver_stride, ::quiver_stride],
            )
            wind_quiver.set_UVC(
                raw_u[idx, ::quiver_stride, ::quiver_stride],
                raw_v[idx, ::quiver_stride, ::quiver_stride],
            )
            measurement_u_quiver.set_UVC(meas_u[idx], np.zeros_like(meas_v[idx]))
            measurement_v_quiver.set_UVC(np.zeros_like(meas_u[idx]), meas_v[idx])
            measurement_wind_quiver.set_UVC(meas_u[idx], meas_v[idx])
            ax.set_title(f"Raw 140m U/V components and wind, time index {idx}")
            writer.grab_frame()
    plt.close(fig)


def write_station_csv(
    out_path: Path,
    meas_u: np.ndarray,
    meas_v: np.ndarray,
    nwp_u: np.ndarray,
    nwp_v: np.ndarray,
) -> None:
    meas_speed = speed(meas_u, meas_v)
    nwp_speed = speed(nwp_u, nwp_v)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_index",
                "station",
                "measurement_u140",
                "measurement_v140",
                "measurement_wind_speed140",
                "nearest_nwp_u140",
                "nearest_nwp_v140",
                "nearest_nwp_wind_speed140",
            ]
        )
        for time_idx in range(meas_u.shape[0]):
            for station_idx, station_name in enumerate(STATION_NAMES):
                writer.writerow(
                    [
                        time_idx,
                        station_name,
                        float(meas_u[time_idx, station_idx]),
                        float(meas_v[time_idx, station_idx]),
                        float(meas_speed[time_idx, station_idx]),
                        float(nwp_u[time_idx, station_idx]),
                        float(nwp_v[time_idx, station_idx]),
                        float(nwp_speed[time_idx, station_idx]),
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw 140m NWP and measurement U/V/wind speed data.")
    parser.add_argument("--split", choices=["offline", "online"], default="offline")
    parser.add_argument("--measurement-path", default=None)
    parser.add_argument("--nwp-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument(
        "--auto-time-index",
        action="store_true",
        help="Choose a GIF/static start time where U and V are both strong over the GIF window.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of time steps to load/plot.")
    parser.add_argument("--quiver-stride", type=int, default=4)
    parser.add_argument("--max-line-points", type=int, default=2500)
    parser.add_argument("--gif", action="store_true", help="Also write an animated GIF over sampled time indices.")
    parser.add_argument("--gif-frames", type=int, default=100)
    parser.add_argument("--gif-step", type=int, default=3, help="Time-step spacing between adjacent GIF frames.")
    parser.add_argument("--gif-fps", type=int, default=5)
    args = parser.parse_args()

    measurement_path = Path(args.measurement_path) if args.measurement_path else DEFAULT_PATHS[args.split]["measurement"]
    nwp_path = Path(args.nwp_path) if args.nwp_path else DEFAULT_PATHS[args.split]["nwp"]
    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "raw_140m_visualization" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    ws_uv = load_mat_variable(measurement_path, "Ws_uv")
    if args.limit is not None:
        ws_uv = ws_uv[: args.limit]
    y = build_z_from_measurements(ws_uv)
    meas_u = y[:, :3]
    meas_v = y[:, 3:6]

    raw_u_hw_t, raw_v_hw_t = load_nwp_uv140(nwp_path, time_limit=args.limit)
    raw_u = np.moveaxis(raw_u_hw_t, 2, 0)
    raw_v = np.moveaxis(raw_v_hw_t, 2, 0)

    n_time = min(raw_u.shape[0], meas_u.shape[0])
    raw_u = raw_u[:n_time]
    raw_v = raw_v[:n_time]
    meas_u = meas_u[:n_time]
    meas_v = meas_v[:n_time]
    if args.auto_time_index:
        time_index = select_balanced_uv_start(raw_u, raw_v, int(args.gif_frames), int(args.gif_step))
    else:
        time_index = int(np.clip(args.time_index if args.time_index >= 0 else n_time + args.time_index, 0, n_time - 1))

    lat_grid = load_mat_variable(nwp_path, "LatValue").astype(np.float32)
    lon_grid = load_mat_variable(nwp_path, "LonValue").astype(np.float32)
    station_lat = load_mat_variable(measurement_path, "LatValue_vec").reshape(-1).astype(np.float32)
    station_lon = load_mat_variable(measurement_path, "LonValue_vec").reshape(-1).astype(np.float32)
    station_yx = nearest_grid_indices(lat_grid, lon_grid, station_lat, station_lon)

    nearest_u = sample_nearest(raw_u, station_yx)
    nearest_v = sample_nearest(raw_v, station_yx)

    plot_nwp_map(
        raw_u,
        raw_v,
        meas_u,
        meas_v,
        station_yx=station_yx,
        time_index=time_index,
        out_path=out_dir / f"nwp_measurement_140m_map_t{time_index}.png",
        quiver_stride=max(1, int(args.quiver_stride)),
    )
    plot_station_timeseries(
        meas_u,
        meas_v,
        nearest_u,
        nearest_v,
        out_path=out_dir / "station_measurement_vs_nearest_nwp_140m.png",
        max_points=max(0, int(args.max_line_points)),
    )
    plot_nwp_summary(raw_u, raw_v, station_yx, out_path=out_dir / "nwp_140m_mean_maps.png")
    write_station_csv(out_dir / "station_measurement_vs_nearest_nwp_140m.csv", meas_u, meas_v, nearest_u, nearest_v)
    outputs = [
        f"nwp_measurement_140m_map_t{time_index}.png",
        "station_measurement_vs_nearest_nwp_140m.png",
        "nwp_140m_mean_maps.png",
        "station_measurement_vs_nearest_nwp_140m.csv",
    ]
    if args.gif:
        make_wind_gif(
            raw_u,
            raw_v,
            meas_u,
            meas_v,
            station_yx=station_yx,
            out_path=out_dir / "nwp_measurement_140m_wind.gif",
            start_index=time_index,
            frame_count=int(args.gif_frames),
            frame_step=int(args.gif_step),
            fps=int(args.gif_fps),
            quiver_stride=max(1, int(args.quiver_stride)),
        )
        outputs.append("nwp_measurement_140m_wind.gif")

    metadata = {
        "split": args.split,
        "measurement_path": str(measurement_path),
        "nwp_path": str(nwp_path),
        "n_time": int(n_time),
        "time_index": int(time_index),
        "auto_time_index": bool(args.auto_time_index),
        "station_nearest_grid_indices": station_yx.tolist(),
        "gif": bool(args.gif),
        "gif_frames": int(args.gif_frames) if args.gif else 0,
        "gif_step": int(args.gif_step) if args.gif else 0,
        "gif_fps": int(args.gif_fps) if args.gif else 0,
        "outputs": outputs,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved raw 140m visualizations to {out_dir}")


if __name__ == "__main__":
    main()
