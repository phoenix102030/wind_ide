from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


DEFAULT_START_TIME = "2021-06-04 00:00:00"
DEFAULT_DT_SECONDS = 600.0


def parse_time(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse time {value!r}; use YYYY-MM-DD HH:MM[:SS].")


def time_at(index: int, start_time: datetime, dt_seconds: float) -> datetime:
    return start_time + timedelta(seconds=float(dt_seconds) * int(index))


def time_label(index: int, start_time: datetime, dt_seconds: float) -> str:
    return time_at(index, start_time, dt_seconds).strftime("%Y-%m-%d %H:%M")


def time_slug(index: int, start_time: datetime, dt_seconds: float) -> str:
    return time_at(index, start_time, dt_seconds).strftime("%Y%m%d_%H%M")


def speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(u * u + v * v, 0.0))


def nearest_indices(lat_grid: np.ndarray, lon_grid: np.ndarray, station_latlon: np.ndarray) -> np.ndarray:
    out = []
    for lat, lon in station_latlon:
        dist = (lat_grid - float(lat)) ** 2 + (lon_grid - float(lon)) ** 2
        out.append(np.unravel_index(int(np.nanargmin(dist)), lat_grid.shape))
    return np.asarray(out, dtype=np.int64)


def load_fields(npz_path: Path, state: str) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        raw_u = np.asarray(data["raw_u"], dtype=np.float64)
        raw_v = np.asarray(data["raw_v"], dtype=np.float64)
        corr_u = np.asarray(data[f"corrected_{state}_u"], dtype=np.float64)
        corr_v = np.asarray(data[f"corrected_{state}_v"], dtype=np.float64)
        lat_grid = np.asarray(data["lat_grid"], dtype=np.float64)
        lon_grid = np.asarray(data["lon_grid"], dtype=np.float64)
        station_latlon = np.asarray(data["station_latlon"], dtype=np.float64)
    return {
        "raw_u": raw_u,
        "raw_v": raw_v,
        "corr_u": corr_u,
        "corr_v": corr_v,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "station_latlon": station_latlon,
    }


def extent_from_latlon(lat_grid: np.ndarray, lon_grid: np.ndarray) -> list[float]:
    return [
        float(np.nanmin(lon_grid)),
        float(np.nanmax(lon_grid)),
        float(np.nanmin(lat_grid)),
        float(np.nanmax(lat_grid)),
    ]


def set_ticks(ax, extent: list[float]) -> None:
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_xticks(np.linspace(lon_min, lon_max, 5))
    ax.set_yticks(np.linspace(lat_min, lat_max, 5))
    ax.tick_params(axis="both", labelsize=11, width=1.2, length=4.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def draw_triptych(
    fields: dict[str, np.ndarray],
    frame: int,
    out_path: Path,
    start_time: datetime,
    dt_seconds: float,
    state: str,
    speed_vmin: float,
    speed_vmax: float,
    corr_vmax: float,
    quiver_stride: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_u = fields["raw_u"][frame]
    raw_v = fields["raw_v"][frame]
    corr_u = fields["corr_u"][frame]
    corr_v = fields["corr_v"][frame]
    lat_grid = fields["lat_grid"]
    lon_grid = fields["lon_grid"]
    station_latlon = fields["station_latlon"]
    extent = extent_from_latlon(lat_grid, lon_grid)
    raw_speed = speed(raw_u, raw_v)
    corrected_speed = speed(corr_u, corr_v)
    correction_mag = speed(corr_u - raw_u, corr_v - raw_v)

    panels = [
        (raw_speed, "Raw NWP wind speed", "m/s", "viridis", speed_vmin, speed_vmax),
        (corrected_speed, "Corrected NWP wind speed", "m/s", "viridis", speed_vmin, speed_vmax),
        (correction_mag, "Correction magnitude", "m/s", "magma", 0.0, corr_vmax),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.7), constrained_layout=True)
    for ax, (field, title, cbar_label, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )
        ax.quiver(
            lon_grid[::quiver_stride, ::quiver_stride],
            lat_grid[::quiver_stride, ::quiver_stride],
            corr_u[::quiver_stride, ::quiver_stride],
            corr_v[::quiver_stride, ::quiver_stride],
            color="white",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.9,
            scale=350,
            width=0.003,
        )
        ax.scatter(
            station_latlon[:, 1],
            station_latlon[:, 0],
            marker="*",
            s=190,
            c="gold",
            edgecolors="black",
            linewidths=1.1,
            zorder=4,
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
        set_ticks(ax, extent)
        cbar = fig.colorbar(image, ax=ax, shrink=0.82)
        cbar.set_label(cbar_label, fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)
    fig.suptitle(
        f"VectorMIDE grid NWP correction ({state}), {time_label(frame, start_time, dt_seconds)}",
        fontsize=19,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_gif(
    fields: dict[str, np.ndarray],
    frames: np.ndarray,
    out_path: Path,
    start_time: datetime,
    dt_seconds: float,
    state: str,
    speed_vmin: float,
    speed_vmax: float,
    corr_vmax: float,
    quiver_stride: int,
    fps: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    lat_grid = fields["lat_grid"]
    lon_grid = fields["lon_grid"]
    station_latlon = fields["station_latlon"]
    extent = extent_from_latlon(lat_grid, lon_grid)
    first = int(frames[0])
    raw_speed = speed(fields["raw_u"][first], fields["raw_v"][first])
    corr_speed = speed(fields["corr_u"][first], fields["corr_v"][first])
    corr_mag = speed(fields["corr_u"][first] - fields["raw_u"][first], fields["corr_v"][first] - fields["raw_v"][first])

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.7), constrained_layout=True)
    panel_configs = [
        (raw_speed, "Raw NWP wind speed", "m/s", "viridis", speed_vmin, speed_vmax),
        (corr_speed, "Corrected NWP wind speed", "m/s", "viridis", speed_vmin, speed_vmax),
        (corr_mag, "Correction magnitude", "m/s", "magma", 0.0, corr_vmax),
    ]
    images = []
    quivers = []
    for ax, (field, title, cbar_label, cmap, vmin, vmax) in zip(axes, panel_configs):
        image = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )
        images.append(image)
        quiver = ax.quiver(
            lon_grid[::quiver_stride, ::quiver_stride],
            lat_grid[::quiver_stride, ::quiver_stride],
            fields["corr_u"][first, ::quiver_stride, ::quiver_stride],
            fields["corr_v"][first, ::quiver_stride, ::quiver_stride],
            color="white",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.9,
            scale=350,
            width=0.003,
        )
        quivers.append(quiver)
        ax.scatter(
            station_latlon[:, 1],
            station_latlon[:, 0],
            marker="*",
            s=190,
            c="gold",
            edgecolors="black",
            linewidths=1.1,
            zorder=4,
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=13, fontweight="bold")
        ax.set_ylabel("Latitude", fontsize=13, fontweight="bold")
        set_ticks(ax, extent)
        cbar = fig.colorbar(image, ax=ax, shrink=0.82)
        cbar.set_label(cbar_label, fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)

    writer = PillowWriter(fps=max(1, int(fps)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_path), dpi=140):
        for frame in frames:
            frame = int(frame)
            images[0].set_data(speed(fields["raw_u"][frame], fields["raw_v"][frame]))
            images[1].set_data(speed(fields["corr_u"][frame], fields["corr_v"][frame]))
            images[2].set_data(
                speed(
                    fields["corr_u"][frame] - fields["raw_u"][frame],
                    fields["corr_v"][frame] - fields["raw_v"][frame],
                )
            )
            for quiver in quivers:
                quiver.set_UVC(
                    fields["corr_u"][frame, ::quiver_stride, ::quiver_stride],
                    fields["corr_v"][frame, ::quiver_stride, ::quiver_stride],
                )
            fig.suptitle(
                f"VectorMIDE grid NWP correction ({state}), {time_label(frame, start_time, dt_seconds)}",
                fontsize=19,
                fontweight="bold",
            )
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw NWP, corrected NWP, and correction magnitude on the NWP grid.")
    parser.add_argument("--npz", type=Path, default=Path("/Users/felix/Downloads/vector_grid_residual_preview_fixed/grid_residual_extension.npz"))
    parser.add_argument("--state", choices=["prediction", "analysis"], default="prediction")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--start-time", default=DEFAULT_START_TIME)
    parser.add_argument("--dt-seconds", type=float, default=DEFAULT_DT_SECONDS)
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/felix/Downloads/vector_grid_residual_preview_fixed/plots_triptych"))
    parser.add_argument("--quiver-stride", type=int, default=4)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-frames", type=int, default=80)
    parser.add_argument("--gif-fps", type=int, default=6)
    args = parser.parse_args()

    fields = load_fields(args.npz, args.state)
    n_time = int(fields["raw_u"].shape[0])
    frame = int(np.clip(args.time_index, 0, n_time - 1))
    start_time = parse_time(args.start_time)
    raw_speed_all = speed(fields["raw_u"], fields["raw_v"])
    corr_speed_all = speed(fields["corr_u"], fields["corr_v"])
    corr_mag_all = speed(fields["corr_u"] - fields["raw_u"], fields["corr_v"] - fields["raw_v"])
    speed_vmin, speed_vmax = np.nanpercentile(np.concatenate([raw_speed_all.ravel(), corr_speed_all.ravel()]), [2, 98])
    corr_vmax = float(np.nanpercentile(corr_mag_all, 98))
    slug = time_slug(frame, start_time, args.dt_seconds)
    png_path = args.output_dir / f"nwp_raw_corrected_correction_{args.state}_{slug}.png"
    draw_triptych(
        fields,
        frame,
        png_path,
        start_time,
        args.dt_seconds,
        args.state,
        float(speed_vmin),
        float(speed_vmax),
        corr_vmax,
        max(1, int(args.quiver_stride)),
    )
    print(f"Wrote {png_path}")
    if args.gif:
        count = max(1, min(int(args.gif_frames), n_time))
        frames = np.linspace(0, n_time - 1, count).round().astype(int)
        gif_path = args.output_dir / f"nwp_raw_corrected_correction_{args.state}.gif"
        make_gif(
            fields,
            frames,
            gif_path,
            start_time,
            args.dt_seconds,
            args.state,
            float(speed_vmin),
            float(speed_vmax),
            corr_vmax,
            max(1, int(args.quiver_stride)),
            int(args.gif_fps),
        )
        print(f"Wrote {gif_path}")


if __name__ == "__main__":
    main()
