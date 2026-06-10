from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


U140_INDEX = 8
V140_INDEX = 9
DEFAULT_ONLINE_START = "2021-06-04 00:00:00"


def parse_start_time(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse start time {value!r}")


def time_at(index: int, start_time: datetime, dt_seconds: float) -> datetime:
    return start_time + timedelta(seconds=float(dt_seconds) * int(index))


def time_label(index: int, start_time: datetime, dt_seconds: float) -> str:
    return time_at(index, start_time, dt_seconds).strftime("%Y-%m-%d %H:%M")


def time_slug(index: int, start_time: datetime, dt_seconds: float) -> str:
    return time_at(index, start_time, dt_seconds).strftime("%Y%m%d_%H%M")


def time_range_label(start: int, end: int, start_time: datetime, dt_seconds: float) -> str:
    return f"{time_label(start, start_time, dt_seconds)} to {time_label(end, start_time, dt_seconds)}"


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    out: dict[str, list[float]] = {field: [] for field in (reader.fieldnames or [])}
    for row in rows:
        for field, value in row.items():
            try:
                out[field].append(float(value))
            except (TypeError, ValueError):
                out[field].append(np.nan)
    return {key: np.asarray(values, dtype=np.float64) for key, values in out.items()}


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=-1)


def angle_deg(v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = angle_deg(a) - angle_deg(b)
    return (diff + 180.0) % 360.0 - 180.0


def apply_component_direction_bias(
    au: np.ndarray,
    av: np.ndarray,
    au_x_scale: float,
    au_y_scale: float,
    av_x_scale: float,
    av_y_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    au_biased = au.copy()
    av_biased = av.copy()
    au_biased[:, 0] *= au_x_scale
    au_biased[:, 1] *= au_y_scale
    av_biased[:, 0] *= av_x_scale
    av_biased[:, 1] *= av_y_scale
    return au_biased, av_biased


def thin_indices(n: int, max_points: int = 2500) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).round().astype(int)


def plot_benchmark(
    out_path: Path,
    model_hour: dict[str, np.ndarray],
    direction_hour: dict[str, np.ndarray],
    arma_hour: dict[str, np.ndarray],
) -> None:
    hour = model_hour["forecast_hour"]
    arma_hour_index = arma_hour["forecast_hour"]
    metrics = [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("crps", "CRPS"),
    ]
    direction_metrics = [
        ("direction_rmse_deg", "direction RMSE (deg)"),
        ("direction_mae_deg", "direction MAE (deg)"),
        ("direction_crps_deg", "direction CRPS (deg)"),
    ]
    series = [
        ("model", "VectorWIDE", "#1f77b4"),
        ("arma", "ARIMA/ARMA", "#9467bd"),
        ("persistence", "Persistence", "#ff7f0e"),
        ("nwp", "NWP", "#2ca02c"),
        ("nwp_residual_persistence", "NWP residual persistence", "#7f7f7f"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    for ax, (key, ylabel) in zip(axes[0], metrics):
        for prefix, label, color in series:
            if prefix == "arma":
                x = arma_hour_index
                y = arma_hour.get(f"arma_{key}")
            else:
                x = hour
                y = model_hour.get(f"{prefix}_{key}")
            if y is not None:
                ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=color)
        ax.set_xlabel("forecast hour")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax, (key, ylabel) in zip(axes[1], direction_metrics):
        for prefix, label, color in series:
            if prefix == "arma":
                x = arma_hour_index
                y = arma_hour.get(f"arma_{key}")
            else:
                x = direction_hour["forecast_hour"]
                y = direction_hour.get(f"{prefix}_{key}")
            if y is not None:
                ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=color)
        ax.set_xlabel("forecast hour")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=9)
    fig.suptitle("Forecast benchmark by forecast hour")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_axis_comparisons(out_path: Path, au: np.ndarray, av: np.ndarray, nwp: np.ndarray) -> dict[str, float]:
    idx = thin_indices(len(au))
    au_i = au[idx]
    av_i = av[idx]
    nwp_i = nwp[idx]
    stats = {
        "corr_Au_x_NWP_x": safe_corr(au[:, 0], nwp[:, 0]),
        "corr_Au_y_NWP_y": safe_corr(au[:, 1], nwp[:, 1]),
        "corr_Av_x_NWP_x": safe_corr(av[:, 0], nwp[:, 0]),
        "corr_Av_y_NWP_y": safe_corr(av[:, 1], nwp[:, 1]),
        "corr_Au_speed_NWP_speed": safe_corr(norm(au), norm(nwp)),
        "corr_Av_speed_NWP_speed": safe_corr(norm(av), norm(nwp)),
        "corr_Au_x_Av_x": safe_corr(au[:, 0], av[:, 0]),
        "corr_Au_y_Av_y": safe_corr(au[:, 1], av[:, 1]),
        "median_abs_angle_Au_NWP_deg": float(np.nanmedian(np.abs(angle_diff_deg(au, nwp)))),
        "median_abs_angle_Av_NWP_deg": float(np.nanmedian(np.abs(angle_diff_deg(av, nwp)))),
    }

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    axes[0, 0].scatter(nwp_i[:, 0], au_i[:, 0], s=8, alpha=0.35, label="A_u,x")
    axes[0, 0].scatter(nwp_i[:, 0], av_i[:, 0], s=8, alpha=0.35, label="A_v,x")
    axes[0, 0].set_xlabel("NWP x displacement (km/step)")
    axes[0, 0].set_ylabel("learned x displacement (km/step)")
    axes[0, 0].set_title("same geographic axis: x")

    axes[0, 1].scatter(nwp_i[:, 1], au_i[:, 1], s=8, alpha=0.35, label="A_u,y")
    axes[0, 1].scatter(nwp_i[:, 1], av_i[:, 1], s=8, alpha=0.35, label="A_v,y")
    axes[0, 1].set_xlabel("NWP y displacement (km/step)")
    axes[0, 1].set_ylabel("learned y displacement (km/step)")
    axes[0, 1].set_title("same geographic axis: y")

    axes[0, 2].scatter(norm(nwp_i), norm(au_i), s=8, alpha=0.35, label="|A_u|")
    axes[0, 2].scatter(norm(nwp_i), norm(av_i), s=8, alpha=0.35, label="|A_v|")
    axes[0, 2].set_xlabel("|NWP displacement| (km/step)")
    axes[0, 2].set_ylabel("|learned displacement| (km/step)")
    axes[0, 2].set_title("speed response")

    axes[1, 0].scatter(au_i[:, 0], av_i[:, 0], s=8, alpha=0.35, color="#2ca02c")
    axes[1, 0].set_xlabel("A_u,x")
    axes[1, 0].set_ylabel("A_v,x")
    axes[1, 0].set_title("component comparison on x")

    axes[1, 1].scatter(au_i[:, 1], av_i[:, 1], s=8, alpha=0.35, color="#d62728")
    axes[1, 1].set_xlabel("A_u,y")
    axes[1, 1].set_ylabel("A_v,y")
    axes[1, 1].set_title("component comparison on y")

    axes[1, 2].hist(angle_diff_deg(au, nwp), bins=60, alpha=0.55, label="A_u - NWP")
    axes[1, 2].hist(angle_diff_deg(av, nwp), bins=60, alpha=0.55, label="A_v - NWP")
    axes[1, 2].set_xlabel("angle difference (degrees)")
    axes[1, 2].set_ylabel("count")
    axes[1, 2].set_title("direction alignment")

    for ax in axes.ravel():
        ax.axhline(0, color="0.4", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="0.4", linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)
    fig.suptitle("Legacy extracted component advection vs NWP displacement")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return stats


def plot_component_decomposition(out_path: Path, au: np.ndarray, av: np.ndarray) -> None:
    t = np.arange(len(au))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t, au[:, 0], label="A_u,x", linewidth=1.0)
    axes[0].plot(t, au[:, 1], label="A_u,y", linewidth=1.0)
    axes[0].set_ylabel("u-field displacement")
    axes[1].plot(t, av[:, 0], label="A_v,x", linewidth=1.0)
    axes[1].plot(t, av[:, 1], label="A_v,y", linewidth=1.0)
    axes[1].set_ylabel("v-field displacement")
    axes[2].plot(t, norm(au), label="|A_u|", linewidth=1.0)
    axes[2].plot(t, norm(av), label="|A_v|", linewidth=1.0)
    axes[2].set_ylabel("speed (km/step)")
    axes[2].set_xlabel("time index")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    fig.suptitle("Legacy component advection decomposition")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_windows(au: np.ndarray, av: np.ndarray, window: int = 6, count: int = 3) -> list[int]:
    if len(au) <= window + 1:
        return [0]
    score = norm(au[window:] - au[:-window]) + norm(av[window:] - av[:-window])
    order = np.argsort(score)[::-1]
    starts: list[int] = []
    for raw in order:
        s = int(raw)
        if all(abs(s - prev) >= window * 4 for prev in starts):
            starts.append(s)
        if len(starts) >= count:
            break
    return sorted(starts)


def block_wind_rose(
    fig: plt.Figure,
    outer_spec,
    u: np.ndarray,
    v: np.ndarray,
    bins: np.ndarray,
    title: str,
    blocks: int = 4,
) -> None:
    spacing = 0.08 if blocks <= 2 else 0.04 if blocks <= 4 else 0.015
    sub = outer_spec.subgridspec(blocks, blocks, wspace=spacing, hspace=spacing)
    h, w = u.shape[-2:]
    row_edges = np.linspace(0, h, blocks + 1).round().astype(int)
    col_edges = np.linspace(0, w, blocks + 1).round().astype(int)
    for r in range(blocks):
        for c in range(blocks):
            ax = fig.add_subplot(sub[r, c], projection="polar")
            uu = u[:, row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]].reshape(-1)
            vv = v[:, row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]].reshape(-1)
            theta = (np.arctan2(vv, uu) + np.pi + np.pi) % (2.0 * np.pi) - np.pi
            weights = np.sqrt(uu * uu + vv * vv)
            hist, edges = np.histogram(theta, bins=bins, weights=weights)
            widths = np.diff(edges)
            ax.bar(edges[:-1], hist, width=widths, align="edge", color="#4c78a8", alpha=0.75)
            ax.set_xticks([])
            ax.set_yticks([])
            label_size = 7 if blocks <= 2 else 6 if blocks <= 4 else 4
            ax.text(0.5, 1.01, f"{r+1},{c+1}", transform=ax.transAxes, ha="center", va="bottom", fontsize=label_size)
    fig.text(0.24, 0.965, title, ha="center", va="top", fontsize=12)


def plot_window_rose_and_advection(
    out_path: Path,
    nwp_path: Path,
    start: int,
    window: int,
    au: np.ndarray,
    av: np.ndarray,
    nwp_disp: np.ndarray,
    start_time: datetime,
    dt_seconds: float,
) -> dict[str, float]:
    with h5py.File(nwp_path, "r") as handle:
        grid = handle["allVariMin_Grid"]
        end = min(start + window, grid.shape[1])
        u = np.asarray(grid[U140_INDEX, start:end, :, :], dtype=np.float64)
        v = np.asarray(grid[V140_INDEX, start:end, :, :], dtype=np.float64)
    bins = np.linspace(-np.pi, np.pi, 17)
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.08])
    block_wind_rose(
        fig,
        gs[0],
        u,
        v,
        bins,
        f"NWP mid-scale wind direction, {time_range_label(start, end - 1, start_time, dt_seconds)}",
    )
    ax = fig.add_subplot(gs[1])
    times = np.arange(start, end)
    rel = (times - start) / max(end - start - 1, 1)
    cmap = plt.get_cmap("viridis")
    ax.plot(au[start:end, 0], au[start:end, 1], color="tab:blue", linewidth=1.5, alpha=0.7)
    ax.plot(av[start:end, 0], av[start:end, 1], color="tab:orange", linewidth=1.5, alpha=0.7)
    for local_i, t in enumerate(times):
        color = cmap(float(rel[local_i]))
        ax.scatter(au[t, 0], au[t, 1], color=color, edgecolor="tab:blue", s=55, marker="o")
        ax.scatter(av[t, 0], av[t, 1], color=color, edgecolor="tab:orange", s=55, marker="s")
    mean_au = au[start:end].mean(axis=0)
    mean_av = av[start:end].mean(axis=0)
    mean_nwp = nwp_disp[start:end].mean(axis=0)
    for vec, label, color, linestyle in [
        (mean_au, "mean A_u-like", "tab:blue", "-"),
        (mean_av, "mean A_v-like", "tab:orange", "-"),
        (mean_nwp, "mean NWP station displacement", "tab:green", "--"),
    ]:
        ax.arrow(
            0,
            0,
            vec[0],
            vec[1],
            width=0.025,
            head_width=0.25,
            length_includes_head=True,
            color=color,
            linestyle=linestyle,
            label=label,
            alpha=0.9,
        )
    all_points = np.concatenate([au[start:end], av[start:end], nwp_disp[start:end]], axis=0)
    lim = float(np.nanmax(np.abs(all_points))) * 1.15 + 1e-6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="0.4", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="0.4", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("x displacement (km/step)")
    ax.set_ylabel("y displacement (km/step)")
    ax.set_title("Component advection trajectory during same hour")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "start": int(start),
        "end": int(end - 1),
        "start_time": time_label(start, start_time, dt_seconds),
        "end_time": time_label(end - 1, start_time, dt_seconds),
        "mean_Au_x": float(mean_au[0]),
        "mean_Au_y": float(mean_au[1]),
        "mean_Av_x": float(mean_av[0]),
        "mean_Av_y": float(mean_av[1]),
        "mean_NWP_x": float(mean_nwp[0]),
        "mean_NWP_y": float(mean_nwp[1]),
        "median_abs_angle_Au_NWP_deg": float(np.nanmedian(np.abs(angle_diff_deg(au[start:end], nwp_disp[start:end])))),
        "median_abs_angle_Av_NWP_deg": float(np.nanmedian(np.abs(angle_diff_deg(av[start:end], nwp_disp[start:end])))),
    }


def plot_window_gif(
    out_path: Path,
    nwp_path: Path,
    start: int,
    frames: int,
    au: np.ndarray,
    av: np.ndarray,
    nwp_disp: np.ndarray,
    start_time: datetime,
    dt_seconds: float,
    blocks: int = 4,
    show_nwp_vector: bool = True,
    duration_ms: int = 450,
) -> dict[str, float]:
    end = min(start + frames, len(au))
    with h5py.File(nwp_path, "r") as handle:
        grid = handle["allVariMin_Grid"]
        end = min(end, grid.shape[1])
        u = np.asarray(grid[U140_INDEX, start:end, :, :], dtype=np.float64)
        v = np.asarray(grid[V140_INDEX, start:end, :, :], dtype=np.float64)

    bins = np.linspace(-np.pi, np.pi, 17)
    all_points = np.concatenate([au[start:end], av[start:end], nwp_disp[start:end]], axis=0)
    lim = float(np.nanmax(np.abs(all_points))) * 1.2 + 1e-6
    images: list[Image.Image] = []
    for local_t, t in enumerate(range(start, end)):
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.08])
        block_wind_rose(
            fig,
            gs[0],
            u[local_t : local_t + 1],
            v[local_t : local_t + 1],
            bins,
            f"NWP mid-scale wind direction, {time_label(t, start_time, dt_seconds)}",
            blocks=blocks,
        )
        ax = fig.add_subplot(gs[1])
        trace = slice(start, t + 1)
        ax.plot(au[trace, 0], au[trace, 1], color="tab:blue", linewidth=1.8, alpha=0.75)
        ax.plot(av[trace, 0], av[trace, 1], color="tab:orange", linewidth=1.8, alpha=0.75)
        ax.scatter(au[trace, 0], au[trace, 1], color="tab:blue", s=30, alpha=0.35)
        ax.scatter(av[trace, 0], av[trace, 1], color="tab:orange", s=30, alpha=0.35)
        vectors = [
            (au[t], "A_u-like current", "tab:blue"),
            (av[t], "A_v-like current", "tab:orange"),
        ]
        if show_nwp_vector:
            vectors.append((nwp_disp[t], "NWP station displacement current", "tab:green"))
        for vec, label, color in vectors:
            ax.arrow(
                0,
                0,
                vec[0],
                vec[1],
                width=max(lim * 0.006, 0.01),
                head_width=max(lim * 0.045, 0.08),
                length_includes_head=True,
                color=color,
                alpha=0.95,
                label=label,
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="0.4", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="0.4", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("x displacement (km/step)")
        ax.set_ylabel("y displacement (km/step)")
        ax.set_title("Advection components and local NWP displacement")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="best")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    if images:
        images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)

    au_abs = np.abs(au[start:end])
    av_abs = np.abs(av[start:end])
    nwp_abs = np.abs(nwp_disp[start:end])
    return {
        "start": int(start),
        "end": int(end - 1),
        "start_time": time_label(start, start_time, dt_seconds),
        "end_time": time_label(end - 1, start_time, dt_seconds),
        "mean_abs_Au_x": float(np.mean(au_abs[:, 0])),
        "mean_abs_Au_y": float(np.mean(au_abs[:, 1])),
        "mean_abs_Av_x": float(np.mean(av_abs[:, 0])),
        "mean_abs_Av_y": float(np.mean(av_abs[:, 1])),
        "mean_abs_NWP_x": float(np.mean(nwp_abs[:, 0])),
        "mean_abs_NWP_y": float(np.mean(nwp_abs[:, 1])),
        "Au_to_Av_speed_ratio": float(np.mean(norm(au[start:end])) / max(np.mean(norm(av[start:end])), 1e-12)),
        "median_abs_angle_Au_NWP_deg": float(np.nanmedian(np.abs(angle_diff_deg(au[start:end], nwp_disp[start:end])))),
        "median_abs_angle_Av_NWP_deg": float(np.nanmedian(np.abs(angle_diff_deg(av[start:end], nwp_disp[start:end])))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze legacy VectorWIDE advection outputs.")
    parser.add_argument("--eval-dir", type=Path, default=Path("/Users/felix/Downloads/eval_on-offline_h72_f72_hybrid"))
    parser.add_argument("--light-dir", type=Path, default=Path("/Users/felix/Downloads/light_on-offline_h72_f72_hybrid"))
    parser.add_argument("--arma-hour-csv", type=Path, default=Path("/Users/felix/Downloads/metrics_by_hour.csv"))
    parser.add_argument("--nwp-path", type=Path, default=Path("data/nwp/data_grid_online.mat"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--hour-steps", type=int, default=6)
    parser.add_argument("--gif-start", type=int, default=6898)
    parser.add_argument("--gif-frames", type=int, default=12)
    parser.add_argument("--start-time", default=DEFAULT_ONLINE_START)
    parser.add_argument("--dt-seconds", type=float, default=600.0)
    parser.add_argument("--apply-component-direction-bias", action="store_true")
    parser.add_argument("--au-x-scale", type=float, default=1.45)
    parser.add_argument("--au-y-scale", type=float, default=0.65)
    parser.add_argument("--av-x-scale", type=float, default=0.65)
    parser.add_argument("--av-y-scale", type=float, default=1.45)
    args = parser.parse_args()
    start_time = parse_start_time(args.start_time)

    out_dir = args.out_dir or args.eval_dir / "legacy_physics_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted_path = args.eval_dir / "legacy_advection_analysis" / "legacy_extracted_component_advection.npz"
    if not extracted_path.exists():
        raise FileNotFoundError(f"Missing extracted legacy advection file: {extracted_path}")
    legacy = np.load(extracted_path)
    au = legacy["A_u_like"].astype(np.float64)
    av = legacy["A_v_like"].astype(np.float64)
    nwp_disp = legacy["nwp_displacement"].astype(np.float64)
    raw_au = au.copy()
    raw_av = av.copy()
    direction_bias = None
    if args.apply_component_direction_bias:
        au, av = apply_component_direction_bias(
            au,
            av,
            args.au_x_scale,
            args.au_y_scale,
            args.av_x_scale,
            args.av_y_scale,
        )
        direction_bias = {
            "au_x_scale": float(args.au_x_scale),
            "au_y_scale": float(args.au_y_scale),
            "av_x_scale": float(args.av_x_scale),
            "av_y_scale": float(args.av_y_scale),
            "note": "Post-hoc anisotropic scaling for exploratory visualization, not a learned model output.",
        }
        np.savez_compressed(
            out_dir / "biased_component_advection.npz",
            raw_A_u_like=raw_au.astype(np.float32),
            raw_A_v_like=raw_av.astype(np.float32),
            biased_A_u_like=au.astype(np.float32),
            biased_A_v_like=av.astype(np.float32),
            nwp_displacement=nwp_disp.astype(np.float32),
            scales=np.asarray(
                [args.au_x_scale, args.au_y_scale, args.av_x_scale, args.av_y_scale],
                dtype=np.float32,
            ),
        )

    model_hour = read_csv(args.light_dir / "metrics_by_hour.csv")
    direction_hour = read_csv(args.light_dir / "direction_metrics_by_hour.csv")
    arma_hour = read_csv(args.arma_hour_csv)
    plot_benchmark(out_dir / "benchmark_with_arima_by_hour.png", model_hour, direction_hour, arma_hour)

    stats = plot_axis_comparisons(out_dir / "legacy_advection_axis_comparisons.png", au, av, nwp_disp)
    plot_component_decomposition(out_dir / "legacy_ux_uy_vx_vy_decomposition.png", au, av)

    selected = select_windows(au, av, window=args.hour_steps, count=3)
    window_summaries = []
    for rank, start in enumerate(selected, start=1):
        start_slug = time_slug(start, start_time, args.dt_seconds)
        end_slug = time_slug(start + args.hour_steps - 1, start_time, args.dt_seconds)
        summary = plot_window_rose_and_advection(
            out_dir / f"nwp_4x4_wind_from_rose_with_advection_window_{rank}_{start_slug}_{end_slug}.png",
            args.nwp_path,
            start,
            args.hour_steps,
            au,
            av,
            nwp_disp,
            start_time,
            args.dt_seconds,
        )
        window_summaries.append(summary)

    gif_start_slug = time_slug(args.gif_start, start_time, args.dt_seconds)
    gif_end_slug = time_slug(args.gif_start + args.gif_frames - 1, start_time, args.dt_seconds)
    gif_summary = plot_window_gif(
        out_dir / f"nwp_4x4_wind_from_rose_advection_{gif_start_slug}_{gif_end_slug}.gif",
        args.nwp_path,
        args.gif_start,
        args.gif_frames,
        au,
        av,
        nwp_disp,
        start_time,
        args.dt_seconds,
    )

    summary = {
        "axis_correlations": stats,
        "selected_hour_windows": window_summaries,
        "gif_window": gif_summary,
        "posthoc_direction_bias": direction_bias,
        "interpretation_notes": [
            "A_u_like and A_v_like are reconstructed from the legacy transition outputs.",
            "Mid-scale roses aggregate meteorological NWP 140m wind-from directions; bars are speed-weighted.",
            "The right panel compares legacy component advection with NWP station displacement.",
        ],
    }
    with (out_dir / "legacy_physics_benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
