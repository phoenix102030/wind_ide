from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.analyze_legacy_advection import (
    U140_INDEX,
    V140_INDEX,
    apply_component_direction_bias,
    angle_diff_deg,
    norm,
    parse_start_time,
    time_label,
    time_slug,
)


def block_means(values: np.ndarray, blocks: int = 4) -> np.ndarray:
    h, w = values.shape[-2:]
    row_edges = np.linspace(0, h, blocks + 1).round().astype(int)
    col_edges = np.linspace(0, w, blocks + 1).round().astype(int)
    out = np.zeros((values.shape[0], blocks, blocks), dtype=np.float64)
    for r in range(blocks):
        for c in range(blocks):
            out[:, r, c] = values[
                :,
                row_edges[r] : row_edges[r + 1],
                col_edges[c] : col_edges[c + 1],
            ].mean(axis=(1, 2))
    return out


def block_vectors(u: np.ndarray, v: np.ndarray, blocks: int = 4) -> tuple[np.ndarray, np.ndarray]:
    return block_means(u, blocks), block_means(v, blocks)


def circular_variance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    speed = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)
    w = speed + 1e-6
    c = (w * np.cos(theta)).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
    s = (w * np.sin(theta)).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
    return 1.0 - np.sqrt(c * c + s * s)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(x, np.ones(window, dtype=np.float64) / window, mode="valid")


def select_cases(u: np.ndarray, v: np.ndarray, frames: int, preferred_strong_start: int = 6898) -> dict[str, int]:
    speed = np.sqrt(u * u + v * v)
    mean_speed = speed.mean(axis=(1, 2))
    block_speed = block_means(speed)
    block_cv = block_speed.std(axis=(1, 2)) / np.maximum(block_speed.mean(axis=(1, 2)), 1e-6)
    circ = circular_variance(u, v)

    win_speed = rolling_mean(mean_speed, frames)
    win_cv = rolling_mean(block_cv, frames)
    win_circ = rolling_mean(circ, frames)
    valid = np.ones_like(win_speed, dtype=bool)
    valid[:24] = False
    valid[-24:] = False

    cases: dict[str, int] = {}
    cases["reference_20210721_2140"] = int(min(max(preferred_strong_start, 0), len(win_speed) - 1))

    excluded = [cases["reference_20210721_2140"]]
    weak_candidates = np.where(valid)[0]
    cases["weak_wind"] = int(weak_candidates[np.argsort(win_speed[weak_candidates])][0])
    excluded.append(cases["weak_wind"])

    for center in excluded:
        valid[max(0, center - 48) : min(len(valid), center + 48)] = False

    complex_candidates = np.where(valid & (win_speed > np.nanmedian(win_speed)))[0]
    cases["complex_direction"] = int(complex_candidates[np.argsort(win_circ[complex_candidates])[::-1]][0])

    valid2 = valid.copy()
    for center in [cases["complex_direction"]]:
        valid2[max(0, center - 48) : min(len(valid2), center + 48)] = False
    hetero_candidates = np.where(valid2 & (win_speed > np.nanmedian(win_speed)))[0]
    cases["speed_heterogeneous"] = int(hetero_candidates[np.argsort(win_cv[hetero_candidates])[::-1]][0])
    return cases


def draw_speed_direction_panel(
    ax: plt.Axes,
    full_speed: np.ndarray,
    block_u: np.ndarray,
    block_v: np.ndarray,
    vmax: float,
    cmap_name: str = "YlOrRd",
) -> None:
    arrow_blocks = int(block_u.shape[0])
    im = ax.imshow(
        full_speed,
        cmap=cmap_name,
        vmin=0.0,
        vmax=vmax,
        origin="upper",
        interpolation="bilinear",
        extent=(-0.5, arrow_blocks - 0.5, arrow_blocks - 0.5, -0.5),
    )
    ax.set_xticks(np.arange(arrow_blocks), labels=[str(i) for i in range(1, arrow_blocks + 1)])
    ax.set_yticks(np.arange(arrow_blocks), labels=[str(i) for i in range(1, arrow_blocks + 1)])
    ax.set_xlabel("arrow block column")
    ax.set_ylabel("arrow block row")
    ax.set_title("NWP mid-scale wind speed")
    for r in range(arrow_blocks):
        for c in range(arrow_blocks):
            u = float(block_u[r, c])
            v = float(block_v[r, c])
            sp = float(np.sqrt(u * u + v * v))
            if sp <= 1e-6:
                continue
            # Vector overlay uses the physical wind-vector direction: blowing toward.
            to_vec = np.asarray([u, v], dtype=np.float64)
            to_vec /= max(np.linalg.norm(to_vec), 1e-12)
            speed_ratio = np.clip(sp / max(vmax, 1e-6), 0.0, 1.0)
            length = 0.12 + 0.48 * speed_ratio
            ax.arrow(
                c,
                r,
                length * to_vec[0],
                -length * to_vec[1],
                color="white",
                edgecolor="black",
                linewidth=1.6,
                head_width=0.11,
                head_length=0.09,
                length_includes_head=True,
            )
            if arrow_blocks <= 4:
                ax.text(
                    c,
                    r + 0.34,
                    f"{sp:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
                )
    ax.set_xlim(-0.5, arrow_blocks - 0.5)
    ax.set_ylim(arrow_blocks - 0.5, -0.5)
    ax.grid(color="white", linewidth=1.0, alpha=0.7)
    return im


def plot_case_gif(
    out_path: Path,
    nwp_path: Path,
    start: int,
    frames: int,
    au: np.ndarray,
    av: np.ndarray,
    nwp_disp: np.ndarray,
    start_time,
    dt_seconds: float,
    case_label: str,
    speed_vmax: float | None = None,
    vector_lim: float | None = None,
    arrow_blocks: int = 4,
    show_nwp_vector: bool = True,
    cmap_name: str = "YlOrRd",
    duration_ms: int = 450,
) -> dict[str, float]:
    end = min(start + frames, len(au))
    with h5py.File(nwp_path, "r") as handle:
        grid = handle["allVariMin_Grid"]
        end = min(end, grid.shape[1])
        u = np.asarray(grid[U140_INDEX, start:end, :, :], dtype=np.float32)
        v = np.asarray(grid[V140_INDEX, start:end, :, :], dtype=np.float32)

    speed = np.sqrt(u * u + v * v)
    bu, bv = block_vectors(u, v, blocks=arrow_blocks)
    bs = block_means(speed, blocks=arrow_blocks)
    vmax = float(speed_vmax if speed_vmax is not None else max(1.0, np.nanpercentile(speed, 98)))
    all_points = np.concatenate([au[start:end], av[start:end], nwp_disp[start:end]], axis=0)
    lim = float(vector_lim if vector_lim is not None else np.nanmax(np.abs(all_points)) * 1.2 + 1e-6)
    images: list[Image.Image] = []
    for local_t, t in enumerate(range(start, end)):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        im = draw_speed_direction_panel(axes[0], speed[local_t], bu[local_t], bv[local_t], vmax, cmap_name=cmap_name)
        cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label("NWP 140m wind speed")
        axes[0].text(0.02, 1.05, time_label(t, start_time, dt_seconds), transform=axes[0].transAxes, ha="left", va="bottom", fontsize=12)
        axes[0].text(
            0.02,
            -0.12,
            "white arrow points toward wind blowing direction",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        ax = axes[1]
        trace = slice(start, t + 1)
        au_color = "#1f77b4"
        av_color = "#d95f02"
        ax.plot(au[trace, 0], au[trace, 1], color="0.55", linewidth=1.4, alpha=0.45, linestyle="--")
        ax.plot(av[trace, 0], av[trace, 1], color="0.65", linewidth=1.4, alpha=0.45, linestyle="--")
        ax.scatter(au[trace, 0], au[trace, 1], color="0.55", s=22, alpha=0.25)
        ax.scatter(av[trace, 0], av[trace, 1], color="0.65", s=22, alpha=0.25)
        vectors = [
            (au[t], "A_u-like current", au_color),
            (av[t], "A_v-like current", av_color),
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
        if show_nwp_vector:
            ax.set_title("Advection components and station-scale NWP displacement")
        else:
            ax.set_title("Advection component comparison")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="best")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    if images:
        images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)

    domain_speed = speed.mean(axis=(1, 2))
    block_cv = bs.std(axis=(1, 2)) / np.maximum(bs.mean(axis=(1, 2)), 1e-6)
    circ = circular_variance(u, v)
    return {
        "case": case_label,
        "arrow_blocks": int(arrow_blocks),
        "cmap": str(cmap_name),
        "start_index": int(start),
        "end_index": int(end - 1),
        "start_time": time_label(start, start_time, dt_seconds),
        "end_time": time_label(end - 1, start_time, dt_seconds),
        "mean_domain_speed": float(domain_speed.mean()),
        "mean_block_speed_cv": float(block_cv.mean()),
        "mean_direction_complexity": float(circ.mean()),
        "mean_abs_Au": float(norm(au[start:end]).mean()),
        "mean_abs_Av": float(norm(av[start:end]).mean()),
        "mean_abs_NWP_station_displacement": float(norm(nwp_disp[start:end]).mean()),
        "median_abs_angle_Au_NWP_deg": float(np.nanmedian(np.abs(angle_diff_deg(au[start:end], nwp_disp[start:end])))),
        "median_abs_angle_Av_NWP_deg": float(np.nanmedian(np.abs(angle_diff_deg(av[start:end], nwp_disp[start:end])))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot legacy advection cases with mid-scale NWP speed/direction maps.")
    parser.add_argument("--eval-dir", type=Path, default=Path("/Users/felix/Downloads/eval_on-offline_h72_f72_hybrid"))
    parser.add_argument("--nwp-path", type=Path, default=Path("data/nwp/data_grid_online.mat"))
    parser.add_argument("--out-dir", type=Path, default=Path("/Users/felix/Downloads/eval_on-offline_h72_f72_hybrid/legacy_physics_speed_cases"))
    parser.add_argument("--start-time", default="2021-06-04 00:00:00")
    parser.add_argument("--dt-seconds", type=float, default=600.0)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--au-x-scale", type=float, default=1.45)
    parser.add_argument("--au-y-scale", type=float, default=0.65)
    parser.add_argument("--av-x-scale", type=float, default=0.65)
    parser.add_argument("--av-y-scale", type=float, default=1.45)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_time = parse_start_time(args.start_time)
    legacy = np.load(args.eval_dir / "legacy_advection_analysis" / "legacy_extracted_component_advection.npz")
    au_raw = legacy["A_u_like"].astype(np.float64)
    av_raw = legacy["A_v_like"].astype(np.float64)
    au, av = apply_component_direction_bias(
        au_raw,
        av_raw,
        args.au_x_scale,
        args.au_y_scale,
        args.av_x_scale,
        args.av_y_scale,
    )
    nwp_disp = legacy["nwp_displacement"].astype(np.float64)

    with h5py.File(args.nwp_path, "r") as handle:
        grid = handle["allVariMin_Grid"]
        t = min(grid.shape[1], len(au))
        u = np.asarray(grid[U140_INDEX, :t, :, :], dtype=np.float32)
        v = np.asarray(grid[V140_INDEX, :t, :, :], dtype=np.float32)
    cases = select_cases(u, v, args.frames)
    case_slices = [slice(start, min(start + args.frames, len(au))) for start in cases.values()]
    all_speed_blocks = []
    all_vectors = []
    for sl in case_slices:
        speed = np.sqrt(u[sl] * u[sl] + v[sl] * v[sl])
        all_speed_blocks.append(block_means(speed).reshape(-1))
        all_vectors.append(au[sl])
        all_vectors.append(av[sl])
        all_vectors.append(nwp_disp[sl])
    shared_speed_vmax = float(max(1.0, np.nanpercentile(np.concatenate(all_speed_blocks), 98)))
    shared_vector_lim = float(np.nanmax(np.abs(np.concatenate(all_vectors, axis=0))) * 1.2 + 1e-6)
    summaries = {}
    for label, start in cases.items():
        path = args.out_dir / (
            f"speed_direction_advection_{label}_"
            f"{time_slug(start, start_time, args.dt_seconds)}_"
            f"{time_slug(start + args.frames - 1, start_time, args.dt_seconds)}.gif"
        )
        summaries[label] = plot_case_gif(
            path,
            args.nwp_path,
            start,
            args.frames,
            au,
            av,
            nwp_disp,
            start_time,
            args.dt_seconds,
            label.replace("_", " "),
            speed_vmax=shared_speed_vmax,
            vector_lim=shared_vector_lim,
        )
        summaries[label]["gif"] = str(path)
        summaries[label]["shared_speed_colorbar_vmax"] = shared_speed_vmax
        summaries[label]["shared_vector_axis_lim"] = shared_vector_lim

    with (args.out_dir / "speed_direction_case_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
