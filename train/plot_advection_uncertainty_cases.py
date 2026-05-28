from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.analyze_vector_mide_physics import (  # noqa: E402
    MODEL_COLOR,
    NWP_COLOR,
    PALETTE,
    advection_covariance_for_flow,
    angle_diff_deg,
    bold_tick_labels,
    load_npz,
    load_optional_dataset,
    mean_station_vector,
    normal_tick_labels,
    polish_axes,
    setup_matplotlib,
    station_speed,
)


def circular_range_deg(angles_deg: np.ndarray) -> float:
    angles = np.asarray(angles_deg, dtype=np.float64)
    angles = angles[np.isfinite(angles)]
    if angles.size <= 1:
        return float("nan")
    angles = np.sort(angles % 360.0)
    gaps = np.diff(np.r_[angles, angles[0] + 360.0])
    return float(360.0 - np.nanmax(gaps))


def circular_std_deg(angles_deg: np.ndarray) -> float:
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    angles = angles[np.isfinite(angles)]
    if angles.size <= 1:
        return float("nan")
    r = np.abs(np.nanmean(np.exp(1j * angles)))
    return float(np.rad2deg(np.sqrt(max(-2.0 * np.log(max(r, 1e-8)), 0.0))))


def wind_dir_deg(vec: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 360.0


def window_metrics(nwp_vec: np.ndarray, nwp_speed: np.ndarray, cov: np.ndarray, window: int) -> list[dict[str, float]]:
    angles = wind_dir_deg(nwp_vec)
    rows: list[dict[str, float]] = []
    for start in range(0, len(nwp_speed) - window + 1):
        end = start + window
        speed_w = nwp_speed[start:end]
        angle_w = angles[start:end]
        vec_w = nwp_vec[start:end]
        cov_w = cov[start:end]
        if not (np.isfinite(speed_w).all() and np.isfinite(vec_w).all() and np.isfinite(cov_w).all()):
            continue
        S = 0.5 * (cov_w + np.swapaxes(cov_w, -1, -2))
        offdiag_abs_mean = float(np.nanmean(np.abs(S[:, 0, 1])))
        offdiag_range = float(np.nanmax(S[:, 0, 1]) - np.nanmin(S[:, 0, 1]))
        trace = S[:, 0, 0] + S[:, 1, 1]
        trace_mean = float(np.nanmean(S[:, 0, 0] + S[:, 1, 1]))
        trace_range = float(np.nanmax(trace) - np.nanmin(trace))
        trace_ratio = float(np.nanmax(trace) / (np.nanmin(trace) + 1e-8))
        axis_error_list = []
        axis_angle_list = []
        for St in S:
            vals, vecs = np.linalg.eigh(St + 1e-8 * np.eye(2))
            axis = vecs[:, int(np.nanargmax(vals))]
            angle = abs(float(np.degrees(np.arctan2(axis[1], axis[0])))) % 180.0
            axis_angle_list.append(angle)
            axis_error_list.append(min(angle, abs(angle - 90.0), abs(angle - 180.0)))
        axis_aligned_error_deg = float(np.nanmean(axis_error_list))
        axis_angle_range_deg = circular_range_deg(np.asarray(axis_angle_list))
        direction_step_sum = float(np.nansum(np.abs(angle_diff_deg(vec_w[1:], vec_w[:-1]))))
        speed_mean = float(np.nanmean(speed_w))
        speed_std = float(np.nanstd(speed_w))
        speed_range = float(np.nanmax(speed_w) - np.nanmin(speed_w))
        speed_cv = speed_std / (abs(speed_mean) + 1e-8)
        dir_range = circular_range_deg(angle_w)
        dir_std = circular_std_deg(angle_w)
        rows.append(
            {
                "start": float(start),
                "end": float(end - 1),
                "speed_mean": speed_mean,
                "speed_std": speed_std,
                "speed_range": speed_range,
                "speed_cv": float(speed_cv),
                "direction_range_deg": dir_range,
                "direction_std_deg": dir_std,
                "direction_step_sum_deg": direction_step_sum,
                "cov_offdiag_abs_mean": offdiag_abs_mean,
                "cov_offdiag_range": offdiag_range,
                "cov_trace_mean": trace_mean,
                "cov_trace_range": trace_range,
                "cov_trace_ratio": trace_ratio,
                "cov_axis_aligned_error_deg": axis_aligned_error_deg,
                "cov_axis_angle_range_deg": axis_angle_range_deg,
            }
        )
    return rows


def select_cases(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    if not rows:
        raise ValueError("No valid one-hour windows found")
    speed_cv = np.asarray([r["speed_cv"] for r in rows])
    speed_range = np.asarray([r["speed_range"] for r in rows])
    dir_range = np.asarray([r["direction_range_deg"] for r in rows])
    dir_std = np.asarray([r["direction_std_deg"] for r in rows])
    speed_mean = np.asarray([r["speed_mean"] for r in rows])
    offdiag = np.asarray([r["cov_offdiag_abs_mean"] for r in rows])
    axis_error = np.asarray([r["cov_axis_aligned_error_deg"] for r in rows])

    # Case A: direction changes strongly while speed remains stable and the covariance ellipse stays near horizontal/vertical.
    stable_speed = speed_cv <= np.nanpercentile(speed_cv, 40)
    active_speed = speed_mean >= np.nanpercentile(speed_mean, 35)
    strong_direction_shift = dir_range >= 20.0
    axis_aligned_cov = (offdiag <= np.nanpercentile(offdiag, 20)) | (axis_error <= np.nanpercentile(axis_error, 20))
    candidate_a = stable_speed & active_speed & strong_direction_shift & axis_aligned_cov
    if not candidate_a.any():
        candidate_a = stable_speed & active_speed & strong_direction_shift
    if not candidate_a.any():
        candidate_a = stable_speed & active_speed
    score_a = dir_range / (speed_cv + 0.02) / (1.0 + 80.0 * offdiag + axis_error / 8.0)
    score_a = np.where(candidate_a, score_a, -np.inf)
    idx_a = int(np.nanargmax(score_a))

    # Case B: speed magnitude changes strongly while direction remains stable.
    stable_direction = dir_std <= np.nanpercentile(dir_std, 30)
    candidate_b = stable_direction & (speed_range >= np.nanpercentile(speed_range, 70))
    if not candidate_b.any():
        candidate_b = stable_direction
    score_b = speed_range / (dir_std + 2.0)
    score_b = np.where(candidate_b, score_b, -np.inf)
    idx_b = int(np.nanargmax(score_b))
    if idx_b == idx_a:
        score_b[idx_b] = -np.inf
        idx_b = int(np.nanargmax(score_b))

    # Case D: covariance changes strongly in both size and orientation/cross-covariance.
    trace_range = np.asarray([r["cov_trace_range"] for r in rows])
    trace_ratio = np.asarray([r["cov_trace_ratio"] for r in rows])
    axis_angle_range = np.asarray([r["cov_axis_angle_range_deg"] for r in rows])
    offdiag_range = np.asarray([r["cov_offdiag_range"] for r in rows])
    candidate_d = (trace_range >= np.nanpercentile(trace_range, 90)) & (
        (axis_angle_range >= np.nanpercentile(axis_angle_range, 80))
        | (offdiag_range >= np.nanpercentile(offdiag_range, 80))
    )
    if not candidate_d.any():
        candidate_d = trace_range >= np.nanpercentile(trace_range, 90)
    score_d = (
        trace_range / (np.nanmedian(trace_range) + 1e-8)
        + 0.35 * trace_ratio / (np.nanmedian(trace_ratio) + 1e-8)
        + axis_angle_range / 30.0
        + offdiag_range / (np.nanmedian(offdiag_range) + 1e-8)
    )
    score_d = np.where(candidate_d, score_d, -np.inf)
    score_d[[idx_a, idx_b]] = -np.inf
    idx_d = int(np.nanargmax(score_d))

    # Case E: large wind-direction change with a visually meaningful wind magnitude.
    direction_step_sum = np.asarray([r["direction_step_sum_deg"] for r in rows])
    candidate_e = (speed_mean >= 8.0) & (dir_range >= 30.0)
    if not candidate_e.any():
        candidate_e = (speed_mean >= 5.0) & (dir_range >= 45.0)
    if not candidate_e.any():
        candidate_e = dir_range >= np.nanpercentile(dir_range, 95)
    score_e = dir_range + 0.25 * direction_step_sum + 1.5 * speed_mean + 2.0 * speed_range
    score_e = np.where(candidate_e, score_e, -np.inf)
    score_e[[idx_a, idx_b, idx_d]] = -np.inf
    idx_e = int(np.nanargmax(score_e))

    return rows[idx_a], rows[idx_b], rows[idx_d], rows[idx_e]


def covariance_entries(cov: np.ndarray, sl: slice) -> dict[str, np.ndarray]:
    S = 0.5 * (cov[sl] + np.swapaxes(cov[sl], -1, -2))
    return {
        "sigma_uu": S[:, 0, 0],
        "sigma_vv": S[:, 1, 1],
        "sigma_uv": S[:, 0, 1],
        "trace": S[:, 0, 0] + S[:, 1, 1],
    }


def ellipse_params(S: np.ndarray) -> tuple[float, float, float, float]:
    S = 0.5 * (S + S.T) + 1e-8 * np.eye(2)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-12, None)
    angle = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    return float(vals[0]), float(vals[1]), float(4.0 * np.sqrt(vals[1])), angle


def selected_time_indices(cases: dict[str, dict[str, float]]) -> np.ndarray:
    times: list[int] = []
    for case in cases.values():
        times.extend(range(int(case["start"]), int(case["end"]) + 1))
    return np.asarray(sorted(set(times)), dtype=np.int64)


def nice_limit(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10.0**exponent)
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= step:
            return float(step * 10.0**exponent)
    return float(10.0 ** (exponent + 1.0))


def global_ellipse_limit(cases: dict[str, dict[str, float]], cov: np.ndarray, flow: np.ndarray) -> float:
    times = selected_time_indices(cases)
    if times.size == 0:
        return 1.0
    max_mean_component = float(np.nanmax(np.abs(flow[times])))
    max_cov_radius = 0.0
    for t in times:
        S = 0.5 * (cov[t] + cov[t].T) + 1e-8 * np.eye(2)
        vals = np.clip(np.linalg.eigvalsh(S), 1e-12, None)
        max_cov_radius = max(max_cov_radius, float(2.0 * np.sqrt(np.nanmax(vals))))
    return nice_limit(1.10 * max(max_mean_component, max_cov_radius, 1e-6))


def global_speed_scale(cases: dict[str, dict[str, float]], optional_data: dict[str, np.ndarray]) -> tuple[float | None, float | None]:
    if not {"raw_u140_grid", "raw_v140_grid"}.issubset(optional_data):
        return None, None
    u = np.asarray(optional_data["raw_u140_grid"])
    v = np.asarray(optional_data["raw_v140_grid"])
    times = selected_time_indices(cases)
    times = times[(times >= 0) & (times < u.shape[0])]
    if times.size == 0:
        return None, None
    speed_grid = np.sqrt(u[times] ** 2 + v[times] ** 2)
    vmin = float(np.nanpercentile(speed_grid, 2))
    vmax = float(np.nanpercentile(speed_grid, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None, None
    return vmin, vmax


def draw_covariance_panel(ax, S: np.ndarray, flow_vec: np.ndarray, title: str, lim: float = 1.0) -> None:
    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(0.5 * (S + S.T) + 1e-8 * np.eye(2))
    vals = np.clip(vals, 1e-12, None)
    angle = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    width = 4.0 * np.sqrt(vals[1])
    height = 4.0 * np.sqrt(vals[0])
    ell = Ellipse((0.0, 0.0), width=width, height=height, angle=angle, facecolor=PALETTE["light_blue"], edgecolor=MODEL_COLOR, alpha=0.55, linewidth=2.0)
    ax.add_patch(ell)
    radius = float(lim)
    ax.arrow(
        0,
        0,
        float(flow_vec[0]),
        float(flow_vec[1]),
        color=MODEL_COLOR,
        width=0.006 * radius,
        head_width=0.045 * radius,
        head_length=0.055 * radius,
        length_includes_head=True,
        label="learned mean",
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(np.linspace(-lim, lim, 5))
    ax.set_yticks(np.linspace(-lim, lim, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    polish_axes(ax)


def save_case_plots(
    case_name: str,
    case: dict[str, float],
    cov: np.ndarray,
    flow: np.ndarray,
    nwp_vec: np.ndarray,
    nwp_speed: np.ndarray,
    out: Path,
    ellipse_lim: float,
) -> None:
    plt = setup_matplotlib()
    start = int(case["start"])
    end = int(case["end"])
    sl = slice(start, end + 1)
    x = np.arange(start, end + 1)
    entries = covariance_entries(cov, sl)
    angles = wind_dir_deg(nwp_vec[sl])

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(x, nwp_speed[sl], color=NWP_COLOR, marker="o", linewidth=2.4)
    axes[0].set_title(f"{case_name}: one-hour wind regime")
    axes[0].set_ylabel("Mean speed")
    polish_axes(axes[0])
    axes[1].plot(x, angles, color=PALETTE["green"], marker="o", linewidth=2.4)
    axes[1].set_ylabel("Direction (deg)")
    polish_axes(axes[1])
    axes[2].plot(x, entries["sigma_uu"], color=MODEL_COLOR, marker="o", linewidth=2.2, label=r"$\Sigma_{UU}$")
    axes[2].plot(x, entries["sigma_vv"], color=PALETTE["purple"], marker="o", linewidth=2.2, label=r"$\Sigma_{VV}$")
    axes[2].plot(x, entries["trace"], color=PALETTE["red"], linestyle="--", marker="o", linewidth=2.2, label=r"$trace(\Sigma)$")
    axes[2].set_ylabel("Diagonal variance")
    axes[2].set_xlabel("Time index")
    axes[2].legend(loc="best", fontsize=10)
    polish_axes(axes[2])
    fig.savefig(out / f"advection_uncertainty_{case_name}_diagonal_covariance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="#333333", linewidth=1.2)
    ax.plot(x, entries["sigma_uv"], color=PALETTE["teal"], marker="o", linewidth=2.6, label=r"$\Sigma_{UV}=\Sigma_{VU}$")
    ax.set_title(f"{case_name}: off-diagonal covariance")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Off-diagonal covariance")
    ax.legend(loc="best", fontsize=10)
    polish_axes(ax)
    fig.savefig(out / f"advection_uncertainty_{case_name}_offdiagonal_covariance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="#333333", linewidth=1.2)
    ax.plot(x, flow[sl, 0], color=MODEL_COLOR, marker="o", linewidth=2.6, label=r"$\mu_U$")
    ax.plot(x, flow[sl, 1], color=PALETTE["purple"], marker="o", linewidth=2.6, label=r"$\mu_V$")
    ax.set_title(f"{case_name}: learned advection mean components")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Mean component")
    ax.legend(loc="best", fontsize=10)
    polish_axes(ax)
    fig.savefig(out / f"advection_uncertainty_{case_name}_mean_components.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(x), figsize=(3.0 * len(x), 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, x):
        draw_covariance_panel(
            ax,
            cov[t],
            flow[t],
            f"t={t}\nmu=({flow[t, 0]:.2f},{flow[t, 1]:.2f})\ntr={entries['trace'][t - start]:.3g}",
            lim=ellipse_lim,
        )
    axes[0].legend(loc="upper left", fontsize=7)
    fig.savefig(out / f"advection_uncertainty_{case_name}_ellipse_sequence.png")
    plt.close(fig)


def save_case_gif(
    case_name: str,
    case: dict[str, float],
    cov: np.ndarray,
    flow: np.ndarray,
    nwp_vec: np.ndarray,
    nwp_speed: np.ndarray,
    optional_data: dict[str, np.ndarray],
    out: Path,
    ellipse_lim: float,
    speed_vmin: float | None,
    speed_vmax: float | None,
) -> None:
    required = ["raw_u140_grid", "raw_v140_grid", "lat_grid", "lon_grid", "baseline_grid_indices"]
    if not all(key in optional_data for key in required):
        return
    plt = setup_matplotlib()
    import matplotlib.animation as animation

    u = np.asarray(optional_data["raw_u140_grid"])
    v = np.asarray(optional_data["raw_v140_grid"])
    lat_grid = np.asarray(optional_data["lat_grid"])
    lon_grid = np.asarray(optional_data["lon_grid"])
    station_idx = np.asarray(optional_data["baseline_grid_indices"], dtype=np.int64)
    station_lats = lat_grid[station_idx[:, 0], station_idx[:, 1]]
    station_lons = lon_grid[station_idx[:, 0], station_idx[:, 1]]
    start = int(case["start"])
    end = int(case["end"])
    times = np.arange(start, end + 1)
    if speed_vmin is None or speed_vmax is None:
        speed_grid = np.sqrt(u[times] ** 2 + v[times] ** 2)
        vmin = float(np.nanpercentile(speed_grid, 2))
        vmax = float(np.nanpercentile(speed_grid, 98))
    else:
        vmin = float(speed_vmin)
        vmax = float(speed_vmax)
    stride = max(1, u.shape[1] // 16)
    lon_sample = lon_grid[::stride, ::stride]
    lat_sample = lat_grid[::stride, ::stride]
    extent = [float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid)), float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    map_ax, ell_ax = axes

    def update(frame_idx: int):
        t = int(times[frame_idx])
        map_ax.clear()
        ell_ax.clear()
        im = map_ax.imshow(
            np.sqrt(u[t] ** 2 + v[t] ** 2),
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=extent,
            aspect="auto",
        )
        map_ax.quiver(
            lon_sample,
            lat_sample,
            u[t, ::stride, ::stride],
            v[t, ::stride, ::stride],
            color="#F2F2F2",
            alpha=0.9,
            angles="uv",
            scale=350,
            width=0.0032,
        )
        map_ax.scatter(station_lons, station_lats, marker="*", s=170, color=PALETTE["red"], edgecolor="white", linewidth=1.0, zorder=5)
        map_ax.set_title(f"{case_name}: NWP wind field, t={t}", fontsize=12, fontweight="bold")
        map_ax.set_xlabel("Longitude")
        map_ax.set_ylabel("Latitude")
        map_ax.tick_params(axis="both", labelsize=8.5)
        normal_tick_labels(map_ax)
        for spine in map_ax.spines.values():
            spine.set_linewidth(1.1)

        draw_covariance_panel(
            ell_ax,
            cov[t],
            flow[t],
            f"Covariance ellipse\nmu_U={flow[t, 0]:.2f}, mu_V={flow[t, 1]:.2f}",
            lim=ellipse_lim,
        )
        return [im]

    update(0)
    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=700, blit=False)
    writer = animation.PillowWriter(fps=1.4)
    ani.save(out / f"advection_uncertainty_{case_name}.gif", writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("/Users/felix/Downloads/cnn_transformer_new/eval_online_h36_f36_o"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("yml_files/VectorMIDE_cuda.yaml"))
    parser.add_argument("--split", default="online")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    out = args.output_dir or (eval_dir / "physics_analysis")
    out.mkdir(parents=True, exist_ok=True)
    forecasts = load_npz(eval_dir / "forecasts.npz")
    advection = load_npz(eval_dir / "advection_parameters.npz")
    flow = advection.get("processed_flow_mu")
    if flow is None or flow.size == 0:
        flow = advection["flow_mu"]
    cov = advection_covariance_for_flow(advection, flow)
    if cov is None:
        raise ValueError("No advection covariance available")
    nwp_vec = mean_station_vector(forecasts["nwp_baseline"])
    nwp_speed = np.nanmean(station_speed(forecasts["nwp_baseline"]), axis=1)
    optional_data = load_optional_dataset(args.config, args.split, expected_t=flow.shape[0])

    rows = window_metrics(nwp_vec, nwp_speed, cov, window=6)
    case_a, case_b, case_d, case_e = select_cases(rows)
    cases = {
        "case_A_direction_shift_speed_stable": case_a,
        "case_B_speed_shift_direction_stable": case_b,
        "case_D_covariance_size_and_direction_shift": case_d,
        "case_E_extreme_direction_shift": case_e,
    }
    ellipse_lim = global_ellipse_limit(cases, cov, flow)
    speed_vmin, speed_vmax = global_speed_scale(cases, optional_data)

    for name, case in cases.items():
        save_case_plots(name, case, cov, flow, nwp_vec, nwp_speed, out, ellipse_lim)
        save_case_gif(name, case, cov, flow, nwp_vec, nwp_speed, optional_data, out, ellipse_lim, speed_vmin, speed_vmax)

    rows_out = []
    for name, case in cases.items():
        row = {"case": name, **case}
        rows_out.append(row)
    csv_header = [
        "case",
        "start",
        "end",
        "speed_mean",
        "speed_std",
        "speed_range",
        "speed_cv",
        "direction_range_deg",
        "direction_std_deg",
        "direction_step_sum_deg",
        "cov_offdiag_abs_mean",
        "cov_offdiag_range",
        "cov_trace_mean",
        "cov_trace_range",
        "cov_trace_ratio",
        "cov_axis_aligned_error_deg",
        "cov_axis_angle_range_deg",
    ]
    with (out / "advection_uncertainty_selected_cases.csv").open("w", encoding="utf-8") as handle:
        handle.write(",".join(csv_header) + "\n")
        for row in rows_out:
            handle.write(",".join(str(row[key]) for key in csv_header) + "\n")
    (out / "advection_uncertainty_selected_cases.json").write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    scale_metadata = {
        "ellipse_axis_limit": ellipse_lim,
        "ellipse_axis_units": "same units as processed_flow_mu",
        "mean_arrow_normalized": False,
        "speed_color_vmin": speed_vmin,
        "speed_color_vmax": speed_vmax,
        "speed_color_units": "m/s",
    }
    (out / "advection_uncertainty_plot_scales.json").write_text(json.dumps(scale_metadata, indent=2), encoding="utf-8")
    print(f"Wrote selected advection uncertainty cases to {out}")


if __name__ == "__main__":
    main()
