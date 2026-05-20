from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STATE_NAMES = [
    "U_E05_140m",
    "U_E06_140m",
    "U_ASOW6_140m",
    "V_E05_140m",
    "V_E06_140m",
    "V_ASOW6_140m",
]
STATION_NAMES = ["E05", "E06", "ASOW6"]
PALETTE = {
    "navy": "#1B365D",
    "blue": "#2878B5",
    "cyan": "#45B5C4",
    "teal": "#008B8B",
    "green": "#4C956C",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "purple": "#7B3F98",
    "gray": "#4A4A4A",
    "light_blue": "#A7C7E7",
    "red": "#C1121F",
}
MODEL_COLOR = PALETTE["blue"]
NWP_COLOR = PALETTE["orange"]
PERSISTENCE_COLOR = PALETTE["gold"]
MEAS_COLOR = "#171717"
INTERVAL_COLOR = PALETTE["light_blue"]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = None
    for arr in arrays:
        arr_mask = np.isfinite(arr).all(axis=tuple(range(1, arr.ndim))) if arr.ndim > 1 else np.isfinite(arr)
        mask = arr_mask if mask is None else (mask & arr_mask)
    return np.asarray(mask, dtype=bool)


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(pred) & np.isfinite(target)
    if mask is not None:
        valid = valid & mask
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((pred[valid] - target[valid]) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(pred) & np.isfinite(target)
    if mask is not None:
        valid = valid & mask
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - target[valid])))


def mean_station_vector(values: np.ndarray) -> np.ndarray:
    """Return [T,2] mean U/V vector across the three stations."""
    return np.stack([np.nanmean(values[:, :3], axis=1), np.nanmean(values[:, 3:], axis=1)], axis=1)


def station_speed(values: np.ndarray) -> np.ndarray:
    return np.sqrt(values[:, :3] ** 2 + values[:, 3:] ** 2)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.degrees(np.arctan2(cross, dot))


def vector_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return dot / denom


def quantile_bins(values: np.ndarray, n_bins: int = 4) -> list[np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    edges = np.unique(np.nanquantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    bins = []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            bins.append((values >= edges[i]) & (values <= edges[i + 1]))
        else:
            bins.append((values >= edges[i]) & (values < edges[i + 1]))
    return bins


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("default")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 11,
            "axes.linewidth": 1.35,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "grid.color": "#B8B8B8",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.24,
            "lines.linewidth": 2.3,
            "patch.linewidth": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    return plt


def bold_tick_labels(ax) -> None:
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")


def normal_tick_labels(ax) -> None:
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("normal")


def polish_axes(ax, grid: bool = True) -> None:
    ax.tick_params(axis="both", which="major", labelsize=15, width=1.35, length=5.5)
    bold_tick_labels(ax)
    if grid:
        ax.grid(True, alpha=0.24)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color("#252525")


def panel_label(ax, label: str, x: float = 0.02, y: float = 0.96) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": PALETTE["navy"],
            "linewidth": 1.2,
            "alpha": 0.95,
        },
        zorder=10,
    )


def load_optional_dataset(
    config_path: Path | None,
    split: str | None,
    expected_t: int,
) -> dict[str, Any]:
    if config_path is None:
        return {}
    try:
        import yaml

        from dataset.vector_data_utils import load_nwp_uv140, load_vector_dataset

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        data = load_vector_dataset(config, split=split or "online")
        out: dict[str, Any] = {}
        for key in ["V_star", "B_star", "coords", "station_latlon", "baseline_grid_indices", "lat_grid", "lon_grid"]:
            value = data.get(key)
            if value is not None:
                out[key] = value
        nwp_key = f"{split or 'online'}_nwp_path"
        nwp_path = config.get("data", {}).get(nwp_key)
        if nwp_path:
            u140, v140 = load_nwp_uv140(nwp_path, time_limit=expected_t)
            if u140.shape[-1] == expected_t:
                out["raw_u140_grid"] = np.moveaxis(u140, -1, 0)
                out["raw_v140_grid"] = np.moveaxis(v140, -1, 0)
        return out
    except Exception as exc:
        print(f"[warn] Could not load optional dataset from {config_path}: {exc}")
        return {}


def plot_skill_curves(metrics: dict[str, np.ndarray], out: Path) -> None:
    plt = setup_matplotlib()
    horizons = metrics["horizons"]
    columns = [str(x) for x in metrics["curve_columns"]]
    rmse_curve = metrics["rmse_curve"]
    mae_curve = metrics["mae_curve"]

    colors = [MODEL_COLOR, PERSISTENCE_COLOR, NWP_COLOR, PALETTE["purple"]]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    for col_idx, name in enumerate(columns):
        axes[0].plot(
            horizons,
            rmse_curve[:, col_idx],
            marker=markers[col_idx % len(markers)],
            color=colors[col_idx % len(colors)],
            markersize=5.5,
            label=name,
        )
        axes[1].plot(
            horizons,
            mae_curve[:, col_idx],
            marker=markers[col_idx % len(markers)],
            color=colors[col_idx % len(colors)],
            markersize=5.5,
            label=name,
        )
    axes[0].set_title("RMSE by forecast horizon")
    axes[1].set_title("MAE by forecast horizon")
    for ax in axes:
        ax.set_xlabel("Horizon")
        polish_axes(ax)
    axes[0].set_ylabel("Error")
    axes[1].legend(loc="upper left", fontsize=8)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.savefig(out / "skill_by_horizon.png")
    plt.close(fig)


def per_station_metrics(
    forecasts: dict[str, np.ndarray],
    station_latlon: np.ndarray | None,
    mean_wind_vec: np.ndarray,
    out: Path,
) -> dict[str, Any]:
    plt = setup_matplotlib()
    target = forecasts["target"]
    pred = forecasts["prediction"]
    nwp = forecasts["nwp_baseline"]
    persistence = forecasts.get("persistence_prediction")

    target_speed = station_speed(target)
    pred_speed = station_speed(pred)
    nwp_speed = station_speed(nwp)
    persistence_speed = station_speed(persistence) if persistence is not None else None

    wind_unit = mean_wind_vec / (np.linalg.norm(mean_wind_vec) + 1e-8)
    downwind_projection = np.full(3, np.nan, dtype=np.float32)
    if station_latlon is not None and station_latlon.shape == (3, 2):
        from dataset.vector_data_utils import latlon_to_xy_km

        origin = (float(np.nanmean(station_latlon[:, 0])), float(np.nanmean(station_latlon[:, 1])))
        xy = latlon_to_xy_km(station_latlon[:, 0], station_latlon[:, 1], origin=origin).astype(np.float32)
        downwind_projection = xy @ wind_unit
    else:
        xy = np.arange(3, dtype=np.float32)[:, None] * np.array([[1.0, 0.0]], dtype=np.float32)

    rows = []
    for i, name in enumerate(STATION_NAMES):
        uv_cols = [i, i + 3]
        row = {
            "station": name,
            "downwind_projection_km": float(downwind_projection[i]),
            "model_uv_rmse": rmse(pred[:, uv_cols], target[:, uv_cols]),
            "model_uv_mae": mae(pred[:, uv_cols], target[:, uv_cols]),
            "nwp_uv_rmse": rmse(nwp[:, uv_cols], target[:, uv_cols]),
            "nwp_uv_mae": mae(nwp[:, uv_cols], target[:, uv_cols]),
            "model_speed_rmse": rmse(pred_speed[:, i], target_speed[:, i]),
            "model_speed_mae": mae(pred_speed[:, i], target_speed[:, i]),
            "nwp_speed_rmse": rmse(nwp_speed[:, i], target_speed[:, i]),
            "nwp_speed_mae": mae(nwp_speed[:, i], target_speed[:, i]),
        }
        if persistence is not None and persistence_speed is not None:
            row.update(
                {
                    "persistence_uv_rmse": rmse(persistence[:, uv_cols], target[:, uv_cols]),
                    "persistence_uv_mae": mae(persistence[:, uv_cols], target[:, uv_cols]),
                    "persistence_speed_rmse": rmse(persistence_speed[:, i], target_speed[:, i]),
                    "persistence_speed_mae": mae(persistence_speed[:, i], target_speed[:, i]),
                }
            )
        row["uv_rmse_improvement_vs_nwp_percent"] = 100.0 * (
            row["nwp_uv_rmse"] - row["model_uv_rmse"]
        ) / max(row["nwp_uv_rmse"], 1e-12)
        row["speed_rmse_improvement_vs_nwp_percent"] = 100.0 * (
            row["nwp_speed_rmse"] - row["model_speed_rmse"]
        ) / max(row["nwp_speed_rmse"], 1e-12)
        rows.append(row)

    order = np.argsort(downwind_projection)
    ordered_names = [STATION_NAMES[i] for i in order]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), constrained_layout=True)
    model_rmse = [rows[i]["model_uv_rmse"] for i in order]
    axes[0].bar(x, model_rmse, 0.58, color=MODEL_COLOR, edgecolor="#222222")
    axes[0].plot(x, model_rmse, color=PALETTE["navy"], marker="o", markersize=7, linewidth=2.2)
    axes[0].set_title("VectorMIDE U/V RMSE by station")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x, ordered_names)
    axes[0].set_xlabel("Upwind -> downwind order")
    polish_axes(axes[0])
    for xi, value in zip(x, model_rmse):
        axes[0].text(xi, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    model_mae = [rows[i]["model_uv_mae"] for i in order]
    axes[1].bar(x, model_mae, 0.58, color=MODEL_COLOR, edgecolor="#222222")
    axes[1].plot(x, model_mae, color=PALETTE["navy"], marker="o", markersize=7, linewidth=2.2)
    axes[1].set_title("VectorMIDE U/V MAE by station")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(x, ordered_names)
    axes[1].set_xlabel("Upwind -> downwind order")
    polish_axes(axes[1])
    for xi, value in zip(x, model_mae):
        axes[1].text(xi, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.savefig(out / "station_rmse_mae_by_downwind_order.png")
    plt.close(fig)

    if station_latlon is not None and station_latlon.shape == (3, 2):
        rmse_values = np.asarray([row["model_uv_rmse"] for row in rows], dtype=np.float32)
        fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
        sc = ax.scatter(
            station_latlon[:, 1],
            station_latlon[:, 0],
            c=rmse_values,
            s=360,
            cmap="YlGnBu_r",
            edgecolor="black",
            linewidth=1.25,
            zorder=3,
        )
        for i, name in enumerate(STATION_NAMES):
            ax.text(station_latlon[i, 1] + 0.015, station_latlon[i, 0] + 0.015, name, fontsize=13, weight="bold")
        lon0 = float(np.nanmean(station_latlon[:, 1]))
        lat0 = float(np.nanmean(station_latlon[:, 0]))
        ax.quiver(
            [lon0],
            [lat0],
            [wind_unit[0]],
            [wind_unit[1]],
            angles="xy",
            scale_units="xy",
            scale=8,
            color=PALETTE["red"],
            width=0.009,
            label="mean NWP wind direction",
        )
        ax.set_title("Station error and dominant advection direction")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        polish_axes(ax)
        ax.legend(loc="best", fontsize=8)
        cbar = fig.colorbar(sc, ax=ax, label="VectorMIDE U/V RMSE")
        cbar.ax.tick_params(labelsize=15, width=1.35, length=5.5)
        bold_tick_labels(cbar.ax)
        fig.savefig(out / "station_error_spatial_map.png")
        plt.close(fig)

    csv_lines = [
        ",".join(
            [
                "station",
                "downwind_projection_km",
                "model_uv_rmse",
                "model_uv_mae",
                "nwp_uv_rmse",
                "nwp_uv_mae",
                "persistence_uv_rmse",
                "persistence_uv_mae",
                "model_speed_rmse",
                "model_speed_mae",
                "nwp_speed_rmse",
                "nwp_speed_mae",
                "uv_rmse_improvement_vs_nwp_percent",
            ]
        )
    ]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    str(row.get("station")),
                    f"{row.get('downwind_projection_km', np.nan):.6g}",
                    f"{row.get('model_uv_rmse', np.nan):.6g}",
                    f"{row.get('model_uv_mae', np.nan):.6g}",
                    f"{row.get('nwp_uv_rmse', np.nan):.6g}",
                    f"{row.get('nwp_uv_mae', np.nan):.6g}",
                    f"{row.get('persistence_uv_rmse', np.nan):.6g}",
                    f"{row.get('persistence_uv_mae', np.nan):.6g}",
                    f"{row.get('model_speed_rmse', np.nan):.6g}",
                    f"{row.get('model_speed_mae', np.nan):.6g}",
                    f"{row.get('nwp_speed_rmse', np.nan):.6g}",
                    f"{row.get('nwp_speed_mae', np.nan):.6g}",
                    f"{row.get('uv_rmse_improvement_vs_nwp_percent', np.nan):.6g}",
                ]
            )
        )
    (out / "station_metrics.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    return {
        "upwind_to_downwind_order": ordered_names,
        "mean_wind_vector_xy": [float(mean_wind_vec[0]), float(mean_wind_vec[1])],
        "stations": rows,
        "corr_downwind_projection_vs_model_uv_rmse": corrcoef(
            downwind_projection,
            np.asarray([row["model_uv_rmse"] for row in rows], dtype=np.float32),
        ),
    }


def plot_flow_physics(
    flow: np.ndarray,
    nwp_vec: np.ndarray,
    nwp_speed_mean: np.ndarray,
    v_star: np.ndarray | None,
    out: Path,
) -> dict[str, Any]:
    plt = setup_matplotlib()
    flow_norm = np.linalg.norm(flow, axis=1)
    nwp_vec_norm = np.linalg.norm(nwp_vec, axis=1)
    mask = finite_mask(flow, nwp_vec) & np.isfinite(nwp_speed_mean)
    angle = np.full(flow.shape[0], np.nan, dtype=np.float32)
    angle[mask] = angle_diff_deg(flow[mask], nwp_vec[mask])
    abs_angle = np.abs(((angle + 180.0) % 360.0) - 180.0)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    ax.scatter(nwp_speed_mean[mask], flow_norm[mask], s=14, alpha=0.28, color=MODEL_COLOR, edgecolors="none")
    ax.set_xlabel("Mean NWP station wind speed")
    ax.set_ylabel("Learned advection norm")
    ax.set_title("Advection strength vs physical wind speed")
    polish_axes(ax)
    fig.savefig(out / "advection_norm_vs_nwp_speed.png")
    plt.close(fig)

    bins = quantile_bins(nwp_speed_mean, 4)
    labels = []
    data = []
    for b in bins:
        b = b & mask
        if np.any(b):
            lo = np.nanmin(nwp_speed_mean[b])
            hi = np.nanmax(nwp_speed_mean[b])
            labels.append(f"{lo:.1f}-{hi:.1f}")
            data.append(flow_norm[b])
    fig, ax = plt.subplots(figsize=(7.6, 5.1), constrained_layout=True)
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.7},
        boxprops={"linewidth": 1.2, "edgecolor": "#222222"},
        whiskerprops={"linewidth": 1.2, "color": "#222222"},
        capprops={"linewidth": 1.2, "color": "#222222"},
    )
    for patch, color in zip(bp["boxes"], ["#DDECCB", "#A7D8C9", "#69B3C4", "#277DA1"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
    ax.set_xlabel("NWP speed regime")
    ax.set_ylabel("Learned advection norm")
    ax.set_title("Learned advection gets stronger by wind regime")
    polish_axes(ax)
    fig.savefig(out / "advection_norm_by_speed_regime.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.9, 5.0), constrained_layout=True)
    ax.hist(
        abs_angle[np.isfinite(abs_angle)],
        bins=np.linspace(0, 180, 37),
        color=PALETTE["teal"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.55,
    )
    ax.set_xlabel("Absolute angle difference (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Learned advection direction vs\nNWP wind direction", pad=8)
    polish_axes(ax)
    fig.savefig(out / "advection_direction_alignment_hist.png")
    plt.close(fig)

    sample = min(720, flow.shape[0])
    start = 0 if flow.shape[0] <= sample else int(np.nanargmax(np.convolve(flow_norm, np.ones(sample), mode="valid")))
    sl = slice(start, start + sample)
    fig, ax1 = plt.subplots(figsize=(13.2, 4.8), constrained_layout=True)
    x = np.arange(start, start + sample)
    ax1.plot(x, flow_norm[sl], label="learned advection norm", color=MODEL_COLOR, linewidth=2.2)
    ax1.set_ylabel("Advection norm")
    ax2 = ax1.twinx()
    ax2.plot(x, nwp_speed_mean[sl], label="mean NWP wind speed", color=NWP_COLOR, linewidth=2.0, alpha=0.9)
    ax2.set_ylabel("NWP speed")
    ax1.set_xlabel("Time index")
    ax1.set_title("Advection parameter follows physical wind regimes")
    polish_axes(ax1)
    polish_axes(ax2, grid=False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.savefig(out / "advection_norm_timeseries_with_nwp_speed.png")
    plt.close(fig)

    summary: dict[str, Any] = {
        "flow_norm_mean": float(np.nanmean(flow_norm)),
        "flow_norm_std": float(np.nanstd(flow_norm)),
        "nwp_speed_mean": float(np.nanmean(nwp_speed_mean)),
        "corr_flow_norm_vs_mean_nwp_speed": corrcoef(flow_norm, nwp_speed_mean),
        "corr_flow_norm_vs_mean_nwp_vector_norm": corrcoef(flow_norm, nwp_vec_norm),
        "median_abs_direction_error_deg_vs_nwp": float(np.nanmedian(abs_angle)),
        "mean_abs_direction_error_deg_vs_nwp": float(np.nanmean(abs_angle)),
        "fraction_direction_within_45deg_vs_nwp": float(np.nanmean(abs_angle <= 45.0)),
    }

    if v_star is not None and v_star.shape[0] == flow.shape[0]:
        v_star_vec = v_star[:, :2]
        v_mask = finite_mask(flow, v_star_vec)
        v_angle = np.full(flow.shape[0], np.nan, dtype=np.float32)
        v_angle[v_mask] = angle_diff_deg(flow[v_mask], v_star_vec[v_mask])
        v_abs_angle = np.abs(((v_angle + 180.0) % 360.0) - 180.0)
        v_star_norm = np.linalg.norm(v_star_vec, axis=1)
        fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
        ax.scatter(v_star_norm[v_mask], flow_norm[v_mask], s=14, alpha=0.28, color=PALETTE["purple"], edgecolors="none")
        ax.set_xlabel("NWP-derived advection label norm")
        ax.set_ylabel("Learned advection norm")
        ax.set_title("Learned advection vs NWP-derived label")
        polish_axes(ax)
        fig.savefig(out / "advection_norm_vs_vstar_norm.png")
        plt.close(fig)
        summary.update(
            {
                "corr_flow_norm_vs_vstar_norm": corrcoef(flow_norm, v_star_norm),
                "median_abs_direction_error_deg_vs_vstar": float(np.nanmedian(v_abs_angle)),
                "fraction_direction_within_45deg_vs_vstar": float(np.nanmean(v_abs_angle <= 45.0)),
            }
        )
    return summary


def direction_deg_from_xy(vec: np.ndarray) -> np.ndarray:
    """Direction of the vector in compass degrees: N=0, E=90."""
    return (np.degrees(np.arctan2(vec[:, 0], vec[:, 1])) + 360.0) % 360.0


def met_from_direction_deg(vec: np.ndarray) -> np.ndarray:
    """Meteorological wind/advection source direction: where the vector comes from."""
    return (direction_deg_from_xy(vec) + 180.0) % 360.0


def draw_rose(
    ax,
    directions_deg: np.ndarray,
    magnitudes: np.ndarray,
    bins: list[float],
    title: str,
    legend_title: str,
) -> dict[str, Any]:
    dirs = np.asarray(directions_deg, dtype=np.float64)
    mags = np.asarray(magnitudes, dtype=np.float64)
    valid = np.isfinite(dirs) & np.isfinite(mags)
    dirs = dirs[valid]
    mags = mags[valid]
    n_sectors = 16
    sector_edges = np.linspace(0.0, 360.0, n_sectors + 1)
    theta = np.deg2rad((sector_edges[:-1] + sector_edges[1:]) / 2.0)
    width = np.deg2rad(360.0 / n_sectors) * 0.92
    colors = ["#FFF7BC", "#C7E9B4", "#7FCDBB", "#41B6C4", "#225EA8"]
    bottom = np.zeros(n_sectors, dtype=np.float64)
    total = max(float(len(dirs)), 1.0)
    labels = []
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j + 1]
        if np.isinf(hi):
            mask = mags >= lo
            labels.append(f">= {lo:g}")
        else:
            mask = (mags >= lo) & (mags < hi)
            labels.append(f"{lo:g}-{hi:g}")
        counts, _ = np.histogram(dirs[mask], bins=sector_edges)
        perc = counts.astype(np.float64) / total * 100.0
        ax.bar(theta, perc, width=width, bottom=bottom, color=colors[j % len(colors)], edgecolor="white", linewidth=0.65, label=labels[-1])
        bottom += perc
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=14, fontweight="bold")
    rmax = max(float(np.nanmax(bottom)) if bottom.size else 1.0, 1.0)
    rticks = np.linspace(0.0, np.ceil(rmax / 2.0) * 2.0, 5)[1:]
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{x:.0f}%" for x in rticks], fontsize=12, fontweight="bold")
    ax.set_title(title, pad=18, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.28)
    ax.legend(title=legend_title, loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=9, title_fontsize=10)
    mode_idx = int(np.nanargmax(bottom)) if bottom.size else 0
    return {
        "dominant_direction_deg": float(np.rad2deg(theta[mode_idx])),
        "dominant_sector_percent": float(bottom[mode_idx]) if bottom.size else float("nan"),
    }


def plot_direction_speed_roses(
    nwp_vec: np.ndarray,
    nwp_speed: np.ndarray,
    flow: np.ndarray,
    out: Path,
) -> dict[str, Any]:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), subplot_kw={"projection": "polar"}, constrained_layout=True)
    nwp_summary = draw_rose(
        axes[0],
        met_from_direction_deg(nwp_vec),
        nwp_speed,
        [0, 5, 10, 15, 20, np.inf],
        "NWP wind rose at 140m\n(from-direction)",
        "Wind speed",
    )
    flow_summary = draw_rose(
        axes[1],
        met_from_direction_deg(flow),
        np.linalg.norm(flow, axis=1),
        [0, 2, 4, 6, 8, np.inf],
        "Learned advection rose\n(from-direction)",
        "Advection norm",
    )
    panel_label(axes[0], "a", x=0.02, y=1.04)
    panel_label(axes[1], "b", x=0.02, y=1.04)
    fig.savefig(out / "wind_rose_nwp_vs_learned_advection.png")
    plt.close(fig)
    return {"nwp": nwp_summary, "learned_advection": flow_summary}


def plot_residual_correction(
    forecasts: dict[str, np.ndarray],
    out: Path,
) -> dict[str, Any]:
    plt = setup_matplotlib()
    target = forecasts["target"]
    pred = forecasts["prediction"]
    nwp = forecasts["nwp_baseline"]
    true_residual = target - nwp
    pred_residual = pred - nwp
    mask = np.isfinite(true_residual) & np.isfinite(pred_residual)

    true_mag = np.sqrt(true_residual[:, :3] ** 2 + true_residual[:, 3:] ** 2)
    pred_mag = np.sqrt(pred_residual[:, :3] ** 2 + pred_residual[:, 3:] ** 2)
    true_vec = np.stack([true_residual[:, :3], true_residual[:, 3:]], axis=-1)
    pred_vec = np.stack([pred_residual[:, :3], pred_residual[:, 3:]], axis=-1)
    cos = vector_cosine(pred_vec, true_vec)
    speed_mask = np.isfinite(true_mag) & np.isfinite(pred_mag)

    fig, ax = plt.subplots(figsize=(6.2, 5.7), constrained_layout=True)
    ax.scatter(true_mag[speed_mask], pred_mag[speed_mask], s=10, alpha=0.18, color=MODEL_COLOR, edgecolors="none")
    lim = float(np.nanpercentile(np.concatenate([true_mag[speed_mask], pred_mag[speed_mask]]), 99))
    ax.plot([0, lim], [0, lim], color="#111111", linestyle="--", linewidth=1.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True NWP residual magnitude")
    ax.set_ylabel("Predicted correction magnitude")
    ax.set_title("Does the model correct the NWP error magnitude?")
    polish_axes(ax)
    fig.savefig(out / "correction_magnitude_vs_true_residual.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    ax.hist(
        cos[np.isfinite(cos)],
        bins=np.linspace(-1, 1, 41),
        color=PALETTE["green"],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.55,
    )
    ax.set_xlabel("Cosine(predicted correction, true NWP residual)")
    ax.set_ylabel("Count")
    ax.set_title("Correction direction alignment")
    polish_axes(ax)
    fig.savefig(out / "correction_direction_alignment_hist.png")
    plt.close(fig)

    summary = {
        "model_rmse": rmse(pred, target),
        "nwp_rmse": rmse(nwp, target),
        "model_mae": mae(pred, target),
        "nwp_mae": mae(nwp, target),
        "correction_magnitude_corr": corrcoef(pred_mag.reshape(-1), true_mag.reshape(-1)),
        "correction_direction_cosine_mean": float(np.nanmean(cos)),
        "correction_direction_cosine_median": float(np.nanmedian(cos)),
        "correction_direction_positive_fraction": float(np.nanmean(cos > 0.0)),
    }
    for i, name in enumerate(["E05", "E06", "ASOW6"]):
        station_mask = mask[:, [i, i + 3]]
        summary[f"{name}_model_rmse"] = rmse(pred[:, [i, i + 3]], target[:, [i, i + 3]], station_mask)
        summary[f"{name}_nwp_rmse"] = rmse(nwp[:, [i, i + 3]], target[:, [i, i + 3]], station_mask)
    return summary


def plot_transition_summary(
    advection: dict[str, np.ndarray],
    transitions: dict[str, np.ndarray],
    out: Path,
) -> dict[str, Any]:
    plt = setup_matplotlib()
    M = transitions["M"]
    M_mean = transitions.get("M_mean", np.nanmean(M, axis=0))
    state_names = [str(x) for x in transitions.get("state_names", np.asarray(STATE_NAMES))]
    fig, ax = plt.subplots(figsize=(7.0, 6.1), constrained_layout=True)
    im = ax.imshow(M_mean, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(state_names)), labels=state_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(state_names)), labels=state_names, fontsize=12)
    bold_tick_labels(ax)
    ax.set_title("Mean learned transition matrix")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=15, width=1.35, length=5.5)
    bold_tick_labels(cbar.ax)
    fig.savefig(out / "mean_transition_matrix.png")
    plt.close(fig)

    flow = advection.get("processed_flow_mu")
    if flow is None or flow.size == 0:
        flow = advection["flow_mu"]
    flow_norm = np.linalg.norm(flow, axis=1)
    offdiag_mask = ~np.eye(M.shape[-1], dtype=bool)
    offdiag_strength = np.nanmean(np.abs(M[:, offdiag_mask]), axis=1)
    diag_strength = np.nanmean(np.abs(np.diagonal(M, axis1=1, axis2=2)), axis=1)
    row_spread = np.nanstd(np.nansum(M, axis=2), axis=1)

    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    ax.scatter(flow_norm, offdiag_strength, s=14, alpha=0.24, color=PALETTE["purple"], edgecolors="none")
    ax.set_xlabel("Learned advection norm")
    ax.set_ylabel("Mean off-diagonal transition")
    ax.set_title("Advection strength vs transition mixing")
    polish_axes(ax)
    fig.savefig(out / "advection_vs_transition_mixing.png")
    plt.close(fig)

    return {
        "transition_diag_abs_mean": float(np.nanmean(diag_strength)),
        "transition_offdiag_abs_mean": float(np.nanmean(offdiag_strength)),
        "transition_rowsum_std_mean": float(np.nanmean(row_spread)),
        "corr_flow_norm_vs_offdiag_transition_strength": corrcoef(flow_norm, offdiag_strength),
    }


def plot_coriolis_rotation_analysis(
    transitions: dict[str, np.ndarray],
    out: Path,
    dt_seconds: float,
    station_latlon: np.ndarray | None = None,
) -> dict[str, Any]:
    plt = setup_matplotlib()
    omega = 7.2921159e-5
    if station_latlon is not None and np.size(station_latlon):
        mean_lat = float(np.nanmean(np.asarray(station_latlon, dtype=np.float64)[:, 0]))
    else:
        mean_lat = 39.5
    coriolis_f = float(2.0 * omega * np.sin(np.deg2rad(mean_lat)))
    coriolis_step = coriolis_f * float(dt_seconds)
    coriolis_period_hours = float(2.0 * np.pi / abs(coriolis_f) / 3600.0)

    matrix_specs = [("M", "final transition", transitions["M"])]
    if "M_base" in transitions:
        matrix_specs.append(("M_base", "base transition", transitions["M_base"]))

    rows: list[list[Any]] = []
    series: dict[str, dict[str, np.ndarray]] = {}
    for key, label, mat in matrix_specs:
        mat = np.asarray(mat, dtype=np.float64)
        n = mat.shape[0]
        uv = np.full((n, 3), np.nan, dtype=np.float64)
        vu = np.full((n, 3), np.nan, dtype=np.float64)
        rot = np.full((n, 3), np.nan, dtype=np.float64)
        sym = np.full((n, 3), np.nan, dtype=np.float64)
        dominance = np.full((n, 3), np.nan, dtype=np.float64)
        period = np.full((n, 3), np.nan, dtype=np.float64)
        for i, station in enumerate(STATION_NAMES):
            uv[:, i] = mat[:, i, i + 3]
            vu[:, i] = mat[:, i + 3, i]
            rot[:, i] = 0.5 * (uv[:, i] - vu[:, i])
            sym[:, i] = 0.5 * (uv[:, i] + vu[:, i])
            dominance[:, i] = np.abs(rot[:, i]) / (np.abs(rot[:, i]) + np.abs(sym[:, i]) + 1e-8)
            rate = rot[:, i] / float(dt_seconds)
            valid_rate = np.isfinite(rate) & (np.abs(rate) > 1e-12)
            period[valid_rate, i] = 2.0 * np.pi / np.abs(rate[valid_rate]) / 3600.0
            sign_consistent = (uv[:, i] > 0.0) & (vu[:, i] < 0.0)
            for t in range(n):
                rows.append(
                    [
                        t,
                        key,
                        label,
                        station,
                        uv[t, i],
                        vu[t, i],
                        rot[t, i],
                        sym[t, i],
                        rate[t],
                        period[t, i],
                        dominance[t, i],
                        bool(sign_consistent[t]),
                    ]
                )
        series[key] = {
            "uv": uv,
            "vu": vu,
            "rotation_step": rot,
            "symmetric_cross": sym,
            "rotation_rate_s_inv": rot / float(dt_seconds),
            "period_hours": period,
            "dominance": dominance,
            "mean_rotation_step": np.nanmean(rot, axis=1),
            "mean_dominance": np.nanmean(dominance, axis=1),
        }

    header = ",".join(
        [
            "time_index",
            "matrix_key",
            "matrix_label",
            "station",
            "M_UV_rowU_colV",
            "M_VU_rowV_colU",
            "rotation_step_half_MUV_minus_MVU",
            "symmetric_cross_half_MUV_plus_MVU",
            "rotation_rate_s_inv",
            "implied_inertial_period_hours",
            "rotation_dominance_abs_rot_over_abs_rot_plus_abs_sym",
            "northern_hemisphere_sign_consistent",
        ]
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    f"{row[4]:.8g}",
                    f"{row[5]:.8g}",
                    f"{row[6]:.8g}",
                    f"{row[7]:.8g}",
                    f"{row[8]:.8g}",
                    f"{row[9]:.8g}",
                    f"{row[10]:.8g}",
                    str(row[11]),
                ]
            )
        )
    (out / "transition_coriolis_rotation_timeseries.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "transition_coriolis_rotation_timeseries.npz",
        dt_seconds=float(dt_seconds),
        mean_latitude_deg=mean_lat,
        coriolis_f_s_inv=coriolis_f,
        coriolis_step=coriolis_step,
        coriolis_period_hours=coriolis_period_hours,
        **{f"{key}_{field}": value for key, values in series.items() for field, value in values.items()},
    )

    summary: dict[str, Any] = {
        "created": True,
        "dt_seconds": float(dt_seconds),
        "mean_latitude_deg": mean_lat,
        "theoretical_coriolis_f_s_inv": coriolis_f,
        "theoretical_coriolis_step_f_dt": coriolis_step,
        "theoretical_inertial_period_hours": coriolis_period_hours,
    }
    for key, label, _ in matrix_specs:
        vals = series[key]
        rot = vals["rotation_step"]
        rate = vals["rotation_rate_s_inv"]
        period = vals["period_hours"]
        uv = vals["uv"]
        vu = vals["vu"]
        dominance = vals["dominance"]
        summary[f"{key}_rotation_step_mean"] = float(np.nanmean(rot))
        summary[f"{key}_rotation_step_median"] = float(np.nanmedian(rot))
        summary[f"{key}_rotation_rate_s_inv_mean"] = float(np.nanmean(rate))
        summary[f"{key}_implied_period_hours_median"] = float(np.nanmedian(period))
        summary[f"{key}_rotation_to_coriolis_step_ratio_mean"] = float(np.nanmean(rot) / coriolis_step)
        summary[f"{key}_sign_consistent_fraction"] = float(np.nanmean((uv > 0.0) & (vu < 0.0)))
        summary[f"{key}_rotation_dominance_mean"] = float(np.nanmean(dominance))

    summary_csv = "metric,value\n" + "\n".join(f"{key},{value}" for key, value in summary.items() if key != "created")
    (out / "transition_coriolis_rotation_summary.csv").write_text(summary_csv + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.2), constrained_layout=True)
    x = np.arange(len(STATION_NAMES))
    width = 0.32
    for offset, (key, label, _) in zip([-width / 2, width / 2], matrix_specs):
        station_means = np.nanmean(series[key]["rotation_step"], axis=0)
        axes[0].bar(x + offset, station_means, width=width, label=label, alpha=0.86)
    axes[0].axhline(coriolis_step, color=PALETTE["red"], linestyle="--", linewidth=2.2, label=r"$f\Delta t$")
    axes[0].set_xticks(x, STATION_NAMES)
    axes[0].set_ylabel(r"Rotation coefficient $(M_{UV}-M_{VU})/2$")
    axes[0].set_title("Local U/V rotational coupling")
    axes[0].legend(fontsize=9)
    polish_axes(axes[0])
    panel_label(axes[0], "a")

    bins = np.linspace(
        min(0.0, min(float(np.nanpercentile(series[key]["mean_rotation_step"], 1)) for key, _, _ in matrix_specs)),
        max(float(np.nanpercentile(series[key]["mean_rotation_step"], 99)) for key, _, _ in matrix_specs),
        45,
    )
    for key, label, _ in matrix_specs:
        axes[1].hist(series[key]["mean_rotation_step"], bins=bins, alpha=0.55, density=True, label=label)
    axes[1].axvline(coriolis_step, color=PALETTE["red"], linestyle="--", linewidth=2.2, label=r"$f\Delta t$")
    axes[1].set_xlabel(r"Mean station rotation coefficient")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Rotation coefficient distribution")
    axes[1].legend(fontsize=9)
    polish_axes(axes[1])
    panel_label(axes[1], "b")

    period_bins = np.linspace(0.0, 80.0, 45)
    for key, label, _ in matrix_specs:
        period_vals = series[key]["period_hours"].reshape(-1)
        period_vals = period_vals[np.isfinite(period_vals) & (period_vals <= 80.0)]
        axes[2].hist(period_vals, bins=period_bins, alpha=0.55, density=True, label=label)
    axes[2].axvline(coriolis_period_hours, color=PALETTE["red"], linestyle="--", linewidth=2.2, label="theoretical inertial period")
    axes[2].set_xlabel("Implied period (hours)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Implied inertial period")
    axes[2].legend(fontsize=9)
    polish_axes(axes[2])
    panel_label(axes[2], "c")
    fig.savefig(out / "transition_coriolis_rotation_analysis.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 6.2), constrained_layout=True)
    for key, label, _ in matrix_specs:
        uv = series[key]["uv"].reshape(-1)
        neg_vu = -series[key]["vu"].reshape(-1)
        ax.scatter(uv, neg_vu, s=11, alpha=0.18, edgecolors="none", label=label)
    lim = float(
        np.nanpercentile(
            np.abs(
                np.concatenate(
                    [series[key]["uv"].reshape(-1) for key, _, _ in matrix_specs]
                    + [-series[key]["vu"].reshape(-1) for key, _, _ in matrix_specs]
                )
            ),
            99.5,
        )
    )
    ax.plot([0, lim], [0, lim], color="#111111", linestyle="--", linewidth=1.7, label=r"$M_{UV}=-M_{VU}$")
    ax.axvline(0, color="#777777", linewidth=1.0)
    ax.axhline(0, color="#777777", linewidth=1.0)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel(r"$M_{UV}$: V contribution to next U")
    ax.set_ylabel(r"$-M_{VU}$: negative U contribution to next V")
    ax.set_title("Anti-symmetric U/V cross coupling")
    ax.legend(fontsize=9, loc="upper left")
    polish_axes(ax)
    fig.savefig(out / "transition_uv_cross_coupling_scatter.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.0, 4.6), constrained_layout=True)
    t = np.arange(transitions["M"].shape[0])
    for key, label, _ in matrix_specs:
        ax.plot(t, series[key]["mean_rotation_step"], linewidth=1.3, alpha=0.72, label=label)
    ax.axhline(coriolis_step, color=PALETTE["red"], linestyle="--", linewidth=2.0, label=r"$f\Delta t$")
    ax.set_xlabel("Time index")
    ax.set_ylabel(r"Mean rotation coefficient")
    ax.set_title("Time-varying U/V rotational coupling")
    ax.legend(fontsize=9, loc="upper right")
    polish_axes(ax)
    fig.savefig(out / "transition_coriolis_rotation_timeseries.png")
    plt.close(fig)

    return summary


def kalman_pred_covariances(
    z: np.ndarray,
    M: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    control: np.ndarray | None = None,
    jitter: float = 1e-5,
) -> np.ndarray:
    state_dim = z.shape[1]
    cov = np.eye(state_dim, dtype=np.float64)
    eye = np.eye(state_dim, dtype=np.float64)
    pred_covs = np.full((z.shape[0], state_dim, state_dim), np.nan, dtype=np.float32)
    for t in range(z.shape[0]):
        M_t = M[t].astype(np.float64, copy=False)
        pred_cov = M_t @ cov @ M_t.T + Q
        pred_cov = 0.5 * (pred_cov + pred_cov.T)
        pred_covs[t] = pred_cov.astype(np.float32)

        obs_mask = np.isfinite(z[t])
        if np.any(obs_mask):
            H_t = eye[obs_mask]
            R_t = R[obs_mask][:, obs_mask]
            F_t = H_t @ pred_cov @ H_t.T + R_t
            F_t = F_t + jitter * np.eye(F_t.shape[0], dtype=np.float64)
            gain = pred_cov @ H_t.T @ np.linalg.pinv(F_t)
            cov = pred_cov - gain @ H_t @ pred_cov
            cov = 0.5 * (cov + cov.T)
        else:
            cov = pred_cov
    return pred_covs


def speed_sigma_from_uv_cov(pred: np.ndarray, covs: np.ndarray) -> np.ndarray:
    sigmas = np.full((pred.shape[0], 3), np.nan, dtype=np.float32)
    for i in range(3):
        u = pred[:, i]
        v = pred[:, i + 3]
        speed = np.sqrt(u**2 + v**2)
        valid = np.isfinite(speed) & (speed > 1e-8)
        grad = np.zeros((pred.shape[0], 2), dtype=np.float64)
        grad[valid, 0] = u[valid] / speed[valid]
        grad[valid, 1] = v[valid] / speed[valid]
        uv_cov = covs[:, [i, i + 3]][:, :, [i, i + 3]].astype(np.float64, copy=False)
        var = np.einsum("ti,tij,tj->t", grad, uv_cov, grad)
        sigmas[:, i] = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
    return sigmas


def plot_station_forecast_intervals(
    forecasts: dict[str, np.ndarray],
    advection: dict[str, np.ndarray],
    transitions: dict[str, np.ndarray],
    flow: np.ndarray,
    out: Path,
) -> dict[str, Any]:
    plt = setup_matplotlib()
    z = forecasts["model_target"]
    M = transitions["M"]
    Q = advection["Q"].astype(np.float64, copy=False)
    R = advection["R"].astype(np.float64, copy=False)
    control = advection.get("transition_control")
    covs = kalman_pred_covariances(z, M, Q, R, control=control)
    speed_sigma = speed_sigma_from_uv_cov(forecasts["prediction"], covs)

    pred_speed = station_speed(forecasts["prediction"])
    target_speed = station_speed(forecasts["target"])
    nwp_speed = station_speed(forecasts["nwp_baseline"])
    lower = pred_speed - 1.96 * speed_sigma
    upper = pred_speed + 1.96 * speed_sigma
    coverage = (target_speed >= lower) & (target_speed <= upper) & np.isfinite(target_speed)

    flow_norm = np.linalg.norm(flow, axis=1)
    sample = min(720, pred_speed.shape[0])
    start = 0 if pred_speed.shape[0] <= sample else int(np.nanargmax(np.convolve(flow_norm, np.ones(sample), mode="valid")))
    sl = slice(start, start + sample)
    x = np.arange(start, start + sample)

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.0), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.fill_between(x, lower[sl, i], upper[sl, i], color=INTERVAL_COLOR, alpha=0.32, label="VectorMIDE 95% interval")
        ax.plot(x, pred_speed[sl, i], color=MODEL_COLOR, linewidth=2.2, label="VectorMIDE mean")
        ax.scatter(
            x,
            target_speed[sl, i],
            color=MEAS_COLOR,
            s=15,
            alpha=0.82,
            linewidths=0,
            label="measurement",
            zorder=4,
        )
        ax.plot(x, nwp_speed[sl, i], color=NWP_COLOR, linewidth=1.9, alpha=0.86, label="NWP")
        ax.set_title(f"{STATION_NAMES[i]} wind speed forecast interval")
        ax.set_ylabel("Speed")
        polish_axes(ax)
        panel_label(ax, chr(ord("a") + i), x=0.012, y=0.30)
    axes[-1].set_xlabel("Time index")
    axes[0].legend(loc="upper left", ncol=4, fontsize=10)
    fig.savefig(out / "station_speed_forecast_95_interval.png")
    plt.close(fig)

    np.savez_compressed(out / "station_speed_forecast_intervals.npz", lower=lower, upper=upper, speed_sigma=speed_sigma)
    return {
        "speed_interval_coverage_overall": float(np.nanmean(coverage)),
        **{
            f"{name}_speed_interval_coverage": float(np.nanmean(coverage[:, i]))
            for i, name in enumerate(STATION_NAMES)
        },
        "mean_speed_interval_half_width": float(np.nanmean(1.96 * speed_sigma)),
    }


def plot_deepmide_style_advection_figure(
    flow: np.ndarray,
    optional_data: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    required = ["raw_u140_grid", "raw_v140_grid", "baseline_grid_indices", "lat_grid", "lon_grid"]
    if not all(key in optional_data for key in required):
        return {"created": False, "reason": "raw NWP grid, lat/lon grid, or station coordinates unavailable"}

    plt = setup_matplotlib()
    from matplotlib.patches import ConnectionPatch
    from matplotlib.gridspec import GridSpec
    from matplotlib.transforms import Bbox

    u = optional_data["raw_u140_grid"]
    v = optional_data["raw_v140_grid"]
    station_indices = np.asarray(optional_data["baseline_grid_indices"], dtype=np.int64)
    lat_grid = np.asarray(optional_data["lat_grid"], dtype=float)
    lon_grid = np.asarray(optional_data["lon_grid"], dtype=float)
    speed = np.sqrt(u**2 + v**2)
    flow_norm = np.linalg.norm(flow, axis=1)
    height, width = speed.shape[1:]
    lon_min, lon_max = float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid))
    lat_min, lat_max = float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))
    lon_pad = max((lon_max - lon_min) * 0.015, 1e-6)
    lat_pad = max((lat_max - lat_min) * 0.015, 1e-6)
    extent = [lon_min, lon_max, lat_min, lat_max]
    lon_ticks = np.linspace(lon_min, lon_max, 5)
    lat_ticks = np.linspace(lat_min, lat_max, 5)
    station_lats = lat_grid[station_indices[:, 0], station_indices[:, 1]]
    station_lons = lon_grid[station_indices[:, 0], station_indices[:, 1]]

    valid = np.where(np.isfinite(flow_norm))[0]
    valid = valid[valid < speed.shape[0] - 1]
    if valid.size < 2:
        return {"created": False, "reason": "not enough finite flow values"}
    high_t = int(valid[np.nanargmax(flow_norm[valid])])
    low_candidates = valid[valid > max(10, int(0.05 * len(valid)))]
    low_t = int(low_candidates[np.nanargmin(flow_norm[low_candidates])]) if low_candidates.size else int(valid[np.nanargmin(flow_norm[valid])])
    chosen = [low_t, low_t + 1, high_t, high_t + 1]

    fig = plt.figure(figsize=(14.8, 8.6), constrained_layout=False)
    gs = GridSpec(
        2,
        5,
        figure=fig,
        height_ratios=[1.0, 1.35],
        width_ratios=[1.0, 1.0, 0.22, 1.0, 1.0],
        left=0.11,
        right=0.86,
        top=0.90,
        bottom=0.10,
        hspace=0.42,
        wspace=0.32,
    )
    map_axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 3, 4]]
    ts_ax = fig.add_subplot(gs[1, :])
    vmax = float(np.nanpercentile(speed[chosen], 98))
    vmin = float(np.nanpercentile(speed[chosen], 2))
    stride = max(1, height // 16)
    lon_sample = lon_grid[::stride, ::stride]
    lat_sample = lat_grid[::stride, ::stride]
    for ax, t, label, color in zip(
        map_axes,
        chosen,
        ["weak wind speed t", "weak wind speed t+1", "strong wind speed t", "strong wind speed t+1"],
        [PALETTE["green"], PALETTE["green"], NWP_COLOR, NWP_COLOR],
    ):
        im = ax.imshow(
            speed[t],
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=extent,
            aspect="auto",
        )
        ax.quiver(
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
        ax.scatter(
            station_lons,
            station_lats,
            marker="*",
            s=190,
            color=PALETTE["red"],
            edgecolor="white",
            linewidth=1.0,
            clip_on=True,
            zorder=5,
        )
        ax.set_title(f"{label}, t={t}", fontsize=10.5, fontweight="bold", pad=5)
        polish_axes(ax, grid=False)
        ax.set_xlabel("Longitude", fontsize=9.5, fontweight="bold", labelpad=2)
        ax.set_ylabel("Latitude", fontsize=9.5, fontweight="bold", labelpad=3)
        ax.tick_params(axis="both", labelsize=8.5, pad=1, width=0.85, length=3)
        ax.set_box_aspect(1)
        ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
        ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels([f"{x:.2f}" for x in lon_ticks])
        ax.set_yticklabels([f"{y:.2f}" for y in lat_ticks])
        normal_tick_labels(ax)
        for spine in ax.spines.values():
            spine.set_visible(False)
        label_text = "a" if ax is map_axes[0] else "b" if ax is map_axes[2] else None
        if label_text:
            ax.text(
                0.04,
                0.93,
                label_text,
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                va="top",
                ha="left",
                bbox={
                    "boxstyle": "square,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": PALETTE["navy"],
                    "linewidth": 0.9,
                    "alpha": 0.95,
                },
                zorder=10,
            )
    cax = fig.add_axes([0.885, 0.585, 0.014, 0.285])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Speed (m/s)", fontsize=10.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=8.5, width=0.85, length=3)
    normal_tick_labels(cbar.ax)

    x = np.arange(flow.shape[0])
    ts_ax.plot(x, flow_norm, color=MODEL_COLOR, linewidth=1.85)
    ts_ax.scatter([high_t, low_t], [flow_norm[high_t], flow_norm[low_t]], s=76, color=[NWP_COLOR, PALETTE["green"]], edgecolor="white", linewidth=1.1, zorder=3)
    top = float(np.nanmax(flow_norm) * 1.16)
    ts_ax.set_ylim(float(np.nanmin(flow_norm) - 0.2), top)
    # ts_ax.annotate(
    #     "strong advection interval",
    #     xy=(high_t, flow_norm[high_t]),
    #     xytext=(high_t, np.nanmax(flow_norm) * 1.04),
    #     arrowprops={"arrowstyle": "->", "color": NWP_COLOR, "lw": 2.2},
    #     color=NWP_COLOR,
    #     ha="center",
    #     fontsize=11,
    #     fontweight="bold",
    # )
    # ts_ax.annotate(
    #     "weak advection interval",
    #     xy=(low_t, flow_norm[low_t]),
    #     xytext=(low_t, np.nanmax(flow_norm) * 0.88),
    #     arrowprops={"arrowstyle": "->", "color": PALETTE["green"], "lw": 2.2},
    #     color=PALETTE["green"],
    #     ha="center",
    #     fontsize=11,
    #     fontweight="bold",
    # )
    ts_ax.set_title("Norm of learned advection parameter with NWP wind-map examples", fontsize=13.5, fontweight="bold")
    ts_ax.set_xlabel("Time index", fontsize=11.5, fontweight="bold")
    ts_ax.set_ylabel("Norm", fontsize=11.5, fontweight="bold", labelpad=7)
    polish_axes(ts_ax)
    ts_ax.tick_params(axis="both", labelsize=10, pad=2)
    normal_tick_labels(ts_ax)
    panel_label(ts_ax, "c", x=0.015, y=0.93)
    fig.canvas.draw()
    group_specs = [
        (map_axes[:2], PALETTE["green"], low_t, flow_norm[low_t]),
        (map_axes[2:], NWP_COLOR, high_t, flow_norm[high_t]),
    ]
    for axes_group, color, target_t, target_y in group_specs:
        bboxes = [ax.get_position() for ax in axes_group]
        bbox = Bbox.union(bboxes)
        x0 = max(0.018, bbox.x0 - 0.040)
        y0 = max(0.515, bbox.y0 - 0.060)
        x1 = min(0.965, bbox.x1 + 0.020)
        y1 = min(0.940, bbox.y1 + 0.045)
        bbox = Bbox.from_extents(x0, y0, x1, y1)
        frame = plt.Rectangle(
            (bbox.x0, bbox.y0),
            bbox.width,
            bbox.height,
            transform=fig.transFigure,
            fill=False,
            edgecolor=color,
            linewidth=3.0,
            linestyle=(0, (5, 3)),
            clip_on=False,
            zorder=1,
        )
        fig.add_artist(frame)
        con = ConnectionPatch(
            xyA=(bbox.x0 + bbox.width / 2.0, bbox.y0),
            coordsA=fig.transFigure,
            xyB=(target_t, target_y),
            coordsB=ts_ax.transData,
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.4,
            color=color,
            linestyle=(0, (5, 3)),
            zorder=25,
        )
        fig.add_artist(con)
    fig.savefig(out / "deepmide_style_advection_norm_with_nwp_maps.png", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return {
        "created": True,
        "high_advection_time_index": high_t,
        "low_advection_time_index": low_t,
        "high_advection_norm": float(flow_norm[high_t]),
        "low_advection_norm": float(flow_norm[low_t]),
    }


def advection_covariance_for_flow(
    advection: dict[str, np.ndarray],
    flow: np.ndarray,
) -> np.ndarray | None:
    sigma = advection.get("flow_Sigma")
    if sigma is None or sigma.size == 0 or sigma.shape[0] != flow.shape[0]:
        return None
    processed = advection.get("processed_flow_mu")
    B = advection.get("B")
    if processed is not None and processed.size and np.allclose(flow, processed, equal_nan=True) and B is not None and B.size:
        return np.einsum("tij,tjk,tlk->til", B, sigma, B).astype(np.float32)
    return sigma.astype(np.float32, copy=False)


def plot_advection_covariance_calibration(
    flow: np.ndarray,
    advection: dict[str, np.ndarray],
    v_star: np.ndarray | None,
    out: Path,
) -> dict[str, Any]:
    cov = advection_covariance_for_flow(advection, flow)
    if v_star is None or v_star.shape[0] != flow.shape[0] or cov is None:
        return {"created": False, "reason": "requires V_star and learned flow covariance"}

    plt = setup_matplotlib()
    target = v_star[:, :2].astype(np.float64, copy=False)
    mean = flow.astype(np.float64, copy=False)
    cov = cov.astype(np.float64, copy=False)
    err = target - mean
    valid = finite_mask(target, mean, cov)
    d2 = np.full(flow.shape[0], np.nan, dtype=np.float64)
    trace = np.full(flow.shape[0], np.nan, dtype=np.float64)
    det = np.full(flow.shape[0], np.nan, dtype=np.float64)
    anisotropy = np.full(flow.shape[0], np.nan, dtype=np.float64)
    angle_err = np.full(flow.shape[0], np.nan, dtype=np.float64)
    for t in np.where(valid)[0]:
        S = 0.5 * (cov[t] + cov[t].T) + 1e-6 * np.eye(2)
        inv = np.linalg.pinv(S)
        d2[t] = float(err[t].T @ inv @ err[t])
        vals = np.linalg.eigvalsh(S)
        vals = np.clip(vals, 1e-10, None)
        trace[t] = float(vals.sum())
        det[t] = float(vals.prod())
        anisotropy[t] = float(vals[-1] / vals[0])
        angle_err[t] = abs(float(angle_diff_deg(mean[t : t + 1], target[t : t + 1])[0]))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), constrained_layout=True)
    bins = np.linspace(0.0, min(25.0, np.nanpercentile(d2, 99.5)), 45)
    axes[0].hist(d2[np.isfinite(d2)], bins=bins, density=True, color=MODEL_COLOR, alpha=0.86, edgecolor="white", linewidth=0.45)
    x = np.linspace(0.0, bins[-1], 300)
    axes[0].plot(x, 0.5 * np.exp(-0.5 * x), color=PALETTE["red"], linewidth=2.2, label="Chi-square df=2")
    axes[0].axvline(5.991, color="#111111", linestyle="--", linewidth=1.6, label="95% ellipse")
    axes[0].set_title("Covariance calibration")
    axes[0].set_xlabel("Mahalanobis distance squared")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)
    polish_axes(axes[0])
    panel_label(axes[0], "a")

    err_norm = np.linalg.norm(err, axis=1)
    axes[1].scatter(np.sqrt(trace), err_norm, s=14, color=PALETTE["purple"], alpha=0.25, edgecolors="none")
    axes[1].set_title("Predicted spread vs realized error")
    axes[1].set_xlabel("sqrt(trace(covariance))")
    axes[1].set_ylabel("||V_star - learned mean||")
    polish_axes(axes[1])
    panel_label(axes[1], "b")

    axes[2].scatter(np.sqrt(trace), angle_err, s=14, color=PALETTE["teal"], alpha=0.25, edgecolors="none")
    axes[2].set_title("Spread vs direction mismatch")
    axes[2].set_xlabel("sqrt(trace(covariance))")
    axes[2].set_ylabel("|direction error| (deg)")
    polish_axes(axes[2])
    panel_label(axes[2], "c")
    fig.savefig(out / "advection_covariance_calibration.png")
    plt.close(fig)

    return {
        "created": True,
        "mahalanobis_d2_mean": float(np.nanmean(d2)),
        "mahalanobis_d2_median": float(np.nanmedian(d2)),
        "ellipse_50pct_coverage": float(np.nanmean(d2 <= 1.386)),
        "ellipse_90pct_coverage": float(np.nanmean(d2 <= 4.605)),
        "ellipse_95pct_coverage": float(np.nanmean(d2 <= 5.991)),
        "corr_cov_trace_vs_error_norm": corrcoef(np.sqrt(trace), err_norm),
        "corr_cov_trace_vs_direction_error": corrcoef(np.sqrt(trace), angle_err),
        "cov_trace_mean": float(np.nanmean(trace)),
        "cov_ellipse_area_proxy_mean": float(np.nanmean(np.sqrt(det))),
        "cov_anisotropy_median": float(np.nanmedian(anisotropy)),
    }


def axis_angle_error_deg(axis_angle: np.ndarray, vector_angle: np.ndarray) -> np.ndarray:
    """Smallest angle between an unoriented axis and an oriented vector."""
    return np.abs((axis_angle - vector_angle + 90.0) % 180.0 - 90.0)


def vector_angle_deg(vec: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))


def plot_advection_covariance_anisotropy(
    flow: np.ndarray,
    advection: dict[str, np.ndarray],
    nwp_vec: np.ndarray,
    nwp_speed_mean: np.ndarray,
    v_star: np.ndarray | None,
    out: Path,
) -> dict[str, Any]:
    cov = advection_covariance_for_flow(advection, flow)
    if cov is None:
        return {"created": False, "reason": "requires learned flow covariance"}

    plt = setup_matplotlib()
    from matplotlib.patches import Ellipse

    n = min(flow.shape[0], cov.shape[0], nwp_vec.shape[0], nwp_speed_mean.shape[0])
    flow = flow[:n].astype(np.float64, copy=False)
    cov = cov[:n].astype(np.float64, copy=False)
    nwp_vec = nwp_vec[:n].astype(np.float64, copy=False)
    nwp_speed_mean = nwp_speed_mean[:n].astype(np.float64, copy=False)
    vstar_vec = None
    if v_star is not None and v_star.shape[0] >= n:
        vstar_vec = v_star[:n, :2].astype(np.float64, copy=False)

    major_lambda = np.full(n, np.nan, dtype=np.float64)
    minor_lambda = np.full(n, np.nan, dtype=np.float64)
    anisotropy_ratio = np.full(n, np.nan, dtype=np.float64)
    log_anisotropy = np.full(n, np.nan, dtype=np.float64)
    ellipse_area = np.full(n, np.nan, dtype=np.float64)
    major_axis_angle = np.full(n, np.nan, dtype=np.float64)
    axis_vec = np.full((n, 2), np.nan, dtype=np.float64)
    valid = finite_mask(flow, cov, nwp_vec, nwp_speed_mean)

    for t in np.where(valid)[0]:
        S = 0.5 * (cov[t] + cov[t].T) + 1e-8 * np.eye(2)
        vals, vecs = np.linalg.eigh(S)
        vals = np.clip(vals, 1e-12, None)
        minor_lambda[t] = float(vals[0])
        major_lambda[t] = float(vals[1])
        anisotropy_ratio[t] = float(np.sqrt(vals[1] / vals[0]))
        log_anisotropy[t] = float(np.log(anisotropy_ratio[t]))
        ellipse_area[t] = float(np.pi * np.sqrt(vals[0] * vals[1]))
        axis = vecs[:, 1]
        if np.dot(axis, nwp_vec[t]) < 0:
            axis = -axis
        axis_vec[t] = axis
        major_axis_angle[t] = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)

    nwp_angle = vector_angle_deg(nwp_vec) % 180.0
    nwp_norm = np.linalg.norm(nwp_vec, axis=1)
    axis_nwp_alignment = np.abs(np.nansum(axis_vec * nwp_vec, axis=1)) / (nwp_norm + 1e-8)
    axis_nwp_angle_error = axis_angle_error_deg(major_axis_angle, nwp_angle)
    flow_norm = np.linalg.norm(flow, axis=1)
    flow_angle = vector_angle_deg(flow) % 180.0
    axis_flow_angle_error = axis_angle_error_deg(major_axis_angle, flow_angle)

    axis_vstar_alignment = np.full(n, np.nan, dtype=np.float64)
    axis_vstar_angle_error = np.full(n, np.nan, dtype=np.float64)
    vstar_norm = np.full(n, np.nan, dtype=np.float64)
    if vstar_vec is not None:
        vstar_norm = np.linalg.norm(vstar_vec, axis=1)
        vstar_angle = vector_angle_deg(vstar_vec) % 180.0
        axis_vstar_alignment = np.abs(np.nansum(axis_vec * vstar_vec, axis=1)) / (vstar_norm + 1e-8)
        axis_vstar_angle_error = axis_angle_error_deg(major_axis_angle, vstar_angle)

    table = np.column_stack(
        [
            np.arange(n),
            major_lambda,
            minor_lambda,
            anisotropy_ratio,
            log_anisotropy,
            ellipse_area,
            major_axis_angle,
            nwp_angle,
            axis_nwp_alignment,
            axis_nwp_angle_error,
            nwp_speed_mean,
            flow_norm,
            axis_flow_angle_error,
            vstar_norm,
            axis_vstar_alignment,
            axis_vstar_angle_error,
        ]
    )
    header = ",".join(
        [
            "time_index",
            "cov_lambda_major",
            "cov_lambda_minor",
            "anisotropy_ratio_sqrt_lambda_major_over_minor",
            "log_anisotropy",
            "ellipse_area_1sigma",
            "major_axis_angle_deg_mod180",
            "nwp_wind_angle_deg_mod180",
            "axis_nwp_alignment_abs_cos",
            "axis_nwp_angle_error_deg",
            "nwp_speed_mean",
            "learned_flow_norm",
            "axis_flow_angle_error_deg",
            "vstar_norm",
            "axis_vstar_alignment_abs_cos",
            "axis_vstar_angle_error_deg",
        ]
    )
    np.savetxt(out / "advection_anisotropy_timeseries.csv", table, delimiter=",", header=header, comments="", fmt="%.8g")
    np.savez_compressed(
        out / "advection_anisotropy_timeseries.npz",
        time_index=np.arange(n),
        major_lambda=major_lambda,
        minor_lambda=minor_lambda,
        anisotropy_ratio=anisotropy_ratio,
        log_anisotropy=log_anisotropy,
        ellipse_area=ellipse_area,
        major_axis_angle=major_axis_angle,
        nwp_angle=nwp_angle,
        axis_nwp_alignment=axis_nwp_alignment,
        axis_nwp_angle_error=axis_nwp_angle_error,
        nwp_speed_mean=nwp_speed_mean,
        flow_norm=flow_norm,
        axis_flow_angle_error=axis_flow_angle_error,
        vstar_norm=vstar_norm,
        axis_vstar_alignment=axis_vstar_alignment,
        axis_vstar_angle_error=axis_vstar_angle_error,
    )

    summary = {
        "created": True,
        "anisotropy_ratio_mean": float(np.nanmean(anisotropy_ratio)),
        "anisotropy_ratio_median": float(np.nanmedian(anisotropy_ratio)),
        "anisotropy_ratio_p90": float(np.nanpercentile(anisotropy_ratio, 90)),
        "anisotropy_ratio_p95": float(np.nanpercentile(anisotropy_ratio, 95)),
        "axis_nwp_alignment_mean_abs_cos": float(np.nanmean(axis_nwp_alignment)),
        "axis_nwp_angle_error_median_deg": float(np.nanmedian(axis_nwp_angle_error)),
        "axis_nwp_angle_error_p25_deg": float(np.nanpercentile(axis_nwp_angle_error, 25)),
        "axis_nwp_angle_error_p75_deg": float(np.nanpercentile(axis_nwp_angle_error, 75)),
        "corr_anisotropy_vs_nwp_speed": corrcoef(anisotropy_ratio, nwp_speed_mean),
        "corr_anisotropy_vs_learned_flow_norm": corrcoef(anisotropy_ratio, flow_norm),
        "corr_alignment_vs_nwp_speed": corrcoef(axis_nwp_alignment, nwp_speed_mean),
        "corr_ellipse_area_vs_nwp_speed": corrcoef(ellipse_area, nwp_speed_mean),
    }
    if np.isfinite(axis_vstar_alignment).any():
        summary.update(
            {
                "axis_vstar_alignment_mean_abs_cos": float(np.nanmean(axis_vstar_alignment)),
                "axis_vstar_angle_error_median_deg": float(np.nanmedian(axis_vstar_angle_error)),
                "corr_anisotropy_vs_vstar_norm": corrcoef(anisotropy_ratio, vstar_norm),
            }
        )
    summary_csv = "metric,value\n" + "\n".join(f"{key},{value}" for key, value in summary.items() if key != "created")
    (out / "advection_anisotropy_summary.csv").write_text(summary_csv + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), constrained_layout=True)
    finite_aniso = anisotropy_ratio[np.isfinite(anisotropy_ratio)]
    bins = np.linspace(1.0, min(float(np.nanpercentile(finite_aniso, 99.5)), float(np.nanmax(finite_aniso))), 42)
    axes[0].hist(finite_aniso, bins=bins, color=MODEL_COLOR, alpha=0.86, edgecolor="white", linewidth=0.45)
    axes[0].axvline(summary["anisotropy_ratio_median"], color=PALETTE["red"], linestyle="--", linewidth=2.0, label="median")
    axes[0].set_title("Covariance anisotropy")
    axes[0].set_xlabel(r"$\sqrt{\lambda_{major}/\lambda_{minor}}$")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    polish_axes(axes[0])
    panel_label(axes[0], "a")

    sc = axes[1].scatter(
        nwp_speed_mean,
        anisotropy_ratio,
        c=axis_nwp_alignment,
        cmap="viridis",
        s=15,
        alpha=0.35,
        edgecolors="none",
    )
    axes[1].set_title("Anisotropy by wind regime")
    axes[1].set_xlabel("Mean NWP wind speed")
    axes[1].set_ylabel("Anisotropy ratio")
    polish_axes(axes[1])
    panel_label(axes[1], "b")
    cbar = fig.colorbar(sc, ax=axes[1], shrink=0.88)
    cbar.set_label("axis-wind alignment")
    cbar.ax.tick_params(labelsize=15, width=1.35, length=5.5)
    bold_tick_labels(cbar.ax)

    axes[2].hist(axis_nwp_angle_error[np.isfinite(axis_nwp_angle_error)], bins=np.linspace(0, 90, 37), color=PALETTE["green"], alpha=0.86, edgecolor="white", linewidth=0.45)
    axes[2].axvline(np.nanmedian(axis_nwp_angle_error), color=PALETTE["red"], linestyle="--", linewidth=2.0, label="median")
    axes[2].set_title("Major axis vs NWP wind direction")
    axes[2].set_xlabel("Axis-wind angle error (deg)")
    axes[2].set_ylabel("Count")
    axes[2].legend(fontsize=9)
    polish_axes(axes[2])
    panel_label(axes[2], "c")
    fig.savefig(out / "advection_covariance_anisotropy.png")
    plt.close(fig)

    def moving_average_nan(values: np.ndarray, window_size: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        finite_values = np.isfinite(values)
        weights = np.ones(window_size, dtype=np.float64)
        summed = np.convolve(np.where(finite_values, values, 0.0), weights, mode="same")
        counts = np.convolve(finite_values.astype(np.float64), weights, mode="same")
        out_values = summed / np.maximum(counts, 1.0)
        out_values[counts == 0.0] = np.nan
        return out_values

    fig, ax1 = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    x = np.arange(n)
    window = min(96, max(8, n // 80))
    anisotropy_smooth = moving_average_nan(anisotropy_ratio, window)
    nwp_speed_smooth = moving_average_nan(nwp_speed_mean, window)
    ax1.plot(x, anisotropy_ratio, color=MODEL_COLOR, alpha=0.26, linewidth=1.15, label="anisotropy ratio")
    if window > 1:
        ax1.plot(x, anisotropy_smooth, color=PALETTE["navy"], linestyle="--", linewidth=2.3, label=f"anisotropy {window}-step smooth")
    ax1.set_xlabel("Time index")
    ax1.set_ylabel("Anisotropy ratio")
    polish_axes(ax1)
    ax2 = ax1.twinx()
    ax2.plot(x, nwp_speed_mean, color=NWP_COLOR, alpha=0.20, linewidth=1.05, label="NWP speed")
    if window > 1:
        ax2.plot(x, nwp_speed_smooth, color=NWP_COLOR, linestyle="--", linewidth=2.2, label=f"NWP speed {window}-step smooth")
    ax2.set_ylabel("Mean NWP wind speed")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)
    ax1.set_title("Covariance anisotropy over time")
    fig.savefig(out / "advection_anisotropy_timeseries.png")
    plt.close(fig)

    relationship = np.column_stack([x, anisotropy_smooth, nwp_speed_smooth])
    np.savetxt(
        out / "advection_anisotropy_smoothed_relationship.csv",
        relationship,
        delimiter=",",
        header="time_index,anisotropy_ratio_smooth,nwp_speed_mean_smooth",
        comments="",
        fmt="%.8g",
    )
    summary["corr_smoothed_anisotropy_vs_nwp_speed"] = corrcoef(anisotropy_smooth, nwp_speed_smooth)
    summary_csv = "metric,value\n" + "\n".join(f"{key},{value}" for key, value in summary.items() if key != "created")
    (out / "advection_anisotropy_summary.csv").write_text(summary_csv + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.6, 5.7), constrained_layout=True)
    rel_mask = np.isfinite(anisotropy_smooth) & np.isfinite(nwp_speed_smooth)
    ax.scatter(
        nwp_speed_smooth[rel_mask],
        anisotropy_smooth[rel_mask],
        color="#6C757D",
        s=13,
        alpha=0.16,
        edgecolors="none",
        label="smoothed time steps",
    )
    if int(rel_mask.sum()) >= 20:
        edges = np.unique(np.nanquantile(nwp_speed_smooth[rel_mask], np.linspace(0.0, 1.0, 13)))
        bin_x = []
        bin_y = []
        bin_y_std = []
        for i in range(len(edges) - 1):
            if i == len(edges) - 2:
                in_bin = rel_mask & (nwp_speed_smooth >= edges[i]) & (nwp_speed_smooth <= edges[i + 1])
            else:
                in_bin = rel_mask & (nwp_speed_smooth >= edges[i]) & (nwp_speed_smooth < edges[i + 1])
            if int(in_bin.sum()) > 0:
                bin_x.append(float(np.nanmean(nwp_speed_smooth[in_bin])))
                bin_y.append(float(np.nanmean(anisotropy_smooth[in_bin])))
                bin_y_std.append(float(np.nanstd(anisotropy_smooth[in_bin])))
        if len(bin_x) >= 2:
            ax.fill_between(
                bin_x,
                np.asarray(bin_y) - np.asarray(bin_y_std),
                np.asarray(bin_y) + np.asarray(bin_y_std),
                color=PALETTE["red"],
                alpha=0.13,
                linewidth=0,
                label="binned +/- 1 std",
            )
            ax.plot(bin_x, bin_y, color=PALETTE["red"], marker="o", markersize=5.2, linewidth=2.7, label="binned mean trend")
            np.savetxt(
                out / "advection_anisotropy_smoothed_binned_trend.csv",
                np.column_stack([bin_x, bin_y, bin_y_std]),
                delimiter=",",
                header="nwp_speed_mean_smooth_bin_mean,anisotropy_ratio_smooth_bin_mean,anisotropy_ratio_smooth_bin_std",
                comments="",
                fmt="%.8g",
            )
    ax.text(
        0.03,
        0.96,
        f"r = {summary['corr_smoothed_anisotropy_vs_nwp_speed']:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#333333", "alpha": 0.92},
    )
    ax.set_xlabel(f"Smoothed mean NWP wind speed ({window}-step)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Smoothed anisotropy ratio ({window}-step)", fontsize=13, fontweight="bold")
    ax.set_title("Smoothed wind speed vs covariance anisotropy", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    bold_tick_labels(ax)
    ax.legend(loc="lower right", fontsize=9)
    polish_axes(ax)
    fig.savefig(out / "advection_anisotropy_smoothed_vs_nwp_speed.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    finite_idx = np.where(np.isfinite(anisotropy_ratio) & np.isfinite(axis_nwp_alignment))[0]
    if finite_idx.size:
        high_aniso = anisotropy_ratio >= np.nanpercentile(anisotropy_ratio[finite_idx], 90)
        high_align = axis_nwp_alignment >= np.nanpercentile(axis_nwp_alignment[finite_idx], 75)
        high_aniso_aligned_idx = finite_idx[high_aniso[finite_idx] & high_align[finite_idx]]
        if high_aniso_aligned_idx.size:
            score = (
                (anisotropy_ratio[high_aniso_aligned_idx] - np.nanmedian(anisotropy_ratio[finite_idx]))
                / (np.nanstd(anisotropy_ratio[finite_idx]) + 1e-8)
                + (axis_nwp_alignment[high_aniso_aligned_idx] - np.nanmedian(axis_nwp_alignment[finite_idx]))
                / (np.nanstd(axis_nwp_alignment[finite_idx]) + 1e-8)
            )
            high_aniso_aligned_t = int(high_aniso_aligned_idx[np.nanargmax(score)])
        else:
            score = np.nan_to_num(anisotropy_ratio[finite_idx], nan=0.0) * np.nan_to_num(axis_nwp_alignment[finite_idx], nan=0.0)
            high_aniso_aligned_t = int(finite_idx[np.nanargmax(score)])
        candidate_times = [
            int(finite_idx[np.nanargmin(anisotropy_ratio[finite_idx])]),
            int(finite_idx[np.nanargmax(anisotropy_ratio[finite_idx])]),
            int(finite_idx[np.nanargmax(axis_nwp_alignment[finite_idx])]),
            high_aniso_aligned_t,
        ]
        selected = []
        for t in candidate_times:
            if t not in selected:
                selected.append(t)
        selected = selected[:4]

        fig, axes = plt.subplots(1, len(selected), figsize=(4.3 * len(selected), 4.2), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, t, label in zip(axes, selected, ["lowest anisotropy", "highest anisotropy", "best wind-aligned", "high anisotropy + aligned"]):
            S = 0.5 * (cov[t] + cov[t].T) + 1e-8 * np.eye(2)
            vals, vecs = np.linalg.eigh(S)
            vals = np.clip(vals, 1e-12, None)
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            width = 4.0 * np.sqrt(vals[1])
            height = 4.0 * np.sqrt(vals[0])
            ell = Ellipse((0.0, 0.0), width=width, height=height, angle=angle, facecolor=PALETTE["light_blue"], edgecolor=MODEL_COLOR, linewidth=2.2, alpha=0.55)
            ax.add_patch(ell)
            radius = max(width, height) * 0.55
            wind_unit = nwp_vec[t] / (np.linalg.norm(nwp_vec[t]) + 1e-8)
            flow_unit = flow[t] / (np.linalg.norm(flow[t]) + 1e-8)
            ax.arrow(0, 0, wind_unit[0] * radius, wind_unit[1] * radius, color=NWP_COLOR, width=0.018 * radius, head_width=0.10 * radius, length_includes_head=True, label="NWP wind")
            ax.arrow(0, 0, flow_unit[0] * radius * 0.82, flow_unit[1] * radius * 0.82, color=MODEL_COLOR, width=0.014 * radius, head_width=0.08 * radius, length_includes_head=True, label="learned advection mean")
            lim = radius * 1.25
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{label}\nt={t}, ratio={anisotropy_ratio[t]:.2f}, align={axis_nwp_alignment[t]:.2f}", fontsize=11)
            ax.set_xlabel("U direction")
            ax.set_ylabel("V direction")
            polish_axes(ax)
        axes[0].legend(loc="upper left", fontsize=8)
        fig.savefig(out / "advection_covariance_ellipse_examples.png")
        plt.close(fig)

    return summary


def load_optional_vstar(config_path: Path | None, split: str | None, expected_t: int) -> np.ndarray | None:
    if config_path is None:
        return None
    try:
        import yaml

        from dataset.vector_data_utils import load_vector_dataset

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        data = load_vector_dataset(config, split=split or "online")
        v_star = data.get("V_star")
        if v_star is None:
            return None
        if v_star.shape[0] != expected_t:
            return None
        return v_star.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"[warn] Could not load V_star from {config_path}: {exc}")
        return None


def build_markdown_report(eval_dir: Path, out: Path, summary: dict[str, Any], results: dict[str, Any] | None) -> None:
    lines = [
        f"# VectorMIDE Physics Analysis",
        "",
        f"Source eval directory: `{eval_dir}`",
        "",
        "## What This Checks",
        "",
        "- Whether learned advection strength grows under stronger NWP wind regimes.",
        "- Whether learned advection direction agrees with the physical NWP wind direction.",
        "- Whether learned covariance anisotropy aligns with physical wind direction and wind regimes.",
        "- Whether the model correction points toward the true NWP residual.",
        "- Whether stronger advection is associated with stronger transition mixing.",
        "- Whether U/V transition cross-coupling is consistent with Northern Hemisphere inertial/Coriolis turning.",
        "- Whether forecast skill changes across horizons.",
        "",
        "## Key Numbers",
        "",
    ]
    for section, values in summary.items():
        lines.append(f"### {section}")
        for key, value in values.items():
            if isinstance(value, float):
                lines.append(f"- `{key}`: {value:.4f}")
            else:
                lines.append(f"- `{key}`: {value}")
        lines.append("")
    if results:
        lines.extend(
            [
                "## Existing Evaluation",
                "",
                f"- target mode: `{results.get('target_mode')}`",
                f"- history window size: `{results.get('history_window_size')}`",
                f"- forecast horizon: `{results.get('forecast_horizon')}`",
                f"- one-step model vs NWP RMSE improvement: `{results.get('model_vs_nwp_improvement', {}).get('rmse_percent')}`",
                f"- one-step model vs persistence RMSE improvement: `{results.get('model_vs_persistence_improvement', {}).get('rmse_percent')}`",
                "",
            ]
        )
    (out / "physics_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VectorMIDE learned advection against physical wind signals.")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Directory containing forecasts.npz and advection_parameters.npz")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Optional config used to load V_star labels")
    parser.add_argument("--split", default="online", choices=["online", "offline"])
    parser.add_argument("--dt-seconds", type=float, default=None, help="Time step in seconds for transition-rate physics checks")
    parser.add_argument(
        "--flow-kind",
        default="processed",
        choices=["processed", "raw"],
        help="Use B @ flow when available, otherwise raw shared flow.",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    out = args.output_dir or (eval_dir / "physics_analysis")
    out.mkdir(parents=True, exist_ok=True)

    forecasts = load_npz(eval_dir / "forecasts.npz")
    advection = load_npz(eval_dir / "advection_parameters.npz")
    transitions = load_npz(eval_dir / "transition_matrices.npz")
    metrics = load_npz(eval_dir / "multi_step_metrics.npz")
    results = None
    results_path = eval_dir / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    flow = advection.get("flow_mu")
    if args.flow_kind == "processed" and advection.get("processed_flow_mu") is not None and advection["processed_flow_mu"].size:
        flow = advection["processed_flow_mu"]
    if flow is None or flow.size == 0:
        raise ValueError("No learned flow found in advection_parameters.npz")

    optional_data = load_optional_dataset(args.config, args.split, expected_t=flow.shape[0])
    v_star = optional_data.get("V_star")
    if v_star is not None and v_star.shape[0] != flow.shape[0]:
        v_star = None
    dt_seconds = args.dt_seconds
    if dt_seconds is None:
        dt_seconds = 600.0
        if args.config is not None and args.config.exists():
            try:
                import yaml

                with args.config.open("r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
                dt_seconds = float(config.get("dt_seconds", dt_seconds))
            except Exception as exc:
                print(f"[warn] Could not read dt_seconds from {args.config}: {exc}")

    nwp_vec = mean_station_vector(forecasts["nwp_baseline"])
    nwp_speed_mean = np.nanmean(station_speed(forecasts["nwp_baseline"]), axis=1)
    mean_wind_vec = np.nanmean(nwp_vec, axis=0)

    summary: dict[str, Any] = {}
    plot_skill_curves(metrics, out)
    summary["station_metrics"] = per_station_metrics(
        forecasts,
        optional_data.get("station_latlon"),
        mean_wind_vec,
        out,
    )
    summary["wind_rose"] = plot_direction_speed_roses(nwp_vec, nwp_speed_mean, flow, out)
    summary["advection_physics"] = plot_flow_physics(flow, nwp_vec, nwp_speed_mean, v_star, out)
    summary["advection_covariance"] = plot_advection_covariance_calibration(flow, advection, v_star, out)
    summary["advection_anisotropy"] = plot_advection_covariance_anisotropy(
        flow,
        advection,
        nwp_vec,
        nwp_speed_mean,
        v_star,
        out,
    )
    summary["residual_correction"] = plot_residual_correction(forecasts, out)
    summary["transition_physics"] = plot_transition_summary(advection, transitions, out)
    summary["transition_coriolis_rotation"] = plot_coriolis_rotation_analysis(
        transitions,
        out,
        dt_seconds=float(dt_seconds),
        station_latlon=optional_data.get("station_latlon"),
    )
    summary["forecast_intervals"] = plot_station_forecast_intervals(
        forecasts,
        advection,
        transitions,
        flow,
        out,
    )
    summary["deepmide_style_figure"] = plot_deepmide_style_advection_figure(
        flow,
        optional_data,
        out,
    )

    (out / "physics_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    build_markdown_report(eval_dir, out, summary, results)
    print(f"Wrote physics analysis to {out}")


if __name__ == "__main__":
    main()
