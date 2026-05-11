from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.grid_residual_data_utils import (
    bilinear_sample_grid,
    build_grid_residual_features,
    feature_channel_count,
    load_grid_residual_dataset,
    static_tensors,
)
from dataset.vector_data_utils import build_z_from_measurements, load_mat_variable
from train.train_grid_residual import build_model, load_config, make_batch, station_masks
from train.train_vector_offline import print_device_info, resolve_device


def load_component_prediction(
    checkpoint_path: str | Path,
    config_path: str | None,
    split: str,
    device: torch.device,
    limit: int | None,
    chunk_size: int,
    holdout_station: int | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = load_config(config_path) if config_path else checkpoint.get("config")
    if config is None:
        raise ValueError(f"No config found in {checkpoint_path}; pass an explicit config.")

    data = load_grid_residual_dataset(config, split=split, time_limit=limit)
    static = static_tensors(data, device)
    n_stations = int(data["obs_values"].shape[1])
    inferred_holdout = checkpoint.get("best", {}).get("holdout_station") if checkpoint.get("best") else None
    effective_holdout = holdout_station if holdout_station is not None else inferred_holdout
    station_input_mask, _ = station_masks(n_stations, config, effective_holdout, device)
    in_channels = int(
        checkpoint.get(
            "in_channels",
            feature_channel_count(data["nwp_input"].shape[1], n_stations, config.get("features", {})),
        )
    )
    model = build_model(config, in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    residual_chunks = []
    raw_chunks = []
    with torch.no_grad():
        for start in range(0, int(data["nwp_input"].shape[0]), chunk_size):
            end = min(start + chunk_size, int(data["nwp_input"].shape[0]))
            batch = make_batch(data, [start], end - start, device)
            nwp_at_stations = bilinear_sample_grid(batch["nwp_target"], static["sample_yx"])
            station_residuals = batch["obs_values"] - nwp_at_stations
            features = build_grid_residual_features(
                nwp_input=batch["nwp_input"],
                station_residuals=station_residuals,
                u=batch["u"],
                v=batch["v"],
                static=static,
                feature_cfg=config.get("features", {}),
                station_input_mask=station_input_mask,
            )
            pred_residual = model(features)
            residual_chunks.append(pred_residual.squeeze(0).squeeze(1).cpu().numpy().astype(np.float32))
            raw_chunks.append(batch["nwp_target"].squeeze(0).squeeze(1).cpu().numpy().astype(np.float32))

    arrays = {
        "raw": np.concatenate(raw_chunks, axis=0),
        "residual": np.concatenate(residual_chunks, axis=0),
        "corrected": np.concatenate(raw_chunks, axis=0) + np.concatenate(residual_chunks, axis=0),
    }
    return data, arrays


def load_observed_uv(config: dict[str, Any], split: str, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = config.get("data", {})
    measurement_path = Path(data_cfg[f"{split}_measurement_path"])
    ws_uv = load_mat_variable(measurement_path, "Ws_uv")
    if limit is not None:
        ws_uv = ws_uv[:limit]
    y = build_z_from_measurements(ws_uv)
    return y[:, :3].astype(np.float32), y[:, 3:6].astype(np.float32)


def speed_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u * u + v * v)
    direction = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    return speed, direction


def sample_field_at_stations(field: np.ndarray, sample_yx: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(field[:, None]).unsqueeze(0).to(device)
    samples = bilinear_sample_grid(tensor, torch.from_numpy(sample_yx).to(device))
    return samples.squeeze(0).cpu().numpy()


def nearest_station_values(field: np.ndarray, sample_yx: np.ndarray) -> np.ndarray:
    h, w = field.shape[1:]
    rows = np.clip(np.rint(sample_yx[:, 0]).astype(int), 0, h - 1)
    cols = np.clip(np.rint(sample_yx[:, 1]).astype(int), 0, w - 1)
    return np.stack([field[:, row, col] for row, col in zip(rows, cols)], axis=1)


def representative_points(height: int, width: int, count: int) -> list[tuple[int, int]]:
    count = max(1, int(count))
    grid_cols = int(np.ceil(np.sqrt(count)))
    grid_rows = int(np.ceil(count / grid_cols))
    rows = np.linspace(0, height - 1, grid_rows).round().astype(int)
    cols = np.linspace(0, width - 1, grid_cols).round().astype(int)
    points = [(int(row), int(col)) for row in rows for col in cols]
    return points[:count]


def plot_wind_maps(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    corr_u: np.ndarray,
    corr_v: np.ndarray,
    station_yx: np.ndarray,
    time_index: int,
    out_path: Path,
    quiver_stride: int,
) -> None:
    raw_speed, raw_dir = speed_direction(raw_u[time_index], raw_v[time_index])
    corr_speed, corr_dir = speed_direction(corr_u[time_index], corr_v[time_index])
    residual_speed, _ = speed_direction(corr_u[time_index] - raw_u[time_index], corr_v[time_index] - raw_v[time_index])
    speed_delta = corr_speed - raw_speed

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        (raw_speed, "Raw NWP wind speed"),
        (corr_speed, "Corrected wind speed"),
        (speed_delta, "Corrected - raw speed"),
        (raw_dir, "Raw NWP direction"),
        (corr_dir, "Corrected direction"),
        (residual_speed, "Residual vector magnitude"),
    ]
    yy, xx = np.mgrid[0 : raw_speed.shape[0], 0 : raw_speed.shape[1]]
    for ax, (field, title) in zip(axes.ravel(), panels):
        image = ax.imshow(field, origin="lower", cmap="viridis")
        ax.scatter(station_yx[:, 1], station_yx[:, 0], c="white", edgecolors="black", s=55, zorder=3)
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
    fig.suptitle(f"Full-grid wind correction at time index {time_index}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_representative_point_lines(
    raw_speed: np.ndarray,
    corr_speed: np.ndarray,
    points: list[tuple[int, int]],
    out_path: Path,
) -> None:
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
    obs_speed: np.ndarray,
    raw_station_speed: np.ndarray,
    corrected_station_speed: np.ndarray,
    nearest_corrected_speed: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    time = np.arange(obs_speed.shape[0])
    for idx, ax in enumerate(axes):
        ax.plot(time, obs_speed[:, idx], color="#111111", linewidth=1.2, label="measurement")
        ax.plot(time, raw_station_speed[:, idx], color="#4c78a8", linewidth=1.0, label="NWP sampled")
        ax.plot(time, corrected_station_speed[:, idx], color="#f58518", linewidth=1.0, label="NWP + residual sampled")
        ax.plot(
            time,
            nearest_corrected_speed[:, idx],
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
    fig.suptitle("Observed stations vs raw NWP and corrected NWP")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize full-grid U/V residual wind corrections.")
    parser.add_argument("--u-checkpoint", required=True)
    parser.add_argument("--v-checkpoint", required=True)
    parser.add_argument("--u-config", default=None)
    parser.add_argument("--v-config", default=None)
    parser.add_argument("--split", default="offline", choices=["offline", "online"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--num-points", type=int, default=40)
    parser.add_argument("--quiver-stride", type=int, default=4)
    parser.add_argument("--holdout-station", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    checkpoint_u = torch.load(args.u_checkpoint, map_location="cpu", weights_only=False)
    base_config = load_config(args.u_config) if args.u_config else checkpoint_u.get("config")
    if base_config is None:
        raise ValueError("No config found in U checkpoint; pass --u-config.")
    device_name = args.device if args.device is not None else base_config.get("device", "auto")
    device = resolve_device(device_name, allow_fallback=bool(base_config.get("allow_device_fallback", True)))
    print_device_info(device)

    data_u, pred_u = load_component_prediction(
        args.u_checkpoint,
        args.u_config,
        split=args.split,
        device=device,
        limit=args.limit,
        chunk_size=args.chunk_size,
        holdout_station=args.holdout_station,
    )
    data_v, pred_v = load_component_prediction(
        args.v_checkpoint,
        args.v_config,
        split=args.split,
        device=device,
        limit=args.limit,
        chunk_size=args.chunk_size,
        holdout_station=args.holdout_station,
    )

    n_time = min(pred_u["raw"].shape[0], pred_v["raw"].shape[0])
    raw_u = pred_u["raw"][:n_time]
    raw_v = pred_v["raw"][:n_time]
    corr_u = pred_u["corrected"][:n_time]
    corr_v = pred_v["corrected"][:n_time]
    raw_speed, _ = speed_direction(raw_u, raw_v)
    corr_speed, _ = speed_direction(corr_u, corr_v)

    time_index = args.time_index if args.time_index >= 0 else n_time + args.time_index
    time_index = int(np.clip(time_index, 0, n_time - 1))
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / "grid_residual_visualization" / f"{Path(args.u_checkpoint).stem}_{Path(args.v_checkpoint).stem}_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    station_yx = data_u["sample_yx"]
    obs_u, obs_v = load_observed_uv(base_config, args.split, args.limit)
    obs_u = obs_u[:n_time]
    obs_v = obs_v[:n_time]
    obs_speed, _ = speed_direction(obs_u, obs_v)
    raw_station_u = sample_field_at_stations(raw_u, station_yx, device)
    raw_station_v = sample_field_at_stations(raw_v, station_yx, device)
    corr_station_u = sample_field_at_stations(corr_u, station_yx, device)
    corr_station_v = sample_field_at_stations(corr_v, station_yx, device)
    raw_station_speed, _ = speed_direction(raw_station_u, raw_station_v)
    corr_station_speed, _ = speed_direction(corr_station_u, corr_station_v)
    nearest_corr_u = nearest_station_values(corr_u, station_yx)
    nearest_corr_v = nearest_station_values(corr_v, station_yx)
    nearest_corr_speed, _ = speed_direction(nearest_corr_u, nearest_corr_v)

    points = representative_points(raw_speed.shape[1], raw_speed.shape[2], args.num_points)
    plot_wind_maps(
        raw_u,
        raw_v,
        corr_u,
        corr_v,
        station_yx=station_yx,
        time_index=time_index,
        out_path=out_dir / f"wind_maps_t{time_index}.png",
        quiver_stride=max(1, int(args.quiver_stride)),
    )
    plot_representative_point_lines(
        raw_speed,
        corr_speed,
        points,
        out_path=out_dir / f"{len(points)}_grid_point_speed_lines.png",
    )
    plot_station_lines(
        obs_speed,
        raw_station_speed,
        corr_station_speed,
        nearest_corr_speed,
        out_path=out_dir / "station_measurement_vs_corrected.png",
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
    metadata = {
        "split": args.split,
        "n_time": n_time,
        "time_index": time_index,
        "num_points": len(points),
        "points": points,
        "outputs": [
            f"wind_maps_t{time_index}.png",
            f"{len(points)}_grid_point_speed_lines.png",
            "station_measurement_vs_corrected.png",
            "representative_grid_point_speed.csv",
        ],
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved visualization artifacts to {out_dir}")


if __name__ == "__main__":
    main()
