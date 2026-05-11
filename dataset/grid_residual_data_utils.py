from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .vector_data_utils import (
    NWP_CHANNELS,
    Standardizer,
    build_z_from_measurements,
    latlon_to_xy_km,
    load_mat_variable,
    load_nwp_maps,
    load_nwp_uv140,
    standardize_maps,
)


TARGET_COMPONENT_SLICES = {
    "u": slice(0, 3),
    "v": slice(3, 6),
}
DEFAULT_TARGET_CHANNEL = {
    "u": "u140",
    "v": "v140",
}


def station_latlon_from_config(
    data_cfg: dict[str, Any],
    measurement_path: str | Path,
) -> np.ndarray:
    if data_cfg.get("station_latlon"):
        raw = data_cfg["station_latlon"]
        if isinstance(raw, dict):
            values = [raw[name] for name in sorted(raw)]
        else:
            values = raw
        station_latlon = np.asarray(values, dtype=np.float32)
    elif data_cfg.get("use_measurement_station_coords", True):
        station_lat = load_mat_variable(measurement_path, "LatValue_vec").reshape(-1)
        station_lon = load_mat_variable(measurement_path, "LonValue_vec").reshape(-1)
        station_latlon = np.stack([station_lat, station_lon], axis=1).astype(np.float32)
    else:
        raise ValueError(
            "Grid residual training needs station coordinates. Set "
            "data.use_measurement_station_coords: true or provide data.station_latlon."
        )
    if station_latlon.ndim != 2 or station_latlon.shape[1] != 2:
        raise ValueError(f"Expected station_latlon shape [N,2], got {station_latlon.shape}")
    return station_latlon


def grid_xy_from_latlon(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    origin = (float(np.nanmean(grid_lat)), float(np.nanmean(grid_lon)))
    grid_xy = latlon_to_xy_km(grid_lat, grid_lon, origin=origin)
    return grid_xy[..., 0], grid_xy[..., 1], origin


def station_xy_from_latlon(
    station_latlon: np.ndarray,
    origin: tuple[float, float],
) -> np.ndarray:
    return latlon_to_xy_km(station_latlon[:, 0], station_latlon[:, 1], origin=origin)


def station_sample_yx_from_latlon(
    station_latlon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> np.ndarray:
    """Map station lat/lon to fractional grid ``[row, col]`` coordinates.

    The NWP grid is mildly rotated, so this uses a local affine inverse on every
    grid cell and chooses the best in-cell or nearest clipped solution.
    """
    grid_x, grid_y, origin = grid_xy_from_latlon(grid_lat, grid_lon)
    station_xy = station_xy_from_latlon(station_latlon, origin=origin)
    grid_xy = np.stack([grid_x, grid_y], axis=-1).astype(np.float64)
    h, w = grid_lat.shape
    positions: list[list[float]] = []

    for target in station_xy.astype(np.float64):
        best: tuple[float, int, int, np.ndarray] | None = None
        for i in range(h - 1):
            for j in range(w - 1):
                p00 = grid_xy[i, j]
                row_vec = grid_xy[i + 1, j] - p00
                col_vec = grid_xy[i, j + 1] - p00
                basis = np.stack([row_vec, col_vec], axis=1)
                det = float(np.linalg.det(basis))
                if abs(det) < 1.0e-10:
                    continue
                frac = np.linalg.solve(basis, target - p00)
                clipped = np.clip(frac, 0.0, 1.0)
                approx = p00 + basis @ clipped
                error = float(np.linalg.norm(target - approx))
                inside_penalty = 0.0 if np.all((frac >= 0.0) & (frac <= 1.0)) else 1.0e3
                score = inside_penalty + error
                if best is None or score < best[0]:
                    best = (score, i, j, clipped)
        if best is None:
            flat = np.nanargmin(((grid_xy - target) ** 2).sum(axis=-1))
            i, j = np.unravel_index(int(flat), (h, w))
            positions.append([float(i), float(j)])
        else:
            _, i, j, clipped = best
            positions.append([float(i + clipped[0]), float(j + clipped[1])])

    return np.asarray(positions, dtype=np.float32)


def bilinear_sample_grid(
    grid_field: Tensor,
    sample_yx: Tensor,
    padding_mode: str = "border",
) -> Tensor:
    """Sample grid fields at fractional station locations.

    Args:
        grid_field: ``[B,T,H,W]`` or ``[B,T,C,H,W]``.
        sample_yx: ``[N,2]`` fractional ``[row, col]`` coordinates.

    Returns:
        ``[B,T,N]`` for scalar input or ``[B,T,C,N]`` for multi-channel input.
    """
    squeeze_channel = False
    if grid_field.ndim == 4:
        grid_field = grid_field.unsqueeze(2)
        squeeze_channel = True
    if grid_field.ndim != 5:
        raise ValueError(f"Expected grid_field [B,T,H,W] or [B,T,C,H,W], got {tuple(grid_field.shape)}")

    bsz, steps, channels, height, width = grid_field.shape
    sample_yx = sample_yx.to(device=grid_field.device, dtype=grid_field.dtype)
    if sample_yx.ndim != 2 or sample_yx.shape[1] != 2:
        raise ValueError(f"Expected sample_yx shape [N,2], got {tuple(sample_yx.shape)}")

    y = sample_yx[:, 0]
    x = sample_yx[:, 1]
    norm_x = 2.0 * x / max(width - 1, 1) - 1.0
    norm_y = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, -1, 1, 2)
    grid = grid.expand(bsz * steps, -1, -1, -1)

    flat = grid_field.reshape(bsz * steps, channels, height, width)
    sampled = F.grid_sample(
        flat,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    sampled = sampled.squeeze(-1).reshape(bsz, steps, channels, -1)
    if squeeze_channel or channels == 1:
        return sampled[:, :, 0, :]
    return sampled


def build_coordinate_features(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    x_center = 0.5 * (float(np.nanmin(grid_x)) + float(np.nanmax(grid_x)))
    y_center = 0.5 * (float(np.nanmin(grid_y)) + float(np.nanmax(grid_y)))
    x_scale = max(float(np.nanmax(grid_x) - np.nanmin(grid_x)), 1.0e-6)
    y_scale = max(float(np.nanmax(grid_y) - np.nanmin(grid_y)), 1.0e-6)
    x_norm = 2.0 * (grid_x - x_center) / x_scale
    y_norm = 2.0 * (grid_y - y_center) / y_scale
    return np.stack([x_norm, y_norm], axis=0).astype(np.float32)


def build_distance_features(
    station_xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    distance_scale_km: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    distances = []
    for x_i, y_i in station_xy:
        dist = np.sqrt((grid_x - float(x_i)) ** 2 + (grid_y - float(y_i)) ** 2)
        distances.append(dist)
    distance_km = np.stack(distances, axis=0).astype(np.float32)
    if distance_scale_km is None or distance_scale_km <= 0.0:
        distance_scale_km = float(np.nanmax(distance_km))
    distance_norm = distance_km / max(float(distance_scale_km), 1.0e-6)
    return distance_km, distance_norm.astype(np.float32)


def build_residual_broadcast(
    station_residuals: Tensor,
    height: int,
    width: int,
    station_input_mask: Optional[Tensor] = None,
) -> Tensor:
    if station_residuals.ndim != 3:
        raise ValueError(f"Expected station_residuals [B,T,N], got {tuple(station_residuals.shape)}")
    residuals = torch.nan_to_num(station_residuals, nan=0.0, posinf=0.0, neginf=0.0)
    if station_input_mask is not None:
        mask = station_input_mask.to(device=residuals.device, dtype=residuals.dtype).view(1, 1, -1)
        residuals = residuals * mask
    return residuals[:, :, :, None, None].expand(-1, -1, -1, height, width)


def build_advection_alignment(
    station_xy: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    u: Tensor,
    v: Tensor,
    eps: float = 1.0e-6,
) -> Tensor:
    if u.shape != v.shape or u.ndim != 4:
        raise ValueError("u and v must both have shape [B,T,H,W]")
    station_xy = station_xy.to(device=u.device, dtype=u.dtype)
    grid_x = grid_x.to(device=u.device, dtype=u.dtype)
    grid_y = grid_y.to(device=u.device, dtype=u.dtype)

    wind_norm = torch.sqrt(u.pow(2) + v.pow(2) + eps)
    alignments = []
    for station in station_xy:
        dx = grid_x - station[0]
        dy = grid_y - station[1]
        dist = torch.sqrt(dx.pow(2) + dy.pow(2) + eps)
        dot = u * dx.view(1, 1, *dx.shape) + v * dy.view(1, 1, *dy.shape)
        alignments.append(dot / (wind_norm * dist.view(1, 1, *dist.shape) + eps))
    return torch.stack(alignments, dim=2)


def build_advective_weight(
    distance_km: Tensor,
    alignment: Tensor,
    length_scale_km: float,
) -> Tensor:
    distance = distance_km.to(device=alignment.device, dtype=alignment.dtype).view(
        1,
        1,
        *distance_km.shape,
    )
    spatial_decay = torch.exp(-distance / max(float(length_scale_km), 1.0e-6))
    return spatial_decay * torch.relu(alignment)


def feature_channel_count(nwp_channels: int, n_stations: int, feature_cfg: dict[str, Any]) -> int:
    count = int(nwp_channels)
    if feature_cfg.get("use_coords", True):
        count += 2
    if feature_cfg.get("use_station_residual_broadcast", True):
        count += n_stations
    if feature_cfg.get("use_distance_features", True):
        count += n_stations
    if feature_cfg.get("use_advection_alignment", True):
        count += n_stations
    if feature_cfg.get("use_advective_weight", True):
        count += n_stations
    return count


def build_grid_residual_features(
    nwp_input: Tensor,
    station_residuals: Tensor,
    u: Tensor,
    v: Tensor,
    static: dict[str, Tensor],
    feature_cfg: dict[str, Any],
    station_input_mask: Optional[Tensor] = None,
) -> Tensor:
    if nwp_input.ndim != 5:
        raise ValueError(f"Expected nwp_input [B,T,C,H,W], got {tuple(nwp_input.shape)}")
    bsz, steps, _, height, width = nwp_input.shape
    features = [nwp_input]

    def expand_static(value: Tensor) -> Tensor:
        return value.to(device=nwp_input.device, dtype=nwp_input.dtype).unsqueeze(0).unsqueeze(0).expand(
            bsz,
            steps,
            -1,
            -1,
            -1,
        )

    if feature_cfg.get("use_coords", True):
        features.append(expand_static(static["coord_features"]))
    if feature_cfg.get("use_station_residual_broadcast", True):
        features.append(
            build_residual_broadcast(
                station_residuals,
                height,
                width,
                station_input_mask=station_input_mask,
            )
        )
    if feature_cfg.get("use_distance_features", True):
        features.append(expand_static(static["distance_features"]))
    alignment = None
    if feature_cfg.get("use_advection_alignment", True) or feature_cfg.get("use_advective_weight", True):
        alignment = build_advection_alignment(
            static["station_xy"],
            static["grid_x"],
            static["grid_y"],
            u,
            v,
        )
    if feature_cfg.get("use_advection_alignment", True):
        features.append(alignment.to(device=nwp_input.device, dtype=nwp_input.dtype))
    if feature_cfg.get("use_advective_weight", True):
        if alignment is None:
            raise RuntimeError("alignment should have been computed")
        features.append(
            build_advective_weight(
                static["distance_km"],
                alignment,
                length_scale_km=float(feature_cfg.get("advective_length_scale_km", 100.0)),
            ).to(device=nwp_input.device, dtype=nwp_input.dtype)
        )
    return torch.cat(features, dim=2)


def load_grid_residual_dataset(
    config: dict[str, Any],
    split: str = "offline",
    time_limit: Optional[int] = None,
) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    if split not in {"offline", "online"}:
        raise ValueError("split must be 'offline' or 'online'")

    measurement_path = Path(data_cfg[f"{split}_measurement_path"])
    nwp_path = Path(data_cfg[f"{split}_nwp_path"])
    target_component = str(data_cfg.get("target_component", "u")).lower()
    if target_component not in TARGET_COMPONENT_SLICES:
        raise ValueError("data.target_component must be 'u' or 'v' for the current measurement files")
    target_channel = str(data_cfg.get("target_channel") or DEFAULT_TARGET_CHANNEL[target_component]).lower()
    if target_channel not in NWP_CHANNELS:
        raise KeyError(f"Unknown target_channel {target_channel!r}")

    channel_names = config.get("nwp_channel_names") or data_cfg.get("nwp_channel_names")
    if channel_names is None:
        channel_names = ("u100", "v100", "u140", "v140", "u180", "v180")
    x = load_nwp_maps(nwp_path, channel_names=channel_names, time_limit=time_limit)
    target_grid = load_nwp_maps(nwp_path, channel_names=[target_channel], time_limit=time_limit)

    u140_hw_t, v140_hw_t = load_nwp_uv140(nwp_path, time_limit=time_limit)
    u = np.moveaxis(u140_hw_t, 2, 0)
    v = np.moveaxis(v140_hw_t, 2, 0)

    ws_uv = load_mat_variable(measurement_path, "Ws_uv")
    if time_limit is not None:
        ws_uv = ws_uv[:time_limit]
    y = build_z_from_measurements(ws_uv)
    obs_values = y[:, TARGET_COMPONENT_SLICES[target_component]]

    t = min(x.shape[0], target_grid.shape[0], u.shape[0], v.shape[0], obs_values.shape[0])
    x = x[:t]
    target_grid = target_grid[:t]
    u = u[:t]
    v = v[:t]
    obs_values = obs_values[:t]
    obs_mask = np.isfinite(obs_values)
    obs_values = np.nan_to_num(obs_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    grid_lat = load_mat_variable(nwp_path, "LatValue").astype(np.float32)
    grid_lon = load_mat_variable(nwp_path, "LonValue").astype(np.float32)
    station_latlon = station_latlon_from_config(data_cfg, measurement_path)
    sample_yx = station_sample_yx_from_latlon(station_latlon, grid_lat, grid_lon)
    grid_x, grid_y, origin = grid_xy_from_latlon(grid_lat, grid_lon)
    station_xy = station_xy_from_latlon(station_latlon, origin=origin)
    coord_features = build_coordinate_features(grid_x, grid_y)
    distance_km, distance_features = build_distance_features(
        station_xy,
        grid_x,
        grid_y,
        distance_scale_km=data_cfg.get("distance_scale_km"),
    )

    x_standardizer: Standardizer | None
    if data_cfg.get("standardize_x", True):
        x, x_standardizer = standardize_maps(x)
    else:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x_standardizer = None

    target_grid = np.nan_to_num(target_grid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return {
        "nwp_input": x,
        "nwp_target": target_grid,
        "obs_values": obs_values,
        "obs_mask": obs_mask.astype(np.float32),
        "u": u,
        "v": v,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "grid_x": grid_x.astype(np.float32),
        "grid_y": grid_y.astype(np.float32),
        "coord_features": coord_features,
        "distance_km": distance_km,
        "distance_features": distance_features,
        "station_latlon": station_latlon.astype(np.float32),
        "station_xy": station_xy.astype(np.float32),
        "sample_yx": sample_yx,
        "target_component": target_component,
        "target_channel": target_channel,
        "nwp_channel_names": tuple(channel_names),
        "x_standardizer": x_standardizer,
    }


def static_tensors(data: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    keys = [
        "grid_x",
        "grid_y",
        "coord_features",
        "distance_km",
        "distance_features",
        "station_xy",
        "sample_yx",
    ]
    return {
        key: torch.from_numpy(np.asarray(data[key], dtype=np.float32)).to(device)
        for key in keys
    }
