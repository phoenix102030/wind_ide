from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


NWP_CHANNELS = {
    "ws100": 0,
    "ws140": 1,
    "ws180": 2,
    "t2": 3,
    "rh2": 4,
    "slp": 5,
    "u100": 6,
    "v100": 7,
    "u140": 8,
    "v140": 9,
    "u180": 10,
    "v180": 11,
}

MEASUREMENT_140M_UV_COLUMNS = [6, 8, 10, 7, 9, 11]
EARTH_RADIUS_KM = 6371.0


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def _loadmat_scipy(path: Path, variable_name: str) -> Optional[np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    try:
        data = loadmat(path, variable_names=[variable_name])
    except NotImplementedError:
        return None
    if variable_name not in data:
        raise KeyError(f"{variable_name!r} not found in {path}")
    return np.asarray(data[variable_name])


def _loadmat_h5py(path: Path, variable_name: str) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Install scipy or h5py to load MATLAB files: python -m pip install scipy h5py"
        ) from exc
    with h5py.File(path, "r") as handle:
        if variable_name not in handle:
            raise KeyError(f"{variable_name!r} not found in {path}")
        arr = np.asarray(handle[variable_name])
    if arr.ndim > 1:
        arr = arr.transpose(tuple(reversed(range(arr.ndim))))
    return arr


def load_mat_variable(path: str | Path, variable_name: str) -> np.ndarray:
    path = Path(path)
    arr = _loadmat_scipy(path, variable_name)
    if arr is None:
        arr = _loadmat_h5py(path, variable_name)
    return np.asarray(arr)


def imputed_measurement_path(path: str | Path) -> Path:
    """Return the sibling ``*_imputed.mat`` path for a measurement file."""
    path = Path(path)
    if path.stem.endswith("_imputed"):
        return path
    return path.with_name(f"{path.stem}_imputed{path.suffix}")


def resolve_measurement_path(data_cfg: dict[str, Any], split: str) -> tuple[Path, Path]:
    """Resolve the configured measurement file, preferring imputed data by default."""
    configured = Path(data_cfg[f"{split}_measurement_path"])
    if bool(data_cfg.get("prefer_imputed_measurements", True)):
        candidate = imputed_measurement_path(configured)
        if candidate.exists():
            return configured, candidate
    return configured, configured


def finite_summary(values: np.ndarray) -> dict[str, int | float]:
    values = np.asarray(values)
    total = int(values.size)
    finite = int(np.isfinite(values).sum())
    return {
        "total": total,
        "finite": finite,
        "missing": total - finite,
        "finite_fraction": float(finite / total) if total else 1.0,
    }


def impute_time_series_columns(values: np.ndarray) -> np.ndarray:
    """Linearly fill missing values column-by-column along time.

    This is a fallback for experiments without precomputed ``*_imputed.mat``
    files. Columns that are entirely missing are filled with zero.
    """
    arr = np.asarray(values, dtype=np.float32).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D time series array, got {arr.shape}")
    time = np.arange(arr.shape[0], dtype=np.float64)
    for col in range(arr.shape[1]):
        series = arr[:, col]
        finite = np.isfinite(series)
        if finite.all():
            continue
        if finite.any():
            arr[:, col] = np.interp(time, time[finite], series[finite]).astype(np.float32)
        else:
            arr[:, col] = 0.0
    return arr


def _open_hdf5_mat(path: Path):
    try:
        import h5py
    except ImportError:
        return None
    try:
        return h5py.File(path, "r")
    except OSError:
        return None


def _ordered_hdf5_channel_read(dataset: Any, indices: list[int]) -> np.ndarray:
    order = np.argsort(indices)
    sorted_indices = np.asarray(indices, dtype=np.int64)[order]
    inverse = np.argsort(order)
    arr = np.asarray(dataset[sorted_indices.tolist(), ...])
    return arr[inverse]


def load_nwp_maps(
    path: str | Path,
    channel_names: Optional[list[str] | tuple[str, ...]] = None,
    time_limit: Optional[int] = None,
) -> np.ndarray:
    """Load selected NWP channels as ``[T,C,H,W]`` without reading all variables."""
    path = Path(path)
    if channel_names is None:
        channel_names = ("u100", "v100", "u140", "v140", "u180", "v180")
    indices = [NWP_CHANNELS[name.lower()] for name in channel_names]

    handle = _open_hdf5_mat(path)
    if handle is not None:
        with handle:
            dataset = handle["allVariMin_Grid"]
            if dataset.ndim != 4:
                raise ValueError(f"Expected 4D allVariMin_Grid, got {dataset.shape}")
            if dataset.shape[0] >= max(indices) + 1:
                time_sel = slice(None, time_limit)
                order = np.argsort(indices)
                sorted_indices = np.asarray(indices, dtype=np.int64)[order]
                inverse = np.argsort(order)
                arr = np.asarray(dataset[sorted_indices.tolist(), time_sel, :, :])
                arr = arr[inverse]
                return np.transpose(arr, (1, 0, 2, 3)).astype(np.float32, copy=False)
            if dataset.shape[-1] >= max(indices) + 1:
                order = np.argsort(indices)
                sorted_indices = np.asarray(indices, dtype=np.int64)[order]
                inverse = np.argsort(order)
                arr = np.asarray(dataset[:, :, slice(None, time_limit), sorted_indices.tolist()])
                arr = arr[..., inverse]
                x = np.moveaxis(arr, 2, 0)
                x = np.moveaxis(x, -1, 1)
                return x.astype(np.float32, copy=False)
            raise ValueError(f"Cannot locate variable dimension in allVariMin_Grid {dataset.shape}")

    nwp_grid = load_mat_variable(path, "allVariMin_Grid")
    if time_limit is not None:
        nwp_grid = nwp_grid[:, :, :time_limit, :]
    return build_x_from_nwp_grid(nwp_grid, channel_names)


def load_nwp_uv140(path: str | Path, time_limit: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Load 140m U/V as two arrays with shape ``[H,W,T]``."""
    path = Path(path)
    handle = _open_hdf5_mat(path)
    if handle is not None:
        with handle:
            dataset = handle["allVariMin_Grid"]
            if dataset.shape[0] > NWP_CHANNELS["v140"]:
                indices = [NWP_CHANNELS["u140"], NWP_CHANNELS["v140"]]
                order = np.argsort(indices)
                sorted_indices = np.asarray(indices, dtype=np.int64)[order]
                inverse = np.argsort(order)
                arr = np.asarray(dataset[sorted_indices.tolist(), slice(None, time_limit), :, :])
                arr = arr[inverse]
                u = np.transpose(arr[0], (1, 2, 0))
                v = np.transpose(arr[1], (1, 2, 0))
                return u.astype(np.float32, copy=False), v.astype(np.float32, copy=False)
            if dataset.shape[-1] > NWP_CHANNELS["v140"]:
                arr = np.asarray(
                    dataset[:, :, slice(None, time_limit), [NWP_CHANNELS["u140"], NWP_CHANNELS["v140"]]]
                )
                return (
                    arr[..., 0].astype(np.float32, copy=False),
                    arr[..., 1].astype(np.float32, copy=False),
                )
            raise ValueError(f"Cannot locate variable dimension in allVariMin_Grid {dataset.shape}")

    nwp_grid = load_mat_variable(path, "allVariMin_Grid")
    if time_limit is not None:
        nwp_grid = nwp_grid[:, :, :time_limit, :]
    return (
        nwp_grid[..., NWP_CHANNELS["u140"]].astype(np.float32, copy=False),
        nwp_grid[..., NWP_CHANNELS["v140"]].astype(np.float32, copy=False),
    )


def build_z_from_measurements(ws_uv: np.ndarray) -> np.ndarray:
    """Extract 140m U/V observations in state order ``[U1,U2,U3,V1,V2,V3]``."""
    ws_uv = np.asarray(ws_uv)
    if ws_uv.ndim != 2 or ws_uv.shape[1] < 12:
        raise ValueError(f"Expected measurement shape [T,>=12], got {ws_uv.shape}")
    return ws_uv[:, MEASUREMENT_140M_UV_COLUMNS].astype(np.float32, copy=False)


def build_x_from_nwp_grid(
    nwp_grid: np.ndarray,
    channel_names: Optional[list[str] | tuple[str, ...]] = None,
) -> np.ndarray:
    """Convert NWP grid ``[H,W,T,V]`` to PyTorch-ready ``[T,C,H,W]``."""
    if channel_names is None:
        channel_names = ("u100", "v100", "u140", "v140", "u180", "v180")
    nwp_grid = np.asarray(nwp_grid)
    if nwp_grid.ndim != 4:
        raise ValueError(f"Expected NWP grid shape [H,W,T,V], got {nwp_grid.shape}")
    indices = []
    for name in channel_names:
        key = name.lower()
        if key not in NWP_CHANNELS:
            raise KeyError(f"Unknown NWP channel {name!r}")
        indices.append(NWP_CHANNELS[key])
    x = np.moveaxis(nwp_grid[..., indices], 2, 0)
    x = np.moveaxis(x, -1, 1)
    return x.astype(np.float32, copy=False)


def latlon_to_xy_km(
    lat: np.ndarray,
    lon: np.ndarray,
    origin: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Convert latitude/longitude degrees to local planar km coordinates."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if origin is None:
        lat0 = float(np.nanmean(lat))
        lon0 = float(np.nanmean(lon))
    else:
        lat0, lon0 = origin

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    x = EARTH_RADIUS_KM * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = EARTH_RADIUS_KM * (lat_rad - lat0_rad)
    return np.stack([x, y], axis=-1).astype(np.float32)


def coords_from_station_latlon(
    station_latlon: dict[str, list[float]] | list[list[float]],
    origin: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    if isinstance(station_latlon, dict):
        values = [station_latlon[name] for name in sorted(station_latlon)]
    else:
        values = station_latlon
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3, 2):
        raise ValueError("station_latlon must describe exactly three [lat, lon] pairs")
    return latlon_to_xy_km(arr[:, 0], arr[:, 1], origin=origin)


def coords_from_grid_indices(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    station_grid_indices: list[list[int]] | tuple[tuple[int, int], ...],
) -> np.ndarray:
    xy_grid = latlon_to_xy_km(lat_grid, lon_grid)
    coords = []
    h, w = lat_grid.shape
    for i, j in station_grid_indices:
        ii = int(np.clip(i, 0, h - 1))
        jj = int(np.clip(j, 0, w - 1))
        coords.append(xy_grid[ii, jj])
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (3, 2):
        raise ValueError("station_grid_indices must contain exactly three [row, col] pairs")
    return coords


def nearest_grid_indices_from_station_latlon(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    station_lat: np.ndarray,
    station_lon: np.ndarray,
) -> list[list[int]]:
    """Find the nearest NWP grid point for each measurement station."""
    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    lon_grid = np.asarray(lon_grid, dtype=np.float64)
    station_lat = np.asarray(station_lat, dtype=np.float64).reshape(-1)
    station_lon = np.asarray(station_lon, dtype=np.float64).reshape(-1)
    if station_lat.shape[0] != 3 or station_lon.shape[0] != 3:
        raise ValueError("Expected exactly three station lat/lon values")

    indices: list[list[int]] = []
    for lat, lon in zip(station_lat, station_lon):
        lat0 = np.deg2rad(float(lat))
        dy = (lat_grid - lat) * 111.0
        dx = (lon_grid - lon) * 111.0 * np.cos(lat0)
        flat_index = int(np.nanargmin(dx * dx + dy * dy))
        i, j = np.unravel_index(flat_index, lat_grid.shape)
        indices.append([int(i), int(j)])
    return indices


def build_nwp_station_baseline(
    u140: np.ndarray,
    v140: np.ndarray,
    station_grid_indices: list[list[int]] | tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Extract nearest-grid 140m NWP baseline in state order [U1,U2,U3,V1,V2,V3]."""
    u140 = np.asarray(u140)
    v140 = np.asarray(v140)
    if u140.shape != v140.shape or u140.ndim != 3:
        raise ValueError("u140 and v140 must both have shape [H,W,T]")

    h, w, _ = u140.shape
    u_series = []
    v_series = []
    for i, j in station_grid_indices:
        ii = int(np.clip(i, 0, h - 1))
        jj = int(np.clip(j, 0, w - 1))
        u_series.append(u140[ii, jj])
        v_series.append(v140[ii, jj])
    if len(u_series) != 3:
        raise ValueError("station_grid_indices must contain exactly three [row, col] pairs")
    return np.stack([*u_series, *v_series], axis=1).astype(np.float32, copy=False)


def standardize_maps(
    x: np.ndarray,
    standardizer: Optional[Standardizer] = None,
) -> tuple[np.ndarray, Standardizer]:
    if standardizer is None:
        mean = np.nanmean(x, axis=(0, 2, 3), keepdims=True)
        std = np.nanstd(x, axis=(0, 2, 3), keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        standardizer = Standardizer(mean=mean, std=np.maximum(std, 1.0e-6))
    x_scaled = standardizer.transform(x)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return x_scaled.astype(np.float32), standardizer


def standardize_series(
    z: np.ndarray,
    standardizer: Optional[Standardizer] = None,
) -> tuple[np.ndarray, Standardizer]:
    if standardizer is None:
        mean = np.nanmean(z, axis=0, keepdims=True)
        std = np.nanstd(z, axis=0, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        standardizer = Standardizer(mean=mean, std=np.maximum(std, 1.0e-6))
    return standardizer.transform(z).astype(np.float32), standardizer


def _patch_mean(field: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    h, w, _ = field.shape
    i, j = center
    i0, i1 = max(i - radius, 0), min(i + radius + 1, h)
    j0, j1 = max(j - radius, 0), min(j + radius + 1, w)
    return np.nanmean(field[i0:i1, j0:j1], axis=(0, 1))


def _patch_mean_2d(field: np.ndarray, center: tuple[int, int], radius: int) -> float:
    h, w = field.shape
    i, j = center
    i0, i1 = max(i - radius, 0), min(i + radius + 1, h)
    j0, j1 = max(j - radius, 0), min(j + radius + 1, w)
    return float(np.nanmean(field[i0:i1, j0:j1]))


def _latlon_grid_spacing_km(lat_grid: np.ndarray, lon_grid: np.ndarray) -> tuple[float, float]:
    xy = latlon_to_xy_km(lat_grid, lon_grid)
    dx = float(np.nanmean(np.abs(np.diff(xy[..., 0], axis=1))))
    dy = float(np.nanmean(np.abs(np.diff(xy[..., 1], axis=0))))
    return max(dx, 1.0e-6), max(dy, 1.0e-6)


def build_simple_advection_labels(
    nwp_grid: np.ndarray,
    station_grid_indices: Optional[list[list[int]]] = None,
    dt_seconds: float = 600.0,
    patch_radius: int = 2,
) -> np.ndarray:
    """Build simple pseudo labels from mean 140m NWP U/V fields.

    Returns ``[T,4]`` in km per time step:
    ``[u_bar, v_bar, u_bar, v_bar]``.
    """
    u140 = nwp_grid[..., NWP_CHANNELS["u140"]]
    v140 = nwp_grid[..., NWP_CHANNELS["v140"]]
    scale = dt_seconds / 1000.0
    if station_grid_indices:
        uv = []
        for center in station_grid_indices:
            center_tuple = (int(center[0]), int(center[1]))
            u_mean = _patch_mean(u140, center_tuple, patch_radius)
            v_mean = _patch_mean(v140, center_tuple, patch_radius)
            uv.append(np.stack([u_mean, v_mean], axis=-1))
        uv_mean = np.nanmean(np.stack(uv, axis=0), axis=0)
        u_bar = uv_mean[:, 0]
        v_bar = uv_mean[:, 1]
    else:
        u_bar = np.nanmean(u140, axis=(0, 1))
        v_bar = np.nanmean(v140, axis=(0, 1))
    labels = np.stack([u_bar, v_bar, u_bar, v_bar], axis=-1)
    return (labels * scale).astype(np.float32)


def build_simple_advection_labels_from_uv(
    u140: np.ndarray,
    v140: np.ndarray,
    station_grid_indices: Optional[list[list[int]]] = None,
    dt_seconds: float = 600.0,
    patch_radius: int = 2,
) -> np.ndarray:
    """Same as ``build_simple_advection_labels`` for preloaded 140m U/V."""
    scale = dt_seconds / 1000.0
    if station_grid_indices:
        uv = []
        for center in station_grid_indices:
            center_tuple = (int(center[0]), int(center[1]))
            u_mean = _patch_mean(u140, center_tuple, patch_radius)
            v_mean = _patch_mean(v140, center_tuple, patch_radius)
            uv.append(np.stack([u_mean, v_mean], axis=-1))
        uv_mean = np.nanmean(np.stack(uv, axis=0), axis=0)
        u_bar = uv_mean[:, 0]
        v_bar = uv_mean[:, 1]
    else:
        u_bar = np.nanmean(u140, axis=(0, 1))
        v_bar = np.nanmean(v140, axis=(0, 1))
    labels = np.stack([u_bar, v_bar, u_bar, v_bar], axis=-1)
    return (labels * scale).astype(np.float32)


def build_optical_flow_advection_labels(
    nwp_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    dt_seconds: float = 600.0,
    stride: int = 2,
    ridge: float = 1.0e-4,
) -> np.ndarray:
    """Estimate advection labels by least-squares optical flow on NWP U/V."""
    xy = latlon_to_xy_km(lat_grid, lon_grid)
    dx = float(np.nanmean(np.abs(np.diff(xy[..., 0], axis=1))))
    dy = float(np.nanmean(np.abs(np.diff(xy[..., 1], axis=0))))
    dx = max(dx, 1.0e-6)
    dy = max(dy, 1.0e-6)

    u = nwp_grid[..., NWP_CHANNELS["u140"]].astype(np.float64)
    v = nwp_grid[..., NWP_CHANNELS["v140"]].astype(np.float64)
    T = u.shape[2]
    labels = np.zeros((T, 4), dtype=np.float32)

    def solve_component(field: np.ndarray, t: int) -> np.ndarray:
        d_dt = (field[..., t + 1] - field[..., t]) / dt_seconds
        d_y, d_x = np.gradient(field[..., t], dy, dx)
        A = np.stack([d_x[::stride, ::stride].ravel(), d_y[::stride, ::stride].ravel()], axis=1)
        b = -d_dt[::stride, ::stride].ravel()
        valid = np.isfinite(A).all(axis=1) & np.isfinite(b)
        if valid.sum() < 2:
            return np.zeros(2, dtype=np.float32)
        lhs = A[valid].T @ A[valid] + ridge * np.eye(2)
        rhs = A[valid].T @ b[valid]
        return np.linalg.solve(lhs, rhs).astype(np.float32)

    for t in range(T - 1):
        cu = solve_component(u, t) * dt_seconds
        cv = solve_component(v, t) * dt_seconds
        labels[t] = [cu[0], cu[1], cv[0], cv[1]]
    if T > 1:
        labels[-1] = labels[-2]
    return labels


def build_optical_flow_advection_labels_from_uv(
    u140: np.ndarray,
    v140: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    dt_seconds: float = 600.0,
    stride: int = 2,
    ridge: float = 1.0e-4,
) -> np.ndarray:
    xy = latlon_to_xy_km(lat_grid, lon_grid)
    dx = float(np.nanmean(np.abs(np.diff(xy[..., 0], axis=1))))
    dy = float(np.nanmean(np.abs(np.diff(xy[..., 1], axis=0))))
    dx = max(dx, 1.0e-6)
    dy = max(dy, 1.0e-6)

    u = u140.astype(np.float64)
    v = v140.astype(np.float64)
    T = u.shape[2]
    labels = np.zeros((T, 4), dtype=np.float32)

    def solve_component(field: np.ndarray, t: int) -> np.ndarray:
        d_dt = (field[..., t + 1] - field[..., t]) / dt_seconds
        d_y, d_x = np.gradient(field[..., t], dy, dx)
        A = np.stack([d_x[::stride, ::stride].ravel(), d_y[::stride, ::stride].ravel()], axis=1)
        b = -d_dt[::stride, ::stride].ravel()
        valid = np.isfinite(A).all(axis=1) & np.isfinite(b)
        if valid.sum() < 2:
            return np.zeros(2, dtype=np.float32)
        lhs = A[valid].T @ A[valid] + ridge * np.eye(2)
        rhs = A[valid].T @ b[valid]
        return np.linalg.solve(lhs, rhs).astype(np.float32)

    for t in range(T - 1):
        cu = solve_component(u, t) * dt_seconds
        cv = solve_component(v, t) * dt_seconds
        labels[t] = [cu[0], cu[1], cv[0], cv[1]]
    if T > 1:
        labels[-1] = labels[-2]
    return labels


def build_shared_optical_flow_advection_labels_from_uv(
    u140: np.ndarray,
    v140: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    dt_seconds: float = 600.0,
    stride: int = 2,
    ridge: float = 1.0e-4,
) -> np.ndarray:
    """Estimate one shared 2D flow from both NWP U and V optical-flow equations."""
    dx, dy = _latlon_grid_spacing_km(lat_grid, lon_grid)
    u = u140.astype(np.float64)
    v = v140.astype(np.float64)
    T = u.shape[2]
    labels = np.zeros((T, 4), dtype=np.float32)

    for t in range(T - 1):
        du_dt = (u[..., t + 1] - u[..., t]) / dt_seconds
        dv_dt = (v[..., t + 1] - v[..., t]) / dt_seconds
        du_dy, du_dx = np.gradient(u[..., t], dy, dx)
        dv_dy, dv_dx = np.gradient(v[..., t], dy, dx)

        A_u = np.stack([du_dx[::stride, ::stride].ravel(), du_dy[::stride, ::stride].ravel()], axis=1)
        A_v = np.stack([dv_dx[::stride, ::stride].ravel(), dv_dy[::stride, ::stride].ravel()], axis=1)
        A = np.concatenate([A_u, A_v], axis=0)
        b = np.concatenate(
            [
                -du_dt[::stride, ::stride].ravel(),
                -dv_dt[::stride, ::stride].ravel(),
            ],
            axis=0,
        )
        valid = np.isfinite(A).all(axis=1) & np.isfinite(b)
        if valid.sum() >= 2:
            lhs = A[valid].T @ A[valid] + ridge * np.eye(2)
            rhs = A[valid].T @ b[valid]
            flow = np.linalg.solve(lhs, rhs).astype(np.float32) * dt_seconds
            labels[t] = [flow[0], flow[1], flow[0], flow[1]]
    if T > 1:
        labels[-1] = labels[-2]
    return labels


def build_deformation_labels_from_uv(
    u140: np.ndarray,
    v140: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    station_grid_indices: Optional[list[list[int]]] = None,
    dt_seconds: float = 600.0,
    patch_radius: int = 2,
    delta_clip: float = 0.3,
) -> np.ndarray:
    """Build near-identity 2x2 deformation labels from local NWP velocity gradients."""
    dx, dy = _latlon_grid_spacing_km(lat_grid, lon_grid)
    u = u140.astype(np.float64)
    v = v140.astype(np.float64)
    T = u.shape[2]
    labels = np.zeros((T, 2, 2), dtype=np.float32)
    identity = np.eye(2, dtype=np.float32)
    scale = dt_seconds / 1000.0

    for t in range(T):
        du_dy, du_dx = np.gradient(u[..., t], dy, dx)
        dv_dy, dv_dx = np.gradient(v[..., t], dy, dx)
        if station_grid_indices:
            grads = []
            for center in station_grid_indices:
                center_tuple = (int(center[0]), int(center[1]))
                grads.append(
                    [
                        _patch_mean_2d(du_dx, center_tuple, patch_radius),
                        _patch_mean_2d(du_dy, center_tuple, patch_radius),
                        _patch_mean_2d(dv_dx, center_tuple, patch_radius),
                        _patch_mean_2d(dv_dy, center_tuple, patch_radius),
                    ]
                )
            grad = np.nanmean(np.asarray(grads, dtype=np.float64), axis=0)
        else:
            grad = np.asarray(
                [
                    np.nanmean(du_dx),
                    np.nanmean(du_dy),
                    np.nanmean(dv_dx),
                    np.nanmean(dv_dy),
                ],
                dtype=np.float64,
            )

        delta = scale * grad.reshape(2, 2)
        if delta_clip > 0.0:
            delta = np.clip(delta, -delta_clip, delta_clip)
        labels[t] = identity + delta.astype(np.float32)
    return labels


def load_vector_dataset(
    config: dict[str, Any],
    split: str = "offline",
    time_limit: Optional[int] = None,
) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    if split not in {"offline", "online"}:
        raise ValueError("split must be 'offline' or 'online'")

    configured_measurement_path, measurement_path = resolve_measurement_path(data_cfg, split)
    nwp_path = Path(data_cfg[f"{split}_nwp_path"])
    ws_uv = load_mat_variable(measurement_path, "Ws_uv")
    if time_limit is not None:
        ws_uv = ws_uv[:time_limit]
    measurement_summary_before = finite_summary(ws_uv)
    if measurement_summary_before["missing"]:
        missing_policy = str(data_cfg.get("measurement_missing_policy", "error")).lower()
        if missing_policy in {"interpolate", "linear", "impute"}:
            ws_uv = impute_time_series_columns(ws_uv)
        elif missing_policy in {"allow", "mask"}:
            pass
        else:
            candidate = imputed_measurement_path(measurement_path)
            hint = (
                f" Use {candidate} or set data.measurement_missing_policy: interpolate "
                "if you want the loader to fill missing measurements."
            )
            raise ValueError(
                f"{measurement_path} contains {measurement_summary_before['missing']} missing "
                f"Ws_uv values out of {measurement_summary_before['total']}."
                + hint
            )
    measurement_summary_after = finite_summary(ws_uv)
    station_lat_vec = None
    station_lon_vec = None
    if data_cfg.get("use_measurement_station_coords", True):
        try:
            station_lat_vec = load_mat_variable(measurement_path, "LatValue_vec").reshape(-1)
            station_lon_vec = load_mat_variable(measurement_path, "LonValue_vec").reshape(-1)
        except KeyError:
            station_lat_vec = None
            station_lon_vec = None
    lat_grid = load_mat_variable(nwp_path, "LatValue")
    lon_grid = load_mat_variable(nwp_path, "LonValue")

    z = build_z_from_measurements(ws_uv)
    x = load_nwp_maps(nwp_path, config.get("nwp_channel_names"), time_limit=time_limit)
    T = min(x.shape[0], z.shape[0])
    x = x[:T]
    z = z[:T]
    u140, v140 = load_nwp_uv140(nwp_path, time_limit=time_limit)
    u140 = u140[:, :, :T]
    v140 = v140[:, :, :T]

    if data_cfg.get("station_latlon"):
        station_latlon_cfg = data_cfg["station_latlon"]
        coords = coords_from_station_latlon(station_latlon_cfg)
        if isinstance(station_latlon_cfg, dict):
            station_latlon_values = [station_latlon_cfg[name] for name in sorted(station_latlon_cfg)]
        else:
            station_latlon_values = station_latlon_cfg
        station_latlon_arr = np.asarray(station_latlon_values, dtype=np.float64)
        baseline_grid_indices = nearest_grid_indices_from_station_latlon(
            lat_grid,
            lon_grid,
            station_latlon_arr[:, 0],
            station_latlon_arr[:, 1],
        )
    elif station_lat_vec is not None and station_lon_vec is not None:
        station_latlon = np.stack([station_lat_vec, station_lon_vec], axis=1)
        coords = coords_from_station_latlon(station_latlon)
        station_latlon_arr = station_latlon.astype(np.float64, copy=False)
        baseline_grid_indices = nearest_grid_indices_from_station_latlon(
            lat_grid,
            lon_grid,
            station_lat_vec,
            station_lon_vec,
        )
    else:
        station_indices = data_cfg.get("station_grid_indices")
        if station_indices is None:
            station_indices = [[18, 18], [20, 20], [22, 22]]
        coords = coords_from_grid_indices(lat_grid, lon_grid, station_indices)
        baseline_grid_indices = [[int(i), int(j)] for i, j in station_indices]
        station_latlon_arr = np.asarray(
            [[lat_grid[int(i), int(j)], lon_grid[int(i), int(j)]] for i, j in baseline_grid_indices],
            dtype=np.float64,
        )

    if data_cfg.get("nwp_baseline_grid_indices") is not None:
        baseline_grid_indices = [
            [int(i), int(j)]
            for i, j in data_cfg["nwp_baseline_grid_indices"]
        ]
    nwp_baseline = build_nwp_station_baseline(u140, v140, baseline_grid_indices)
    target_mode = str(data_cfg.get("target_mode", config.get("target_mode", "measurement"))).lower()
    if target_mode in {"measurement", "direct"}:
        model_target = z
    elif target_mode in {"residual_nwp", "nwp_residual", "residual"}:
        model_target = z - nwp_baseline
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    target_summary = finite_summary(model_target)
    if target_summary["missing"] and bool(data_cfg.get("require_complete_target", True)):
        raise ValueError(
            f"VectorMIDE target contains {target_summary['missing']} missing values after "
            f"building target_mode={target_mode!r}. Check measurement and NWP baseline inputs, "
            "or set data.require_complete_target: false to keep masked-observation training."
        )

    label_grid_indices = [[int(i), int(j)] for i, j in baseline_grid_indices]
    label_mode = data_cfg.get("advection_label_mode", "simple")
    if label_mode == "simple":
        v_star = build_simple_advection_labels_from_uv(
            u140,
            v140,
            station_grid_indices=label_grid_indices,
            dt_seconds=float(config.get("dt_seconds", 600.0)),
            patch_radius=int(data_cfg.get("patch_radius", 2)),
        )[:T]
    elif label_mode in {"shared_optical_flow", "joint_optical_flow"}:
        v_star = build_shared_optical_flow_advection_labels_from_uv(
            u140,
            v140,
            lat_grid,
            lon_grid,
            dt_seconds=float(config.get("dt_seconds", 600.0)),
            stride=int(data_cfg.get("optical_flow_stride", 2)),
            ridge=float(data_cfg.get("optical_flow_ridge", 1.0e-4)),
        )[:T]
    elif label_mode == "optical_flow":
        v_star = build_optical_flow_advection_labels_from_uv(
            u140,
            v140,
            lat_grid,
            lon_grid,
            dt_seconds=float(config.get("dt_seconds", 600.0)),
            stride=int(data_cfg.get("optical_flow_stride", 2)),
            ridge=float(data_cfg.get("optical_flow_ridge", 1.0e-4)),
        )[:T]
    elif label_mode in {"none", None}:
        v_star = None
    else:
        raise ValueError(f"Unknown advection_label_mode: {label_mode}")

    deformation_label_mode = data_cfg.get("deformation_label_mode", "none")
    if deformation_label_mode == "nwp_gradient":
        B_star = build_deformation_labels_from_uv(
            u140,
            v140,
            lat_grid,
            lon_grid,
            station_grid_indices=label_grid_indices,
            dt_seconds=float(config.get("dt_seconds", 600.0)),
            patch_radius=int(data_cfg.get("patch_radius", 2)),
            delta_clip=float(data_cfg.get("deformation_label_clip", config.get("deformation_scale", 0.3))),
        )[:T]
    elif deformation_label_mode in {"none", None}:
        B_star = None
    else:
        raise ValueError(f"Unknown deformation_label_mode: {deformation_label_mode}")

    if data_cfg.get("standardize_x", True):
        x, x_standardizer = standardize_maps(x)
    else:
        x_standardizer = None
    if data_cfg.get("standardize_z", False):
        model_target, z_standardizer = standardize_series(model_target)
    else:
        z_standardizer = None

    return {
        "X": x,
        "Z": model_target.astype(np.float32, copy=False),
        "Y": z.astype(np.float32, copy=False),
        "nwp_baseline": nwp_baseline.astype(np.float32, copy=False),
        "V_star": v_star,
        "B_star": B_star,
        "coords": coords,
        "station_latlon": station_latlon_arr.astype(np.float32, copy=False),
        "target_mode": target_mode,
        "baseline_grid_indices": np.asarray(baseline_grid_indices, dtype=np.int64),
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "x_standardizer": x_standardizer,
        "z_standardizer": z_standardizer,
        "configured_measurement_path": str(configured_measurement_path),
        "measurement_path": str(measurement_path),
        "measurement_summary_before": measurement_summary_before,
        "measurement_summary_after": measurement_summary_after,
        "target_summary": target_summary,
    }


class VectorWindowDataset:
    """Simple rolling-window dataset for PyTorch training loops."""

    def __init__(
        self,
        x: np.ndarray,
        z: np.ndarray,
        v_star: Optional[np.ndarray] = None,
        window_size: int = 1008,
        stride: int = 24,
    ) -> None:
        self.x = x
        self.z = z
        self.v_star = v_star
        self.window_size = window_size
        self.starts = list(range(0, max(1, len(x) - window_size + 1), stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        start = self.starts[index]
        end = start + self.window_size
        item = {"X": self.x[start:end], "Z": self.z[start:end]}
        if self.v_star is not None:
            item["V_star"] = self.v_star[start:end]
        return item
