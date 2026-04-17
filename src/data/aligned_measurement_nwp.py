from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.mat_loader import load_mat_auto, normalize_nwp_grid_shape


MEAS_COLS = {
    "E05":   {"100": (0, 1),  "140": (6, 7),   "180": (12, 13)},
    "E06":   {"100": (2, 3),  "140": (8, 9),   "180": (14, 15)},
    "ASOW6": {"100": (4, 5),  "140": (10, 11), "180": (16, 17)},
}

NWP_VAR_MAP = {
    "WS_100m": 0, "WS_140m": 1, "WS_180m": 2,
    "T2": 3, "RH2": 4, "SLP": 5,
    "U_100m": 6, "V_100m": 7,
    "U_140m": 8, "V_140m": 9,
    "U_180m": 10, "V_180m": 11,
}

NWP_CHANNELS_BY_MODE = {
    "uv6": [
        "U_100m",
        "V_100m",
        "U_140m",
        "V_140m",
        "U_180m",
        "V_180m",
    ],
    "all12": [
        "WS_100m",
        "WS_140m",
        "WS_180m",
        "T2",
        "RH2",
        "SLP",
        "U_100m",
        "V_100m",
        "U_140m",
        "V_140m",
        "U_180m",
        "V_180m",
    ],
}


@dataclass
class AlignedDataBundle:
    z_meas: np.ndarray      # [T, 3, 2]
    meas_lat: np.ndarray    # [3]
    meas_lon: np.ndarray    # [3]
    nwp_uv: np.ndarray      # [T, C, Y, X]
    nwp_lat: np.ndarray     # [Y, X]
    nwp_lon: np.ndarray     # [Y, X]


@dataclass
class NormalizationStats:
    z_mean: np.ndarray      # [3, 2]
    z_std: np.ndarray       # [3, 2]
    nwp_mean: np.ndarray    # [C]
    nwp_std: np.ndarray     # [C]

    def to_config_dict(self):
        return {
            "z_mean": self.z_mean.tolist(),
            "z_std": self.z_std.tolist(),
            "nwp_mean": self.nwp_mean.tolist(),
            "nwp_std": self.nwp_std.tolist(),
        }


def _safe_std(std, min_std=1e-6):
    std = np.asarray(std, dtype=np.float32)
    return np.where(np.isfinite(std) & (std >= min_std), std, 1.0).astype(np.float32)


def normalize_nwp_channel_mode(channel_mode):
    channel_mode = str(channel_mode).lower()
    if channel_mode not in NWP_CHANNELS_BY_MODE:
        supported = ", ".join(sorted(NWP_CHANNELS_BY_MODE))
        raise ValueError(f"Unsupported nwp channel mode {channel_mode!r}. Expected one of: {supported}.")
    return channel_mode


def get_nwp_channel_names(channel_mode="uv6"):
    channel_mode = normalize_nwp_channel_mode(channel_mode)
    return list(NWP_CHANNELS_BY_MODE[channel_mode])


def get_nwp_uv_channel_indices(channel_mode="uv6", height="140"):
    channel_names = get_nwp_channel_names(channel_mode)
    height = str(height)
    return (
        channel_names.index(f"U_{height}m"),
        channel_names.index(f"V_{height}m"),
    )


def fit_bundle_normalization(bundle: AlignedDataBundle, end_time=None):
    if end_time is None:
        end_time = bundle.z_meas.shape[0]
    end_time = int(max(1, min(end_time, bundle.z_meas.shape[0], bundle.nwp_uv.shape[0])))

    z_ref = bundle.z_meas[:end_time]
    nwp_ref = bundle.nwp_uv[:end_time]
    return NormalizationStats(
        z_mean=np.nanmean(z_ref, axis=0).astype(np.float32),
        z_std=_safe_std(np.nanstd(z_ref, axis=0)),
        nwp_mean=np.nanmean(nwp_ref, axis=(0, 2, 3)).astype(np.float32),
        nwp_std=_safe_std(np.nanstd(nwp_ref, axis=(0, 2, 3))),
    )


def normalization_stats_from_config(cfg):
    payload = cfg.get("normalization")
    if not payload:
        return None
    return NormalizationStats(
        z_mean=np.asarray(payload["z_mean"], dtype=np.float32),
        z_std=_safe_std(payload["z_std"]),
        nwp_mean=np.asarray(payload["nwp_mean"], dtype=np.float32),
        nwp_std=_safe_std(payload["nwp_std"]),
    )


def _as_like(array, like):
    if torch.is_tensor(like):
        return torch.as_tensor(array, dtype=like.dtype, device=like.device)
    return np.asarray(array, dtype=like.dtype if hasattr(like, "dtype") else np.float32)


def normalize_z(z, stats: NormalizationStats):
    mean = _as_like(stats.z_mean, z)
    std = _as_like(stats.z_std, z)
    return (z - mean) / std


def denormalize_z(z, stats: NormalizationStats):
    mean = _as_like(stats.z_mean, z)
    std = _as_like(stats.z_std, z)
    return z * std + mean


def normalize_nwp(nwp, stats: NormalizationStats):
    shape = [1] * nwp.ndim
    shape[-3] = stats.nwp_mean.shape[0]
    mean = _as_like(stats.nwp_mean.reshape(shape), nwp)
    std = _as_like(stats.nwp_std.reshape(shape), nwp)
    return (nwp - mean) / std


def denormalize_nwp(nwp, stats: NormalizationStats):
    shape = [1] * nwp.ndim
    shape[-3] = stats.nwp_mean.shape[0]
    mean = _as_like(stats.nwp_mean.reshape(shape), nwp)
    std = _as_like(stats.nwp_std.reshape(shape), nwp)
    return nwp * std + mean


def denormalize_nwp_uv_pairs(uv_pairs, stats: NormalizationStats, channel_indices):
    channel_indices = list(channel_indices)
    mean = _as_like(stats.nwp_mean[channel_indices].reshape(*([1] * (uv_pairs.ndim - 1)), len(channel_indices)), uv_pairs)
    std = _as_like(stats.nwp_std[channel_indices].reshape(*([1] * (uv_pairs.ndim - 1)), len(channel_indices)), uv_pairs)
    return uv_pairs * std + mean


def apply_bundle_normalization(bundle: AlignedDataBundle, stats: NormalizationStats, normalize_z_values=True, normalize_nwp_values=True):
    return AlignedDataBundle(
        z_meas=(normalize_z(bundle.z_meas, stats) if normalize_z_values else bundle.z_meas).astype(np.float32),
        meas_lat=bundle.meas_lat.copy(),
        meas_lon=bundle.meas_lon.copy(),
        nwp_uv=(normalize_nwp(bundle.nwp_uv, stats) if normalize_nwp_values else bundle.nwp_uv).astype(np.float32),
        nwp_lat=bundle.nwp_lat.copy(),
        nwp_lon=bundle.nwp_lon.copy(),
    )


def load_measurement_140m(meas_file):
    data = load_mat_auto(meas_file)
    ws_uv = np.asarray(data["Ws_uv"]).astype(np.float32)

    lat = np.asarray(data["LatValue_vec"]).reshape(-1).astype(np.float32)
    lon = np.asarray(data["LonValue_vec"]).reshape(-1).astype(np.float32)

    print("[load_measurement_140m] raw Ws_uv shape:", ws_uv.shape)

    # 目标必须变成 (T, 18)
    if ws_uv.ndim != 2:
        raise ValueError(f"Ws_uv should be 2D, got {ws_uv.shape}")

    if ws_uv.shape[1] == 18:
        pass
    elif ws_uv.shape[0] == 18:
        ws_uv = ws_uv.T
        print("[load_measurement_140m] transposed Ws_uv to:", ws_uv.shape)
    else:
        raise ValueError(f"Unexpected Ws_uv shape: {ws_uv.shape}")

    T = ws_uv.shape[0]
    z = np.zeros((T, 3, 2), dtype=np.float32)
    stations = ["E05", "E06", "ASOW6"]

    for i, st in enumerate(stations):
        u_col, v_col = MEAS_COLS[st]["140"]
        z[:, i, 0] = ws_uv[:, u_col]
        z[:, i, 1] = ws_uv[:, v_col]

    print("[load_measurement_140m] final z shape:", z.shape)
    return z, lat, lon


def load_nwp_uv_3heights(nwp_file, channel_mode="uv6"):
    channel_mode = normalize_nwp_channel_mode(channel_mode)
    data = load_mat_auto(nwp_file)
    grid = normalize_nwp_grid_shape(np.asarray(data["allVariMin_Grid"]))
    lat = np.asarray(data["LatValue"]).astype(np.float32)
    lon = np.asarray(data["LonValue"]).astype(np.float32)

    channels = [grid[:, :, :, NWP_VAR_MAP[name]] for name in get_nwp_channel_names(channel_mode)]
    channels = [c.transpose(2, 0, 1).astype(np.float32) for c in channels]
    x = np.stack(channels, axis=1)  # [T, C, Y, X]
    return x, lat, lon


def build_aligned_bundle(meas_file, nwp_file, nwp_channel_mode="uv6"):
    z_meas, meas_lat, meas_lon = load_measurement_140m(meas_file)
    nwp_uv, nwp_lat, nwp_lon = load_nwp_uv_3heights(nwp_file, channel_mode=nwp_channel_mode)

    T = min(z_meas.shape[0], nwp_uv.shape[0])
    return AlignedDataBundle(
        z_meas=z_meas[:T],
        meas_lat=meas_lat,
        meas_lon=meas_lon,
        nwp_uv=nwp_uv[:T],
        nwp_lat=nwp_lat,
        nwp_lon=nwp_lon,
    )


class IDEBaselineDataset(Dataset):
    """
    Stage 1: IDE only.
    Only returns measurement sequence chunks.
    """
    def __init__(self, bundle: AlignedDataBundle, chunk_len=16):
        self.bundle = bundle
        self.chunk_len = chunk_len
        self.valid = []

        T = bundle.z_meas.shape[0]
        for start in range(0, T - chunk_len):
            z = bundle.z_meas[start:start + chunk_len + 1]
            if np.isnan(z).any():
                continue
            self.valid.append(start)

        print(f"[IDEBaselineDataset] valid={len(self.valid)}")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, i):
        start = self.valid[i]
        z = self.bundle.z_meas[start:start + self.chunk_len + 1]

        return {
            "z_seq_full": torch.from_numpy(z).float(),        # [chunk_len+1, 3, 2]
            "site_lat": torch.from_numpy(self.bundle.meas_lat).float(),
            "site_lon": torch.from_numpy(self.bundle.meas_lon).float(),
            "time_idx_start": torch.tensor(start, dtype=torch.long),
        }


class MLMuDataset(Dataset):
    """
    Stage 2: train ML with frozen IDE.
    Returns:
      - full NWP context for rolling mu_t
      - aligned measurement sequence for IDE likelihood
    """
    def __init__(self, bundle: AlignedDataBundle, seq_len=4, chunk_len=16):
        self.bundle = bundle
        self.seq_len = seq_len
        self.chunk_len = chunk_len
        self.valid = []

        T = bundle.z_meas.shape[0]
        # Need seq_len context + chunk_len transitions
        min_len = seq_len + chunk_len

        for start in range(0, T - min_len + 1):
            end = start + min_len
            z = bundle.z_meas[start:end]         # [seq_len+chunk_len, 3, 2]
            x = bundle.nwp_uv[start:end - 1]     # [seq_len+chunk_len-1, 6, Y, X]

            if np.isnan(z).any():
                continue
            if np.isnan(x).any():
                continue
            self.valid.append(start)

        print(f"[MLMuDataset] valid={len(self.valid)}")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, i):
        start = self.valid[i]
        end = start + self.seq_len + self.chunk_len

        z = self.bundle.z_meas[start:end]        # [seq_len+chunk_len, 3, 2]
        x = self.bundle.nwp_uv[start:end - 1]    # [seq_len+chunk_len-1, 6, Y, X]

        return {
            "z_seq_full": torch.from_numpy(z).float(),
            "nwp_seq_full": torch.from_numpy(x).float(),
            "site_lat": torch.from_numpy(self.bundle.meas_lat).float(),
            "site_lon": torch.from_numpy(self.bundle.meas_lon).float(),
            "time_idx_start": torch.tensor(start, dtype=torch.long),
        }
