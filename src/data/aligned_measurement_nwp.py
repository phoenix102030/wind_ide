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


@dataclass
class AlignedDataBundle:
    z_meas: np.ndarray      # [T, 3, 2]
    meas_lat: np.ndarray    # [3]
    meas_lon: np.ndarray    # [3]
    nwp_uv: np.ndarray      # [T, 6, Y, X]
    nwp_lat: np.ndarray     # [Y, X]
    nwp_lon: np.ndarray     # [Y, X]


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


def load_nwp_uv_3heights(nwp_file):
    data = load_mat_auto(nwp_file)
    grid = normalize_nwp_grid_shape(np.asarray(data["allVariMin_Grid"]))
    lat = np.asarray(data["LatValue"]).astype(np.float32)
    lon = np.asarray(data["LonValue"]).astype(np.float32)

    channels = [
        grid[:, :, :, NWP_VAR_MAP["U_100m"]],
        grid[:, :, :, NWP_VAR_MAP["V_100m"]],
        grid[:, :, :, NWP_VAR_MAP["U_140m"]],
        grid[:, :, :, NWP_VAR_MAP["V_140m"]],
        grid[:, :, :, NWP_VAR_MAP["U_180m"]],
        grid[:, :, :, NWP_VAR_MAP["V_180m"]],
    ]
    channels = [c.transpose(2, 0, 1).astype(np.float32) for c in channels]
    x = np.stack(channels, axis=1)  # [T, 6, Y, X]
    return x, lat, lon


def build_aligned_bundle(meas_file, nwp_file):
    z_meas, meas_lat, meas_lon = load_measurement_140m(meas_file)
    nwp_uv, nwp_lat, nwp_lon = load_nwp_uv_3heights(nwp_file)

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
        }