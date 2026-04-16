from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from .mat_loader import load_mat_auto, normalize_nwp_grid_shape

NWP_VAR_MAP = {
    "WS_100m": 0, "WS_140m": 1, "WS_180m": 2,
    "T2": 3, "RH2": 4, "SLP": 5,
    "U_100m": 6, "V_100m": 7, "U_140m": 8, "V_140m": 9, "U_180m": 10, "V_180m": 11,
}

@dataclass
class OfflineDataBundle:
    lat: np.ndarray
    lon: np.ndarray
    ws: np.ndarray
    u: np.ndarray
    v: np.ndarray
    t2: np.ndarray
    rh2: np.ndarray
    slp: np.ndarray


def load_offline_nwp_one_height(nwp_file, height=100):
    data = load_mat_auto(nwp_file)
    grid = normalize_nwp_grid_shape(np.asarray(data["allVariMin_Grid"]))
    lat = np.asarray(data["LatValue"]).astype(np.float32)
    lon = np.asarray(data["LonValue"]).astype(np.float32)
    ws = grid[:, :, :, NWP_VAR_MAP[f"WS_{height}m"]].transpose(2, 0, 1).astype(np.float32)
    u = grid[:, :, :, NWP_VAR_MAP[f"U_{height}m"]].transpose(2, 0, 1).astype(np.float32)
    v = grid[:, :, :, NWP_VAR_MAP[f"V_{height}m"]].transpose(2, 0, 1).astype(np.float32)
    t2 = grid[:, :, :, NWP_VAR_MAP["T2"]].transpose(2, 0, 1).astype(np.float32)
    rh2 = grid[:, :, :, NWP_VAR_MAP["RH2"]].transpose(2, 0, 1).astype(np.float32)
    slp = grid[:, :, :, NWP_VAR_MAP["SLP"]].transpose(2, 0, 1).astype(np.float32)
    return OfflineDataBundle(lat=lat, lon=lon, ws=ws, u=u, v=v, t2=t2, rh2=rh2, slp=slp)


class OfflineIDEPatchDataset(Dataset):
    def __init__(self, bundle, seq_len=4):
        self.bundle = bundle
        self.seq_len = seq_len
        self.T = bundle.ws.shape[0]
        if self.T < seq_len + 1:
            raise ValueError("Not enough time steps for requested seq_len")

    def __len__(self):
        return self.T - self.seq_len

    def __getitem__(self, idx):
        t = idx + self.seq_len - 1
        seq_slice = slice(t - self.seq_len + 1, t + 1)
        z_seq = np.stack(
            [self.bundle.t2[seq_slice], self.bundle.rh2[seq_slice], self.bundle.slp[seq_slice]],
            axis=1,
        ).astype(np.float32)
        return {
            "y_t": torch.from_numpy(self.bundle.ws[t]),
            "y_next": torch.from_numpy(self.bundle.ws[t + 1]),
            "u_t": torch.from_numpy(self.bundle.u[t]),
            "v_t": torch.from_numpy(self.bundle.v[t]),
            "z_seq": torch.from_numpy(z_seq),
            "lat": torch.from_numpy(self.bundle.lat),
            "lon": torch.from_numpy(self.bundle.lon),
        }