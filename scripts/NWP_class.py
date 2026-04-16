import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta

VAR_NAMES = [
    "WS_100m", "WS_140m", "WS_180m",
    "T2", "RH2", "SLP",
    "U_100m", "V_100m",
    "U_140m", "V_140m",
    "U_180m", "V_180m"
]

VAR_MAP = {name: i for i, name in enumerate(VAR_NAMES)}

class NWPGridReader:
    def __init__(self, mat_path, start_time_str, freq_minutes=10):
        self.mat_path = Path(mat_path).expanduser().resolve()
        self.start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        self.freq_minutes = freq_minutes

        with h5py.File(str(self.mat_path), "r") as f:
            self.lat = np.array(f["LatValue"])
            self.lon = np.array(f["LonValue"])
            raw = np.array(f["allVariMin_Grid"])

        # (12, time, 40, 40) -> (40, 40, time, 12)
        self.grid = np.transpose(raw, (2, 3, 1, 0))

        self.ny, self.nx, self.nt, self.nv = self.grid.shape
        self.time_index = [
            self.start_time + timedelta(minutes=i * self.freq_minutes)
            for i in range(self.nt)
        ]

    def get_field(self, var_name, t):
        vidx = VAR_MAP[var_name]
        return self.grid[:, :, t, vidx]

    def get_timeseries(self, y, x, var_name):
        vidx = VAR_MAP[var_name]
        return self.grid[y, x, :, vidx]

    def get_value(self, y, x, t, var_name):
        vidx = VAR_MAP[var_name]
        return self.grid[y, x, t, vidx]

    def get_latlon(self, y, x):
        return self.lat[y, x], self.lon[y, x]

    def get_time(self, t):
        return self.time_index[t]

    def summary(self):
        print("file:", self.mat_path)
        print("grid shape:", self.grid.shape)
        print("lat shape:", self.lat.shape)
        print("lon shape:", self.lon.shape)
        print("time zone:", self.time_index[0], "->", self.time_index[-1])
        print("variables:", VAR_NAMES)