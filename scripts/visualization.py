import numpy as np
import h5py
import matplotlib.pyplot as plt
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

        # h5py 常见输出: (12, time, 40, 40)
        # 转成: (40, 40, time, 12)
        if raw.shape[0] == 12:
            self.grid = np.transpose(raw, (2, 3, 1, 0))
        else:
            self.grid = raw

        self.ny, self.nx, self.nt, self.nv = self.grid.shape
        self.time_index = [
            self.start_time + timedelta(minutes=i * self.freq_minutes)
            for i in range(self.nt)
        ]

    def get_field(self, var_name, t):
        return self.grid[:, :, t, VAR_MAP[var_name]]

    def get_wind_layer(self, height, t):
        ws = self.get_field(f"WS_{height}m", t)
        u = self.get_field(f"U_{height}m", t)
        v = self.get_field(f"V_{height}m", t)
        return ws, u, v

    def get_time_str(self, t):
        return self.time_index[t].strftime("%Y-%m-%d %H:%M:%S")


def plot_all_fields(reader, t, stride=3, save_path=None):
    """
    一张图画所有主要场:
    第一行: WS+UV at 100m / 140m / 180m
    第二行: T2 / RH2 / SLP
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    heights = [100, 140, 180]

    # 为三个 WS 统一色标，便于对比
    ws_all = []
    for h in heights:
        ws, _, _ = reader.get_wind_layer(h, t)
        ws_all.append(ws)
    ws_min = min(np.nanmin(x) for x in ws_all)
    ws_max = max(np.nanmax(x) for x in ws_all)

    # 第一行：风场
    for ax, h in zip(axes[0], heights):
        ws, u, v = reader.get_wind_layer(h, t)

        im = ax.pcolormesh(
            reader.lon, reader.lat, ws,
            shading="auto",
            vmin=ws_min, vmax=ws_max
        )

        lon_s = reader.lon[::stride, ::stride]
        lat_s = reader.lat[::stride, ::stride]
        u_s = u[::stride, ::stride]
        v_s = v[::stride, ::stride]

        # 红色画 U：水平箭头
        ax.quiver(
            lon_s, lat_s,
            u_s, np.zeros_like(u_s),
            color="red",
            angles="xy",
            scale_units="xy",
            scale=250,
            width=0.004,
            headwidth=4,
            headlength=3,
            headaxislength=2.5
        )

    # 蓝色画 V：竖直箭头
        ax.quiver(
            lon_s, lat_s,
            np.zeros_like(v_s), v_s,
            color="blue",
            angles="xy",
            scale_units="xy",
            scale=250,
            width=0.004,
            headwidth=4,
            headlength=3,
            headaxislength=2.5
        )

        ax.set_title(f"Wind field at {h}m\n(WS + U(red) + V(blue))")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # 第一行统一色条
    cbar1 = fig.colorbar(im, ax=axes[0, :], shrink=0.9)
    cbar1.set_label("Wind Speed")

    # 第二行：T2 / RH2 / SLP
    scalar_vars = ["T2", "RH2", "SLP"]
    scalar_titles = {
        "T2": "T2 (2m Air Temperature)",
        "RH2": "RH2 (2m Relative Humidity)",
        "SLP": "SLP (Sea Level Pressure)"
    }

    for ax, var in zip(axes[1], scalar_vars):
        field = reader.get_field(var, t)
        im2 = ax.pcolormesh(
            reader.lon, reader.lat, field,
            shading="auto"
        )
        ax.set_title(scalar_titles[var])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        cbar = fig.colorbar(im2, ax=ax, shrink=0.9)
        cbar.set_label(var)

    fig.suptitle(f"All fields at t = {reader.get_time_str(t)}", fontsize=16)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[OK] saved: {save_path}")

    plt.show()


def check_ws_from_uv(reader, t, height):
    """
    简单检查 WS 是否约等于 sqrt(U^2 + V^2)
    """
    ws, u, v = reader.get_wind_layer(height, t)
    ws_calc = np.sqrt(u**2 + v**2)
    diff = ws - ws_calc

    print(f"\n[Check] {height}m")
    print("max abs diff =", np.nanmax(np.abs(diff)))
    print("mean abs diff =", np.nanmean(np.abs(diff)))


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    mat_file = base_dir.parent / "data" / "nwp" / "data_grid_online.mat"

    reader = NWPGridReader(
        mat_file,
        start_time_str="2021-06-04 00:00:00"
    )

    # 画第 0 个时间步
    t = 0

    plot_all_fields(
        reader,
        t=t,
        stride=3,
        save_path=base_dir / "figures" / f"all_fields_t{t:05d}.png"
    )

    # 检查 WS 和 sqrt(U^2 + V^2) 是否一致
    for h in [100, 140, 180]:
        check_ws_from_uv(reader, t, h)