import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


MEAS_COLS = {
    "E05":   {"100": (0, 1),  "140": (6, 7),  "180": (12, 13)},
    "E06":   {"100": (2, 3),  "140": (8, 9),  "180": (14, 15)},
    "ASOW6": {"100": (4, 5),  "140": (10, 11), "180": (16, 17)},
}

NWP_VAR_MAP = {
    "WS_100m": 0,
    "WS_140m": 1,
    "WS_180m": 2,
    "T2": 3,
    "RH2": 4,
    "SLP": 5,
    "U_100m": 6,
    "V_100m": 7,
    "U_140m": 8,
    "V_140m": 9,
    "U_180m": 10,
    "V_180m": 11,
}


def load_mat_auto(mat_path):
    mat_path = Path(mat_path).expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(f"文件不存在: {mat_path}")

    # 先尝试 scipy
    try:
        from scipy.io import loadmat
        data = loadmat(str(mat_path))
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        print(f"[OK] scipy 读取成功: {mat_path.name}")
        return data
    except Exception:
        pass

    # 再尝试 h5py
    out = {}
    with h5py.File(str(mat_path), "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    print(f"[OK] h5py 读取成功: {mat_path.name}")
    return out


def normalize_nwp_grid_shape(grid):
    # 目标: (40, 40, time, 12)
    if grid.ndim != 4:
        raise ValueError(f"NWP grid 应为4维, 实际 {grid.shape}")

    if grid.shape[0] == 40 and grid.shape[1] == 40 and grid.shape[3] == 12:
        return grid

    if grid.shape[0] == 12:
        # h5py 常见: (12, time, 40, 40)
        return np.transpose(grid, (2, 3, 1, 0))

    raise ValueError(f"无法识别 NWP 维度顺序: {grid.shape}")


def build_time_index(start_time_str, n_time, freq_minutes=10):
    start = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    return [start + timedelta(minutes=freq_minutes * i) for i in range(n_time)]


def compute_ws(u, v):
    return np.sqrt(u ** 2 + v ** 2)


def find_nearest_grid(lat_grid, lon_grid, target_lat, target_lon):
    dist2 = (lat_grid - target_lat) ** 2 + (lon_grid - target_lon) ** 2
    idx = np.argmin(dist2)
    y, x = np.unravel_index(idx, lat_grid.shape)
    return y, x, np.sqrt(dist2[y, x])


def summarize_pair(obs, nwp):
    mask = np.isfinite(obs) & np.isfinite(nwp)
    if mask.sum() == 0:
        return {
            "n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "corr": np.nan
        }

    e = nwp[mask] - obs[mask]
    corr = np.corrcoef(obs[mask], nwp[mask])[0, 1] if mask.sum() > 1 else np.nan

    return {
        "n": int(mask.sum()),
        "bias": float(np.mean(e)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "corr": float(corr),
    }


def load_measurement_dataset(mat_path, start_time_str):
    data = load_mat_auto(mat_path)

    ws_uv = data["Ws_uv"]
    lat_vec = np.array(data["LatValue_vec"]).reshape(-1)
    lon_vec = np.array(data["LonValue_vec"]).reshape(-1)

    stations = ["E05", "E06", "ASOW6"]
    coords = {st: (float(lat_vec[i]), float(lon_vec[i])) for i, st in enumerate(stations)}

    time_index = build_time_index(start_time_str, ws_uv.shape[0], 10)

    return {
        "Ws_uv": ws_uv,
        "coords": coords,
        "time_index": time_index,
    }


def load_nwp_dataset(mat_path, start_time_str):
    data = load_mat_auto(mat_path)

    grid = normalize_nwp_grid_shape(np.array(data["allVariMin_Grid"]))
    lat = np.array(data["LatValue"])
    lon = np.array(data["LonValue"])

    time_index = build_time_index(start_time_str, grid.shape[2], 10)

    return {
        "grid": grid,
        "lat": lat,
        "lon": lon,
        "time_index": time_index,
    }


def extract_measurement_series(meas, station, height):
    u_col, v_col = MEAS_COLS[station][height]
    u = meas["Ws_uv"][:, u_col]
    v = meas["Ws_uv"][:, v_col]
    ws = compute_ws(u, v)
    return u, v, ws


def extract_nwp_series_at_grid(nwp, y, x, height):
    u = nwp["grid"][y, x, :, NWP_VAR_MAP[f"U_{height}m"]]
    v = nwp["grid"][y, x, :, NWP_VAR_MAP[f"V_{height}m"]]
    ws = nwp["grid"][y, x, :, NWP_VAR_MAP[f"WS_{height}m"]]
    return u, v, ws


def compare_one_dataset(meas, nwp, label):
    stations = ["E05", "E06", "ASOW6"]
    heights = ["100", "140", "180"]

    # 时间长度取交集
    nt = min(len(meas["time_index"]), len(nwp["time_index"]))

    print(f"\n===== {label} =====")
    print(f"time steps used: {nt}")

    nearest_info = {}
    results = []

    for st in stations:
        lat0, lon0 = meas["coords"][st]
        y, x, d = find_nearest_grid(nwp["lat"], nwp["lon"], lat0, lon0)
        nearest_info[st] = {
            "obs_lat": lat0,
            "obs_lon": lon0,
            "grid_y": y,
            "grid_x": x,
            "grid_lat": float(nwp["lat"][y, x]),
            "grid_lon": float(nwp["lon"][y, x]),
            "dist_deg": float(d),
        }

        print(f"\n[{label}] {st}")
        print(f"  obs : ({lat0:.6f}, {lon0:.6f})")
        print(f"  nwp : ({nwp['lat'][y, x]:.6f}, {nwp['lon'][y, x]:.6f})")
        print(f"  dist: {d:.6f} degree")

        for h in heights:
            obs_u, obs_v, obs_ws = extract_measurement_series(meas, st, h)
            nwp_u, nwp_v, nwp_ws = extract_nwp_series_at_grid(nwp, y, x, h)

            obs_u = obs_u[:nt]
            obs_v = obs_v[:nt]
            obs_ws = obs_ws[:nt]
            nwp_u = nwp_u[:nt]
            nwp_v = nwp_v[:nt]
            nwp_ws = nwp_ws[:nt]

            stat_u = summarize_pair(obs_u, nwp_u)
            stat_v = summarize_pair(obs_v, nwp_v)
            stat_ws = summarize_pair(obs_ws, nwp_ws)

            results.append({
                "dataset": label,
                "station": st,
                "height": h,
                "variable": "U",
                **stat_u
            })
            results.append({
                "dataset": label,
                "station": st,
                "height": h,
                "variable": "V",
                **stat_v
            })
            results.append({
                "dataset": label,
                "station": st,
                "height": h,
                "variable": "WS",
                **stat_ws
            })

            print(
                f"  {h}m | "
                f"U rmse={stat_u['rmse']:.3f}, corr={stat_u['corr']:.3f} | "
                f"V rmse={stat_v['rmse']:.3f}, corr={stat_v['corr']:.3f} | "
                f"WS rmse={stat_ws['rmse']:.3f}, corr={stat_ws['corr']:.3f}"
            )

    return nearest_info, results


def plot_station_mapping(nwp, nearest_info, title, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(nwp["lon"], nwp["lat"], np.zeros_like(nwp["lat"]), shading="auto")

    for st, info in nearest_info.items():
        ax.scatter(info["obs_lon"], info["obs_lat"], marker="o", s=80, label=f"{st} obs")
        ax.scatter(info["grid_lon"], info["grid_lat"], marker="x", s=80, label=f"{st} nwp")
        ax.plot([info["obs_lon"], info["grid_lon"]], [info["obs_lat"], info["grid_lat"]], "--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    plt.show()


def plot_timeseries_comparison(meas, nwp, nearest_info, station, height, label, save_path=None):
    nt = min(len(meas["time_index"]), len(nwp["time_index"]))
    times = meas["time_index"][:nt]

    obs_u, obs_v, obs_ws = extract_measurement_series(meas, station, height)
    info = nearest_info[station]
    nwp_u, nwp_v, nwp_ws = extract_nwp_series_at_grid(nwp, info["grid_y"], info["grid_x"], height)

    obs_u, obs_v, obs_ws = obs_u[:nt], obs_v[:nt], obs_ws[:nt]
    nwp_u, nwp_v, nwp_ws = nwp_u[:nt], nwp_v[:nt], nwp_ws[:nt]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].plot(times, obs_u, label="measurement U")
    axes[0].plot(times, nwp_u, label="nwp U")
    axes[0].set_ylabel("U")
    axes[0].legend()

    axes[1].plot(times, obs_v, label="measurement V")
    axes[1].plot(times, nwp_v, label="nwp V")
    axes[1].set_ylabel("V")
    axes[1].legend()

    axes[2].plot(times, obs_ws, label="measurement WS")
    axes[2].plot(times, nwp_ws, label="nwp WS")
    axes[2].set_ylabel("WS")
    axes[2].legend()

    fig.suptitle(f"{label} | {station} | {height}m")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    plt.show()


def plot_scatter_comparison(meas, nwp, nearest_info, station, height, label, save_path=None):
    nt = min(len(meas["time_index"]), len(nwp["time_index"]))

    obs_u, obs_v, obs_ws = extract_measurement_series(meas, station, height)
    info = nearest_info[station]
    nwp_u, nwp_v, nwp_ws = extract_nwp_series_at_grid(nwp, info["grid_y"], info["grid_x"], height)

    pairs = [
        ("U", obs_u[:nt], nwp_u[:nt]),
        ("V", obs_v[:nt], nwp_v[:nt]),
        ("WS", obs_ws[:nt], nwp_ws[:nt]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for ax, (name, obs, mod) in zip(axes, pairs):
        mask = np.isfinite(obs) & np.isfinite(mod)
        ax.scatter(obs[mask], mod[mask], s=8, alpha=0.5)

        if mask.sum() > 0:
            mn = min(np.nanmin(obs[mask]), np.nanmin(mod[mask]))
            mx = max(np.nanmax(obs[mask]), np.nanmax(mod[mask]))
            ax.plot([mn, mx], [mn, mx], "--")

        ax.set_xlabel(f"measurement {name}")
        ax.set_ylabel(f"nwp {name}")
        ax.set_title(name)

    fig.suptitle(f"{label} scatter | {station} | {height}m")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"

    # 按你的实际目录改
    meas_offline_file = data_dir / "measurement" / "wv_h100_180_offline.mat"
    meas_online_file  = data_dir / "measurement" / "wv_h100_180_online.mat"
    nwp_offline_file  = data_dir / "nwp" / "data_grid_offline.mat"
    nwp_online_file   = data_dir / "nwp" / "data_grid_online.mat"

    out_dir = base_dir / "figures_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 时间起点按 README
    meas_offline = load_measurement_dataset(meas_offline_file, "2020-02-28 00:00:00")
    meas_online  = load_measurement_dataset(meas_online_file,  "2021-06-04 00:00:00")
    nwp_offline  = load_nwp_dataset(nwp_offline_file, "2020-02-28 00:00:00")
    nwp_online   = load_nwp_dataset(nwp_online_file,  "2021-06-04 00:00:00")

    # offline 对比
    offline_nearest, offline_results = compare_one_dataset(meas_offline, nwp_offline, "offline")
    plot_station_mapping(
        nwp_offline,
        offline_nearest,
        "Offline: measurement stations vs nearest NWP grid points",
        save_path=out_dir / "offline_station_mapping.png"
    )

    # online 对比
    online_nearest, online_results = compare_one_dataset(meas_online, nwp_online, "online")
    plot_station_mapping(
        nwp_online,
        online_nearest,
        "Online: measurement stations vs nearest NWP grid points",
        save_path=out_dir / "online_station_mapping.png"
    )

    # 例子：给每个数据集画一个站点一个层高
    plot_timeseries_comparison(
        meas_offline, nwp_offline, offline_nearest,
        station="E05", height="100", label="offline",
        save_path=out_dir / "offline_E05_100m_timeseries.png"
    )
    plot_scatter_comparison(
        meas_offline, nwp_offline, offline_nearest,
        station="E05", height="100", label="offline",
        save_path=out_dir / "offline_E05_100m_scatter.png"
    )

    plot_timeseries_comparison(
        meas_online, nwp_online, online_nearest,
        station="E05", height="100", label="online",
        save_path=out_dir / "online_E05_100m_timeseries.png"
    )
    plot_scatter_comparison(
        meas_online, nwp_online, online_nearest,
        station="E05", height="100", label="online",
        save_path=out_dir / "online_E05_100m_scatter.png"
    )