import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

VAR_NAMES = [
    "WS_100m", "WS_140m", "WS_180m",
    "T2", "RH2", "SLP",
    "U_100m", "V_100m",
    "U_140m", "V_140m",
    "U_180m", "V_180m"
]

def load_mat_auto(mat_path):
    mat_path = Path(mat_path).expanduser().resolve()

    if not mat_path.exists():
        raise FileNotFoundError(f"文件不存在: {mat_path}")

    scipy_err = None
    h5py_err = None

    try:
        from scipy.io import loadmat
        data = loadmat(str(mat_path))
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        print(f"[OK] 用 scipy.io.loadmat 读取成功: {mat_path}")
        return data, "scipy"
    except Exception as e:
        scipy_err = str(e)
        print(f"[INFO] scipy 读取失败，尝试 h5py: {scipy_err}")

    try:
        import h5py

        out = {}
        with h5py.File(str(mat_path), "r") as f:
            print("\n=== HDF5 顶层 keys ===")
            for k in f.keys():
                print(" -", k)

            for k in f.keys():
                out[k] = np.array(f[k])

        print(f"[OK] 用 h5py 读取成功: {mat_path}")
        return out, "h5py"

    except Exception as e:
        h5py_err = str(e)

    raise RuntimeError(
        f"无法读取 MAT 文件。\n"
        f"scipy 错误: {scipy_err}\n"
        f"h5py 错误: {h5py_err}"
    )


def normalize_grid_shape(grid):
    """
    统一成 (40, 40, time, 12)
    已知 h5py 读出来常见为 (12, time, 40, 40)
    """
    shape = grid.shape

    if len(shape) != 4:
        raise ValueError(f"allVariMin_Grid 不是4维，实际 shape={shape}")

    # 已经是目标顺序
    if shape[0] == 40 and shape[1] == 40 and shape[3] == 12:
        return grid

    # h5py 常见顺序: (12, time, 40, 40) -> (40, 40, time, 12)
    if shape[0] == 12 and shape[2] == 40 and shape[3] == 40:
        return np.transpose(grid, (2, 3, 1, 0))

    raise ValueError(f"无法识别 allVariMin_Grid 维度顺序: {shape}")


def print_time_info(n_time, start_time_str):
    start = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end = start + timedelta(minutes=10 * (n_time - 1))
    print("\n=== 时间信息 ===")
    print("起始时间:", start)
    print("结束时间:", end)
    print("时间步数:", n_time)
    print("时间分辨率: 10 min")


def summarize_grid_mat(mat_path, start_time_str):
    data, engine = load_mat_auto(mat_path)

    print("\n=== 文件中的变量 ===")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: type={type(v)}")

    grid = data.get("allVariMin_Grid")
    lat = data.get("LatValue")
    lon = data.get("LonValue")

    if grid is None:
        raise KeyError("没找到 allVariMin_Grid")

    print("\n=== 原始 shape ===")
    print("allVariMin_Grid raw shape:", grid.shape)

    grid = normalize_grid_shape(grid)

    print("\n=== 标准化后 shape ===")
    print("allVariMin_Grid normalized shape:", grid.shape)
    print("维度含义: (y, x, time, var)")

    ny, nx, nt, nv = grid.shape

    if lat is not None:
        print("\n=== 经纬度网格 ===")
        print("LatValue shape:", lat.shape, "min/max:", np.nanmin(lat), np.nanmax(lat))
        print("LonValue shape:", lon.shape, "min/max:", np.nanmin(lon), np.nanmax(lon))

        print("左上角(lat, lon):", lat[0, 0], lon[0, 0])
        print("右下角(lat, lon):", lat[-1, -1], lon[-1, -1])

    print_time_info(nt, start_time_str)

    print("\n=== 变量统计 ===")
    for vidx, name in enumerate(VAR_NAMES):
        arr = grid[:, :, :, vidx]
        nan_count = np.isnan(arr).sum()
        total = arr.size
        print(
            f"{vidx:2d} {name:8s} | "
            f"shape={arr.shape} | "
            f"min={np.nanmin(arr):.4f} "
            f"max={np.nanmax(arr):.4f} "
            f"mean={np.nanmean(arr):.4f} "
            f"std={np.nanstd(arr):.4f} "
            f"nan={nan_count}/{total}"
        )

    print("\n=== 示例访问 ===")
    t0 = 0
    y0, x0 = 20, 20

    print(f"网格点 ({y0}, {x0}) 的经纬度:")
    print("lat/lon =", lat[y0, x0], lon[y0, x0])

    print(f"\n第 {t0} 个时刻该点的12个变量:")
    for vidx, name in enumerate(VAR_NAMES):
        print(f"{name:8s}: {grid[y0, x0, t0, vidx]}")

    print("\n=== 示例：取某个变量整个时间序列 ===")
    ws100_ts = grid[y0, x0, :, 0]
    print("WS_100m time series shape:", ws100_ts.shape)
    print("前10个值:", ws100_ts[:10])

    return {
        "grid": grid,
        "lat": lat,
        "lon": lon,
        "engine": engine,
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    mat_file = BASE_DIR.parent / "data" / "nwp" / "data_grid_online.mat"

    result = summarize_grid_mat(
        mat_file,
        start_time_str="2021-06-04 00:00:00",
    )