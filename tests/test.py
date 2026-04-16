from pathlib import Path
import numpy as np
import h5py


MEAS_FILE = Path("~/Projects/wind_ide/data/measurement/wv_h100_180_offline_imputed.mat").expanduser().resolve()

STATIONS = ["E05", "E06", "ASOW6"]
HEIGHTS = ["100", "140", "180"]

# 你目前采用的列映射
MEAS_COLS = {
    "E05":   {"100": (0, 1),  "140": (6, 7),   "180": (12, 13)},
    "E06":   {"100": (2, 3),  "140": (8, 9),   "180": (14, 15)},
    "ASOW6": {"100": (4, 5),  "140": (10, 11), "180": (16, 17)},
}


def load_mat_auto(mat_path: Path):
    if not mat_path.exists():
        raise FileNotFoundError(f"文件不存在: {mat_path}")

    # 先尝试 scipy
    try:
        from scipy.io import loadmat
        data = loadmat(str(mat_path))
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        print(f"[OK] scipy 读取成功: {mat_path}")
        return data
    except Exception as e:
        print(f"[INFO] scipy 读取失败，尝试 h5py: {e}")

    # 再尝试 h5py
    out = {}
    with h5py.File(str(mat_path), "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    print(f"[OK] h5py 读取成功: {mat_path}")
    return out


def summarize_array(name, arr):
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if np.issubdtype(arr.dtype, np.number):
        print(f"  nan count = {np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 'N/A'}")
        try:
            print(f"  min={np.nanmin(arr):.6f}, max={np.nanmax(arr):.6f}, mean={np.nanmean(arr):.6f}")
        except Exception:
            pass


def inspect_ws_uv(ws_uv):
    print("\n" + "=" * 80)
    print("检查 Ws_uv")
    print("=" * 80)
    summarize_array("Ws_uv", ws_uv)

    if ws_uv.ndim != 2:
        raise ValueError(f"Ws_uv 应该是二维矩阵 (T, 18)，但实际是 {ws_uv.shape}")

    T, C = ws_uv.shape
    print(f"T={T}, C={C}")

    if C != 18:
        print("[WARN] Ws_uv 列数不是 18，请重新确认列映射。")

    print("\n前 5 行前 10 列：")
    print(ws_uv[:5, :10])

    # 逐站点逐高度逐分量检查
    print("\n" + "=" * 80)
    print("按站点/高度/UV 拆分检查")
    print("=" * 80)

    for st in STATIONS:
        for h in HEIGHTS:
            u_col, v_col = MEAS_COLS[st][h]
            u = ws_uv[:, u_col]
            v = ws_uv[:, v_col]
            ws = np.sqrt(u ** 2 + v ** 2)

            print(f"\n[{st} - {h}m]")
            print(f"U col={u_col}, V col={v_col}")
            print(f"  U: nan={np.isnan(u).sum()}, min={np.nanmin(u):.6f}, max={np.nanmax(u):.6f}, mean={np.nanmean(u):.6f}")
            print(f"  V: nan={np.isnan(v).sum()}, min={np.nanmin(v):.6f}, max={np.nanmax(v):.6f}, mean={np.nanmean(v):.6f}")
            print(f" WS: nan={np.isnan(ws).sum()}, min={np.nanmin(ws):.6f}, max={np.nanmax(ws):.6f}, mean={np.nanmean(ws):.6f}")
            print("  first 5 U:", u[:5])
            print("  first 5 V:", v[:5])
            print("  first 5 WS:", ws[:5])


def build_measurement_140m(ws_uv):
    """
    按当前列映射，构造 z_meas: (T, 3, 2)
    """
    T = ws_uv.shape[0]
    z = np.zeros((T, 3, 2), dtype=np.float32)

    for i, st in enumerate(STATIONS):
        u_col, v_col = MEAS_COLS[st]["140"]
        z[:, i, 0] = ws_uv[:, u_col]
        z[:, i, 1] = ws_uv[:, v_col]

    return z


def inspect_140m_tensor(z):
    print("\n" + "=" * 80)
    print("检查 140m 构造后的 z_meas")
    print("=" * 80)
    summarize_array("z_meas_140m", z)

    if z.ndim != 3:
        raise ValueError(f"z 应该是三维 (T, 3, 2)，但实际是 {z.shape}")

    T, S, C = z.shape
    print(f"T={T}, S={S}, C={C}")

    if S != 3 or C != 2:
        print("[WARN] z 的后两维不是 (3,2)，请检查构造逻辑。")

    print("\n前 3 个时间点：")
    print(z[:3])

    for i, st in enumerate(STATIONS):
        u = z[:, i, 0]
        v = z[:, i, 1]
        ws = np.sqrt(u ** 2 + v ** 2)
        print(f"\n[{st} - 140m in z_meas]")
        print(f"  U nan={np.isnan(u).sum()}, mean={np.nanmean(u):.6f}")
        print(f"  V nan={np.isnan(v).sum()}, mean={np.nanmean(v):.6f}")
        print(f" WS nan={np.isnan(ws).sum()}, mean={np.nanmean(ws):.6f}")


def main():
    print("=" * 80)
    print("检查 measurement 文件")
    print("=" * 80)
    print("file:", MEAS_FILE)

    data = load_mat_auto(MEAS_FILE)

    print("\n" + "=" * 80)
    print("文件中的变量")
    print("=" * 80)
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            summarize_array(k, v)
        else:
            print(f"{k}: type={type(v)}")

    # 坐标检查
    if "LatValue_vec" in data:
        lat = np.asarray(data["LatValue_vec"]).reshape(-1)
        print("\nLatValue_vec reshaped:", lat.shape)
        print("values:", lat)

    if "LonValue_vec" in data:
        lon = np.asarray(data["LonValue_vec"]).reshape(-1)
        print("\nLonValue_vec reshaped:", lon.shape)
        print("values:", lon)

    # 主数据检查
    if "Ws_uv" not in data:
        raise KeyError("文件里没有 Ws_uv")

    ws_uv = np.asarray(data["Ws_uv"]).astype(np.float32)
    inspect_ws_uv(ws_uv)

    print("\n原始 Ws_uv shape:", ws_uv.shape)

    if ws_uv.shape[1] == 18:
        pass
    elif ws_uv.shape[0] == 18:
        ws_uv = ws_uv.T
        print("转置后 Ws_uv shape:", ws_uv.shape)
    else:
        raise ValueError(f"无法识别 Ws_uv shape: {ws_uv.shape}")

    # 140m 构造检查
    z_140 = build_measurement_140m(ws_uv)
    inspect_140m_tensor(z_140)

    print("\n" + "=" * 80)
    print("结论提示")
    print("=" * 80)
    print("1. 重点确认 Ws_uv 的 shape 是否是 (T, 18)")
    print("2. 重点确认构造后的 z_meas_140m 是否是 (T, 3, 2)")
    print("3. 如果 T 不是几万而是 18，说明你之前把列维度误当成时间维度了")
    print("4. 如果这里一切正常，再去接训练脚本")


if __name__ == "__main__":
    main()

    