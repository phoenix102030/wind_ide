from pathlib import Path
import numpy as np
import h5py


def load_mat_auto(mat_path):
    mat_path = Path(mat_path).expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    try:
        from scipy.io import loadmat
        data = loadmat(str(mat_path))
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception:
        pass
    out = {}
    with h5py.File(str(mat_path), "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    return out


def normalize_nwp_grid_shape(grid):
    if grid.ndim != 4:
        raise ValueError(f"Expected 4D NWP grid, got {grid.shape}")
    if grid.shape[0] == 40 and grid.shape[1] == 40 and grid.shape[3] == 12:
        return grid
    if grid.shape[0] == 12:
        return np.transpose(grid, (2, 3, 1, 0))
    raise ValueError(f"Unrecognized NWP grid shape: {grid.shape}")