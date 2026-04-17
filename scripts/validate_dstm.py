import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aligned_measurement_nwp import build_aligned_bundle, MLMuDataset
from src.models.advection_mean_net import AdvectionMeanNet
from src.models.ide_state_space import IDEStateSpaceModel
from src.trainers.train_ml_mu import build_dynamics_sequence
from src.utils.misc import set_seed


STATIONS = ["E05", "E06", "ASOW6"]
VARS = ["U", "V", "WS"]


def auto_device(name):
    if name is not None:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to ide_best.pt or joint_best.pt")
    p.add_argument("--measurement-file", type=str, default=None)
    p.add_argument("--nwp-file", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--dataset-label", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="outputs/validation")
    return p.parse_args()


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def to_numpy(x):
    return x.detach().cpu().numpy()


def circular_abs_diff_deg(a, b):
    diff = (b - a + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def _scalar_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "corr": np.nan, "n": 0}

    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 else np.nan
    return {
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "corr": corr,
        "n": int(yt.size),
    }


def _vector_metrics(y_true, y_pred):
    err = y_pred - y_true
    err_norm = np.sqrt((err ** 2).sum(axis=-1))
    return {
        "vector_mae": float(np.mean(np.abs(err_norm))),
        "vector_rmse": float(np.sqrt(np.mean(err_norm ** 2))),
    }


def _direction_metrics(y_true, y_pred):
    dir_true = (np.degrees(np.arctan2(y_true[..., 1], y_true[..., 0])) + 360.0) % 360.0
    dir_pred = (np.degrees(np.arctan2(y_pred[..., 1], y_pred[..., 0])) + 360.0) % 360.0
    diff = circular_abs_diff_deg(dir_true, dir_pred)
    return {
        "dir_mae_deg": float(np.mean(diff)),
        "dir_rmse_deg": float(np.sqrt(np.mean(diff ** 2))),
    }


def compute_metrics(y_true, y_pred):
    ws_true = np.sqrt((y_true ** 2).sum(axis=-1))
    ws_pred = np.sqrt((y_pred ** 2).sum(axis=-1))

    results = {}
    for idx, station in enumerate(STATIONS):
        station_true = y_true[:, idx]
        station_pred = y_pred[:, idx]
        results[station] = {
            "U": _scalar_metrics(station_true[:, 0], station_pred[:, 0]),
            "V": _scalar_metrics(station_true[:, 1], station_pred[:, 1]),
            "WS": _scalar_metrics(ws_true[:, idx], ws_pred[:, idx]),
            **_vector_metrics(station_true, station_pred),
            **_direction_metrics(station_true, station_pred),
        }

    results["overall"] = {
        "U": _scalar_metrics(y_true[..., 0].reshape(-1), y_pred[..., 0].reshape(-1)),
        "V": _scalar_metrics(y_true[..., 1].reshape(-1), y_pred[..., 1].reshape(-1)),
        "WS": _scalar_metrics(ws_true.reshape(-1), ws_pred.reshape(-1)),
        **_vector_metrics(y_true.reshape(-1, 2), y_pred.reshape(-1, 2)),
        **_direction_metrics(y_true.reshape(-1, 2), y_pred.reshape(-1, 2)),
    }
    return results


def find_nearest_grid(bundle):
    indices = []
    mapping = {}
    for idx, station in enumerate(STATIONS):
        lat0 = float(bundle.meas_lat[idx])
        lon0 = float(bundle.meas_lon[idx])
        dist2 = (bundle.nwp_lat - lat0) ** 2 + (bundle.nwp_lon - lon0) ** 2
        y, x = np.unravel_index(np.argmin(dist2), dist2.shape)
        indices.append((int(y), int(x)))
        mapping[station] = {
            "obs_lat": lat0,
            "obs_lon": lon0,
            "grid_lat": float(bundle.nwp_lat[y, x]),
            "grid_lon": float(bundle.nwp_lon[y, x]),
            "grid_y": int(y),
            "grid_x": int(x),
        }
    return indices, mapping


def extract_nwp_baseline(nwp_seq_full, seq_len, nearest_grid):
    # nwp_seq_full: [B, seq_len+chunk_len-1, 6, Y, X]
    current = nwp_seq_full[:, seq_len - 1:]  # [B, chunk_len, 6, Y, X]
    preds = []
    for station_idx, (gy, gx) in enumerate(nearest_grid):
        u = current[:, :, 2, gy, gx]
        v = current[:, :, 3, gy, gx]
        preds.append(torch.stack([u, v], dim=-1))
    return torch.stack(preds, dim=2)  # [B, chunk_len, 3, 2]


def reconstruct_models(ckpt, device):
    cfg = ckpt["config"]
    ide_model = IDEStateSpaceModel(
        dt=cfg.get("dt", 1.0),
        total_steps=cfg.get("ide_total_steps", 1),
        param_window=cfg.get("ide_param_window", 4),
        param_mode=cfg.get("ide_param_mode", "absolute"),
        init_log_q_proc=cfg.get("init_log_q_proc", -2.0),
        init_log_r_obs=cfg.get("init_log_r_obs", -2.0),
        init_log_p0=cfg.get("init_log_p0", 0.0),
        init_log_damping=cfg.get("init_log_damping", 0.0),
    ).to(device)

    mean_model = None
    if "ide_model_state" in ckpt:
        ide_model.load_state_dict(ckpt["ide_model_state"])
        mean_model = AdvectionMeanNet(
            hidden_dim=cfg.get("hidden_dim", 32),
            embed_dim=cfg.get("embed_dim", 32),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            ff_dim=cfg.get("ff_dim", 64),
            dropout=cfg.get("dropout", 0.1),
            mu_scale=cfg.get("mu_scale", 0.5),
            chol_offdiag_scale=cfg.get("chol_offdiag_scale", 0.15),
            chol_diag_max=cfg.get("chol_diag_max", 1.5),
            chol_eps=cfg.get("chol_eps", 1e-4),
        ).to(device)
        mean_model.load_state_dict(ckpt["mean_model_state"])
    else:
        ide_model.load_state_dict(ckpt["model_state"])

    ide_model.eval()
    if mean_model is not None:
        mean_model.eval()
    return cfg, ide_model, mean_model


@torch.no_grad()
def evaluate_dataset(loader, ide_model, mean_model, seq_len, device, nearest_grid, max_batches=None):
    preds_by_name = {
        "static_ide": [],
        "persistence": [],
        "nwp_current": [],
    }
    if mean_model is not None:
        preds_by_name["joint_dstm"] = []

    truths = []
    flat_preds_cache = {name: [] for name in preds_by_name.keys()}
    sample_plot = None
    evaluated_chunks = 0
    nll_stats = {
        "static_ide": [],
    }
    if mean_model is not None:
        nll_stats["joint_dstm"] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        z_full = batch["z_seq_full"]
        nwp_full = batch["nwp_seq_full"]
        evaluated_chunks += z_full.shape[0]

        chunk_len = nwp_full.shape[1] - seq_len + 1
        z_aligned = z_full[:, -(chunk_len + 1):]
        start_idx = batch["time_idx_start"] + seq_len - 1
        y_true = z_aligned[:, 1:]

        static_pred = ide_model.predict_sequence(
            z_seq=z_aligned,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            dynamics_seq=None,
            start_idx=start_idx,
        )
        nll_stats["static_ide"].append(
            float(
                ide_model.sequence_nll(
                    z_seq=z_aligned,
                    site_lon=batch["site_lon"],
                    site_lat=batch["site_lat"],
                    dynamics_seq=None,
                    start_idx=start_idx,
                ).cpu()
            )
        )

        static_np = to_numpy(static_pred.reshape(-1, 3, 2))
        pers_np = to_numpy(z_aligned[:, :-1].reshape(-1, 3, 2))
        nwp_np = to_numpy(extract_nwp_baseline(nwp_full, seq_len, nearest_grid).reshape(-1, 3, 2))
        truth_np = to_numpy(y_true.reshape(-1, 3, 2))

        preds_by_name["static_ide"].append(static_np)
        preds_by_name["persistence"].append(pers_np)
        preds_by_name["nwp_current"].append(nwp_np)
        flat_preds_cache["static_ide"].append(static_np)
        flat_preds_cache["persistence"].append(pers_np)
        flat_preds_cache["nwp_current"].append(nwp_np)
        truths.append(truth_np)

        if mean_model is not None:
            dynamics_seq = build_dynamics_sequence(mean_model, nwp_full, seq_len=seq_len)
            joint_pred = ide_model.predict_sequence(
                z_seq=z_aligned,
                site_lon=batch["site_lon"],
                site_lat=batch["site_lat"],
                dynamics_seq=dynamics_seq,
                start_idx=start_idx,
            )
            nll_stats["joint_dstm"].append(
                float(
                    ide_model.sequence_nll(
                        z_seq=z_aligned,
                        site_lon=batch["site_lon"],
                        site_lat=batch["site_lat"],
                        dynamics_seq=dynamics_seq,
                        start_idx=start_idx,
                    ).cpu()
                )
            )
            joint_np = to_numpy(joint_pred.reshape(-1, 3, 2))
            preds_by_name["joint_dstm"].append(joint_np)
            flat_preds_cache["joint_dstm"].append(joint_np)

        if sample_plot is None:
            sample_plot = {
                "truth": to_numpy(y_true[0]),
                "static_ide": to_numpy(static_pred[0]),
                "persistence": to_numpy(z_aligned[0, :-1]),
                "nwp_current": to_numpy(extract_nwp_baseline(nwp_full[:1], seq_len, nearest_grid)[0]),
            }
            if mean_model is not None:
                sample_plot["joint_dstm"] = to_numpy(joint_pred[0])

    truth = np.concatenate(truths, axis=0)
    results = {}
    for name, pred_list in preds_by_name.items():
        pred = np.concatenate(pred_list, axis=0)
        results[name] = {
            "metrics": compute_metrics(truth, pred),
        }
        if name in nll_stats and nll_stats[name]:
            results[name]["avg_sequence_nll"] = float(np.mean(nll_stats[name]))

    flat_preds_cache = {name: np.concatenate(pred_list, axis=0) for name, pred_list in flat_preds_cache.items()}
    return truth, results, sample_plot, flat_preds_cache, evaluated_chunks


def print_summary(results):
    print("\n" + "=" * 80)
    print("Overall Metrics")
    print("=" * 80)
    for name, payload in results.items():
        overall = payload["metrics"]["overall"]
        nll_str = ""
        if "avg_sequence_nll" in payload:
            nll_str = f" | NLL={payload['avg_sequence_nll']:.4f}"
        print(
            f"{name:>12} | "
            f"U RMSE={overall['U']['rmse']:.4f} | "
            f"V RMSE={overall['V']['rmse']:.4f} | "
            f"WS RMSE={overall['WS']['rmse']:.4f} | "
            f"Vec RMSE={overall['vector_rmse']:.4f} | "
            f"Dir MAE={overall['dir_mae_deg']:.2f} deg"
            f"{nll_str}"
        )


def plot_overall_bars(results, save_path):
    baseline_names = list(results.keys())
    metric_map = {
        "U RMSE": [results[name]["metrics"]["overall"]["U"]["rmse"] for name in baseline_names],
        "V RMSE": [results[name]["metrics"]["overall"]["V"]["rmse"] for name in baseline_names],
        "WS RMSE": [results[name]["metrics"]["overall"]["WS"]["rmse"] for name in baseline_names],
        "Dir MAE": [results[name]["metrics"]["overall"]["dir_mae_deg"] for name in baseline_names],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.reshape(-1)

    for ax, (title, values) in zip(axes, metric_map.items()):
        x = np.arange(len(baseline_names))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(baseline_names, rotation=20)
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")

    fig.suptitle("Overall Validation Metrics", fontsize=14)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_station_ws_rmse(results, save_path):
    baseline_names = list(results.keys())
    x = np.arange(len(STATIONS))
    width = 0.8 / len(baseline_names)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for idx, name in enumerate(baseline_names):
        vals = [results[name]["metrics"][station]["WS"]["rmse"] for station in STATIONS]
        ax.bar(x + (idx - (len(baseline_names) - 1) / 2) * width, vals, width=width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(STATIONS)
    ax.set_ylabel("WS RMSE")
    ax.set_title("Per-Station Wind-Speed RMSE")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_sample_timeseries(sample_plot, save_path, station_idx=0):
    station = STATIONS[station_idx]
    time_axis = np.arange(sample_plot["truth"].shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    series_specs = [
        ("U", lambda x: x[:, station_idx, 0]),
        ("V", lambda x: x[:, station_idx, 1]),
        ("WS", lambda x: np.sqrt((x[:, station_idx] ** 2).sum(axis=-1))),
    ]

    for ax, (label, fn) in zip(axes, series_specs):
        ax.plot(time_axis, fn(sample_plot["truth"]), label="truth", linewidth=2)
        for name, values in sample_plot.items():
            if name == "truth":
                continue
            ax.plot(time_axis, fn(values), label=name, alpha=0.85)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.set_title(f"{station} {label}")

    axes[-1].set_xlabel("Relative Forecast Step")
    axes[0].legend(ncol=4, fontsize=8)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_ws_scatter(truth, results, preds_cache, save_path):
    baseline_names = list(preds_cache.keys())
    ncols = min(2, len(baseline_names))
    nrows = int(np.ceil(len(baseline_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    ws_true = np.sqrt((truth ** 2).sum(axis=-1)).reshape(-1)
    for ax, name in zip(axes, baseline_names):
        ws_pred = np.sqrt((preds_cache[name] ** 2).sum(axis=-1)).reshape(-1)
        ax.scatter(ws_true, ws_pred, s=5, alpha=0.15)
        lim = max(float(np.nanmax(ws_true)), float(np.nanmax(ws_pred)))
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        rmse = results[name]["metrics"]["overall"]["WS"]["rmse"]
        corr = results[name]["metrics"]["overall"]["WS"]["corr"]
        ax.set_title(f"{name} | RMSE={rmse:.3f}, Corr={corr:.3f}")
        ax.set_xlabel("True WS")
        ax.set_ylabel("Predicted WS")
        ax.grid(alpha=0.2)

    for ax in axes[len(baseline_names):]:
        ax.axis("off")

    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_station_mapping(bundle, mapping, save_path):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.pcolormesh(bundle.nwp_lon, bundle.nwp_lat, np.zeros_like(bundle.nwp_lat), shading="auto", alpha=0.15)

    for station in STATIONS:
        info = mapping[station]
        ax.scatter(info["obs_lon"], info["obs_lat"], s=70, label=f"{station} obs")
        ax.scatter(info["grid_lon"], info["grid_lat"], s=70, marker="x", label=f"{station} nwp")
        ax.plot([info["obs_lon"], info["grid_lon"]], [info["obs_lat"], info["grid_lat"]], "--", linewidth=1)

    ax.set_title("Measurement / NWP Station Mapping")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8, ncol=2)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg, ide_model, mean_model = reconstruct_models(ckpt, auto_device(args.device))
    device = next(ide_model.parameters()).device

    set_seed(cfg.get("seed", 42))

    meas_file = args.measurement_file or cfg.get("measurement_file")
    nwp_file = args.nwp_file or cfg.get("nwp_file")
    if meas_file is None or nwp_file is None:
        raise ValueError("Need measurement_file and nwp_file either from args or checkpoint config")

    meas_file = str(Path(meas_file).expanduser().resolve())
    nwp_file = str(Path(nwp_file).expanduser().resolve())
    dataset_label = args.dataset_label or Path(meas_file).stem

    bundle = build_aligned_bundle(meas_file, nwp_file)
    seq_len = cfg.get("seq_len", 4)
    chunk_len = cfg.get("chunk_len", 16)
    dataset = MLMuDataset(bundle, seq_len=seq_len, chunk_len=chunk_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    nearest_grid, mapping = find_nearest_grid(bundle)

    truth, results, sample_plot, preds_cache, evaluated_chunks = evaluate_dataset(
        loader=loader,
        ide_model=ide_model,
        mean_model=mean_model,
        seq_len=seq_len,
        device=device,
        nearest_grid=nearest_grid,
        max_batches=args.max_batches,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = {
        "checkpoint": str(ckpt_path),
        "dataset_label": dataset_label,
        "measurement_file": meas_file,
        "nwp_file": nwp_file,
        "device": str(device),
        "num_chunks": evaluated_chunks,
        "results": results,
        "mapping": mapping,
    }

    with open(out_dir / f"{dataset_label}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    print(f"device: {device}")
    print(f"checkpoint: {ckpt_path}")
    print(f"dataset: {dataset_label}")
    print(f"num_chunks: {evaluated_chunks}")
    print_summary(results)

    plot_overall_bars(results, out_dir / f"{dataset_label}_overall_bars.png")
    plot_station_ws_rmse(results, out_dir / f"{dataset_label}_station_ws_rmse.png")
    plot_sample_timeseries(sample_plot, out_dir / f"{dataset_label}_sample_timeseries.png")
    plot_ws_scatter(truth, results, preds_cache, out_dir / f"{dataset_label}_ws_scatter.png")
    plot_station_mapping(bundle, mapping, out_dir / f"{dataset_label}_station_mapping.png")

    print(f"\nSaved validation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
