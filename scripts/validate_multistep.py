import argparse
import math
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aligned_measurement_nwp import (
    apply_bundle_normalization,
    build_aligned_bundle,
    denormalize_z,
    normalization_stats_from_config,
)
from src.models.advection_mean_net import AdvectionMeanNet
from src.models.ide_state_space import IDEStateSpaceModel
from src.trainers.train_ml_mu import build_dynamics_sequence, zero_dynamics_sequence
from src.utils.misc import set_seed


STATIONS = ["E05", "E06", "ASOW6"]


def auto_device(name):
    if name is not None:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--measurement-file", type=str, required=True)
    parser.add_argument("--nwp-file", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--history-len", type=int, default=24, help="Number of observed transitions used before forecasting")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast steps, defaults to config forecast_horizon")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/validation_multistep")
    return parser.parse_args()


def reconstruct_models(ckpt, device):
    cfg = ckpt["config"]
    ide_model = IDEStateSpaceModel(
        dt=cfg.get("dt", 1.0),
        total_steps=cfg.get("ide_total_steps", 1),
        param_window=cfg.get("ide_param_window", 4),
        param_mode=cfg.get("ide_param_mode", "absolute"),
        init_log_ell_par=cfg.get("init_log_ell_par", 0.5),
        init_log_ell_perp=cfg.get("init_log_ell_perp", 0.0),
        init_log_q_proc=cfg.get("init_log_q_proc", -2.0),
        init_log_r_obs=cfg.get("init_log_r_obs", -2.0),
        init_log_p0=cfg.get("init_log_p0", 0.0),
        init_log_damping=cfg.get("init_log_damping", 0.0),
        q_proc_min=cfg.get("q_proc_min", math.exp(-4.0)),
        q_proc_max=cfg.get("q_proc_max", 0.5),
        r_obs_min=cfg.get("r_obs_min", math.exp(-4.0)),
        r_obs_max=cfg.get("r_obs_max", 0.75),
        damping_min=cfg.get("damping_min", math.exp(-4.0)),
        damping_max=cfg.get("damping_max", 1.0),
    ).to(device)
    ide_model.load_state_dict(ckpt["ide_model_state"] if "ide_model_state" in ckpt else ckpt["model_state"])
    ide_model.eval()

    mean_model = None
    if "mean_model_state" in ckpt:
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
        mean_model.eval()

    return cfg, ide_model, mean_model


def scalar_metrics(y_true, y_pred):
    err = y_pred - y_true
    corr = np.nan
    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "corr": corr,
    }


def ws(y):
    return np.sqrt((y ** 2).sum(axis=-1))


def compute_metrics(y_true, y_pred):
    results = {}
    for i, station in enumerate(STATIONS):
        results[station] = {
            "U": scalar_metrics(y_true[:, i, 0], y_pred[:, i, 0]),
            "V": scalar_metrics(y_true[:, i, 1], y_pred[:, i, 1]),
            "WS": scalar_metrics(ws(y_true[:, i]), ws(y_pred[:, i])),
        }
    results["overall"] = {
        "U": scalar_metrics(y_true[..., 0].reshape(-1), y_pred[..., 0].reshape(-1)),
        "V": scalar_metrics(y_true[..., 1].reshape(-1), y_pred[..., 1].reshape(-1)),
        "WS": scalar_metrics(ws(y_true).reshape(-1), ws(y_pred).reshape(-1)),
    }
    return results


def plot_rmse_by_horizon(metrics_by_horizon, save_path):
    horizons = sorted(metrics_by_horizon.keys())
    labels = ["U", "V", "WS"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for ax, label in zip(axes, labels):
        model_vals = [metrics_by_horizon[h]["model"]["overall"][label]["rmse"] for h in horizons]
        pers_vals = [metrics_by_horizon[h]["persistence"]["overall"][label]["rmse"] for h in horizons]
        ax.plot(horizons, model_vals, marker="o", label="model")
        ax.plot(horizons, pers_vals, marker="s", label="persistence")
        ax.set_title(f"{label} RMSE")
        ax.set_xlabel("Horizon")
        ax.grid(alpha=0.2)

    axes[0].legend()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_sample_forecast(y_true, y_model, y_persistence, save_path, station_idx=0):
    x = np.arange(y_true.shape[0]) + 1
    station = STATIONS[station_idx]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    series_defs = [
        ("U", lambda a: a[:, station_idx, 0]),
        ("V", lambda a: a[:, station_idx, 1]),
        ("WS", lambda a: ws(a[:, station_idx])),
    ]
    for ax, (label, fn) in zip(axes, series_defs):
        ax.plot(x, fn(y_true), label="truth", linewidth=2)
        ax.plot(x, fn(y_model), label="model", linewidth=1.8)
        ax.plot(x, fn(y_persistence), label="persistence", linewidth=1.6)
        ax.set_title(f"{station} {label}")
        ax.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("Forecast Horizon")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def build_sequences(mean_model, nwp_seq, seq_len, context_len):
    start_at = context_len - 1
    total_outputs = nwp_seq.shape[1] - start_at

    if mean_model is None:
        return zero_dynamics_sequence(
            batch_size=nwp_seq.shape[0],
            chunk_len=total_outputs,
            device=nwp_seq.device,
            dtype=nwp_seq.dtype,
        )
    return build_dynamics_sequence(mean_model, nwp_seq, seq_len=seq_len, start_at=start_at)


def main():
    args = parse_args()
    ckpt = torch.load(Path(args.ckpt).expanduser().resolve(), map_location="cpu")
    device = auto_device(args.device)
    cfg, ide_model, mean_model = reconstruct_models(ckpt, device)
    set_seed(cfg.get("seed", 42))
    norm_stats = normalization_stats_from_config(cfg)

    seq_len = cfg.get("seq_len", 4)
    context_len = seq_len
    horizon = args.horizon if args.horizon is not None else cfg.get("forecast_horizon", 12)

    bundle = build_aligned_bundle(
        str(Path(args.measurement_file).expanduser().resolve()),
        str(Path(args.nwp_file).expanduser().resolve()),
    )
    if norm_stats is not None:
        bundle = apply_bundle_normalization(bundle, norm_stats)

    total_windows = bundle.z_meas.shape[0] - (context_len + args.history_len + horizon - 1)
    if total_windows <= 0:
        raise ValueError("Not enough time steps for the requested history_len and horizon")
    max_windows = total_windows if args.max_windows is None else min(total_windows, args.max_windows)

    site_lon = torch.from_numpy(bundle.meas_lon).float().unsqueeze(0).to(device)
    site_lat = torch.from_numpy(bundle.meas_lat).float().unsqueeze(0).to(device)

    model_truths = [[] for _ in range(horizon)]
    model_preds = [[] for _ in range(horizon)]
    pers_preds = [[] for _ in range(horizon)]
    sample_payload = None

    for start in range(max_windows):
        z_hist_np = bundle.z_meas[start + context_len - 1:start + context_len + args.history_len]
        y_future_np = bundle.z_meas[
            start + context_len + args.history_len:
            start + context_len + args.history_len + horizon
        ]
        nwp_np = bundle.nwp_uv[start:start + context_len + args.history_len + horizon - 1]

        z_hist = torch.from_numpy(z_hist_np).float().unsqueeze(0).to(device)
        nwp_seq = torch.from_numpy(nwp_np).float().unsqueeze(0).to(device)

        dynamics_full = build_sequences(
            mean_model=mean_model,
            nwp_seq=nwp_seq,
            seq_len=seq_len,
            context_len=context_len,
        )
        hist_transitions = z_hist.shape[1] - 1
        dynamics_hist = {k: v[:, :hist_transitions] for k, v in dynamics_full.items()}
        dynamics_future = {k: v[:, hist_transitions:hist_transitions + horizon] for k, v in dynamics_full.items()}

        pred = ide_model.forecast_multistep(
            z_hist=z_hist,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_hist=dynamics_hist,
            dynamics_future=dynamics_future,
            start_idx=torch.tensor([start + context_len - 1], device=device),
        )[0].cpu().numpy()
        truth = y_future_np
        persistence = np.repeat(z_hist_np[-1:], horizon, axis=0)
        if norm_stats is not None:
            pred = denormalize_z(pred, norm_stats)
            truth = denormalize_z(truth, norm_stats)
            persistence = denormalize_z(persistence, norm_stats)

        for h in range(horizon):
            model_truths[h].append(truth[h:h + 1])
            model_preds[h].append(pred[h:h + 1])
            pers_preds[h].append(persistence[h:h + 1])

        if sample_payload is None:
            sample_payload = {
                "truth": truth,
                "model": pred,
                "persistence": persistence,
            }

    metrics_by_horizon = {}
    for h in range(horizon):
        y_true = np.concatenate(model_truths[h], axis=0)
        y_model = np.concatenate(model_preds[h], axis=0)
        y_pers = np.concatenate(pers_preds[h], axis=0)
        metrics_by_horizon[h + 1] = {
            "model": compute_metrics(y_true, y_model),
            "persistence": compute_metrics(y_true, y_pers),
        }

    overall_true = np.concatenate([np.concatenate(v, axis=0) for v in model_truths], axis=0)
    overall_model = np.concatenate([np.concatenate(v, axis=0) for v in model_preds], axis=0)
    overall_pers = np.concatenate([np.concatenate(v, axis=0) for v in pers_preds], axis=0)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "validation_multistep.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
                "history_len": args.history_len,
                "horizon": horizon,
                "num_windows": max_windows,
                "seq_len": seq_len,
                "metrics_by_horizon": metrics_by_horizon,
                "overall": {
                    "model": compute_metrics(overall_true, overall_model),
                    "persistence": compute_metrics(overall_true, overall_pers),
                },
            },
            f,
            indent=2,
        )

    plot_rmse_by_horizon(metrics_by_horizon, out_dir / "rmse_by_horizon.png")
    if sample_payload is not None:
        plot_sample_forecast(
            sample_payload["truth"],
            sample_payload["model"],
            sample_payload["persistence"],
            out_dir / "sample_multistep_forecast.png",
        )

    print(f"history_len={args.history_len} horizon={horizon} windows={max_windows}")
    for h in range(1, horizon + 1):
        model_rmse = metrics_by_horizon[h]["model"]["overall"]["WS"]["rmse"]
        pers_rmse = metrics_by_horizon[h]["persistence"]["overall"]["WS"]["rmse"]
        print(f"h={h:02d} model_WS_RMSE={model_rmse:.4f} persistence_WS_RMSE={pers_rmse:.4f}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
