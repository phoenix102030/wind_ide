import argparse
import json
import math
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
from src.trainers.train_ml_mu import build_dynamics_sequence
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
    parser.add_argument("--ckpt", type=str, required=True, help="offline_best.pt, joint_best.pt, or online_roll_last.pt")
    parser.add_argument("--measurement-file", type=str, required=True)
    parser.add_argument("--nwp-file", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--history-len", type=int, default=16, help="Number of transitions used for local online adaptation")
    parser.add_argument("--local-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/validation_online_roll")
    return parser.parse_args()


def scalar_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else np.nan,
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

    mean_model = AdvectionMeanNet(
        in_channels=cfg.get("nwp_in_channels", 6),
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
        mix_scale=cfg.get("mix_scale", 0.35),
        mu_mode=cfg.get("mu_mode", "free"),
        sigma_mode=cfg.get("sigma_mode", "network"),
        init_global_sigma_diag=cfg.get("init_global_sigma_diag", 0.2),
        init_base_scale_par=cfg.get("init_base_scale_par", 1.0),
        init_base_scale_perp=cfg.get("init_base_scale_perp", 1.0),
        base_scale_min=cfg.get("base_scale_min", 0.15),
        base_scale_max=cfg.get("base_scale_max", 2.5),
        wind_anchor_indices=cfg.get("nwp_anchor_channel_indices", None),
    ).to(device)
    if "mean_model_state" in ckpt:
        mean_model.load_state_dict(ckpt["mean_model_state"])
    mean_model.eval()
    for p in mean_model.parameters():
        p.requires_grad = False

    return cfg, ide_model, mean_model


def clone_ide_model(source_model, cfg, device):
    model = IDEStateSpaceModel(
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
    model.load_state_dict(source_model.state_dict())
    return model


def local_adapt(ide_model, z_hist, site_lon, site_lat, dynamics_adapt, start_idx, lr, local_steps, noise_reg_weight):
    opt = torch.optim.Adam(ide_model.parameters(), lr=lr)
    last_loss = None
    snapshot = {k: v.detach().cpu().clone() for k, v in ide_model.state_dict().items()}
    for _ in range(local_steps):
        opt.zero_grad()
        nll = ide_model.sequence_nll(
            z_seq=z_hist,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_seq=dynamics_adapt,
            start_idx=start_idx,
        )
        loss = nll + noise_reg_weight * ide_model.noise_regularization()
        if not torch.isfinite(loss):
            ide_model.load_state_dict(snapshot)
            return float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ide_model.parameters(), 1.0)
        opt.step()
        ide_model.clamp_parameters_()
        last_loss = float(loss.detach().cpu())
        if not np.isfinite(last_loss):
            ide_model.load_state_dict(snapshot)
            return float("nan")
    return last_loss


def plot_rmse(results, save_path):
    methods = list(results.keys())
    labels = ["U", "V", "WS"]
    values = {label: [results[m]["overall"][label]["rmse"] for m in methods] for label in labels}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, label in zip(axes, labels):
        x = np.arange(len(methods))
        ax.bar(x, values[label])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20)
        ax.set_title(f"{label} RMSE")
        ax.grid(alpha=0.2, axis="y")

    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_online_param_path(history, save_path):
    steps = np.arange(len(history["loss"]))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

    axes[0].plot(steps, history["damping"], label="damping")
    axes[0].plot(steps, history["q_proc"], label="q_proc")
    axes[0].plot(steps, history["r_obs"], label="r_obs")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(steps, history["loss"], label="local_loss")
    axes[1].plot(steps, history["mu_norm"], label="mu_norm")
    axes[1].plot(steps, history["sigma_mean"], label="sigma_mean")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    axes[1].set_xlabel("Rolling Window Index")

    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_timeseries(y_true, outputs, save_path, station_idx=0):
    station = STATIONS[station_idx]
    x = np.arange(y_true.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

    series_defs = [
        ("U", lambda a: a[:, station_idx, 0]),
        ("V", lambda a: a[:, station_idx, 1]),
        ("WS", lambda a: ws(a[:, station_idx])),
    ]
    for ax, (label, fn) in zip(axes, series_defs):
        ax.plot(x, fn(y_true), label="truth", linewidth=2)
        for name, pred in outputs.items():
            ax.plot(x, fn(pred), label=name, alpha=0.85)
        ax.set_ylabel(label)
        ax.set_title(f"{station} {label}")
        ax.grid(alpha=0.2)

    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Rolling Forecast Index")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    ckpt = torch.load(Path(args.ckpt).expanduser().resolve(), map_location="cpu")
    ckpt_name = Path(args.ckpt).name
    if ckpt_name.startswith("online_roll"):
        print(
            "[warning] Using an already adapted online checkpoint as the validation starting point can"
            " cause repeated adaptation drift. Prefer starting from offline_best.pt or joint_best.pt."
        )
    device = auto_device(args.device)
    cfg, base_ide_model, mean_model = reconstruct_models(ckpt, device)
    set_seed(cfg.get("seed", 42))
    norm_stats = normalization_stats_from_config(cfg)

    bundle = build_aligned_bundle(
        str(Path(args.measurement_file).expanduser().resolve()),
        str(Path(args.nwp_file).expanduser().resolve()),
        nwp_channel_mode=cfg.get("nwp_input_mode", "uv6"),
    )
    if norm_stats is not None:
        bundle = apply_bundle_normalization(bundle, norm_stats)

    seq_len = cfg.get("seq_len", 4)
    history_len = args.history_len
    total_windows = bundle.z_meas.shape[0] - (seq_len + history_len)
    if total_windows <= 0:
        raise ValueError("Not enough time steps for the requested history_len and seq_len")

    max_windows = total_windows if args.max_windows is None else min(total_windows, args.max_windows)
    lr = args.lr if args.lr is not None else cfg.get("stat_lr", 1e-4)
    noise_reg_weight = cfg.get("noise_reg_weight", 1e-4)

    site_lon = torch.from_numpy(bundle.meas_lon).float().unsqueeze(0).to(device)
    site_lat = torch.from_numpy(bundle.meas_lat).float().unsqueeze(0).to(device)

    online_model = clone_ide_model(base_ide_model, cfg, device)
    offline_model = clone_ide_model(base_ide_model, cfg, device)
    offline_model.eval()

    outputs = {
        "online_roll": [],
        "offline_fixed": [],
        "persistence": [],
    }
    truths = []
    history = {
        "damping": [],
        "q_proc": [],
        "r_obs": [],
        "loss": [],
        "mu_norm": [],
        "sigma_mean": [],
    }

    for start in range(max_windows):
        z_hist_np = bundle.z_meas[start + seq_len - 1:start + seq_len + history_len]
        target_np = bundle.z_meas[start + seq_len + history_len]
        nwp_np = bundle.nwp_uv[start:start + seq_len + history_len]

        z_hist = torch.from_numpy(z_hist_np).float().unsqueeze(0).to(device)
        target = torch.from_numpy(target_np).float().unsqueeze(0).to(device)
        nwp_seq = torch.from_numpy(nwp_np).float().unsqueeze(0).to(device)

        with torch.no_grad():
            dynamics_full = build_dynamics_sequence(mean_model, nwp_seq, seq_len=seq_len)
            dynamics_adapt = {
                "mu": dynamics_full["mu"][:, :-1],
                "sigma": dynamics_full["sigma"][:, :-1],
            }

        hist_start_idx = torch.tensor([start + seq_len - 1], device=device)

        last_loss = local_adapt(
            ide_model=online_model,
            z_hist=z_hist,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_adapt=dynamics_adapt,
            start_idx=hist_start_idx,
            lr=lr,
            local_steps=args.local_steps,
            noise_reg_weight=noise_reg_weight,
        )

        with torch.no_grad():
            pred_online = online_model.forecast_next(
                z_hist=z_hist,
                site_lon=site_lon,
                site_lat=site_lat,
                dynamics_seq=dynamics_full,
                start_idx=hist_start_idx,
            )
            pred_offline = offline_model.forecast_next(
                z_hist=z_hist,
                site_lon=site_lon,
                site_lat=site_lat,
                dynamics_seq=dynamics_full,
                start_idx=hist_start_idx,
            )
            pred_persistence = z_hist[:, -1]

        truths.append(target.cpu().numpy())
        outputs["online_roll"].append(pred_online.cpu().numpy())
        outputs["offline_fixed"].append(pred_offline.cpu().numpy())
        outputs["persistence"].append(pred_persistence.cpu().numpy())

        history["damping"].append(float(online_model.damping.detach().cpu()))
        history["q_proc"].append(float(online_model.q_proc.detach().cpu()))
        history["r_obs"].append(float(online_model.r_obs.detach().cpu()))
        history["loss"].append(last_loss)
        history["mu_norm"].append(float(dynamics_full["mu"].norm(dim=-1).mean().detach().cpu()))
        history["sigma_mean"].append(float(dynamics_full["sigma"].diagonal(dim1=-2, dim2=-1).mean().detach().cpu()))

        if start % 200 == 0:
            print(
                f"[VAL-ONLINE][window {start}] "
                f"loss={last_loss:.6f} "
                f"damping={history['damping'][-1]:.4f}"
            )

    y_true = np.concatenate(truths, axis=0)
    pred_arrays = {name: np.concatenate(preds, axis=0) for name, preds in outputs.items()}
    if norm_stats is not None:
        y_true = denormalize_z(y_true, norm_stats)
        pred_arrays = {name: denormalize_z(pred, norm_stats) for name, pred in pred_arrays.items()}
    results = {name: compute_metrics(y_true, pred) for name, pred in pred_arrays.items()}

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "validation_online_roll.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
                "num_windows": int(y_true.shape[0]),
                "results": results,
                "history": history,
            },
            f,
            indent=2,
        )

    plot_rmse(results, out_dir / "online_roll_rmse.png")
    plot_online_param_path(history, out_dir / "online_roll_param_path.png")
    plot_timeseries(y_true, pred_arrays, out_dir / "online_roll_timeseries.png")

    print("\nOverall rolling one-step-ahead metrics:")
    for name, payload in results.items():
        overall = payload["overall"]
        print(
            f"{name:>12} | "
            f"U RMSE={overall['U']['rmse']:.4f} "
            f"V RMSE={overall['V']['rmse']:.4f} "
            f"WS RMSE={overall['WS']['rmse']:.4f}"
        )
    print(f"\nSaved validation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
