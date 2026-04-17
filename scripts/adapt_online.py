import argparse
import json
import math
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
    parser.add_argument("--ckpt", type=str, required=True, help="offline_best.pt or joint_best.pt")
    parser.add_argument("--measurement-file", type=str, required=True)
    parser.add_argument("--nwp-file", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/measurement_140m_online")
    parser.add_argument("--local-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def to_numpy(x):
    return x.detach().cpu().numpy()


def compute_ws(y):
    return np.sqrt((y ** 2).sum(axis=-1))


def scalar_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else np.nan,
    }


def overall_metrics(y_true, y_pred):
    ws_true = compute_ws(y_true)
    ws_pred = compute_ws(y_pred)
    return {
        "U": scalar_metrics(y_true[..., 0].reshape(-1), y_pred[..., 0].reshape(-1)),
        "V": scalar_metrics(y_true[..., 1].reshape(-1), y_pred[..., 1].reshape(-1)),
        "WS": scalar_metrics(ws_true.reshape(-1), ws_pred.reshape(-1)),
    }


def local_adapt_step(ide_model, batch, dynamics_seq, lr, local_steps, noise_reg_weight):
    z_full = batch["z_seq_full"]
    chunk_len = dynamics_seq["mu"].shape[1]
    z_aligned = z_full[:, -(chunk_len + 1):]
    start_idx = batch["time_idx_start"] + batch["nwp_seq_full"].shape[1] - chunk_len

    optimizer = torch.optim.Adam(ide_model.parameters(), lr=lr)
    last_loss = None
    snapshot = {k: v.detach().cpu().clone() for k, v in ide_model.state_dict().items()}
    for _ in range(local_steps):
        optimizer.zero_grad()
        nll = ide_model.sequence_nll(
            z_seq=z_aligned,
            site_lon=batch["site_lon"],
            site_lat=batch["site_lat"],
            dynamics_seq=dynamics_seq,
            start_idx=start_idx,
        )
        loss = nll + noise_reg_weight * ide_model.noise_regularization()
        if not torch.isfinite(loss):
            ide_model.load_state_dict(snapshot)
            return z_aligned, start_idx, float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ide_model.parameters(), 1.0)
        optimizer.step()
        ide_model.clamp_parameters_()
        last_loss = float(loss.detach().cpu())
        if not np.isfinite(last_loss):
            ide_model.load_state_dict(snapshot)
            return z_aligned, start_idx, float("nan")
    return z_aligned, start_idx, last_loss


def plot_param_trajectory(history, save_path):
    steps = np.arange(len(history["damping"]))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

    axes[0].plot(steps, history["damping"], label="damping")
    axes[0].plot(steps, history["q_proc"], label="q_proc")
    axes[0].plot(steps, history["r_obs"], label="r_obs")
    axes[0].legend()
    axes[0].set_ylabel("IDE Params")
    axes[0].grid(alpha=0.2)

    axes[1].plot(steps, history["mu_norm"], label="mu_norm")
    axes[1].plot(steps, history["sigma_mean"], label="sigma_mean")
    axes[1].plot(steps, history["loss"], label="local_loss")
    axes[1].legend()
    axes[1].set_ylabel("Advection / Loss")
    axes[1].set_xlabel("Online Window Index")
    axes[1].grid(alpha=0.2)

    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    ckpt = torch.load(Path(args.ckpt).expanduser().resolve(), map_location="cpu")
    cfg = ckpt["config"]
    set_seed(cfg.get("seed", 42))
    device = auto_device(args.device)

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
    if "mean_model_state" in ckpt:
        mean_model.load_state_dict(ckpt["mean_model_state"])
    mean_model.eval()
    for p in mean_model.parameters():
        p.requires_grad = False

    bundle = build_aligned_bundle(
        str(Path(args.measurement_file).expanduser().resolve()),
        str(Path(args.nwp_file).expanduser().resolve()),
    )
    dataset = MLMuDataset(bundle, seq_len=cfg.get("seq_len", 4), chunk_len=cfg.get("chunk_len", 16))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lr = args.lr if args.lr is not None else cfg.get("stat_lr", 1e-4)
    noise_reg_weight = cfg.get("noise_reg_weight", 1e-4)

    truth_list = []
    pred_list = []
    history = {
        "damping": [],
        "q_proc": [],
        "r_obs": [],
        "mu_norm": [],
        "sigma_mean": [],
        "loss": [],
    }

    for window_idx, batch in enumerate(loader):
        if args.max_windows is not None and window_idx >= args.max_windows:
            break

        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            dynamics_seq = build_dynamics_sequence(mean_model, batch["nwp_seq_full"], seq_len=cfg.get("seq_len", 4))

        z_aligned, aligned_start_idx, last_loss = local_adapt_step(
            ide_model=ide_model,
            batch=batch,
            dynamics_seq=dynamics_seq,
            lr=lr,
            local_steps=args.local_steps,
            noise_reg_weight=noise_reg_weight,
        )

        with torch.no_grad():
            pred_seq = ide_model.predict_sequence(
                z_seq=z_aligned,
                site_lon=batch["site_lon"],
                site_lat=batch["site_lat"],
                dynamics_seq=dynamics_seq,
                start_idx=aligned_start_idx,
            )

        truth = to_numpy(z_aligned[:, -1]).reshape(1, 3, 2)
        pred = to_numpy(pred_seq[:, -1]).reshape(1, 3, 2)
        truth_list.append(truth)
        pred_list.append(pred)

        mu_norm = float(dynamics_seq["mu"].norm(dim=-1).mean().detach().cpu())
        sigma_mean = float(dynamics_seq["sigma"].diagonal(dim1=-2, dim2=-1).mean().detach().cpu())
        history["damping"].append(float(ide_model.damping.detach().cpu()))
        history["q_proc"].append(float(ide_model.q_proc.detach().cpu()))
        history["r_obs"].append(float(ide_model.r_obs.detach().cpu()))
        history["mu_norm"].append(mu_norm)
        history["sigma_mean"].append(sigma_mean)
        history["loss"].append(last_loss)

        if window_idx % 100 == 0:
            print(
                f"[ONLINE-ROLL][window {window_idx}] "
                f"loss={last_loss:.6f} "
                f"damping={history['damping'][-1]:.4f} "
                f"q_proc={history['q_proc'][-1]:.4f} "
                f"r_obs={history['r_obs'][-1]:.4f} "
                f"mu_norm={mu_norm:.4f}"
            )

    truth = np.concatenate(truth_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    metrics = overall_metrics(truth, pred)

    payload = {
        "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
        "num_windows": int(truth.shape[0]),
        "metrics": metrics,
        "history": history,
    }
    with open(out_dir / "online_roll_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    torch.save(
        {
            "mean_model_state": mean_model.state_dict(),
            "ide_model_state": ide_model.state_dict(),
            "config": cfg,
        },
        out_dir / "online_roll_last.pt",
    )

    plot_param_trajectory(history, out_dir / "online_roll_params.png")

    print("\nOverall online rolling metrics:")
    for var in ("U", "V", "WS"):
        print(
            f"{var}: RMSE={metrics[var]['rmse']:.4f} "
            f"MAE={metrics[var]['mae']:.4f} "
            f"Corr={metrics[var]['corr']:.4f}"
        )
    print(f"\nSaved online rolling outputs to: {out_dir}")


if __name__ == "__main__":
    main()
