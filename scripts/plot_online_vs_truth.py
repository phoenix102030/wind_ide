import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aligned_measurement_nwp import build_aligned_bundle
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
    parser.add_argument("--ckpt", type=str, required=True, help="Use offline_best.pt or joint_best.pt as the rolling start state")
    parser.add_argument("--measurement-file", type=str, required=True)
    parser.add_argument("--nwp-file", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--local-steps", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/online_prediction_compare")
    return parser.parse_args()


def compute_ws(y):
    return np.sqrt((y ** 2).sum(axis=-1))


def scalar_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else np.nan,
    }


def compute_metrics(y_true, y_pred):
    results = {}
    for i, station in enumerate(STATIONS):
        results[station] = {
            "U": scalar_metrics(y_true[:, i, 0], y_pred[:, i, 0]),
            "V": scalar_metrics(y_true[:, i, 1], y_pred[:, i, 1]),
            "WS": scalar_metrics(compute_ws(y_true[:, i]), compute_ws(y_pred[:, i])),
        }
    results["overall"] = {
        "U": scalar_metrics(y_true[..., 0].reshape(-1), y_pred[..., 0].reshape(-1)),
        "V": scalar_metrics(y_true[..., 1].reshape(-1), y_pred[..., 1].reshape(-1)),
        "WS": scalar_metrics(compute_ws(y_true).reshape(-1), compute_ws(y_pred).reshape(-1)),
    }
    return results


def reconstruct_models(ckpt, device):
    cfg = ckpt["config"]
    ide_model = IDEStateSpaceModel(
        dt=cfg.get("dt", 1.0),
        init_log_ell_par=cfg.get("init_log_ell_par", 0.5),
        init_log_ell_perp=cfg.get("init_log_ell_perp", 0.0),
        init_log_q_proc=cfg.get("init_log_q_proc", -2.0),
        init_log_r_obs=cfg.get("init_log_r_obs", -2.0),
        init_log_p0=cfg.get("init_log_p0", 0.0),
        init_log_damping=cfg.get("init_log_damping", 0.0),
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
    return cfg, ide_model, mean_model


def local_adapt(ide_model, z_hist, site_lon, site_lat, dynamics_adapt, lr, local_steps, noise_reg_weight):
    opt = torch.optim.Adam(ide_model.parameters(), lr=lr)
    snapshot = {k: v.detach().cpu().clone() for k, v in ide_model.state_dict().items()}
    for _ in range(local_steps):
        opt.zero_grad()
        nll = ide_model.sequence_nll(
            z_seq=z_hist,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_seq=dynamics_adapt,
        )
        loss = nll + noise_reg_weight * ide_model.noise_regularization()
        if not torch.isfinite(loss):
            ide_model.load_state_dict(snapshot)
            return
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ide_model.parameters(), 1.0)
        opt.step()
        ide_model.clamp_parameters_()


def plot_predictions(y_true, y_model, y_persistence, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True, constrained_layout=True)
    x = np.arange(y_true.shape[0])

    series_defs = [
        ("U", lambda arr, idx: arr[:, idx, 0]),
        ("V", lambda arr, idx: arr[:, idx, 1]),
        ("WS", lambda arr, idx: compute_ws(arr[:, idx])),
    ]

    for row, station in enumerate(STATIONS):
        for col, (label, fn) in enumerate(series_defs):
            ax = axes[row, col]
            ax.plot(x, fn(y_true, row), label="truth", linewidth=2, color="black")
            ax.plot(x, fn(y_model, row), label="online_model", linewidth=1.8, color="#1f77b4")
            ax.plot(x, fn(y_persistence, row), label="persistence", linewidth=1.4, color="#ff7f0e")
            ax.set_title(f"{station} {label}")
            ax.grid(alpha=0.2)

    axes[0, 0].legend(ncol=3, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Forecast Step")
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    ckpt = torch.load(Path(args.ckpt).expanduser().resolve(), map_location="cpu")
    device = auto_device(args.device)
    cfg, ide_model, mean_model = reconstruct_models(ckpt, device)
    set_seed(cfg.get("seed", 42))

    bundle = build_aligned_bundle(
        str(Path(args.measurement_file).expanduser().resolve()),
        str(Path(args.nwp_file).expanduser().resolve()),
    )

    site_lon = torch.from_numpy(bundle.meas_lon).float().unsqueeze(0).to(device)
    site_lat = torch.from_numpy(bundle.meas_lat).float().unsqueeze(0).to(device)
    seq_len = cfg.get("seq_len", 4)
    lr = args.lr if args.lr is not None else cfg.get("stat_lr", 1e-4)
    noise_reg_weight = cfg.get("noise_reg_weight", 1e-4)

    total_available = bundle.z_meas.shape[0] - (seq_len + args.history_len)
    num_steps = min(args.num_steps, total_available)

    true_list = []
    model_list = []
    persistence_list = []

    for start in range(num_steps):
        z_hist_np = bundle.z_meas[start + seq_len - 1:start + seq_len + args.history_len]
        target_np = bundle.z_meas[start + seq_len + args.history_len]
        nwp_np = bundle.nwp_uv[start:start + seq_len + args.history_len]

        z_hist = torch.from_numpy(z_hist_np).float().unsqueeze(0).to(device)
        target = torch.from_numpy(target_np).float().unsqueeze(0).to(device)
        nwp_seq = torch.from_numpy(nwp_np).float().unsqueeze(0).to(device)

        with torch.no_grad():
            dynamics_full = build_dynamics_sequence(mean_model, nwp_seq, seq_len=seq_len)
            dynamics_adapt = {
                "mu": dynamics_full["mu"][:, :-1],
                "sigma": dynamics_full["sigma"][:, :-1],
            }

        local_adapt(
            ide_model=ide_model,
            z_hist=z_hist,
            site_lon=site_lon,
            site_lat=site_lat,
            dynamics_adapt=dynamics_adapt,
            lr=lr,
            local_steps=args.local_steps,
            noise_reg_weight=noise_reg_weight,
        )

        with torch.no_grad():
            pred = ide_model.forecast_next(
                z_hist=z_hist,
                site_lon=site_lon,
                site_lat=site_lat,
                dynamics_seq=dynamics_full,
            )
            persistence = z_hist[:, -1]

        true_list.append(target.cpu().numpy())
        model_list.append(pred.cpu().numpy())
        persistence_list.append(persistence.cpu().numpy())

    y_true = np.concatenate(true_list, axis=0)
    y_model = np.concatenate(model_list, axis=0)
    y_persistence = np.concatenate(persistence_list, axis=0)

    model_metrics = compute_metrics(y_true, y_model)
    persistence_metrics = compute_metrics(y_true, y_persistence)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_predictions(
        y_true=y_true,
        y_model=y_model,
        y_persistence=y_persistence,
        save_path=out_dir / "first_100_online_vs_truth.png",
    )

    payload = {
        "num_steps": int(num_steps),
        "model_metrics": model_metrics,
        "persistence_metrics": persistence_metrics,
        "truth": y_true.tolist(),
        "model_pred": y_model.tolist(),
        "persistence_pred": y_persistence.tolist(),
    }
    with open(out_dir / "first_100_online_vs_truth.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"num_steps: {num_steps}")
    print(
        "model overall:",
        f"U_RMSE={model_metrics['overall']['U']['rmse']:.4f}",
        f"V_RMSE={model_metrics['overall']['V']['rmse']:.4f}",
        f"WS_RMSE={model_metrics['overall']['WS']['rmse']:.4f}",
    )
    print(
        "persistence overall:",
        f"U_RMSE={persistence_metrics['overall']['U']['rmse']:.4f}",
        f"V_RMSE={persistence_metrics['overall']['V']['rmse']:.4f}",
        f"WS_RMSE={persistence_metrics['overall']['WS']['rmse']:.4f}",
    )
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
