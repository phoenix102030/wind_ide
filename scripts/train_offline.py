import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, random_split
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aligned_measurement_nwp import build_aligned_bundle, IDEBaselineDataset, MLMuDataset
from src.models.advection_mean_net import AdvectionMeanNet
from src.models.ide_state_space import IDEStateSpaceModel
from src.trainers.train_ide_baseline import train_ide_baseline_one_epoch, eval_ide_baseline
from src.trainers.train_ml_mu import (
    train_statistical_one_epoch,
    eval_statistical,
    train_advection_one_epoch,
    eval_advection,
)
from src.utils.misc import set_seed, count_parameters


def auto_device(name):
    if name is not None:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(Path(args.config).expanduser(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_pair_checkpoint(path, mean_model, ide_model, cfg):
    torch.save(
        {
            "mean_model_state": mean_model.state_dict(),
            "ide_model_state": ide_model.state_dict(),
            "config": cfg,
        },
        path,
    )


def main():
    cfg = parse_config()
    set_seed(cfg.get("seed", 42))
    device = auto_device(cfg.get("device", None))

    meas_file = str(Path(cfg["measurement_file"]).expanduser().resolve())
    nwp_file = str(Path(cfg["nwp_file"]).expanduser().resolve())
    seq_len = cfg.get("seq_len", 4)
    chunk_len = cfg.get("chunk_len", 16)

    bundle = build_aligned_bundle(meas_file, nwp_file)
    out_dir = Path(cfg.get("out_dir", "outputs/measurement_140m_two_stage"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print(f"advection_seq_len={seq_len}")

    ide_base_ds = IDEBaselineDataset(bundle, chunk_len=chunk_len)
    n_total = len(ide_base_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    ide_train_ds, ide_val_ds = random_split(ide_base_ds, [n_train, n_val])

    ide_train_loader = DataLoader(ide_train_ds, batch_size=cfg.get("ide_batch_size", 32), shuffle=True)
    ide_val_loader = DataLoader(ide_val_ds, batch_size=cfg.get("ide_batch_size", 32), shuffle=False)

    ide_model = IDEStateSpaceModel(
        dt=cfg.get("dt", 1.0),
        init_log_q_proc=cfg.get("init_log_q_proc", -2.0),
        init_log_r_obs=cfg.get("init_log_r_obs", -2.0),
        init_log_p0=cfg.get("init_log_p0", 0.0),
        init_log_damping=cfg.get("init_log_damping", 0.0),
    ).to(device)
    ide_opt = torch.optim.Adam(ide_model.parameters(), lr=cfg.get("ide_lr", 1e-3))

    print("ide_model params:", count_parameters(ide_model))

    best_ide_val = float("inf")
    for epoch in range(cfg.get("ide_epochs", 20)):
        tr = train_ide_baseline_one_epoch(
            ide_model=ide_model,
            loader=ide_train_loader,
            optimizer=ide_opt,
            device=device,
            max_steps=cfg.get("ide_max_steps", None),
            noise_reg_weight=cfg.get("noise_reg_weight", 1e-4),
        )
        va = eval_ide_baseline(
            ide_model=ide_model,
            loader=ide_val_loader,
            device=device,
            max_steps=cfg.get("ide_max_steps", None),
            noise_reg_weight=cfg.get("noise_reg_weight", 1e-4),
        )

        coupling = ide_model.base_coupling.detach().cpu()
        print(
            f"[OFFLINE-STAT-ZERO][epoch {epoch}] "
            f"train={tr:.6f} val={va:.6f} "
            f"damping={float(ide_model.damping.detach().cpu()):.4f} "
            f"q_proc={float(ide_model.q_proc.detach().cpu()):.4f} "
            f"r_obs={float(ide_model.r_obs.detach().cpu()):.4f} "
            f"b12={float(coupling[0, 1]):.4f} "
            f"b21={float(coupling[1, 0]):.4f}"
        )

        torch.save({"model_state": ide_model.state_dict(), "config": cfg}, out_dir / "ide_last.pt")
        if va < best_ide_val:
            best_ide_val = va
            torch.save({"model_state": ide_model.state_dict(), "config": cfg}, out_dir / "ide_best.pt")

    ml_ds = MLMuDataset(bundle, seq_len=seq_len, chunk_len=chunk_len)
    n_total = len(ml_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    ml_train_ds, ml_val_ds = random_split(ml_ds, [n_train, n_val])

    ml_train_loader = DataLoader(ml_train_ds, batch_size=cfg.get("ml_batch_size", 16), shuffle=True)
    ml_val_loader = DataLoader(ml_val_ds, batch_size=cfg.get("ml_batch_size", 16), shuffle=False)

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

    stat_opt = torch.optim.Adam(ide_model.parameters(), lr=cfg.get("stat_lr", 1e-4))
    adv_opt = torch.optim.Adam(mean_model.parameters(), lr=cfg.get("ml_lr", 1e-3))

    print("mean_model params:", count_parameters(mean_model))

    best_offline_val = float("inf")
    offline_rounds = cfg.get("offline_rounds", 3)
    stat_epochs = cfg.get("stat_epochs_per_round", 5)
    adv_epochs = cfg.get("adv_epochs_per_round", cfg.get("ml_epochs", 20))
    stat_max_steps = cfg.get("stat_max_steps", cfg.get("ml_max_steps", None))
    adv_max_steps = cfg.get("adv_max_steps", cfg.get("ml_max_steps", None))
    use_advection_in_stat = cfg.get("use_advection_in_stat", True)

    for round_idx in range(offline_rounds):
        for epoch in range(stat_epochs):
            tr = train_statistical_one_epoch(
                mean_model=mean_model,
                ide_model=ide_model,
                loader=ml_train_loader,
                optimizer=stat_opt,
                device=device,
                seq_len=seq_len,
                max_steps=stat_max_steps,
                noise_reg_weight=cfg.get("noise_reg_weight", 1e-4),
                use_advection=use_advection_in_stat,
            )
            va = eval_statistical(
                mean_model=mean_model,
                ide_model=ide_model,
                loader=ml_val_loader,
                device=device,
                seq_len=seq_len,
                max_steps=stat_max_steps,
                noise_reg_weight=cfg.get("noise_reg_weight", 1e-4),
                use_advection=use_advection_in_stat,
            )

            coupling = ide_model.base_coupling.detach().cpu()
            print(
                f"[OFFLINE-STAT][round {round_idx} epoch {epoch}] "
                f"train={tr['loss']:.6f} val={va['loss']:.6f} "
                f"train_nll={tr['nll']:.6f} val_nll={va['nll']:.6f} "
                f"damping={float(ide_model.damping.detach().cpu()):.4f} "
                f"q_proc={float(ide_model.q_proc.detach().cpu()):.4f} "
                f"r_obs={float(ide_model.r_obs.detach().cpu()):.4f} "
                f"b12={float(coupling[0, 1]):.4f} "
                f"b21={float(coupling[1, 0]):.4f}"
            )

        for epoch in range(adv_epochs):
            tr = train_advection_one_epoch(
                mean_model=mean_model,
                ide_model=ide_model,
                loader=ml_train_loader,
                optimizer=adv_opt,
                device=device,
                seq_len=seq_len,
                max_steps=adv_max_steps,
                smoothness_weight=cfg.get("smoothness_weight", 1e-3),
            )
            va = eval_advection(
                mean_model=mean_model,
                ide_model=ide_model,
                loader=ml_val_loader,
                device=device,
                seq_len=seq_len,
                max_steps=adv_max_steps,
                smoothness_weight=cfg.get("smoothness_weight", 1e-3),
            )

            print(
                f"[OFFLINE-ADV][round {round_idx} epoch {epoch}] "
                f"train={tr['loss']:.6f} val={va['loss']:.6f} "
                f"train_nll={tr['nll']:.6f} val_nll={va['nll']:.6f} "
                f"smooth={tr['smoothness']:.6f} "
                f"sigma_mean={tr['sigma_mean']:.6f}"
            )

            save_pair_checkpoint(out_dir / "offline_last.pt", mean_model, ide_model, cfg)
            save_pair_checkpoint(out_dir / "joint_last.pt", mean_model, ide_model, cfg)
            save_pair_checkpoint(out_dir / "mu_last.pt", mean_model, ide_model, cfg)

            if va["loss"] < best_offline_val:
                best_offline_val = va["loss"]
                save_pair_checkpoint(out_dir / "offline_best.pt", mean_model, ide_model, cfg)
                save_pair_checkpoint(out_dir / "joint_best.pt", mean_model, ide_model, cfg)
                save_pair_checkpoint(out_dir / "mu_best.pt", mean_model, ide_model, cfg)

    print("best ide val:", best_ide_val)
    print("best offline val:", best_offline_val)


if __name__ == "__main__":
    main()
