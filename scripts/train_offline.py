import argparse
import math
import os
from pathlib import Path
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aligned_measurement_nwp import (
    apply_bundle_normalization,
    build_aligned_bundle,
    fit_bundle_normalization,
    get_nwp_uv_channel_indices,
    IDEBaselineDataset,
    MLMuDataset,
)
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


def auto_device(name=None, local_rank=None):
    if local_rank is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unwrap_model(model):
    return getattr(model, "module", model)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_distributed()) or dist.get_rank() == 0


def maybe_barrier():
    if is_distributed():
        dist.barrier()


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def setup_process_group(local_rank, world_size, master_port=None):
    if is_distributed():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if master_port is not None:
        os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("RANK", str(local_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    dist.init_process_group(backend=backend, init_method="env://", rank=local_rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_process_group():
    if is_distributed():
        dist.destroy_process_group()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(Path(args.config).expanduser(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_pair_checkpoint(path, mean_model, ide_model, cfg):
    torch.save(
        {
            "mean_model_state": unwrap_model(mean_model).state_dict(),
            "ide_model_state": unwrap_model(ide_model).state_dict(),
            "config": cfg,
        },
        path,
    )


def save_offline_pair_checkpoints(out_dir, mean_model, ide_model, cfg):
    save_pair_checkpoint(out_dir / "offline_last.pt", mean_model, ide_model, cfg)
    save_pair_checkpoint(out_dir / "joint_last.pt", mean_model, ide_model, cfg)
    save_pair_checkpoint(out_dir / "mu_last.pt", mean_model, ide_model, cfg)


def save_best_offline_pair_checkpoints(out_dir, mean_model, ide_model, cfg):
    save_pair_checkpoint(out_dir / "offline_best.pt", mean_model, ide_model, cfg)
    save_pair_checkpoint(out_dir / "joint_best.pt", mean_model, ide_model, cfg)
    save_pair_checkpoint(out_dir / "mu_best.pt", mean_model, ide_model, cfg)


def configure_ide_trainability(model, train_ell_params=True):
    del train_ell_params
    model.log_ell_par_knots.requires_grad_(False)
    model.log_ell_perp_knots.requires_grad_(False)


def dataset_sample_span(dataset):
    if hasattr(dataset, "seq_len"):
        return int(dataset.seq_len + dataset.chunk_len)
    return int(dataset.chunk_len + 1)


def subset_window_starts(subset):
    if isinstance(subset, Subset):
        return [subset.dataset.valid[i] for i in subset.indices]
    return list(subset.valid)


def summarize_knot_coverage(subset, sample_span, param_window):
    starts = subset_window_starts(subset)
    if not starts:
        return {"windows": 0, "knot_count": 0, "knot_min": None, "knot_max": None}

    knots = set()
    for start in starts:
        end = start + sample_span - 1
        for t in range(start, end + 1):
            knots.add(t // param_window)

    return {
        "windows": len(starts),
        "start_min": int(min(starts)),
        "start_max": int(max(starts)),
        "knot_count": len(knots),
        "knot_min": int(min(knots)),
        "knot_max": int(max(knots)),
        "knot_set": knots,
    }


def contiguous_time_split(dataset, val_fraction=0.2):
    starts = list(dataset.valid)
    if len(starts) < 2:
        raise ValueError("Need at least two valid windows to create a contiguous train/val split.")

    sample_span = dataset_sample_span(dataset)
    preferred_cut = int(round((1.0 - val_fraction) * len(starts)))
    preferred_cut = min(max(preferred_cut, 1), len(starts) - 1)

    candidate_cuts = []
    for delta in range(len(starts)):
        right = preferred_cut + delta
        left = preferred_cut - delta
        if right < len(starts):
            candidate_cuts.append(right)
        if delta > 0 and left > 0:
            candidate_cuts.append(left)

    for cut in candidate_cuts:
        split_time = starts[cut]
        train_indices = [i for i, start in enumerate(starts[:cut]) if start + sample_span <= split_time]
        val_indices = [cut + i for i, start in enumerate(starts[cut:]) if start >= split_time]
        if train_indices and val_indices:
            info = {
                "split_time": int(split_time),
                "sample_span": sample_span,
                "train_windows": len(train_indices),
                "val_windows": len(val_indices),
            }
            return Subset(dataset, train_indices), Subset(dataset, val_indices), info

    raise ValueError("Failed to construct a contiguous non-overlapping train/val split.")


def build_loader(dataset, batch_size, shuffle, distributed, num_workers=0):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def maybe_wrap_ddp(model, device, distributed):
    if not distributed:
        return model
    if device.type != "cuda":
        return DDP(model)
    return DDP(model, device_ids=[device.index], output_device=device.index)


def should_auto_spawn(cfg):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return False
    if cfg.get("device", None) not in (None, "cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return False
    requested = int(cfg.get("num_gpus", world_size))
    return requested > 1


def run_training(cfg, local_rank=None, world_size=1, master_port=None):
    distributed = world_size > 1
    if distributed:
        setup_process_group(local_rank=local_rank, world_size=world_size, master_port=master_port)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = auto_device(cfg.get("device", None), local_rank=local_rank if distributed else None)
    num_workers = int(cfg.get("num_workers", 0))

    meas_file = str(Path(cfg["measurement_file"]).expanduser().resolve())
    nwp_file = str(Path(cfg["nwp_file"]).expanduser().resolve())
    seq_len = int(cfg.get("seq_len", 4))
    chunk_len = int(cfg.get("chunk_len", 16))
    val_fraction = float(cfg.get("val_fraction", 0.2))

    cfg["nwp_input_mode"] = str(cfg.get("nwp_input_mode", "uv6")).lower()
    raw_bundle = build_aligned_bundle(meas_file, nwp_file, nwp_channel_mode=cfg["nwp_input_mode"])
    cfg = dict(cfg)
    cfg["ide_total_steps"] = int(raw_bundle.z_meas.shape[0])
    cfg["nwp_in_channels"] = int(raw_bundle.nwp_uv.shape[1])
    cfg["nwp_anchor_channel_indices"] = list(get_nwp_uv_channel_indices(cfg["nwp_input_mode"], height="140"))
    cfg["ide_param_mode"] = str(cfg.get("ide_param_mode", "absolute")).lower()
    cfg["skip_ide_warmup"] = bool(cfg.get("skip_ide_warmup", False))
    cfg["train_ell_params"] = bool(cfg.get("train_ell_params", True))
    cfg["normalize_z"] = bool(cfg.get("normalize_z", True))
    cfg["normalize_nwp"] = bool(cfg.get("normalize_nwp", True))
    cfg["mu_mode"] = str(cfg.get("mu_mode", "free")).lower()
    cfg["sigma_mode"] = str(cfg.get("sigma_mode", "network")).lower()
    out_dir = Path(cfg.get("out_dir", "outputs/measurement_140m_two_stage"))
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print("device:", device)
        print("distributed_world_size:", world_size)
        print(f"advection_seq_len={seq_len}")
        print(f"ide_param_mode={cfg['ide_param_mode']}")
        print(f"skip_ide_warmup={cfg['skip_ide_warmup']}")
        print(f"train_ell_params={cfg['train_ell_params']}")
        print(f"ide_param_window={cfg.get('ide_param_window', 12)}")
        print(f"chunk_len={chunk_len}")
        print(f"normalize_z={cfg['normalize_z']}")
        print(f"normalize_nwp={cfg['normalize_nwp']}")
        print(f"mu_mode={cfg['mu_mode']}")
        print(f"sigma_mode={cfg['sigma_mode']}")
        print(f"nwp_input_mode={cfg['nwp_input_mode']}")
        if cfg["nwp_input_mode"] != "all12":
            print(
                "[warning] nwp_input_mode!=all12 means the advection net only sees wind U/V channels "
                "instead of the full environmental NWP context."
            )
        if cfg["train_ell_params"]:
            print(
                "[warning] train_ell_params is deprecated after the theorem-inspired kernel refactor; "
                "legacy ell parameters stay frozen for checkpoint compatibility."
            )
        if "mix_scale" in cfg:
            print("[warning] mix_scale is deprecated after the theorem-based pair-kernel refactor and is ignored.")
        if cfg["sigma_mode"] == "global":
            print(
                "[warning] sigma_mode=global keeps the joint 4x4 advection covariance fixed over time, "
                "so NWP cannot modulate pairwise kernel shape through Sigma_t."
            )

    raw_ide_base_ds = IDEBaselineDataset(raw_bundle, chunk_len=chunk_len)
    _, _, raw_ide_split_info = contiguous_time_split(raw_ide_base_ds, val_fraction=val_fraction)
    raw_ml_ds = MLMuDataset(raw_bundle, seq_len=seq_len, chunk_len=chunk_len)
    _, _, raw_ml_split_info = contiguous_time_split(raw_ml_ds, val_fraction=val_fraction)

    bundle = raw_bundle
    if cfg["normalize_z"] or cfg["normalize_nwp"]:
        norm_train_stop = min(raw_ide_split_info["split_time"], raw_ml_split_info["split_time"])
        norm_stats = fit_bundle_normalization(raw_bundle, end_time=norm_train_stop)
        cfg["normalization"] = norm_stats.to_config_dict()
        bundle = apply_bundle_normalization(
            raw_bundle,
            norm_stats,
            normalize_z_values=cfg["normalize_z"],
            normalize_nwp_values=cfg["normalize_nwp"],
        )
        if is_main_process():
            print(
                "[normalization] "
                f"fit_until_t={norm_train_stop} "
                f"z_std_mean={float(norm_stats.z_std.mean()):.4f} "
                f"nwp_std_mean={float(norm_stats.nwp_std.mean()):.4f}"
            )
    else:
        cfg["normalization"] = None

    ide_base_ds = IDEBaselineDataset(bundle, chunk_len=chunk_len)
    ide_train_ds, ide_val_ds, ide_split_info = contiguous_time_split(ide_base_ds, val_fraction=val_fraction)
    if is_main_process():
        print(
            "[IDE split] "
            f"split_time={ide_split_info['split_time']} "
            f"sample_span={ide_split_info['sample_span']} "
            f"train={ide_split_info['train_windows']} "
            f"val={ide_split_info['val_windows']}"
        )
        if cfg["ide_param_mode"] == "absolute":
            ide_coverage_train = summarize_knot_coverage(
                ide_train_ds,
                sample_span=ide_split_info["sample_span"],
                param_window=int(cfg.get("ide_param_window", 12)),
            )
            ide_coverage_val = summarize_knot_coverage(
                ide_val_ds,
                sample_span=ide_split_info["sample_span"],
                param_window=int(cfg.get("ide_param_window", 12)),
            )
            overlap = len(ide_coverage_train["knot_set"] & ide_coverage_val["knot_set"])
            print(
                "[IDE knot coverage] "
                f"train={ide_coverage_train['knot_min']}..{ide_coverage_train['knot_max']} "
                f"({ide_coverage_train['knot_count']} knots) "
                f"val={ide_coverage_val['knot_min']}..{ide_coverage_val['knot_max']} "
                f"({ide_coverage_val['knot_count']} knots) "
                f"overlap={overlap}"
            )
            if overlap == 0:
                print(
                    "[warning] ide_param_mode=absolute with a contiguous time split means the validation "
                    "windows use a disjoint set of learned IDE time knots; val loss can stay flat even "
                    "while train loss drops."
                )

    ide_train_loader, ide_train_sampler = build_loader(
        ide_train_ds,
        batch_size=cfg.get("ide_batch_size", 32),
        shuffle=True,
        distributed=distributed,
        num_workers=num_workers,
    )
    ide_val_loader, _ = build_loader(
        ide_val_ds,
        batch_size=cfg.get("ide_batch_size", 32),
        shuffle=False,
        distributed=distributed,
        num_workers=num_workers,
    )

    ide_model = IDEStateSpaceModel(
        dt=cfg.get("dt", 1.0),
        total_steps=cfg.get("ide_total_steps", 1),
        param_window=cfg.get("ide_param_window", 12),
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
    configure_ide_trainability(ide_model, train_ell_params=cfg["train_ell_params"])
    ide_model = maybe_wrap_ddp(ide_model, device, distributed)
    ide_opt = torch.optim.Adam(
        [p for p in unwrap_model(ide_model).parameters() if p.requires_grad],
        lr=cfg.get("ide_lr", 1e-3),
    )

    if is_main_process():
        print("ide_model params:", count_parameters(unwrap_model(ide_model)))
        print("ide_model trainable params:", sum(p.numel() for p in unwrap_model(ide_model).parameters() if p.requires_grad))
        raw_ide = unwrap_model(ide_model)
        print(
            "[IDE init] "
            f"damping={float(raw_ide.damping.detach().cpu()):.4f} "
            f"q_proc={float(raw_ide.q_proc.detach().cpu()):.4f} "
            f"r_obs={float(raw_ide.r_obs.detach().cpu()):.4f}"
        )

    warmup_epochs = int(cfg.get("ide_epochs", 20))
    skip_ide_warmup = cfg["skip_ide_warmup"] or warmup_epochs <= 0
    best_ide_val = float("nan") if skip_ide_warmup else float("inf")
    if skip_ide_warmup:
        raw_ide = unwrap_model(ide_model)
        if is_main_process():
            print("[IDE warmup] skipped; starting directly from alternating optimization.")
            torch.save({"model_state": raw_ide.state_dict(), "config": cfg}, out_dir / "ide_last.pt")
            torch.save({"model_state": raw_ide.state_dict(), "config": cfg}, out_dir / "ide_best.pt")
        maybe_barrier()
    else:
        for epoch in range(warmup_epochs):
            if ide_train_sampler is not None:
                ide_train_sampler.set_epoch(epoch)

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

            raw_ide = unwrap_model(ide_model)
            if is_main_process():
                print(
                    f"[OFFLINE-STAT-ZERO][epoch {epoch}] "
                    f"train={tr:.6f} val={va:.6f} "
                    f"ell_par={float(raw_ide.ell_par.detach().cpu()):.4f} "
                    f"ell_perp={float(raw_ide.ell_perp.detach().cpu()):.4f} "
                    f"damping={float(raw_ide.damping.detach().cpu()):.4f} "
                    f"q_proc={float(raw_ide.q_proc.detach().cpu()):.4f} "
                    f"r_obs={float(raw_ide.r_obs.detach().cpu()):.4f}"
                )
                torch.save({"model_state": raw_ide.state_dict(), "config": cfg}, out_dir / "ide_last.pt")
                if va < best_ide_val:
                    best_ide_val = va
                    torch.save({"model_state": raw_ide.state_dict(), "config": cfg}, out_dir / "ide_best.pt")
            maybe_barrier()

    ml_ds = MLMuDataset(bundle, seq_len=seq_len, chunk_len=chunk_len)
    ml_train_ds, ml_val_ds, ml_split_info = contiguous_time_split(ml_ds, val_fraction=val_fraction)
    if is_main_process():
        print(
            "[ML split] "
            f"split_time={ml_split_info['split_time']} "
            f"sample_span={ml_split_info['sample_span']} "
            f"train={ml_split_info['train_windows']} "
            f"val={ml_split_info['val_windows']}"
        )

    ml_train_loader, ml_train_sampler = build_loader(
        ml_train_ds,
        batch_size=cfg.get("ml_batch_size", 16),
        shuffle=True,
        distributed=distributed,
        num_workers=num_workers,
    )
    ml_val_loader, _ = build_loader(
        ml_val_ds,
        batch_size=cfg.get("ml_batch_size", 16),
        shuffle=False,
        distributed=distributed,
        num_workers=num_workers,
    )

    mean_model = AdvectionMeanNet(
        in_channels=cfg.get("nwp_in_channels", raw_bundle.nwp_uv.shape[1]),
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
        wind_anchor_indices=cfg.get("nwp_anchor_channel_indices", None),
    ).to(device)
    mean_model = maybe_wrap_ddp(mean_model, device, distributed)

    stat_opt = torch.optim.Adam(
        [p for p in unwrap_model(ide_model).parameters() if p.requires_grad],
        lr=cfg.get("stat_lr", 1e-4),
    )
    adv_opt = torch.optim.Adam(unwrap_model(mean_model).parameters(), lr=cfg.get("ml_lr", 1e-3))

    if is_main_process():
        print("mean_model params:", count_parameters(unwrap_model(mean_model)))

    best_offline_val = float("inf")
    offline_rounds = cfg.get("offline_rounds", 3)
    stat_epochs = cfg.get("stat_epochs_per_round", 5)
    adv_epochs = cfg.get("adv_epochs_per_round", cfg.get("ml_epochs", 20))
    stat_max_steps = cfg.get("stat_max_steps", cfg.get("ml_max_steps", None))
    adv_max_steps = cfg.get("adv_max_steps", cfg.get("ml_max_steps", None))
    use_advection_in_stat = cfg.get("use_advection_in_stat", True)

    for round_idx in range(offline_rounds):
        for epoch in range(adv_epochs):
            if ml_train_sampler is not None:
                ml_train_sampler.set_epoch(round_idx * (stat_epochs + adv_epochs) + epoch)

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

            if is_main_process():
                print(
                    f"[OFFLINE-ADV][round {round_idx} epoch {epoch}] "
                    f"train={tr['loss']:.6f} val={va['loss']:.6f} "
                    f"train_nll={tr['nll']:.6f} val_nll={va['nll']:.6f} "
                    f"smooth={tr['smoothness']:.6f} "
                    f"mu_norm={tr['mu_norm']:.6f} "
                    f"mu_abs={tr['mu_abs_mean']:.6f} "
                    f"sigma_mean={tr['sigma_mean']:.6f} "
                    f"sigma_trace={tr['sigma_trace']:.6f} "
                    f"sigma_diag_min={tr['sigma_diag_min']:.6f}"
                )
                save_offline_pair_checkpoints(out_dir, mean_model, ide_model, cfg)
                if va["loss"] < best_offline_val:
                    best_offline_val = va["loss"]
                    save_best_offline_pair_checkpoints(out_dir, mean_model, ide_model, cfg)
            maybe_barrier()

        for epoch in range(stat_epochs):
            if ml_train_sampler is not None:
                ml_train_sampler.set_epoch(round_idx * (stat_epochs + adv_epochs) + adv_epochs + epoch)

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

            raw_ide = unwrap_model(ide_model)
            if is_main_process():
                print(
                    f"[OFFLINE-STAT][round {round_idx} epoch {epoch}] "
                    f"train={tr['loss']:.6f} val={va['loss']:.6f} "
                    f"train_nll={tr['nll']:.6f} val_nll={va['nll']:.6f} "
                    f"ell_par={float(raw_ide.ell_par.detach().cpu()):.4f} "
                    f"ell_perp={float(raw_ide.ell_perp.detach().cpu()):.4f} "
                    f"damping={float(raw_ide.damping.detach().cpu()):.4f} "
                    f"q_proc={float(raw_ide.q_proc.detach().cpu()):.4f} "
                    f"r_obs={float(raw_ide.r_obs.detach().cpu()):.4f}"
                )
                save_offline_pair_checkpoints(out_dir, mean_model, ide_model, cfg)
                if va["loss"] < best_offline_val:
                    best_offline_val = va["loss"]
                    save_best_offline_pair_checkpoints(out_dir, mean_model, ide_model, cfg)
            maybe_barrier()

    if is_main_process():
        print("best ide val:", best_ide_val)
        print("best offline val:", best_offline_val)

    cleanup_process_group()


def main_worker(local_rank, world_size, cfg, master_port):
    run_training(cfg=cfg, local_rank=local_rank, world_size=world_size, master_port=master_port)


def main():
    cfg = parse_config()

    if should_auto_spawn(cfg):
        world_size = min(int(cfg.get("num_gpus", torch.cuda.device_count())), torch.cuda.device_count())
        master_port = int(cfg.get("ddp_port", find_free_port()))
        print(f"Launching offline training with DDP on {world_size} GPUs (port={master_port})")
        mp.spawn(main_worker, args=(world_size, cfg, master_port), nprocs=world_size, join=True)
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    run_training(cfg=cfg, local_rank=local_rank if world_size > 1 else None, world_size=world_size, master_port=None)


if __name__ == "__main__":
    main()
