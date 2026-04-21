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
    bundle_to_shared_tensors,
    build_aligned_bundle,
    fit_bundle_normalization,
    get_nwp_uv_channel_indices,
    MLMuDataset,
)
from src.models.advection_mean_net import AdvectionMeanNet
from src.models.ide_state_space import IDEStateSpaceModel
from src.trainers.train_ml_mu import (
    build_dynamics_sequence,
    train_joint_one_epoch,
    eval_joint,
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


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


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


def _stack_transition_diagnostics(diag_steps):
    stacked = {}
    for key in diag_steps[0]:
        value0 = diag_steps[0][key]
        if torch.is_tensor(value0):
            stacked[key] = torch.stack([step[key] for step in diag_steps], dim=1)
        else:
            stacked[key] = [step[key] for step in diag_steps]
    return stacked


@torch.no_grad()
def save_transition_dump(
    out_dir,
    split_name,
    epoch,
    mean_model,
    ide_model,
    loader,
    device,
    seq_len,
    cfg,
):
    raw_mean = unwrap_model(mean_model)
    raw_ide = unwrap_model(ide_model)
    was_mean_training = mean_model.training
    was_ide_training = ide_model.training
    mean_model.eval()
    ide_model.eval()
    try:
        try:
            batch = next(iter(loader))
        except StopIteration:
            return None

        batch = move_batch_to_device(batch, device)
        z_full = batch["z_seq_full"]
        nwp_full = batch["nwp_seq_full"]
        start_idx = batch["time_idx_start"]
        chunk_len = nwp_full.shape[1] - seq_len + 1
        dynamics_seq = build_dynamics_sequence(mean_model, nwp_full, seq_len=seq_len)
        z_aligned = z_full[:, -(chunk_len + 1):]
        aligned_start_idx = start_idx + seq_len - 1
        trans_idx = raw_ide._time_indices(aligned_start_idx, chunk_len)

        diag_steps = []
        for t in range(chunk_len):
            dynamics_t = {
                "mu": dynamics_seq["mu"][:, t],
                "base_scales": dynamics_seq["base_scales"][:, t],
                "transport_gates": dynamics_seq["transport_gates"][:, t],
                "state_bias": dynamics_seq["state_bias"][:, t],
                "sigma": dynamics_seq["sigma"][:, t],
            }
            diag_steps.append(
                raw_ide.transition_diagnostics(
                    site_lon=batch["site_lon"],
                    site_lat=batch["site_lat"],
                    dynamics_t=dynamics_t,
                    transition_idx=trans_idx[:, t],
                    device=device,
                    dtype=z_aligned.dtype,
                    apply_damping=True,
                )
            )

        diagnostics = _stack_transition_diagnostics(diag_steps)
        dump = {
            "epoch": int(epoch),
            "split": split_name,
            "config": cfg,
            "state_order": [f"{comp}(site{i+1})" for i in range(raw_ide.num_sites) for comp in ("u", "v")],
            "site_lon": batch["site_lon"].detach().cpu(),
            "site_lat": batch["site_lat"].detach().cpu(),
            "time_idx_start": batch["time_idx_start"].detach().cpu(),
            "aligned_start_idx": aligned_start_idx.detach().cpu(),
            "z_seq_full": batch["z_seq_full"].detach().cpu(),
            "nwp_seq_full": batch["nwp_seq_full"].detach().cpu(),
            "z_aligned": z_aligned.detach().cpu(),
            "dynamics_seq": {key: value.detach().cpu() for key, value in dynamics_seq.items()},
            "transition_diagnostics": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in diagnostics.items()},
            "ide_parameter_series": {
                "q_proc_series": raw_ide.q_proc_series.detach().cpu(),
                "r_obs_series": raw_ide.r_obs_series.detach().cpu(),
                "damping_series": raw_ide.damping_series.detach().cpu(),
                "p0_series": raw_ide.p0_series.detach().cpu(),
                "init_mean_series": raw_ide.init_mean_series.detach().cpu(),
                "q_adv_scale": raw_ide.q_adv_scale.detach().cpu(),
                "nll_sigma_scale": raw_ide.nll_sigma_scale.detach().cpu(),
            },
            "mean_model_parameters": {name: tensor.detach().cpu() for name, tensor in raw_mean.state_dict().items()},
            "ide_model_parameters": {name: tensor.detach().cpu() for name, tensor in raw_ide.state_dict().items()},
            "normalization": cfg.get("normalization"),
            "normalize_z": bool(cfg.get("normalize_z", True)),
            "normalize_nwp": bool(cfg.get("normalize_nwp", True)),
        }

        diagnostics_dir = out_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostics_dir / f"{split_name}_epoch_{int(epoch):03d}_transition_dump.pt"
        torch.save(dump, path)
        return path
    finally:
        mean_model.train(was_mean_training)
        ide_model.train(was_ide_training)


def configure_ide_trainability(
    model,
    train_ell_params=True,
    train_q_proc=True,
    train_r_obs=True,
    train_p0=False,
    train_damping=True,
    train_init_mean=False,
):
    del train_ell_params
    model.log_ell_par_knots.requires_grad_(False)
    model.log_ell_perp_knots.requires_grad_(False)
    model.log_q_proc_knots.requires_grad_(bool(train_q_proc))
    model.log_r_obs_knots.requires_grad_(bool(train_r_obs))
    model.log_p0_knots.requires_grad_(bool(train_p0))
    model.log_damping_knots.requires_grad_(bool(train_damping))
    model.init_mean_knots.requires_grad_(bool(train_init_mean))


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


def prepare_training_bundle(cfg):
    cfg = dict(cfg)
    meas_file = str(Path(cfg["measurement_file"]).expanduser().resolve())
    nwp_file = str(Path(cfg["nwp_file"]).expanduser().resolve())
    seq_len = int(cfg.get("seq_len", 4))
    chunk_len = int(cfg.get("chunk_len", 16))
    val_fraction = float(cfg.get("val_fraction", 0.2))

    cfg["nwp_input_mode"] = str(cfg.get("nwp_input_mode", "uv6")).lower()
    raw_bundle = build_aligned_bundle(meas_file, nwp_file, nwp_channel_mode=cfg["nwp_input_mode"])
    cfg["ide_total_steps"] = int(raw_bundle.z_meas.shape[0])
    cfg["num_sites"] = int(raw_bundle.z_meas.shape[1])
    cfg["nwp_in_channels"] = int(raw_bundle.nwp_uv.shape[1])
    cfg["nwp_anchor_channel_indices"] = list(get_nwp_uv_channel_indices(cfg["nwp_input_mode"], height="140"))
    cfg["ide_param_mode"] = str(cfg.get("ide_param_mode", "absolute")).lower()
    cfg["train_ell_params"] = bool(cfg.get("train_ell_params", False))
    cfg["normalize_z"] = bool(cfg.get("normalize_z", True))
    cfg["normalize_nwp"] = bool(cfg.get("normalize_nwp", True))
    cfg["mu_mode"] = str(cfg.get("mu_mode", "anchored")).lower()
    cfg["sigma_mode"] = str(cfg.get("sigma_mode", "network")).lower()

    raw_ml_ds = MLMuDataset(raw_bundle, seq_len=seq_len, chunk_len=chunk_len)
    _, _, raw_ml_split_info = contiguous_time_split(raw_ml_ds, val_fraction=val_fraction)

    bundle = raw_bundle
    if cfg["normalize_z"] or cfg["normalize_nwp"]:
        norm_train_stop = raw_ml_split_info["split_time"]
        norm_stats = fit_bundle_normalization(raw_bundle, end_time=norm_train_stop)
        cfg["normalization"] = norm_stats.to_config_dict()
        bundle = apply_bundle_normalization(
            raw_bundle,
            norm_stats,
            normalize_z_values=cfg["normalize_z"],
            normalize_nwp_values=cfg["normalize_nwp"],
        )
        cfg["normalization_summary"] = {
            "fit_until_t": int(norm_train_stop),
            "z_std_mean": float(norm_stats.z_std.mean()),
            "nwp_std_mean": float(norm_stats.nwp_std.mean()),
        }
    else:
        cfg["normalization"] = None
        cfg["normalization_summary"] = None

    return cfg, bundle


def run_training(cfg, local_rank=None, world_size=1, master_port=None, preloaded_bundle=None):
    distributed = world_size > 1
    if distributed:
        setup_process_group(local_rank=local_rank, world_size=world_size, master_port=master_port)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = auto_device(cfg.get("device", None), local_rank=local_rank if distributed else None)
    num_workers = int(cfg.get("num_workers", 0))

    seq_len = int(cfg.get("seq_len", 4))
    chunk_len = int(cfg.get("chunk_len", 16))
    val_fraction = float(cfg.get("val_fraction", 0.2))
    cfg = dict(cfg)
    batch_size = int(cfg.get("batch_size", cfg.get("ml_batch_size", 16)))
    mean_lr = float(cfg.get("mean_lr", cfg.get("ml_lr", 1e-3)))
    ide_lr = float(cfg.get("ide_lr", cfg.get("joint_ide_lr", cfg.get("stat_lr", 1e-4))))
    joint_epochs = int(
        cfg.get(
            "epochs",
            cfg.get("joint_epochs", cfg.get("offline_rounds", 3) * cfg.get("adv_epochs_per_round", cfg.get("ml_epochs", 20))),
        )
    )
    joint_max_steps = cfg.get("max_steps", cfg.get("joint_max_steps", cfg.get("adv_max_steps", cfg.get("ml_max_steps", None))))
    out_dir = Path(cfg.get("out_dir", "outputs/measurement_140m_two_stage"))
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print("device:", device)
        print("distributed_world_size:", world_size)
        print(f"advection_seq_len={seq_len}")
        print(f"ide_param_mode={cfg['ide_param_mode']}")
        print(f"ide_param_window={cfg.get('ide_param_window', 12)}")
        print(f"chunk_len={chunk_len}")
        print(f"batch_size={batch_size}")
        print(f"epochs={joint_epochs}")
        print(f"max_steps={joint_max_steps}")
        print(f"mean_lr={mean_lr}")
        print(f"ide_lr={ide_lr}")
        print(f"normalize_z={cfg['normalize_z']}")
        print(f"normalize_nwp={cfg['normalize_nwp']}")
        print(f"mu_mode={cfg['mu_mode']}")
        print(f"sigma_mode={cfg['sigma_mode']}")
        print(f"nwp_input_mode={cfg['nwp_input_mode']}")
        print(f"adv_history_len={cfg.get('adv_history_len', 6)}")
        print(f"adv_one_step_weight={cfg.get('adv_one_step_weight', 0.25)}")
        print(f"adv_rollout_weight={cfg.get('adv_rollout_weight', 1.0)}")
        print(f"step_nll_weight={cfg.get('step_nll_weight', 0.1)}")
        print(f"prob_nll_weight={cfg.get('prob_nll_weight', 1.0)}")
        print(f"asymmetry_weight={cfg.get('asymmetry_weight', 0.01)}")
        print(f"transport_reg_weight={cfg.get('transport_reg_weight', 1e-2)}")
        print(f"transport_floor_weight={cfg.get('transport_floor_weight', 0.0)}")
        print(f"min_transport_gate={cfg.get('min_transport_gate', 0.0)}")
        print(f"state_bias_reg_weight={cfg.get('state_bias_reg_weight', 1e-3)}")
        print(f"sigma_cross_reg_weight={cfg.get('sigma_cross_reg_weight', 1e-4)}")
        print(f"sigma_floor_weight={cfg.get('sigma_floor_weight', 0.0)}")
        print(f"min_sigma_diag={cfg.get('min_sigma_diag', 0.0)}")
        print(f"scale_floor_weight={cfg.get('scale_floor_weight', 0.0)}")
        print(f"min_q_adv_scale={cfg.get('min_q_adv_scale', 0.0)}")
        print(f"min_nll_sigma_scale={cfg.get('min_nll_sigma_scale', 0.0)}")
        print(f"init_transport_gate={cfg.get('init_transport_gate', 0.05)}")
        print(f"transport_gate_max={cfg.get('transport_gate_max', 0.35)}")
        print(f"state_bias_scale={cfg.get('state_bias_scale', 1.0)}")
        print(f"gate_warmup_fraction={cfg.get('gate_warmup_fraction', 0.2)}")
        print(f"force_gate_start_value={cfg.get('force_gate_start_value', cfg.get('force_gate_value', 0.2))}")
        print(f"force_gate_end_value={cfg.get('force_gate_end_value', cfg.get('gate_floor_value', 0.05))}")
        print(f"gate_floor_fraction={cfg.get('gate_floor_fraction', 0.2)}")
        print(f"gate_floor_value={cfg.get('gate_floor_value', 0.05)}")
        print(f"save_transition_dump={cfg.get('save_transition_dump', True)}")
        print(f"transition_dump_split={cfg.get('transition_dump_split', 'val')}")
        print(f"transition_dump_every={cfg.get('transition_dump_every', 1)}")
        print(f"train_q_proc={cfg.get('train_q_proc', True)}")
        print(f"train_r_obs={cfg.get('train_r_obs', True)}")
        print(f"train_damping={cfg.get('train_damping', True)}")
        print(f"train_p0={cfg.get('train_p0', False)}")
        print(f"train_init_mean={cfg.get('train_init_mean', False)}")
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
        if cfg["mu_mode"] == "free":
            print("[warning] mu_mode=free bypasses the PDF's anchored alpha*bias advection design.")

    if preloaded_bundle is None:
        cfg, bundle = prepare_training_bundle(cfg)
    else:
        bundle = preloaded_bundle
    if is_main_process() and cfg.get("normalization_summary") is not None:
        print(
            "[normalization] "
            f"fit_until_t={cfg['normalization_summary']['fit_until_t']} "
            f"z_std_mean={cfg['normalization_summary']['z_std_mean']:.4f} "
            f"nwp_std_mean={cfg['normalization_summary']['nwp_std_mean']:.4f}"
        )

    ml_ds = MLMuDataset(bundle, seq_len=seq_len, chunk_len=chunk_len)
    ml_train_ds, ml_val_ds, ml_split_info = contiguous_time_split(ml_ds, val_fraction=val_fraction)
    if is_main_process():
        print(
            "[Joint split] "
            f"split_time={ml_split_info['split_time']} "
            f"sample_span={ml_split_info['sample_span']} "
            f"train={ml_split_info['train_windows']} "
            f"val={ml_split_info['val_windows']}"
        )
        if cfg["ide_param_mode"] == "absolute":
            coverage_train = summarize_knot_coverage(
                ml_train_ds,
                sample_span=ml_split_info["sample_span"],
                param_window=int(cfg.get("ide_param_window", 12)),
            )
            coverage_val = summarize_knot_coverage(
                ml_val_ds,
                sample_span=ml_split_info["sample_span"],
                param_window=int(cfg.get("ide_param_window", 12)),
            )
            overlap = len(coverage_train["knot_set"] & coverage_val["knot_set"])
            print(
                "[IDE knot coverage] "
                f"train={coverage_train['knot_min']}..{coverage_train['knot_max']} "
                f"({coverage_train['knot_count']} knots) "
                f"val={coverage_val['knot_min']}..{coverage_val['knot_max']} "
                f"({coverage_val['knot_count']} knots) "
                f"overlap={overlap}"
            )
            if overlap == 0:
                print(
                    "[warning] ide_param_mode=absolute with a contiguous time split means the validation "
                    "windows use a disjoint set of learned IDE time knots; use joint val curves as a trend, "
                    "and rely on held-out forecast validation for the final check."
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
    configure_ide_trainability(
        ide_model,
        train_ell_params=cfg["train_ell_params"],
        train_q_proc=cfg.get("train_q_proc", True),
        train_r_obs=cfg.get("train_r_obs", True),
        train_p0=cfg.get("train_p0", False),
        train_damping=cfg.get("train_damping", True),
        train_init_mean=cfg.get("train_init_mean", False),
    )
    ide_model = maybe_wrap_ddp(ide_model, device, distributed)

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

    ml_train_loader, ml_train_sampler = build_loader(
        ml_train_ds,
        batch_size=batch_size,
        shuffle=True,
        distributed=distributed,
        num_workers=num_workers,
    )
    ml_val_loader, _ = build_loader(
        ml_val_ds,
        batch_size=batch_size,
        shuffle=False,
        distributed=distributed,
        num_workers=num_workers,
    )

    mean_model = AdvectionMeanNet(
        in_channels=cfg.get("nwp_in_channels", bundle.nwp_uv.shape[1]),
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
        num_sites=cfg.get("num_sites", bundle.z_meas.shape[1]),
        init_transport_gate=cfg.get("init_transport_gate", 0.05),
        transport_gate_max=cfg.get("transport_gate_max", 0.35),
        state_bias_scale=cfg.get("state_bias_scale", 1.0),
        wind_anchor_indices=cfg.get("nwp_anchor_channel_indices", None),
    ).to(device)
    mean_model = maybe_wrap_ddp(mean_model, device, distributed)

    mean_params = list(unwrap_model(mean_model).parameters())
    ide_params = [p for p in unwrap_model(ide_model).parameters() if p.requires_grad]
    joint_opt = torch.optim.Adam(
        [
            {"params": mean_params, "lr": mean_lr},
            {"params": ide_params, "lr": ide_lr},
        ]
    )

    if is_main_process():
        print("mean_model params:", count_parameters(unwrap_model(mean_model)))
        print("joint_trainable params:", sum(p.numel() for p in mean_params + ide_params if p.requires_grad))

    best_offline_val = float("inf")
    warmup_epochs = max(int(round(joint_epochs * float(cfg.get("gate_warmup_fraction", 0.2)))), 0)
    transport_gate_cap = float(cfg.get("transport_gate_max", 0.35))
    force_gate_start_value = float(cfg.get("force_gate_start_value", cfg.get("force_gate_value", 0.2)))
    force_gate_end_value = float(cfg.get("force_gate_end_value", cfg.get("gate_floor_value", 0.05)))
    force_gate_start_value = min(max(force_gate_start_value, 0.0), transport_gate_cap)
    force_gate_end_value = min(max(force_gate_end_value, 0.0), transport_gate_cap)
    gate_floor_epochs = max(int(round(joint_epochs * float(cfg.get("gate_floor_fraction", 0.2)))), 0)
    gate_floor_value = float(cfg.get("gate_floor_value", 0.05))
    dump_every = max(int(cfg.get("transition_dump_every", 1)), 1)
    dump_enabled = bool(cfg.get("save_transition_dump", True))
    dump_split = str(cfg.get("transition_dump_split", "val")).lower()

    for epoch in range(joint_epochs):
        if ml_train_sampler is not None:
            ml_train_sampler.set_epoch(epoch)
        raw_mean = unwrap_model(mean_model)
        current_gate_override = None
        current_gate_mix = None
        current_gate_floor = None
        in_warmup = epoch < warmup_epochs
        if in_warmup:
            if warmup_epochs > 1:
                warmup_progress = epoch / float(warmup_epochs - 1)
            else:
                warmup_progress = 0.0
            scheduled_force_gate = (
                (1.0 - warmup_progress) * force_gate_start_value
                + warmup_progress * force_gate_end_value
            )
            raw_mean.force_gate_value = float(scheduled_force_gate)
            current_gate_override = float(scheduled_force_gate)
        else:
            raw_mean.force_gate_value = None
        if in_warmup and warmup_epochs > 1:
            raw_mean.force_gate_mix = max(0.0, 1.0 - (epoch / float(warmup_epochs - 1)))
        elif in_warmup:
            raw_mean.force_gate_mix = 1.0
        else:
            raw_mean.force_gate_mix = None
        current_gate_mix = None if raw_mean.force_gate_mix is None else float(raw_mean.force_gate_mix)
        if epoch < warmup_epochs:
            raw_mean.gate_floor_value = None
        elif gate_floor_epochs > 0 and epoch < warmup_epochs + gate_floor_epochs:
            decay_progress = (epoch - warmup_epochs) / max(gate_floor_epochs, 1)
            raw_mean.gate_floor_value = gate_floor_value * max(0.0, 1.0 - decay_progress)
        else:
            raw_mean.gate_floor_value = None
        current_gate_floor = None if raw_mean.gate_floor_value is None else float(raw_mean.gate_floor_value)

        tr = train_joint_one_epoch(
            mean_model=mean_model,
            ide_model=ide_model,
            loader=ml_train_loader,
            optimizer=joint_opt,
            device=device,
            seq_len=seq_len,
            max_steps=joint_max_steps,
            smoothness_weight=cfg.get("smoothness_weight", 1e-3),
            one_step_weight=cfg.get("adv_one_step_weight", 0.25),
            rollout_weight=cfg.get("adv_rollout_weight", 1.0),
            rollout_history=cfg.get("adv_history_len", 6),
            step_nll_weight=cfg.get("step_nll_weight", 0.1),
            prob_nll_weight=cfg.get("prob_nll_weight", 1.0),
            noise_reg_weight=cfg.get("noise_reg_weight", 1e-4),
            transport_reg_weight=cfg.get("transport_reg_weight", 1e-2),
            transport_floor_weight=cfg.get("transport_floor_weight", 0.0),
            min_transport_gate=cfg.get("min_transport_gate", 0.0),
            state_bias_reg_weight=cfg.get("state_bias_reg_weight", 1e-3),
            asymmetry_weight=cfg.get("asymmetry_weight", 0.01),
            sigma_cross_reg_weight=cfg.get("sigma_cross_reg_weight", 1e-4),
            sigma_floor_weight=cfg.get("sigma_floor_weight", 0.0),
            min_sigma_diag=cfg.get("min_sigma_diag", 0.0),
            scale_floor_weight=cfg.get("scale_floor_weight", 0.0),
            min_q_adv_scale=cfg.get("min_q_adv_scale", 0.0),
            min_nll_sigma_scale=cfg.get("min_nll_sigma_scale", 0.0),
        )
        raw_mean.force_gate_value = None
        raw_mean.force_gate_mix = None
        raw_mean.gate_floor_value = None
        va = eval_joint(
            mean_model=mean_model,
            ide_model=ide_model,
            loader=ml_val_loader,
            device=device,
            seq_len=seq_len,
            max_steps=joint_max_steps,
            smoothness_weight=cfg.get("smoothness_weight", 1e-3),
            one_step_weight=cfg.get("adv_one_step_weight", 0.25),
            rollout_weight=cfg.get("adv_rollout_weight", 1.0),
            rollout_history=cfg.get("adv_history_len", 6),
            step_nll_weight=cfg.get("step_nll_weight", 0.1),
            prob_nll_weight=cfg.get("prob_nll_weight", 1.0),
            noise_reg_weight=cfg.get("noise_reg_weight", 1e-4),
            transport_reg_weight=cfg.get("transport_reg_weight", 1e-2),
            transport_floor_weight=cfg.get("transport_floor_weight", 0.0),
            min_transport_gate=cfg.get("min_transport_gate", 0.0),
            state_bias_reg_weight=cfg.get("state_bias_reg_weight", 1e-3),
            asymmetry_weight=cfg.get("asymmetry_weight", 0.01),
            sigma_cross_reg_weight=cfg.get("sigma_cross_reg_weight", 1e-4),
            sigma_floor_weight=cfg.get("sigma_floor_weight", 0.0),
            min_sigma_diag=cfg.get("min_sigma_diag", 0.0),
            scale_floor_weight=cfg.get("scale_floor_weight", 0.0),
            min_q_adv_scale=cfg.get("min_q_adv_scale", 0.0),
            min_nll_sigma_scale=cfg.get("min_nll_sigma_scale", 0.0),
        )

        raw_ide = unwrap_model(ide_model)
        if is_main_process():
            print(
                f"[OFFLINE-JOINT][epoch {epoch}] "
                f"train={tr['loss']:.6f} val={va['loss']:.6f} "
                f"train_prob={tr['prob_nll']:.6f} val_prob={va['prob_nll']:.6f} "
                f"train_roll={tr['rollout_mse']:.6f} val_roll={va['rollout_mse']:.6f} "
                f"train_step={tr['one_step_mse']:.6f} val_step={va['one_step_mse']:.6f} "
                f"train_nll={tr['step_nll']:.6f} val_nll={va['step_nll']:.6f} "
                f"noise={tr['noise_reg']:.6f} "
                f"smooth={tr['smoothness']:.6f} "
                f"mu_norm={tr['mu_norm']:.6f} "
                f"mu_abs={tr['mu_abs_mean']:.6f} "
                f"base_mean={tr['base_scale_mean']:.6f} "
                f"base_aniso={tr['base_scale_anisotropy']:.6f} "
                f"transport={tr['transport_gate_mean']:.6f} "
                f"bias_abs={tr['state_bias_abs_mean']:.6f} "
                f"transport_reg={tr['transport_reg']:.6f} "
                f"transport_floor_pen={tr['transport_floor_pen']:.3e} "
                f"bias_reg={tr['state_bias_reg']:.6f} "
                f"sigma_cross_reg={tr['sigma_cross_reg']:.3e} "
                f"sigma_floor_pen={tr['sigma_floor_pen']:.3e} "
                f"scale_floor_pen={tr['scale_floor_pen']:.3e} "
                f"asym={tr['asym_loss']:.3e} "
                f"damping={float(raw_ide.damping.detach().cpu()):.6f} "
                f"q_proc={float(raw_ide.q_proc.detach().cpu()):.6f} "
                f"r_obs={float(raw_ide.r_obs.detach().cpu()):.6f} "
                f"q_adv_mean={tr['q_adv_mean']:.6f} "
                f"nll_sigma_scale={tr['nll_sigma_scale_mean']:.6f} "
                f"sigma_mean={tr['sigma_mean']:.6f} "
                f"sigma_trace={tr['sigma_trace']:.6f} "
                f"sigma_diag_min={tr['sigma_diag_min']:.3e} "
                f"sigma_state_var_mean={tr['sigma_state_var_mean']:.6f} "
                f"asym_active={tr['asym_active_frac']:.6f} "
                f"gate_override={current_gate_override if current_gate_override is not None else -1.0:.2f} "
                f"gate_mix={current_gate_mix if current_gate_mix is not None else -1.0:.4f} "
                f"gate_floor={current_gate_floor if current_gate_floor is not None else -1.0:.4f}"
            )
            save_offline_pair_checkpoints(out_dir, mean_model, ide_model, cfg)
            if va["loss"] < best_offline_val:
                best_offline_val = va["loss"]
                save_best_offline_pair_checkpoints(out_dir, mean_model, ide_model, cfg)
            if dump_enabled and ((epoch + 1) % dump_every == 0 or epoch == joint_epochs - 1):
                dump_loader = ml_val_loader if dump_split == "val" else ml_train_loader
                dump_path = save_transition_dump(
                    out_dir=out_dir,
                    split_name=dump_split,
                    epoch=epoch,
                    mean_model=mean_model,
                    ide_model=ide_model,
                    loader=dump_loader,
                    device=device,
                    seq_len=seq_len,
                    cfg=cfg,
                )
                if dump_path is not None:
                    print(f"[transition-dump] saved {dump_path}")
        maybe_barrier()

    if is_main_process():
        print("best offline val:", best_offline_val)

    cleanup_process_group()


def main_worker(local_rank, world_size, cfg, master_port, preloaded_bundle):
    run_training(
        cfg=cfg,
        local_rank=local_rank,
        world_size=world_size,
        master_port=master_port,
        preloaded_bundle=preloaded_bundle,
    )


def main():
    cfg = parse_config()

    if should_auto_spawn(cfg):
        world_size = min(int(cfg.get("num_gpus", torch.cuda.device_count())), torch.cuda.device_count())
        master_port = int(cfg.get("ddp_port", find_free_port()))
        prepared_cfg, prepared_bundle = prepare_training_bundle(cfg)
        shared_bundle = bundle_to_shared_tensors(prepared_bundle)
        print(f"Launching offline training with DDP on {world_size} GPUs (port={master_port})")
        mp.spawn(main_worker, args=(world_size, prepared_cfg, master_port, shared_bundle), nprocs=world_size, join=True)
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    run_training(cfg=cfg, local_rank=local_rank if world_size > 1 else None, world_size=world_size, master_port=None)


if __name__ == "__main__":
    main()
