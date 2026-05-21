from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.vector_data_utils import load_vector_dataset
from model.covariance import smoothness_loss
from model.vector_dstm import VectorMIDE


class VectorMIDETrainingModule(torch.nn.Module):
    """DDP-friendly wrapper whose forward computes the training loss dict."""

    def __init__(self, model: VectorMIDE) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        coords: torch.Tensor,
        v_star: torch.Tensor | None,
        B_star: torch.Tensor | None,
        stage: str,
        loss_kwargs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if stage == "adv":
            if v_star is None:
                outputs = self.model(x, coords)
                zero = outputs["M"].new_tensor(0.0)
                return {
                    "loss": zero,
                    "loss_adv": zero,
                    "loss_deform": zero,
                    "loss_smooth": zero,
                    **outputs,
                }
            outputs = self.model(x, coords)
            loss_adv = self.model.advection_supervision_loss(v_star, outputs)
            loss_deform = self.model.deformation_supervision_loss(B_star, outputs)
            smooth_mu = outputs.get("flow_mu", outputs["mu"])
            smooth_matrix = outputs.get("B", outputs["A"])
            loss_smooth = smoothness_loss(smooth_mu, smooth_matrix)
            loss = (
                loss_adv
                + float(loss_kwargs.get("lambda_deform", 0.0)) * loss_deform
                + float(loss_kwargs.get("lambda_smooth", 0.001)) * loss_smooth
            )
            return {
                "loss": loss,
                "loss_adv": loss_adv,
                "loss_deform": loss_deform,
                "loss_smooth": loss_smooth,
                **outputs,
            }
        if stage == "kf":
            return self.model.training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=None,
                lambda_adv=0.0,
                lambda_smooth=0.0,
                lambda_reg=float(loss_kwargs.get("lambda_reg", 0.0001)),
                lambda_multistep=float(loss_kwargs.get("lambda_multistep", 0.0)),
                multistep_horizons=loss_kwargs.get("multistep_horizons", ()),
                multistep_max_origins=int(loss_kwargs.get("multistep_max_origins", 256)),
            )
        if stage == "joint":
            return self.model.training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                B_star=B_star,
                **loss_kwargs,
            )
        raise ValueError(f"Unknown stage: {stage}")


ModelLike = VectorMIDE | VectorMIDETrainingModule | DistributedDataParallel


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(name: str, allow_fallback: bool = True) -> torch.device:
    """Resolve a configured device name across Mac and CUDA servers."""
    name = str(name).strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(name)
        if allow_fallback:
            return resolve_device("auto", allow_fallback=False)
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if allow_fallback:
            return resolve_device("auto", allow_fallback=False)
        raise RuntimeError("MPS was requested but is not available.")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device(name)


def print_device_info(device: torch.device) -> None:
    if device.type == "cuda":
        count = torch.cuda.device_count()
        current = torch.cuda.current_device()
        name = torch.cuda.get_device_name(current)
        print(f"Using device: {device} ({name}); visible CUDA devices: {count}")
    else:
        print(f"Using device: {device}")


def distributed_requested() -> bool:
    return "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup_distributed(config: dict[str, Any], requested_device: str) -> tuple[bool, int, int, torch.device]:
    if not distributed_requested():
        device = resolve_device(
            requested_device,
            allow_fallback=bool(config.get("allow_device_fallback", True)),
        )
        return False, 0, 1, device

    if not torch.cuda.is_available():
        raise RuntimeError("torchrun multi-process training requires CUDA for this script.")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )
    torch.cuda.set_device(local_rank)
    backend = str(config.get("distributed_backend", "nccl"))
    dist.init_process_group(backend=backend)
    return True, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)


def unwrap_model(model: ModelLike) -> VectorMIDE:
    if isinstance(model, DistributedDataParallel):
        return unwrap_model(model.module)
    if isinstance(model, VectorMIDETrainingModule):
        return model.model
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any]) -> VectorMIDE:
    gamma = 0.0 if config.get("time_mode", "target_only") == "target_only" else float(config["gamma"])
    return VectorMIDE(
        n_sites=int(config["n_dim"]),
        in_channels=int(config["in_channels"]),
        hidden_dim=int(config.get("hidden_dim", 64)),
        mu_scale_init=float(config.get("mu_scale_init", 1.0)),
        component_mixing_floor=float(config.get("component_mixing_floor", 0.0)),
        network_type=str(config.get("network_type", "cnn_transformer")),
        transformer_d_model=int(config.get("transformer_d_model", 128)),
        transformer_nhead=int(config.get("transformer_nhead", 4)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        transformer_dim_feedforward=int(config.get("transformer_dim_feedforward", 256)),
        transformer_dropout=float(config.get("transformer_dropout", 0.1)),
        transformer_causal=bool(config.get("transformer_causal", True)),
        transformer_max_len=int(config.get("transformer_max_len", 4096)),
        component_specific_mu=bool(config.get("component_specific_mu", False)),
        advection_mode=str(config.get("advection_mode", "component")),
        deformation_scale=float(config.get("deformation_scale", 0.3)),
        dt=float(config.get("dt", 1.0)),
        gamma=gamma,
        row_normalize=bool(config.get("row_normalize", True)),
        use_spectral_scaling=bool(config.get("use_spectral_scaling", False)),
        kernel_jitter=float(config.get("kernel_jitter", 1.0e-5)),
        ell_init=float(config.get("ell_init", 1.0)),
        ell_min=float(config.get("ell_min", 0.05)),
        ell_max=float(config.get("ell_max", 10.0)),
        learnable_gamma=bool(config.get("learnable_gamma", False)),
        q_init=float(config.get("q_init", 0.2)),
        r_init=float(config.get("r_init", 0.2)),
        kalman_jitter=float(config.get("kalman_jitter", 1.0e-5)),
        transition_kernel_weight=bool(config.get("transition_kernel_weight", False)),
        transition_kernel_weight_init=float(config.get("transition_kernel_weight_init", 1.0)),
        transition_kernel_weight_min=float(config.get("transition_kernel_weight_min", 0.0)),
        transition_kernel_weight_max=float(config.get("transition_kernel_weight_max", 1.0)),
        transition_residual_decay=bool(config.get("transition_residual_decay", False)),
        transition_residual_decay_init=float(config.get("transition_residual_decay_init", 1.0)),
        transition_residual_decay_min=float(config.get("transition_residual_decay_min", 0.0)),
        transition_residual_decay_max=float(config.get("transition_residual_decay_max", 1.0)),
        transition_control=bool(config.get("transition_control", False)),
        transition_control_scale=float(config.get("transition_control_scale", 0.0)),
    )


def build_optimizer(model: ModelLike, config: dict[str, Any]) -> torch.optim.Optimizer:
    base_model = unwrap_model(model)
    groups = [
        {"params": base_model.net.backbone.parameters(), "lr": float(config["lr_cnn"])},
        {"params": list(base_model.net.head_parameters()), "lr": float(config["lr_heads"])},
        {"params": base_model.kernel.parameters(), "lr": float(config["lr_kernel"])},
        {"params": base_model.qr_params.parameters(), "lr": float(config["lr_qr"])},
    ]
    return torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 1.0e-4)))


def set_module_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def configure_stage(model: ModelLike, stage: str) -> None:
    base_model = unwrap_model(model)
    set_module_grad(base_model, True)
    if stage == "kf":
        set_module_grad(base_model.net, False)
    elif stage == "adv":
        set_module_grad(base_model.kernel, False)
        set_module_grad(base_model.qr_params, False)


def sample_window(
    arrays: dict[str, np.ndarray | None],
    window_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    T = arrays["X"].shape[0]
    if T <= window_size:
        start = 0
        end = T
    else:
        start = np.random.randint(0, T - window_size + 1)
        end = start + window_size
    x = torch.from_numpy(arrays["X"][start:end]).to(device)
    z = torch.from_numpy(arrays["Z"][start:end]).to(device)
    v_star_np = arrays.get("V_star")
    v_star = torch.from_numpy(v_star_np[start:end]).to(device) if v_star_np is not None else None
    B_star_np = arrays.get("B_star")
    B_star = torch.from_numpy(B_star_np[start:end]).to(device) if B_star_np is not None else None
    return x, z, v_star, B_star


def slice_arrays(
    arrays: dict[str, np.ndarray | None],
    start: int,
    end: int,
) -> dict[str, np.ndarray | None]:
    return {
        key: value[start:end] if value is not None else None
        for key, value in arrays.items()
    }


def make_fixed_validation_starts(
    n_time: int,
    window_size: int,
    num_windows: int,
) -> list[int]:
    if n_time <= 0:
        return []
    if n_time <= window_size:
        return [0]
    max_start = n_time - window_size
    count = max(1, min(num_windows, max_start + 1))
    return np.linspace(0, max_start, count).round().astype(int).tolist()


def split_train_validation(
    arrays: dict[str, np.ndarray | None],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray | None], dict[str, np.ndarray | None] | None, list[int]]:
    if not bool(config.get("validation_enabled", True)):
        return arrays, None, []

    n_time = int(arrays["X"].shape[0])
    window_size = int(config.get("validation_window_size") or config.get("window_size", 1008))
    val_fraction = float(config.get("validation_fraction", 0.15))
    val_min_windows = int(config.get("validation_min_windows", 2))
    val_num_windows = int(config.get("validation_num_windows", 8))

    min_val_len = min(n_time, max(window_size, val_min_windows * window_size))
    val_len = max(min_val_len, int(round(n_time * val_fraction)))
    val_len = min(val_len, n_time)
    train_end = max(0, n_time - val_len)

    # Keep enough training data for at least one window. For tiny --limit runs,
    # validation uses the same short sequence rather than making training empty.
    if train_end < min(window_size, n_time):
        return arrays, arrays, make_fixed_validation_starts(n_time, window_size, val_num_windows)

    train_arrays = slice_arrays(arrays, 0, train_end)
    val_arrays = slice_arrays(arrays, train_end, n_time)
    starts = make_fixed_validation_starts(val_arrays["X"].shape[0], window_size, val_num_windows)
    return train_arrays, val_arrays, starts


def print_data_input_summary(data: dict[str, Any], split: str) -> None:
    configured = data.get("configured_measurement_path")
    resolved = data.get("measurement_path")
    if configured and resolved and configured != resolved:
        print(f"Using imputed {split} measurements: {resolved} (configured: {configured})")
    elif resolved:
        print(f"Using {split} measurements: {resolved}")
    target_summary = data.get("target_summary")
    if target_summary:
        print(
            f"{split} target finite values: "
            f"{target_summary['finite']}/{target_summary['total']} "
            f"({target_summary['finite_fraction']:.2%})"
        )


def multistep_horizons(config: dict[str, Any]) -> list[int]:
    raw = config.get("multistep_horizons", [3, 6, 12])
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(item) for item in raw]


def parse_int_list(raw: Any) -> list[int]:
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(item) for item in raw]


def lambda_multistep_for_stage(config: dict[str, Any], stage: str) -> float:
    stages = config.get("multistep_stages", ["joint"])
    if isinstance(stages, str):
        stages = [item.strip() for item in stages.split(",") if item.strip()]
    if stage not in stages:
        return 0.0
    return float(config.get("lambda_multistep", 0.0))


def training_loss_kwargs(config: dict[str, Any], stage: str) -> dict[str, Any]:
    prefix = f"{stage}_"
    horizons = config.get(f"{prefix}multistep_horizons")
    if horizons is None:
        horizons = multistep_horizons(config)
    return {
        "lambda_adv": float(config.get(f"{prefix}lambda_adv", config.get("lambda_adv", 0.1))),
        "lambda_deform": float(config.get(f"{prefix}lambda_deform", config.get("lambda_deform", 0.0))),
        "lambda_smooth": float(config.get(f"{prefix}lambda_smooth", config.get("lambda_smooth", 0.001))),
        "lambda_reg": float(config.get(f"{prefix}lambda_reg", config.get("lambda_reg", 0.0001))),
        "lambda_multistep": float(
            config.get(f"{prefix}lambda_multistep", lambda_multistep_for_stage(config, stage))
        ),
        "multistep_horizons": parse_int_list(horizons),
        "multistep_max_origins": int(
            config.get(f"{prefix}multistep_max_origins", config.get("multistep_max_origins", 256))
        ),
    }


def validation_losses(
    model: ModelLike,
    val_arrays: dict[str, np.ndarray | None],
    val_starts: list[int],
    coords: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    if not val_arrays or not val_starts:
        return {}

    base_model = unwrap_model(model)
    base_model.eval()
    window_size = int(config.get("validation_window_size") or config.get("window_size", 1008))
    sums: dict[str, float] = {}
    with torch.no_grad():
        for start in val_starts:
            end = min(start + window_size, val_arrays["X"].shape[0])
            x = torch.from_numpy(val_arrays["X"][start:end]).to(device)
            z = torch.from_numpy(val_arrays["Z"][start:end]).to(device)
            v_star_np = val_arrays.get("V_star")
            v_star = torch.from_numpy(v_star_np[start:end]).to(device) if v_star_np is not None else None
            B_star_np = val_arrays.get("B_star")
            B_star = torch.from_numpy(B_star_np[start:end]).to(device) if B_star_np is not None else None
            losses = base_model.training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                B_star=B_star,
                **training_loss_kwargs(config, "joint"),
            )
            for key, value in losses.items():
                if key.startswith("loss"):
                    sums[f"val_{key}"] = sums.get(f"val_{key}", 0.0) + float(value.detach().cpu())

    denom = max(len(val_starts), 1)
    return {key: value / denom for key, value in sums.items()}


def run_epoch(
    model: ModelLike,
    arrays: dict[str, np.ndarray | None],
    coords: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    stage: str,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    configure_stage(model, stage)
    base_model = unwrap_model(model)
    steps = int(config.get("steps_per_epoch", 50))
    window_size = int(config.get("window_size", 1008))
    grad_clip = float(config.get("grad_clip", 1.0))
    sums: dict[str, float] = {}

    for _ in range(steps):
        x, z, v_star, B_star = sample_window(arrays, window_size, device)
        optimizer.zero_grad(set_to_none=True)

        if isinstance(model, (VectorMIDETrainingModule, DistributedDataParallel)):
            losses = model(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                B_star=B_star,
                stage=stage,
                loss_kwargs=training_loss_kwargs(config, stage),
            )
        else:
            losses = VectorMIDETrainingModule(base_model)(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                B_star=B_star,
                stage=stage,
                loss_kwargs=training_loss_kwargs(config, stage),
            )
        loss = losses["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key, value in losses.items():
            if key.startswith("loss"):
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())

    denom = max(steps, 1)
    return {key: value / denom for key, value in sums.items()}


def save_checkpoint(
    model: ModelLike,
    config: dict[str, Any],
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": unwrap_model(model).state_dict(), "config": config}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def checkpoint_name_with_suffix(filename: str, suffix: str) -> str:
    path = Path(filename)
    return f"{path.stem}{suffix}{path.suffix or '.pt'}"


def checkpoint_score(metrics: dict[str, float], monitor: str) -> float:
    if monitor in metrics:
        return float(metrics[monitor])
    if "loss_kf" in metrics:
        return float(metrics["loss_kf"])
    return float(metrics["loss"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VectorMIDE offline.")
    parser.add_argument("--config", default="yml_files/VectorMIDE.yaml")
    parser.add_argument("--device", default=None, help="Override config device: auto, cpu, mps, cuda, cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional time limit for quick runs.")
    parser.add_argument("--dry-run", action="store_true", help="Only run one forward/loss pass.")
    args = parser.parse_args()

    config = load_config(args.config)
    device_name = args.device if args.device is not None else config.get("device", "auto")
    distributed, local_rank, world_size, device = setup_distributed(config, str(device_name))
    set_seed(int(config.get("seed", 123)) + local_rank)
    if distributed:
        rank_zero_print(f"Distributed training enabled: world_size={world_size}")
    if is_main_process():
        print_device_info(device)

    data = load_vector_dataset(config, split="offline", time_limit=args.limit)
    if is_main_process():
        print_data_input_summary(data, "offline")

    arrays = {"X": data["X"], "Z": data["Z"], "V_star": data["V_star"], "B_star": data.get("B_star")}
    train_arrays, val_arrays, val_starts = split_train_validation(arrays, config)
    if val_arrays is not None and is_main_process():
        rank_zero_print(
            "Validation enabled: "
            f"train_T={train_arrays['X'].shape[0]}, "
            f"val_T={val_arrays['X'].shape[0]}, "
            f"val_windows={len(val_starts)}, "
            f"val_every={int(config.get('validation_every_epochs', 5))} epoch(s)"
        )
    coords = torch.from_numpy(data["coords"]).to(device)
    base_model = build_model(config).to(device)
    train_model: ModelLike = VectorMIDETrainingModule(base_model)
    if distributed:
        train_model = DistributedDataParallel(
            train_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    optimizer = build_optimizer(train_model, config)

    if args.dry_run:
        x, z, v_star, B_star = sample_window(arrays, min(16, arrays["X"].shape[0]), device)
        with torch.enable_grad():
            losses = unwrap_model(train_model).training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                B_star=B_star,
                **training_loss_kwargs(config, "joint"),
            )
        if is_main_process():
            print({key: float(value.detach().cpu()) for key, value in losses.items() if key.startswith("loss")})
        cleanup_distributed(distributed)
        return

    schedule = [
        ("adv", int(config.get("offline_epochs_pretrain_adv", 0))),
        ("kf", int(config.get("offline_epochs_kf", 0))),
        ("joint", int(config.get("offline_epochs_finetune", 0))),
    ]
    active_stages = [stage for stage, epochs in schedule if epochs > 0]
    monitor_stage = str(config.get("checkpoint_stage", active_stages[-1] if active_stages else "joint"))
    monitor_metric = str(config.get("checkpoint_metric", "val_loss_kf"))
    validation_every = int(config.get("validation_every_epochs", 5))
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    best_ckpt_path = ckpt_dir / config.get("offline_checkpoint_name", "vector_mide_offline.pt")
    last_ckpt_path = ckpt_dir / config.get(
        "last_offline_checkpoint_name",
        checkpoint_name_with_suffix(best_ckpt_path.name, "_last"),
    )
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_info: dict[str, Any] | None = None

    try:
        for stage, epochs in schedule:
            for epoch in range(epochs):
                metrics = run_epoch(train_model, train_arrays, coords, optimizer, config, stage, device)
                if (
                    val_arrays is not None
                    and stage == monitor_stage
                    and validation_every > 0
                    and ((epoch + 1) % validation_every == 0 or epoch + 1 == epochs)
                    and is_main_process()
                ):
                    metrics.update(validation_losses(train_model, val_arrays, val_starts, coords, config, device))
                if distributed:
                    dist.barrier()
                if is_main_process():
                    rank_zero_print(f"{stage} epoch {epoch + 1}/{epochs}: {metrics}")
                    record = {"stage": stage, "epoch": epoch + 1, "epochs": epochs, **metrics}
                    history.append(record)

                    if stage == monitor_stage and monitor_metric in metrics:
                        score = checkpoint_score(metrics, monitor_metric)
                        if score < best_score:
                            best_score = score
                            best_info = {
                                "stage": stage,
                                "epoch": epoch + 1,
                                "score": score,
                                "monitor_metric": monitor_metric,
                                "metrics": metrics,
                            }
                            save_checkpoint(
                                train_model,
                                config,
                                best_ckpt_path,
                                extra={"best": best_info, "history": history},
                            )
                            rank_zero_print(
                                f"Saved best checkpoint to {best_ckpt_path} "
                                f"({monitor_stage}/{monitor_metric}={score:.6g})"
                            )

        if is_main_process():
            save_checkpoint(
                train_model,
                config,
                last_ckpt_path,
                extra={"best": best_info, "history": history},
            )
            if best_info is None:
                save_checkpoint(train_model, config, best_ckpt_path, extra={"history": history})
                rank_zero_print(f"Saved checkpoint to {best_ckpt_path}")
            rank_zero_print(f"Saved last checkpoint to {last_ckpt_path}")
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
