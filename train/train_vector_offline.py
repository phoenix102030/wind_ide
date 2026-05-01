from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.vector_data_utils import load_vector_dataset
from model.covariance import advection_nll_loss, smoothness_loss
from model.vector_dstm import VectorMIDE


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
    )


def build_optimizer(model: VectorMIDE, config: dict[str, Any]) -> torch.optim.Optimizer:
    groups = [
        {"params": model.net.backbone.parameters(), "lr": float(config["lr_cnn"])},
        {"params": list(model.net.head_parameters()), "lr": float(config["lr_heads"])},
        {"params": model.kernel.parameters(), "lr": float(config["lr_kernel"])},
        {"params": model.qr_params.parameters(), "lr": float(config["lr_qr"])},
    ]
    return torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 1.0e-4)))


def set_module_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def configure_stage(model: VectorMIDE, stage: str) -> None:
    set_module_grad(model, True)
    if stage == "kf":
        set_module_grad(model.net, False)
    elif stage == "adv":
        set_module_grad(model.kernel, False)
        set_module_grad(model.qr_params, False)


def sample_window(
    arrays: dict[str, np.ndarray | None],
    window_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
    return x, z, v_star


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


def multistep_horizons(config: dict[str, Any]) -> list[int]:
    raw = config.get("multistep_horizons", [3, 6, 12])
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
    return {
        "lambda_adv": float(config.get("lambda_adv", 0.1)),
        "lambda_smooth": float(config.get("lambda_smooth", 0.001)),
        "lambda_reg": float(config.get("lambda_reg", 0.0001)),
        "lambda_multistep": lambda_multistep_for_stage(config, stage),
        "multistep_horizons": multistep_horizons(config),
        "multistep_max_origins": int(config.get("multistep_max_origins", 256)),
    }


def validation_losses(
    model: VectorMIDE,
    val_arrays: dict[str, np.ndarray | None],
    val_starts: list[int],
    coords: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    if not val_arrays or not val_starts:
        return {}

    model.eval()
    window_size = int(config.get("validation_window_size") or config.get("window_size", 1008))
    sums: dict[str, float] = {}
    with torch.no_grad():
        for start in val_starts:
            end = min(start + window_size, val_arrays["X"].shape[0])
            x = torch.from_numpy(val_arrays["X"][start:end]).to(device)
            z = torch.from_numpy(val_arrays["Z"][start:end]).to(device)
            v_star_np = val_arrays.get("V_star")
            v_star = torch.from_numpy(v_star_np[start:end]).to(device) if v_star_np is not None else None
            losses = model.training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                **training_loss_kwargs(config, "joint"),
            )
            for key, value in losses.items():
                if key.startswith("loss"):
                    sums[f"val_{key}"] = sums.get(f"val_{key}", 0.0) + float(value.detach().cpu())

    denom = max(len(val_starts), 1)
    return {key: value / denom for key, value in sums.items()}


def run_epoch(
    model: VectorMIDE,
    arrays: dict[str, np.ndarray | None],
    coords: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    stage: str,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    configure_stage(model, stage)
    steps = int(config.get("steps_per_epoch", 50))
    window_size = int(config.get("window_size", 1008))
    grad_clip = float(config.get("grad_clip", 1.0))
    sums: dict[str, float] = {}

    for _ in range(steps):
        x, z, v_star = sample_window(arrays, window_size, device)
        optimizer.zero_grad(set_to_none=True)

        if stage == "adv":
            if v_star is None:
                continue
            outputs = model.net(x)
            loss_adv = advection_nll_loss(v_star, outputs["mu"], outputs["Sigma"])
            loss_smooth = smoothness_loss(outputs["mu"], outputs["A"])
            loss = loss_adv + float(config.get("lambda_smooth", 0.001)) * loss_smooth
            losses = {"loss": loss, "loss_adv": loss_adv, "loss_smooth": loss_smooth}
        elif stage == "kf":
            losses = model.training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=None,
                lambda_adv=0.0,
                lambda_smooth=0.0,
                lambda_reg=float(config.get("lambda_reg", 0.0001)),
                lambda_multistep=lambda_multistep_for_stage(config, stage),
                multistep_horizons=multistep_horizons(config),
                multistep_max_origins=int(config.get("multistep_max_origins", 256)),
            )
            loss = losses["loss"]
        elif stage == "joint":
            losses = model.training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                **training_loss_kwargs(config, stage),
            )
            loss = losses["loss"]
        else:
            raise ValueError(f"Unknown stage: {stage}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key, value in losses.items():
            if key.startswith("loss"):
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())

    denom = max(steps, 1)
    return {key: value / denom for key, value in sums.items()}


def save_checkpoint(
    model: VectorMIDE,
    config: dict[str, Any],
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "config": config}
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
    set_seed(int(config.get("seed", 123)))
    device_name = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(
        device_name,
        allow_fallback=bool(config.get("allow_device_fallback", True)),
    )
    print_device_info(device)

    data = load_vector_dataset(config, split="offline", time_limit=args.limit)

    arrays = {"X": data["X"], "Z": data["Z"], "V_star": data["V_star"]}
    train_arrays, val_arrays, val_starts = split_train_validation(arrays, config)
    if val_arrays is not None:
        print(
            "Validation enabled: "
            f"train_T={train_arrays['X'].shape[0]}, "
            f"val_T={val_arrays['X'].shape[0]}, "
            f"val_windows={len(val_starts)}, "
            f"val_every={int(config.get('validation_every_epochs', 5))} epoch(s)"
        )
    coords = torch.from_numpy(data["coords"]).to(device)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)

    if args.dry_run:
        x, z, v_star = sample_window(arrays, min(16, arrays["X"].shape[0]), device)
        with torch.enable_grad():
            losses = model.training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                **training_loss_kwargs(config, "joint"),
            )
        print({key: float(value.detach().cpu()) for key, value in losses.items() if key.startswith("loss")})
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

    for stage, epochs in schedule:
        for epoch in range(epochs):
            metrics = run_epoch(model, train_arrays, coords, optimizer, config, stage, device)
            if (
                val_arrays is not None
                and stage == monitor_stage
                and validation_every > 0
                and ((epoch + 1) % validation_every == 0 or epoch + 1 == epochs)
            ):
                metrics.update(validation_losses(model, val_arrays, val_starts, coords, config, device))
            print(f"{stage} epoch {epoch + 1}/{epochs}: {metrics}")
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
                        model,
                        config,
                        best_ckpt_path,
                        extra={"best": best_info, "history": history},
                    )
                    print(
                        f"Saved best checkpoint to {best_ckpt_path} "
                        f"({monitor_stage}/{monitor_metric}={score:.6g})"
                    )

    save_checkpoint(
        model,
        config,
        last_ckpt_path,
        extra={"best": best_info, "history": history},
    )
    if best_info is None:
        save_checkpoint(model, config, best_ckpt_path, extra={"history": history})
        print(f"Saved checkpoint to {best_ckpt_path}")
    print(f"Saved last checkpoint to {last_ckpt_path}")


if __name__ == "__main__":
    main()
