from __future__ import annotations

import argparse
import math
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
from model.vector_dstm import VectorMIDE
from train.train_vector_offline import (
    build_model,
    checkpoint_name_with_suffix,
    print_device_info,
    resolve_device,
    save_checkpoint,
    set_seed,
    training_loss_kwargs,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_online_trainable(model: VectorMIDE, update_ell: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.net.head_parameters():
        param.requires_grad = True
    for param in model.qr_params.parameters():
        param.requires_grad = True
    if update_ell:
        model.kernel.raw_ell.requires_grad = True
        if getattr(model.kernel, "learnable_gamma", False):
            model.kernel.raw_gamma.requires_grad = True


def trainable_named_parameters(model: VectorMIDE) -> dict[str, torch.Tensor]:
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def anchor_loss(
    current: dict[str, torch.Tensor],
    anchor: dict[str, torch.Tensor],
) -> torch.Tensor:
    first = next(iter(current.values()))
    loss = first.new_tensor(0.0)
    for name, param in current.items():
        if name in anchor:
            loss = loss + (param - anchor[name].to(param.device, param.dtype)).pow(2).mean()
    return loss


def build_online_optimizer(model: VectorMIDE, config: dict[str, Any]) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=float(config.get("lr_heads", 5.0e-4)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )


def window_tensors(
    data: dict[str, Any],
    start: int,
    end: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = torch.from_numpy(data["X"][start:end]).to(device)
    z = torch.from_numpy(data["Z"][start:end]).to(device)
    v_star = data["V_star"]
    v = torch.from_numpy(v_star[start:end]).to(device) if v_star is not None else None
    return x, z, v


def evaluate_window_loss(
    model: VectorMIDE,
    data: dict[str, Any],
    start: int,
    end: int,
    coords: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    if end <= start:
        return {}
    model.eval()
    with torch.no_grad():
        x, z, v_star = window_tensors(data, start, end, device)
        losses = model.training_losses(
            x=x,
            z=z,
            coords=coords,
            v_star=v_star,
            **training_loss_kwargs(config, "online"),
        )
    return {
        f"val_{key}": float(value.detach().cpu())
        for key, value in losses.items()
        if key.startswith("loss")
    }


def metric_value(metrics: dict[str, float], name: str) -> float | None:
    if name in metrics:
        return float(metrics[name])
    if "val_loss_kf" in metrics:
        return float(metrics["val_loss_kf"])
    if "val_loss" in metrics:
        return float(metrics["val_loss"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling online VectorMIDE adaptation.")
    parser.add_argument("--config", default="yml_files/VectorMIDE.yaml")
    parser.add_argument("--checkpoint", default=None, help="Offline checkpoint path.")
    parser.add_argument("--device", default=None, help="Override config device: auto, cpu, mps, cuda, cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional online time limit.")
    parser.add_argument("--no-update-ell", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 123)))
    device_name = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(
        device_name,
        allow_fallback=bool(config.get("allow_device_fallback", True)),
    )
    print_device_info(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = Path(config.get("checkpoint_dir", "checkpoints")) / config.get(
            "offline_checkpoint_name",
            "vector_mide_offline.pt",
        )
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    configure_online_trainable(model, update_ell=not args.no_update_ell)
    anchor = {name: param.detach().cpu().clone() for name, param in trainable_named_parameters(model).items()}
    optimizer = build_online_optimizer(model, config)

    data = load_vector_dataset(config, split="online", time_limit=args.limit)

    coords = torch.from_numpy(data["coords"]).to(device)
    window_size = int(config.get("online_window_size", 168))
    update_every = int(config.get("online_update_every", 6))
    online_steps = int(config.get("online_steps", 10))
    online_val_window = int(config.get("online_validation_window_size", window_size))
    online_val_every = int(config.get("online_validation_every_updates", 10))
    online_val_gap = int(config.get("online_validation_gap", 0))
    online_monitor_metric = str(config.get("online_checkpoint_metric", "val_loss_kf"))
    online_early_stop_patience = int(config.get("online_early_stop_patience", 0))
    online_min_delta = float(config.get("online_min_delta", 0.0))
    lambda_anchor = float(config.get("lambda_anchor", 0.01))
    grad_clip = float(config.get("grad_clip", 1.0))

    metrics = []
    best_score = math.inf
    best_info: dict[str, Any] | None = None
    bad_validations = 0
    T = data["X"].shape[0]
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    best_path = ckpt_dir / config.get("online_checkpoint_name", "vector_mide_online.pt")
    last_path = ckpt_dir / config.get(
        "last_online_checkpoint_name",
        checkpoint_name_with_suffix(best_path.name, "_last"),
    )

    for update_idx, end in enumerate(range(window_size, T + 1, update_every), start=1):
        start = end - window_size
        model.train()
        for _ in range(online_steps):
            x, z, v_star = window_tensors(data, start, end, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_losses(
                x=x,
                z=z,
                coords=coords,
                v_star=v_star,
                **training_loss_kwargs(config, "online"),
            )
            current = trainable_named_parameters(model)
            loss = losses["loss"] + lambda_anchor * anchor_loss(current, anchor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        record = {
            "update": update_idx,
            "train_start": start,
            "end": end,
            "loss": float(losses["loss"].detach().cpu()),
            "loss_forecast": float(losses["loss_forecast"].detach().cpu()),
            "loss_kf": float(losses["loss_kf"].detach().cpu()),
            "loss_adv": float(losses["loss_adv"].detach().cpu()),
            "loss_smooth": float(losses["loss_smooth"].detach().cpu()),
            "loss_reg": float(losses["loss_reg"].detach().cpu()),
            "loss_multistep": float(losses["loss_multistep"].detach().cpu()),
        }

        should_validate = online_val_every > 0 and (
            update_idx % online_val_every == 0 or end + online_val_gap + online_val_window >= T
        )
        if should_validate:
            val_start = end + online_val_gap
            val_end = min(val_start + online_val_window, T)
            if val_end > val_start:
                record.update(
                    evaluate_window_loss(
                        model,
                        data,
                        val_start,
                        val_end,
                        coords,
                        config,
                        device,
                    )
                )
                score = metric_value(record, online_monitor_metric)
                if score is not None and score < best_score - online_min_delta:
                    best_score = score
                    bad_validations = 0
                    best_info = {
                        "update": update_idx,
                        "end": end,
                        "validation_start": val_start,
                        "validation_end": val_end,
                        "score": score,
                        "monitor_metric": online_monitor_metric,
                        "metrics": record,
                    }
                    save_checkpoint(
                        model,
                        config,
                        best_path,
                        extra={"best": best_info, "metrics": metrics + [record]},
                    )
                    print(
                        f"Saved best online checkpoint to {best_path} "
                        f"({online_monitor_metric}={score:.6g}, val={val_start}:{val_end})"
                    )
                elif score is not None:
                    bad_validations += 1
            else:
                record["val_skipped"] = 1.0

        metrics.append(record)
        print(record)

        if online_early_stop_patience > 0 and bad_validations >= online_early_stop_patience:
            print(f"Early stopping online adaptation after {bad_validations} stale validations.")
            break

    save_checkpoint(
        model,
        config,
        last_path,
        extra={"best": best_info, "metrics": metrics},
    )
    if best_info is None:
        save_checkpoint(model, config, best_path, extra={"metrics": metrics})
        print(f"Saved online checkpoint to {best_path}")
    print(f"Saved last online checkpoint to {last_path}")


if __name__ == "__main__":
    main()
