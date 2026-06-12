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
    build_covariance_proxy_arrays,
    build_model,
    checkpoint_name_with_suffix,
    print_data_input_summary,
    print_device_info,
    resolve_device,
    save_checkpoint,
    set_seed,
    training_loss_kwargs,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_compatible_state_dict(model: VectorMIDE, checkpoint: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    state = checkpoint["model_state"]
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), list(unexpected), skipped


def configure_online_trainable(
    model: VectorMIDE,
    update_ell: bool = True,
    scope: str = "full_head",
) -> None:
    scope = scope.replace("-", "_").lower()
    for param in model.parameters():
        param.requires_grad = False
    if scope == "full_head":
        for param in model.net.head_parameters():
            param.requires_grad = True
    elif scope == "output_head":
        for param in model.net.output_head_parameters():
            param.requires_grad = True
    elif scope == "ide_only":
        pass
    else:
        raise ValueError("online adaptation scope must be full_head, output_head, or ide_only")
    for param in model.qr_params.parameters():
        param.requires_grad = True
    if update_ell:
        model.kernel.raw_ell.requires_grad = True
        if getattr(model.kernel, "learnable_gamma", False):
            model.kernel.raw_gamma.requires_grad = True
        if getattr(model.kernel, "learnable_sigma_scale", False):
            model.kernel.raw_sigma_scale.requires_grad = True


def trainable_parameter_summary(model: VectorMIDE) -> str:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return f"trainable parameters: {trainable:,}/{total:,} ({trainable / max(total, 1):.2%})"


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


def parameter_drift_summary(
    current: dict[str, torch.Tensor],
    anchor: dict[str, torch.Tensor],
) -> dict[str, float]:
    if not current:
        return {}
    sq_delta = 0.0
    sq_anchor = 0.0
    max_abs = 0.0
    for name, param in current.items():
        if name not in anchor:
            continue
        base = anchor[name].to(device=param.device, dtype=param.dtype)
        delta = (param.detach() - base).float()
        sq_delta += float(delta.pow(2).sum().detach().cpu())
        sq_anchor += float(base.detach().float().pow(2).sum().detach().cpu())
        max_abs = max(max_abs, float(delta.abs().max().detach().cpu()))
    return {
        "param_drift_l2": math.sqrt(sq_delta),
        "param_drift_relative_l2": math.sqrt(sq_delta) / max(math.sqrt(sq_anchor), 1.0e-12),
        "param_drift_max_abs": max_abs,
    }


def build_online_optimizer(model: VectorMIDE, config: dict[str, Any]) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=float(config.get("online_lr", config.get("lr_heads", 5.0e-4))),
        weight_decay=float(config.get("online_weight_decay", config.get("weight_decay", 1.0e-4))),
    )


def window_tensors(
    data: dict[str, Any],
    start: int,
    end: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    x = torch.from_numpy(data["X"][start:end]).to(device)
    z = torch.from_numpy(data["Z"][start:end]).to(device)
    nwp = data.get("nwp_baseline")
    nwp_baseline = torch.from_numpy(nwp[start:end]).to(device) if nwp is not None else None
    v_star = data["V_star"]
    v = torch.from_numpy(v_star[start:end]).to(device) if v_star is not None else None
    anchor_np = data.get("A_anchor")
    advection_anchor = torch.from_numpy(anchor_np[start:end]).to(device) if anchor_np is not None else None
    B_star = data.get("B_star")
    B = torch.from_numpy(B_star[start:end]).to(device) if B_star is not None else None
    covariance_proxy_np = data.get("covariance_proxy")
    covariance_proxy = torch.from_numpy(covariance_proxy_np[start:end]).to(device) if covariance_proxy_np is not None else None
    covariance_proxy_primary_np = data.get("covariance_proxy_primary")
    covariance_proxy_primary = (
        torch.from_numpy(covariance_proxy_primary_np[start:end]).to(device)
        if covariance_proxy_primary_np is not None
        else None
    )
    covariance_proxy_secondary_np = data.get("covariance_proxy_secondary")
    covariance_proxy_secondary = (
        torch.from_numpy(covariance_proxy_secondary_np[start:end]).to(device)
        if covariance_proxy_secondary_np is not None
        else None
    )
    return (
        x,
        z,
        nwp_baseline,
        v,
        advection_anchor,
        B,
        covariance_proxy,
        covariance_proxy_primary,
        covariance_proxy_secondary,
    )


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
        (
            x,
            z,
            nwp_baseline,
            v_star,
            advection_anchor,
            B_star,
            covariance_proxy,
            covariance_proxy_primary,
            covariance_proxy_secondary,
        ) = window_tensors(data, start, end, device)
        losses = model.training_losses(
            x=x,
            z=z,
            coords=coords,
            v_star=v_star,
            advection_anchor=advection_anchor,
            B_star=B_star,
            nwp_baseline=nwp_baseline,
            covariance_proxy=covariance_proxy,
            covariance_proxy_primary=covariance_proxy_primary,
            covariance_proxy_secondary=covariance_proxy_secondary,
            **training_loss_kwargs(config, "online"),
        )
    return {
        f"val_{key}": float(value.detach().cpu())
        for key, value in losses.items()
        if key.startswith("loss")
    }


def evaluate_forecast_metrics(
    model: VectorMIDE,
    data: dict[str, Any],
    start: int,
    end: int,
    coords: torch.Tensor,
    forecast_horizon: int,
    device: torch.device,
) -> dict[str, float]:
    """Validation metrics aligned with multi-step forecast evaluation.

    The metric is computed in model-target space. For residual-NWP targets this
    is equivalent to measurement-space error because the same future NWP
    baseline is added to both prediction and target.
    """
    if end - start < 2 or forecast_horizon < 1:
        return {}
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(data["X"][start:end]).to(device)
        z = torch.from_numpy(data["Z"][start:end]).to(device)
        anchor_np = data.get("A_anchor")
        advection_anchor = torch.from_numpy(anchor_np[start:end]).to(device) if anchor_np is not None else None
        outputs = model(x, coords, advection_anchor=advection_anchor)
        control_seq = outputs.get("transition_control")
        kf = model.dstm.kalman_filter(
            z=z,
            M_seq=outputs["M"],
            control_seq=control_seq,
            return_history=True,
        )

        max_horizon = min(int(forecast_horizon), z.shape[0] - 1)
        rmse_by_horizon: list[torch.Tensor] = []
        mae_by_horizon: list[torch.Tensor] = []
        for horizon in range(1, max_horizon + 1):
            origins = torch.arange(0, z.shape[0] - horizon, device=device)
            if origins.numel() == 0:
                continue
            mean = kf["filter_means"][origins]
            for step in range(1, horizon + 1):
                M_batch = outputs["M"][origins + step]
                mean = torch.bmm(M_batch, mean.unsqueeze(-1)).squeeze(-1)
                if control_seq is not None:
                    mean = mean + control_seq[origins + step]
            target = z[origins + horizon]
            mask = torch.isfinite(mean) & torch.isfinite(target)
            if not mask.any():
                continue
            err = mean[mask] - target[mask]
            rmse_by_horizon.append(torch.sqrt(err.pow(2).mean()))
            mae_by_horizon.append(err.abs().mean())

    if not rmse_by_horizon:
        return {}
    rmse = torch.stack(rmse_by_horizon)
    mae = torch.stack(mae_by_horizon)
    return {
        "val_forecast_rmse_mean": float(rmse.mean().detach().cpu()),
        "val_forecast_mae_mean": float(mae.mean().detach().cpu()),
        "val_forecast_rmse_horizon": float(rmse[-1].detach().cpu()),
        "val_forecast_mae_horizon": float(mae[-1].detach().cpu()),
        "val_forecast_horizon": float(max_horizon),
    }


def evaluate_online_validation(
    model: VectorMIDE,
    data: dict[str, Any],
    start: int,
    end: int,
    coords: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    forecast_horizon: int,
) -> dict[str, float]:
    metrics = evaluate_window_loss(model, data, start, end, coords, config, device)
    metrics.update(
        evaluate_forecast_metrics(
            model,
            data,
            start,
            end,
            coords,
            forecast_horizon=forecast_horizon,
            device=device,
        )
    )
    return metrics


def metric_value(metrics: dict[str, float], name: str) -> float | None:
    if name in metrics:
        return float(metrics[name])
    if "val_forecast_rmse_mean" in metrics:
        return float(metrics["val_forecast_rmse_mean"])
    if "val_loss_kf" in metrics:
        return float(metrics["val_loss_kf"])
    if "val_loss" in metrics:
        return float(metrics["val_loss"])
    return None


def online_update_checkpoint_path(base_path: Path, update_idx: int, end: int) -> Path:
    return (
        base_path.parent
        / "online_updates"
        / f"{base_path.stem}_update{update_idx:05d}_end{end:06d}{base_path.suffix or '.pt'}"
    )


def loss_record(losses: dict[str, torch.Tensor], **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = dict(extra)
    for key in (
        "loss",
        "loss_forecast",
        "loss_kf",
        "loss_adv",
        "loss_deform",
        "loss_smooth",
        "loss_reg",
        "loss_multistep",
    ):
        if key in losses:
            record[key] = float(losses[key].detach().cpu())
    return record


def train_one_window(
    model: VectorMIDE,
    data: dict[str, Any],
    start: int,
    end: int,
    coords: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    anchor: dict[str, torch.Tensor],
    lambda_anchor: float,
    grad_clip: float,
) -> dict[str, torch.Tensor]:
    (
        x,
        z,
        nwp_baseline,
        v_star,
        advection_anchor,
        B_star,
        covariance_proxy,
        covariance_proxy_primary,
        covariance_proxy_secondary,
    ) = window_tensors(data, start, end, device)
    optimizer.zero_grad(set_to_none=True)
    losses = model.training_losses(
        x=x,
        z=z,
        coords=coords,
        v_star=v_star,
        advection_anchor=advection_anchor,
            B_star=B_star,
            nwp_baseline=nwp_baseline,
            covariance_proxy=covariance_proxy,
            covariance_proxy_primary=covariance_proxy_primary,
            covariance_proxy_secondary=covariance_proxy_secondary,
            **training_loss_kwargs(config, "online"),
        )
    current = trainable_named_parameters(model)
    loss = losses["loss"] + lambda_anchor * anchor_loss(current, anchor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [param for param in model.parameters() if param.requires_grad],
        grad_clip,
    )
    optimizer.step()
    return losses


def run_global_finetune(
    model: VectorMIDE,
    data: dict[str, Any],
    coords: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    anchor: dict[str, torch.Tensor],
    best_path: Path,
    initial_path: Path,
    last_path: Path,
    steps: int,
    window_size: int,
    validation_every: int,
    validation_window: int,
    validation_forecast_horizon: int,
    monitor_metric: str,
    lambda_anchor: float,
    grad_clip: float,
    require_improvement_over_initial: bool,
) -> None:
    rng = np.random.default_rng(int(config.get("seed", 123)))
    T = int(data["X"].shape[0])
    if T <= 0:
        raise ValueError("Online dataset is empty.")
    window_size = min(window_size, T)
    max_start = max(T - window_size, 0)
    metrics: list[dict[str, Any]] = []
    best_score = math.inf
    best_info: dict[str, Any] | None = None
    best_source = "none"
    val_end = T
    val_start = max(0, val_end - min(validation_window, T))

    print(
        "Global online finetune: "
        f"T={T}, window_size={window_size}, steps={steps}, "
        f"validation_every={validation_every}"
    )
    if validation_every > 0:
        initial_record = {"step": 0, "train_start": None, "end": None}
        initial_record.update(
            evaluate_online_validation(
                model,
                data,
                val_start,
                val_end,
                coords,
                config,
                device,
                validation_forecast_horizon,
            )
        )
        initial_score = metric_value(initial_record, monitor_metric)
        print({"initial_baseline": True, **initial_record})
        initial_info = {
            "step": 0,
            "score": initial_score,
            "monitor_metric": monitor_metric,
            "validation_start": val_start,
            "validation_end": val_end,
            "metrics": initial_record,
            "initial_baseline": True,
            "best_source": "initial",
        }
        save_checkpoint(
            model,
            config,
            initial_path,
            extra={
                "initial": initial_info,
                "metrics": [initial_record],
                "mode": "global_finetune",
                "checkpoint_role": "initial_offline_baseline",
            },
        )
        print(f"Saved explicit initial online baseline checkpoint to {initial_path}")
        if require_improvement_over_initial and initial_score is not None:
            best_score = initial_score
            best_info = initial_info
            best_source = "initial"
            save_checkpoint(
                model,
                config,
                best_path,
                extra={
                    "best": best_info,
                    "metrics": [initial_record],
                    "mode": "global_finetune",
                    "checkpoint_role": "best",
                    "best_source": best_source,
                    "initial_checkpoint": str(initial_path),
                },
            )
            print(
                f"Saved initial baseline as current best global online checkpoint to {best_path} "
                f"({monitor_metric}={initial_score:.6g})"
            )
        metrics.append(initial_record)

    for step in range(1, steps + 1):
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end = start + window_size
        model.train()
        losses = train_one_window(
            model,
            data,
            start,
            end,
            coords,
            config,
            device,
            optimizer,
            anchor,
            lambda_anchor,
            grad_clip,
        )
        record = loss_record(losses, step=step, train_start=start, end=end)

        should_validate = validation_every > 0 and (step % validation_every == 0 or step == steps)
        if should_validate:
            record.update(
                evaluate_online_validation(
                    model,
                    data,
                    val_start,
                    val_end,
                    coords,
                    config,
                    device,
                    validation_forecast_horizon,
                )
            )
            score = metric_value(record, monitor_metric)
            if score is not None and score < best_score:
                best_score = score
                best_info = {
                    "step": step,
                    "score": score,
                    "monitor_metric": monitor_metric,
                    "validation_start": val_start,
                    "validation_end": val_end,
                    "metrics": record,
                    "best_source": "trained_online",
                }
                best_source = "trained_online"
                save_checkpoint(
                    model,
                    config,
                    best_path,
                    extra={
                        "best": best_info,
                        "metrics": metrics + [record],
                        "mode": "global_finetune",
                        "checkpoint_role": "best",
                        "best_source": best_source,
                        "initial_checkpoint": str(initial_path),
                    },
                )
                print(f"Saved best global online checkpoint to {best_path} ({monitor_metric}={score:.6g})")
            elif score is not None and require_improvement_over_initial:
                record["did_not_beat_initial_baseline"] = 1.0
            record.update(parameter_drift_summary(trainable_named_parameters(model), anchor))

        metrics.append(record)
        if step == 1 or step == steps or (validation_every > 0 and step % validation_every == 0):
            print(record)

    save_checkpoint(
        model,
        config,
        last_path,
        extra={
            "best": best_info,
            "metrics": metrics,
            "mode": "global_finetune",
            "checkpoint_role": "last_trained_online",
            "best_source": best_source,
            "initial_checkpoint": str(initial_path),
            "param_drift": parameter_drift_summary(trainable_named_parameters(model), anchor),
        },
    )
    if best_info is None:
        save_checkpoint(model, config, best_path, extra={"metrics": metrics, "mode": "global_finetune"})
        print(f"Saved global online checkpoint to {best_path}")
    print(f"Saved last global online checkpoint to {last_path}")
    if best_source == "initial":
        print(
            "No trained online checkpoint beat the initial offline baseline on the selected "
            f"metric ({monitor_metric}). Evaluate the *_last.pt checkpoint if you want to inspect "
            "the trained-but-not-best model."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling online VectorMIDE adaptation.")
    parser.add_argument("--config", default="yml_files/VectorMIDE.yaml")
    parser.add_argument("--checkpoint", default=None, help="Offline checkpoint path.")
    parser.add_argument("--device", default=None, help="Override config device: auto, cpu, mps, cuda, cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional online time limit.")
    parser.add_argument("--no-update-ell", action="store_true")
    parser.add_argument("--online-window-size", type=int, default=None, help="Override online rolling train window size.")
    parser.add_argument("--online-update-every", type=int, default=None, help="Override online update stride.")
    parser.add_argument("--online-steps", type=int, default=None, help="Override gradient steps per online update.")
    parser.add_argument("--online-validation-every-updates", type=int, default=None, help="Override validation frequency; 0 disables validation.")
    parser.add_argument("--online-validation-forecast-horizon", type=int, default=None, help="Override forecast horizon used for online checkpoint selection.")
    parser.add_argument("--online-checkpoint-metric", default=None, help="Override online checkpoint selection metric.")
    parser.add_argument("--global-finetune", action="store_true", help="Train on random windows from the full online split for a fixed number of steps.")
    parser.add_argument("--global-steps", type=int, default=None, help="Total optimizer steps for --global-finetune.")
    parser.add_argument("--adaptation-scope", choices=["full-head", "output-head", "ide-only"], default=None, help="Online trainable parameter set.")
    parser.add_argument("--lambda-multistep", type=float, default=None, help="Override lambda_multistep for online training.")
    parser.add_argument("--lambda-adv", type=float, default=None, help="Override lambda_adv for online training.")
    parser.add_argument("--lambda-deform", type=float, default=None, help="Override lambda_deform for online training.")
    parser.add_argument("--lambda-smooth", type=float, default=None, help="Override lambda_smooth for online training.")
    parser.add_argument("--online-lr", type=float, default=None, help="Override online learning rate.")
    parser.add_argument("--allow-worse-than-initial", action="store_true", help="Allow global online best checkpoint to be worse than the initial offline checkpoint on the validation window.")
    parser.add_argument("--save-every-update", action="store_true", help="Save one online checkpoint after every rolling update.")
    parser.add_argument("--online-checkpoint-every", type=int, default=None, help="Save an online checkpoint every N updates; 0 disables update checkpoints.")
    parser.add_argument("--max-updates", type=int, default=None, help="Stop after this many online updates.")
    args = parser.parse_args()

    config = load_config(args.config)
    for key, value in (
        ("online_lambda_multistep", args.lambda_multistep),
        ("online_lambda_adv", args.lambda_adv),
        ("online_lambda_deform", args.lambda_deform),
        ("online_lambda_smooth", args.lambda_smooth),
        ("online_lr", args.online_lr),
    ):
        if value is not None:
            config[key] = value
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
    missing, unexpected, skipped = load_compatible_state_dict(model, checkpoint)
    if missing:
        print(f"Initialized new online parameters not found in checkpoint: {missing}")
    if unexpected:
        print(f"Ignored checkpoint parameters not used by this config: {unexpected}")
    if skipped:
        print(f"Skipped checkpoint parameters with incompatible shapes: {skipped}")
    adaptation_scope = str(args.adaptation_scope or config.get("online_adaptation_scope", "full_head"))
    update_ell = bool(config.get("online_update_ell", config.get("learnable_ell", True))) and not args.no_update_ell
    configure_online_trainable(model, update_ell=update_ell, scope=adaptation_scope)
    print(f"Online adaptation scope: {adaptation_scope}; {trainable_parameter_summary(model)}")
    anchor = {name: param.detach().cpu().clone() for name, param in trainable_named_parameters(model).items()}
    optimizer = build_online_optimizer(model, config)

    data = load_vector_dataset(config, split="online", time_limit=args.limit)
    data.update(build_covariance_proxy_arrays(data, config))
    print_data_input_summary(data, "online")

    coords = torch.from_numpy(data["coords"]).to(device)
    window_size = int(args.online_window_size or config.get("online_window_size", 168))
    update_every = int(args.online_update_every or config.get("online_update_every", 6))
    online_steps = int(args.online_steps or config.get("online_steps", 10))
    online_val_window = int(config.get("online_validation_window_size", window_size))
    online_val_every = int(
        config.get("online_validation_every_updates", 10)
        if args.online_validation_every_updates is None
        else args.online_validation_every_updates
    )
    online_val_gap = int(config.get("online_validation_gap", 0))
    online_monitor_metric = str(args.online_checkpoint_metric or config.get("online_checkpoint_metric", "val_loss_kf"))
    online_val_forecast_horizon = int(
        args.online_validation_forecast_horizon
        or config.get("online_validation_forecast_horizon", config.get("forecast_horizon", 1))
    )
    online_early_stop_patience = int(config.get("online_early_stop_patience", 0))
    online_min_delta = float(config.get("online_min_delta", 0.0))
    lambda_anchor = float(config.get("lambda_anchor", 0.01))
    grad_clip = float(config.get("grad_clip", 1.0))
    if args.save_every_update:
        online_checkpoint_every = 1
    elif args.online_checkpoint_every is not None:
        online_checkpoint_every = int(args.online_checkpoint_every)
    else:
        online_checkpoint_every = int(config.get("online_checkpoint_every_updates", 0))

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

    if args.global_finetune:
        global_best_path = ckpt_dir / config.get(
            "global_online_checkpoint_name",
            checkpoint_name_with_suffix(best_path.name, "_global"),
        )
        global_initial_path = ckpt_dir / config.get(
            "initial_global_online_checkpoint_name",
            checkpoint_name_with_suffix(global_best_path.name, "_initial"),
        )
        global_last_path = ckpt_dir / config.get(
            "last_global_online_checkpoint_name",
            checkpoint_name_with_suffix(global_best_path.name, "_last"),
        )
        run_global_finetune(
            model=model,
            data=data,
            coords=coords,
            config=config,
            device=device,
            optimizer=optimizer,
            anchor=anchor,
            best_path=global_best_path,
            initial_path=global_initial_path,
            last_path=global_last_path,
            steps=int(args.global_steps or config.get("online_global_steps", online_steps)),
            window_size=window_size,
            validation_every=online_val_every,
            validation_window=online_val_window,
            validation_forecast_horizon=online_val_forecast_horizon,
            monitor_metric=online_monitor_metric,
            lambda_anchor=lambda_anchor,
            grad_clip=grad_clip,
            require_improvement_over_initial=bool(
                config.get("online_require_improvement_over_initial", True)
            )
            and not args.allow_worse_than_initial,
        )
        return

    for update_idx, end in enumerate(range(window_size, T + 1, update_every), start=1):
        if args.max_updates is not None and update_idx > args.max_updates:
            break
        start = end - window_size
        model.train()
        for _ in range(online_steps):
            losses = train_one_window(
                model,
                data,
                start,
                end,
                coords,
                config,
                device,
                optimizer,
                anchor,
                lambda_anchor,
                grad_clip,
            )

        record = loss_record(losses, update=update_idx, train_start=start, end=end)

        should_validate = online_val_every > 0 and (
            update_idx % online_val_every == 0 or end + online_val_gap + online_val_window >= T
        )
        if should_validate:
            val_start = end + online_val_gap
            val_end = min(val_start + online_val_window, T)
            if val_end > val_start:
                record.update(
                    evaluate_online_validation(
                        model,
                        data,
                        val_start,
                        val_end,
                        coords,
                        config,
                        device,
                        online_val_forecast_horizon,
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
        if online_checkpoint_every > 0 and update_idx % online_checkpoint_every == 0:
            update_path = online_update_checkpoint_path(best_path, update_idx, end)
            record["checkpoint_path"] = str(update_path)
            save_checkpoint(
                model,
                config,
                update_path,
                extra={
                    "update": update_idx,
                    "end": end,
                    "metrics": metrics,
                    "best": best_info,
                },
            )
            print(f"Saved online update checkpoint to {update_path}")
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
