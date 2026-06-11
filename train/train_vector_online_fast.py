from __future__ import annotations

import argparse
import json
import math
import sys
import time
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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def finite_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    mask = np.isfinite(err)
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "count": 0.0}
    values = err[mask]
    return {
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mae": float(np.mean(np.abs(values))),
        "bias": float(np.mean(values)),
        "count": float(values.size),
    }


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


def configure_fast_trainable(model: VectorMIDE, scope: str) -> None:
    scope = scope.replace("-", "_").lower()
    for param in model.parameters():
        param.requires_grad = False

    for param in model.qr_params.parameters():
        param.requires_grad = True

    if scope in {"qr", "qr_only"}:
        return

    if scope in {"qr_kernel", "kernel"}:
        model.kernel.raw_ell.requires_grad = True
        if getattr(model.kernel, "learnable_gamma", False):
            model.kernel.raw_gamma.requires_grad = True
        if getattr(model.kernel, "learnable_sigma_scale", False):
            model.kernel.raw_sigma_scale.requires_grad = True
        return

    if scope in {"stat_head", "statistical_head"}:
        model.kernel.raw_ell.requires_grad = True
        if getattr(model.kernel, "learnable_gamma", False):
            model.kernel.raw_gamma.requires_grad = True
        if getattr(model.kernel, "learnable_sigma_scale", False):
            model.kernel.raw_sigma_scale.requires_grad = True
        if getattr(model.net, "regime_chol_raw", None) is not None:
            model.net.regime_chol_raw.requires_grad = True
        scale_head = getattr(model.net, "covariance_scale_head", None)
        if scale_head is not None:
            for param in scale_head.parameters():
                param.requires_grad = True
        return

    raise ValueError("fast online scope must be qr, qr-kernel, or stat-head")


def trainable_parameter_summary(model: VectorMIDE) -> str:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return f"trainable parameters: {trainable:,}/{total:,} ({trainable / max(total, 1):.2%})"


def fast_loss_kwargs(config: dict[str, Any], lambda_multistep: float) -> dict[str, Any]:
    kwargs = training_loss_kwargs(config, "online")
    kwargs.update(
        {
            "lambda_adv": 0.0,
            "lambda_deform": 0.0,
            "lambda_smooth": 0.0,
            "lambda_advection_residual": 0.0,
            "lambda_multistep": float(lambda_multistep),
            "lambda_sigma_ratio": 0.0,
            "lambda_covariance_proxy": 0.0,
            "lambda_covariance_correlation": 0.0,
            "lambda_regime_usage": 0.0,
            "lambda_covariance_residual_nll": 0.0,
            "lambda_covariance_shape": 0.0,
            "lambda_calibration": 0.0,
        }
    )
    if lambda_multistep <= 0.0:
        kwargs["multistep_horizons"] = []
    return kwargs


def window_tensors(
    data: dict[str, Any],
    start: int,
    end: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x = torch.from_numpy(data["X"][start:end]).to(device)
    z = torch.from_numpy(data["Z"][start:end]).to(device)
    nwp_np = data.get("nwp_baseline")
    nwp = torch.from_numpy(nwp_np[start:end]).to(device) if nwp_np is not None else None
    anchor_np = data.get("A_anchor")
    anchor = torch.from_numpy(anchor_np[start:end]).to(device) if anchor_np is not None else None
    return x, z, nwp, anchor


def optimize_window(
    model: VectorMIDE,
    data: dict[str, Any],
    start: int,
    end: int,
    coords: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    steps: int,
    grad_clip: float,
    loss_kwargs: dict[str, Any],
) -> dict[str, float]:
    x, z, nwp, anchor = window_tensors(data, start, end, device)
    last_losses: dict[str, torch.Tensor] | None = None
    model.train()
    for _ in range(max(steps, 0)):
        optimizer.zero_grad(set_to_none=True)
        losses = model.training_losses(
            x=x,
            z=z,
            coords=coords,
            advection_anchor=anchor,
            nwp_baseline=nwp,
            **loss_kwargs,
        )
        losses["loss"].backward()
        trainable = [param for param in model.parameters() if param.requires_grad]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        last_losses = losses
    if last_losses is None:
        with torch.no_grad():
            last_losses = model.training_losses(
                x=x,
                z=z,
                coords=coords,
                advection_anchor=anchor,
                nwp_baseline=nwp,
                **loss_kwargs,
            )
    return {
        key: float(value.detach().cpu())
        for key, value in last_losses.items()
        if key.startswith("loss")
    }


def optimize_window_cached_qr(
    model: VectorMIDE,
    data: dict[str, Any],
    start: int,
    end: int,
    coords: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    steps: int,
    grad_clip: float,
    lambda_reg: float,
) -> dict[str, float]:
    x, z, _, anchor = window_tensors(data, start, end, device)
    model.eval()
    with torch.no_grad():
        outputs = model(x, coords, advection_anchor=anchor)
        M_seq = outputs["M"].detach()
        control_seq = outputs.get("transition_control")
        if control_seq is not None:
            control_seq = control_seq.detach()

    trainable = [param for param in model.parameters() if param.requires_grad]
    last_loss = None
    last_kf = None
    for _ in range(max(steps, 0)):
        optimizer.zero_grad(set_to_none=True)
        loss_kf = model.dstm.kalman_nll(z=z, M_seq=M_seq, control_seq=control_seq)
        loss_reg = z.new_tensor(0.0)
        for param in model.qr_params.parameters():
            if param.requires_grad:
                loss_reg = loss_reg + param.pow(2).mean()
        loss = loss_kf + float(lambda_reg) * loss_reg
        loss.backward()
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        last_loss = loss
        last_kf = loss_kf
    if last_loss is None:
        with torch.no_grad():
            last_kf = model.dstm.kalman_nll(z=z, M_seq=M_seq, control_seq=control_seq)
            last_loss = last_kf
    return {
        "loss": float(last_loss.detach().cpu()),
        "loss_forecast": float(last_kf.detach().cpu()),
        "loss_kf": float(last_kf.detach().cpu()),
    }


def forecast_from_window(
    model: VectorMIDE,
    data: dict[str, Any],
    train_start: int,
    train_end: int,
    forecast_end: int,
    coords: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_all, _, _, anchor_all = window_tensors(data, train_start, forecast_end, device)
    z_train = torch.from_numpy(data["Z"][train_start:train_end]).to(device)
    train_len = train_end - train_start
    horizon = forecast_end - train_end
    if horizon <= 0:
        raise ValueError("forecast_end must be greater than train_end")

    model.eval()
    with torch.no_grad():
        outputs = model(x_all, coords, advection_anchor=anchor_all)
        kf = model.dstm.kalman_filter(
            z=z_train,
            M_seq=outputs["M"][:train_len],
            control_seq=(
                outputs["transition_control"][:train_len]
                if outputs.get("transition_control") is not None
                else None
            ),
            return_history=True,
        )
        mean = kf["filter_means"][-1]
        cov = kf["filter_covs"][-1]
        preds = []
        for idx in range(horizon):
            future_idx = train_len + idx
            control = outputs.get("transition_control")
            future_control = None if control is None else control[future_idx]
            mean, cov = model.dstm.get_forecast_dist(
                mean,
                cov,
                outputs["M"][future_idx],
                future_control=future_control,
            )
            preds.append(mean)
    prediction = torch.stack(preds, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    target = data["Z"][train_end:forecast_end].astype(np.float32, copy=False)
    return prediction, target


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast rolling online VectorMIDE adaptation.")
    parser.add_argument("--config", default="yml_files/VectorMIDE_cuda.yaml")
    parser.add_argument("--checkpoint", default=None, help="Offline checkpoint path.")
    parser.add_argument("--device", default=None, help="Override config device: auto, cpu, mps, cuda, cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional online time limit.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scope", choices=["qr", "qr-kernel", "stat-head"], default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None, help="Rolling stride in 10-min steps.")
    parser.add_argument("--forecast-horizon", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-multistep", type=float, default=None)
    parser.add_argument("--max-rolls", type=int, default=None)
    parser.add_argument("--reset-each-roll", action="store_true")
    parser.add_argument("--use-current-config-architecture", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 123)))
    device_name = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(device_name, allow_fallback=bool(config.get("allow_device_fallback", True)))
    print_device_info(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = Path(config.get("checkpoint_dir", "checkpoints")) / config.get(
            "offline_checkpoint_name",
            "vector_mide_offline.pt",
        )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_config = config if args.use_current_config_architecture else checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    missing, unexpected, skipped = load_compatible_state_dict(model, checkpoint)
    if missing:
        print(f"Initialized parameters not found in checkpoint: {missing}")
    if unexpected:
        print(f"Ignored unexpected checkpoint parameters: {unexpected}")
    if skipped:
        print(f"Skipped incompatible checkpoint parameters: {skipped}")

    scope = str(args.scope or config.get("fast_online_scope", "qr"))
    configure_fast_trainable(model, scope=scope)
    print(f"Fast online scope: {scope}; {trainable_parameter_summary(model)}")

    lr = float(args.lr or config.get("fast_online_lr", config.get("online_lr", 5.0e-4)))
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=float(config.get("fast_online_weight_decay", 0.0)),
    )
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    data = load_vector_dataset(config, split="online", time_limit=args.limit)
    data.update(build_covariance_proxy_arrays(data, model_config))
    print_data_input_summary(data, "online")
    coords = torch.from_numpy(data["coords"]).to(device)

    T = int(data["X"].shape[0])
    window_size = int(args.window_size or config.get("fast_online_window_size", config.get("online_window_size", 1008)))
    stride = int(args.stride or config.get("fast_online_stride", config.get("online_update_every", 36)))
    forecast_horizon = int(
        args.forecast_horizon
        or config.get("fast_online_forecast_horizon", config.get("forecast_horizon", 12))
    )
    steps = int(args.steps or config.get("fast_online_steps", config.get("online_steps", 10)))
    grad_clip = float(config.get("grad_clip", 1.0))
    lambda_multistep = float(args.lambda_multistep if args.lambda_multistep is not None else config.get("fast_online_lambda_multistep", 0.0))
    loss_kwargs = fast_loss_kwargs(config, lambda_multistep=lambda_multistep)

    if T <= window_size + 1:
        raise ValueError(f"Online split is too short: T={T}, window_size={window_size}")
    output_dir = Path(args.output_dir or Path(config.get("checkpoint_dir", "outputs")) / "fast_online")
    output_dir.mkdir(parents=True, exist_ok=True)

    roll_records: list[dict[str, Any]] = []
    predictions = []
    targets = []
    origins = []
    horizons = []
    roll_ids = []
    start_time = time.perf_counter()
    max_end = T - 1
    roll_iter = enumerate(range(window_size, max_end + 1, stride), start=1)
    for roll_idx, train_end in roll_iter:
        if args.max_rolls is not None and roll_idx > args.max_rolls:
            break
        if args.reset_each_roll:
            model.load_state_dict(initial_state, strict=True)
        train_start = train_end - window_size
        forecast_end = min(train_end + forecast_horizon, T)
        if forecast_end <= train_end:
            break

        roll_t0 = time.perf_counter()
        if scope.replace("-", "_").lower() in {"qr", "qr_only"}:
            losses = optimize_window_cached_qr(
                model=model,
                data=data,
                start=train_start,
                end=train_end,
                coords=coords,
                device=device,
                optimizer=optimizer,
                steps=steps,
                grad_clip=grad_clip,
                lambda_reg=float(config.get("online_lambda_reg", config.get("lambda_reg", 0.0))),
            )
        else:
            losses = optimize_window(
                model=model,
                data=data,
                start=train_start,
                end=train_end,
                coords=coords,
                device=device,
                optimizer=optimizer,
                steps=steps,
                grad_clip=grad_clip,
                loss_kwargs=loss_kwargs,
            )
        pred, target = forecast_from_window(
            model=model,
            data=data,
            train_start=train_start,
            train_end=train_end,
            forecast_end=forecast_end,
            coords=coords,
            device=device,
        )
        elapsed = time.perf_counter() - roll_t0
        metrics = finite_metrics(pred, target)
        roll_records.append(
            {
                "roll": roll_idx,
                "train_start": train_start,
                "train_end": train_end,
                "forecast_end": forecast_end,
                "forecast_horizon": forecast_end - train_end,
                "seconds": elapsed,
                **losses,
                **{f"forecast_{key}": value for key, value in metrics.items()},
            }
        )
        predictions.append(pred)
        targets.append(target)
        origins.append(np.full((pred.shape[0],), train_end, dtype=np.int64))
        horizons.append(np.arange(1, pred.shape[0] + 1, dtype=np.int64))
        roll_ids.append(np.full((pred.shape[0],), roll_idx, dtype=np.int64))
        if roll_idx == 1 or roll_idx % int(config.get("fast_online_print_every", 10)) == 0:
            print(roll_records[-1])

    prediction_arr = np.concatenate(predictions, axis=0) if predictions else np.empty((0, model.state_dim), dtype=np.float32)
    target_arr = np.concatenate(targets, axis=0) if targets else np.empty((0, model.state_dim), dtype=np.float32)
    origin_arr = np.concatenate(origins, axis=0) if origins else np.empty((0,), dtype=np.int64)
    horizon_arr = np.concatenate(horizons, axis=0) if horizons else np.empty((0,), dtype=np.int64)
    roll_arr = np.concatenate(roll_ids, axis=0) if roll_ids else np.empty((0,), dtype=np.int64)
    overall = finite_metrics(prediction_arr, target_arr)
    total_seconds = time.perf_counter() - start_time
    result = {
        "mode": "fast_rolling_online",
        "scope": scope,
        "checkpoint": str(ckpt_path),
        "window_size": window_size,
        "stride": stride,
        "steps": steps,
        "lr": lr,
        "lambda_multistep": lambda_multistep,
        "forecast_horizon": forecast_horizon,
        "n_rolls": len(roll_records),
        "seconds_total": total_seconds,
        "seconds_per_roll": total_seconds / max(len(roll_records), 1),
        "overall": overall,
        "rolls": roll_records,
    }
    save_json(output_dir / "results.json", result)
    np.savez_compressed(
        output_dir / "forecasts.npz",
        prediction=prediction_arr,
        target=target_arr,
        origin_index=origin_arr,
        horizon=horizon_arr,
        roll=roll_arr,
        state_names=np.asarray([f"state_{idx}" for idx in range(model.state_dim)]),
    )
    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    last_path = ckpt_dir / config.get(
        "fast_online_checkpoint_name",
        checkpoint_name_with_suffix(Path(ckpt_path).name, "_fast_online_last"),
    )
    save_checkpoint(
        model,
        model_config,
        last_path,
        extra={
            "mode": "fast_rolling_online",
            "results": result,
            "checkpoint_role": "last_fast_online",
        },
    )
    print(json.dumps({k: result[k] for k in ("n_rolls", "seconds_total", "seconds_per_roll", "overall")}, indent=2))
    print(f"Saved fast online artifacts to {output_dir}")
    print(f"Saved last fast online checkpoint to {last_path}")


if __name__ == "__main__":
    main()
