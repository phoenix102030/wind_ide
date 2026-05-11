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

from dataset.grid_residual_data_utils import (
    bilinear_sample_grid,
    build_grid_residual_features,
    feature_channel_count,
    load_grid_residual_dataset,
    static_tensors,
)
from model.grid_residual import GridResidualCNN
from train.grid_residual_losses import (
    advection_consistency_loss,
    distance_weighted_prior_loss,
    masked_mse,
    prior_loss,
    spatial_smoothness_loss,
    temporal_smoothness_loss,
)
from train.train_vector_offline import print_device_info, resolve_device


TIME_ARRAY_KEYS = ("nwp_input", "nwp_target", "obs_values", "obs_mask", "u", "v")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any], in_channels: int) -> GridResidualCNN:
    model_cfg = config.get("model", {})
    return GridResidualCNN(
        in_channels=in_channels,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 5)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        max_residual=float(model_cfg.get("max_residual", 0.0)),
    )


def split_train_validation(
    data: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, list[int]]:
    val_cfg = config.get("validation", {})
    if not bool(val_cfg.get("enabled", True)):
        return data, None, []

    n_time = int(data["nwp_input"].shape[0])
    window_size = int(config.get("train", {}).get("window_size", 1))
    val_fraction = float(val_cfg.get("fraction", 0.15))
    val_min_windows = int(val_cfg.get("min_windows", 4))
    val_num_windows = int(val_cfg.get("num_windows", 16))

    min_val_len = min(n_time, max(window_size, val_min_windows * window_size))
    val_len = max(min_val_len, int(round(n_time * val_fraction)))
    val_len = min(val_len, n_time)
    train_end = max(0, n_time - val_len)
    if train_end < min(window_size, n_time):
        starts = make_fixed_starts(n_time, window_size, val_num_windows)
        return data, data, starts

    train_data = slice_time_data(data, 0, train_end)
    val_data = slice_time_data(data, train_end, n_time)
    starts = make_fixed_starts(int(val_data["nwp_input"].shape[0]), window_size, val_num_windows)
    return train_data, val_data, starts


def slice_time_data(data: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    sliced = dict(data)
    for key in TIME_ARRAY_KEYS:
        sliced[key] = data[key][start:end]
    return sliced


def make_fixed_starts(n_time: int, window_size: int, num_windows: int) -> list[int]:
    if n_time <= 0:
        return []
    if n_time <= window_size:
        return [0]
    max_start = n_time - window_size
    count = max(1, min(num_windows, max_start + 1))
    return np.linspace(0, max_start, count).round().astype(int).tolist()


def sample_starts(
    n_time: int,
    window_size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_time <= window_size:
        return np.zeros(batch_size, dtype=np.int64)
    return rng.integers(0, n_time - window_size + 1, size=batch_size, dtype=np.int64)


def make_batch(
    arrays: dict[str, Any],
    starts: np.ndarray | list[int],
    window_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in TIME_ARRAY_KEYS:
        stacked = np.stack([arrays[key][int(start) : int(start) + window_size] for start in starts], axis=0)
        batch[key] = torch.from_numpy(stacked).to(device)
    return batch


def station_masks(
    n_stations: int,
    config: dict[str, Any],
    holdout_station: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_cfg = config.get("train", {})
    input_mask = torch.ones(n_stations, device=device)
    loss_mask = torch.ones(n_stations, device=device)
    if train_cfg.get("train_station_indices") is not None:
        loss_mask.zero_()
        input_mask.zero_()
        for idx in train_cfg["train_station_indices"]:
            loss_mask[int(idx)] = 1.0
            input_mask[int(idx)] = 1.0
    if holdout_station is not None:
        if not 0 <= holdout_station < n_stations:
            raise ValueError(f"holdout_station must be in [0,{n_stations - 1}]")
        loss_mask[holdout_station] = 0.0
        input_mask[holdout_station] = 0.0
    return input_mask, loss_mask


def compute_losses(
    model: GridResidualCNN,
    batch: dict[str, torch.Tensor],
    static: dict[str, torch.Tensor],
    config: dict[str, Any],
    station_input_mask: torch.Tensor,
    station_loss_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    feature_cfg = config.get("features", {})
    loss_cfg = config.get("loss", {})

    nwp_at_stations = bilinear_sample_grid(batch["nwp_target"], static["sample_yx"])
    true_station_residuals = batch["obs_values"] - nwp_at_stations
    features = build_grid_residual_features(
        nwp_input=batch["nwp_input"],
        station_residuals=true_station_residuals,
        u=batch["u"],
        v=batch["v"],
        static=static,
        feature_cfg=feature_cfg,
        station_input_mask=station_input_mask,
    )
    pred_residual = model(features)
    pred_at_stations = bilinear_sample_grid(pred_residual, static["sample_yx"])

    obs_mask = batch["obs_mask"].to(dtype=torch.bool) & station_loss_mask.view(1, 1, -1).to(dtype=torch.bool)
    loss_obs = masked_mse(pred_at_stations, true_station_residuals, obs_mask)
    loss_smooth = spatial_smoothness_loss(pred_residual)
    if bool(loss_cfg.get("distance_weighted_prior", True)):
        loss_prior = distance_weighted_prior_loss(pred_residual, static["distance_km"])
    else:
        loss_prior = prior_loss(pred_residual)
    loss_time = temporal_smoothness_loss(pred_residual)
    if float(loss_cfg.get("lambda_adv", 0.0)) > 0.0:
        loss_adv = advection_consistency_loss(
            pred_residual,
            batch["u"],
            batch["v"],
            grid_x=static["grid_x"],
            grid_y=static["grid_y"],
            dt_seconds=float(config.get("data", {}).get("dt_seconds", config.get("dt_seconds", 600.0))),
        )
    else:
        loss_adv = pred_residual.new_tensor(0.0)

    total = (
        loss_obs
        + float(loss_cfg.get("lambda_smooth", 0.05)) * loss_smooth
        + float(loss_cfg.get("lambda_prior", 0.01)) * loss_prior
        + float(loss_cfg.get("lambda_time", 0.0)) * loss_time
        + float(loss_cfg.get("lambda_adv", 0.0)) * loss_adv
    )
    return {
        "loss": total,
        "loss_obs": loss_obs,
        "loss_smooth": loss_smooth,
        "loss_prior": loss_prior,
        "loss_time": loss_time,
        "loss_adv": loss_adv,
        "pred_residual": pred_residual,
        "pred_at_stations": pred_at_stations,
        "true_station_residuals": true_station_residuals,
    }


def run_epoch(
    model: GridResidualCNN,
    arrays: dict[str, Any],
    static: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    station_input_mask: torch.Tensor,
    station_loss_mask: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    train_cfg = config.get("train", {})
    steps = int(train_cfg.get("steps_per_epoch", 100))
    batch_size = int(train_cfg.get("batch_size", 8))
    window_size = int(train_cfg.get("window_size", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    n_time = int(arrays["nwp_input"].shape[0])
    sums: dict[str, float] = {}

    for _ in range(steps):
        starts = sample_starts(n_time, window_size, batch_size, rng)
        batch = make_batch(arrays, starts, window_size, device)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(
            model,
            batch,
            static,
            config,
            station_input_mask=station_input_mask,
            station_loss_mask=station_loss_mask,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key, value in losses.items():
            if key.startswith("loss"):
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())

    return {key: value / max(steps, 1) for key, value in sums.items()}


def validation_losses(
    model: GridResidualCNN,
    arrays: dict[str, Any],
    starts: list[int],
    static: dict[str, torch.Tensor],
    config: dict[str, Any],
    station_input_mask: torch.Tensor,
    station_loss_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    if not starts:
        return {}
    model.eval()
    window_size = int(config.get("train", {}).get("window_size", 1))
    sums: dict[str, float] = {}
    with torch.no_grad():
        for start in starts:
            batch = make_batch(arrays, [start], window_size, device)
            losses = compute_losses(
                model,
                batch,
                static,
                config,
                station_input_mask=station_input_mask,
                station_loss_mask=station_loss_mask,
            )
            for key, value in losses.items():
                if key.startswith("loss"):
                    sums[f"val_{key}"] = sums.get(f"val_{key}", 0.0) + float(value.detach().cpu())
    return {key: value / max(len(starts), 1) for key, value in sums.items()}


def save_checkpoint(
    model: GridResidualCNN,
    config: dict[str, Any],
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "config": config}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def checkpoint_score(metrics: dict[str, float], monitor: str) -> float:
    if monitor in metrics:
        return float(metrics[monitor])
    if "val_loss_obs" in metrics:
        return float(metrics["val_loss_obs"])
    return float(metrics.get("loss_obs", metrics.get("loss", math.inf)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full-grid residual analysis correction model.")
    parser.add_argument("--config", default="yml_files/GridResidual_u140.yaml")
    parser.add_argument("--device", default=None, help="Override config device: auto, cpu, mps, cuda, cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional time limit for quick runs.")
    parser.add_argument("--dry-run", action="store_true", help="Only run one forward/loss pass.")
    parser.add_argument(
        "--holdout-station",
        type=int,
        default=None,
        help="Leave one station out of residual inputs and supervised loss; use 0, 1, or 2.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 123)))
    device_name = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(device_name, allow_fallback=bool(config.get("allow_device_fallback", True)))
    print_device_info(device)

    data = load_grid_residual_dataset(config, split="offline", time_limit=args.limit)
    train_data, val_data, val_starts = split_train_validation(data, config)
    n_stations = int(data["obs_values"].shape[1])
    station_input_mask, station_loss_mask = station_masks(n_stations, config, args.holdout_station, device)
    static = static_tensors(data, device)

    in_channels = feature_channel_count(
        nwp_channels=int(data["nwp_input"].shape[1]),
        n_stations=n_stations,
        feature_cfg=config.get("features", {}),
    )
    model = build_model(config, in_channels=in_channels).to(device)
    train_cfg = config.get("train", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-5)),
    )
    print(
        "Grid residual data: "
        f"T={data['nwp_input'].shape[0]}, H={data['nwp_input'].shape[2]}, W={data['nwp_input'].shape[3]}, "
        f"target={data['target_channel']}, input_channels={in_channels}, holdout={args.holdout_station}"
    )
    if val_data is not None:
        print(
            "Validation enabled: "
            f"train_T={train_data['nwp_input'].shape[0]}, "
            f"val_T={val_data['nwp_input'].shape[0]}, val_windows={len(val_starts)}"
        )

    window_size = int(train_cfg.get("window_size", 1))
    if args.dry_run:
        batch = make_batch(train_data, [0], min(window_size, train_data["nwp_input"].shape[0]), device)
        with torch.enable_grad():
            losses = compute_losses(
                model,
                batch,
                static,
                config,
                station_input_mask=station_input_mask,
                station_loss_mask=station_loss_mask,
            )
        print({key: float(value.detach().cpu()) for key, value in losses.items() if key.startswith("loss")})
        print(f"pred_residual shape: {tuple(losses['pred_residual'].shape)}")
        return

    rng = np.random.default_rng(int(config.get("seed", 123)))
    epochs = int(train_cfg.get("epochs", 100))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", config.get("checkpoint_dir", "checkpoints")))
    suffix = "" if args.holdout_station is None else f"_holdout{args.holdout_station}"
    ckpt_name = train_cfg.get("checkpoint_name", f"grid_residual_{data['target_channel']}{suffix}.pt")
    if args.holdout_station is not None and suffix not in ckpt_name:
        path = Path(ckpt_name)
        ckpt_name = f"{path.stem}{suffix}{path.suffix or '.pt'}"
    best_path = ckpt_dir / ckpt_name
    last_path = ckpt_dir / train_cfg.get("last_checkpoint_name", f"{Path(ckpt_name).stem}_last.pt")
    monitor = str(train_cfg.get("checkpoint_metric", "val_loss_obs"))
    validation_every = int(config.get("validation", {}).get("every_epochs", 1))
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_info: dict[str, Any] | None = None

    for epoch in range(epochs):
        metrics = run_epoch(
            model,
            train_data,
            static,
            optimizer,
            config,
            station_input_mask=station_input_mask,
            station_loss_mask=station_loss_mask,
            device=device,
            rng=rng,
        )
        if val_data is not None and validation_every > 0 and (
            (epoch + 1) % validation_every == 0 or epoch + 1 == epochs
        ):
            metrics.update(
                validation_losses(
                    model,
                    val_data,
                    val_starts,
                    static,
                    config,
                    station_input_mask=station_input_mask,
                    station_loss_mask=station_loss_mask,
                    device=device,
                )
            )
        print(f"epoch {epoch + 1}/{epochs}: {metrics}")
        record = {"epoch": epoch + 1, "epochs": epochs, **metrics}
        history.append(record)

        if monitor in metrics:
            score = checkpoint_score(metrics, monitor)
            if score < best_score:
                best_score = score
                best_info = {
                    "epoch": epoch + 1,
                    "score": score,
                    "monitor_metric": monitor,
                    "metrics": metrics,
                    "holdout_station": args.holdout_station,
                }
                save_checkpoint(
                    model,
                    config,
                    best_path,
                    extra={"best": best_info, "history": history, "in_channels": in_channels},
                )
                print(f"Saved best checkpoint to {best_path} ({monitor}={score:.6g})")

    save_checkpoint(
        model,
        config,
        last_path,
        extra={"best": best_info, "history": history, "in_channels": in_channels},
    )
    if best_info is None:
        save_checkpoint(model, config, best_path, extra={"history": history, "in_channels": in_channels})
        print(f"Saved checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")


if __name__ == "__main__":
    main()
