from __future__ import annotations

import argparse
import json
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
from train.train_grid_residual import build_model, load_config, make_batch, station_masks
from train.train_vector_offline import print_device_info, resolve_device


def finite_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    if not np.any(valid):
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "corr": float("nan")}
    error = pred[valid] - target[valid]
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    if valid.sum() >= 2 and np.std(pred[valid]) > 0.0 and np.std(target[valid]) > 0.0:
        corr = float(np.corrcoef(pred[valid], target[valid])[0, 1])
    else:
        corr = float("nan")
    return {"rmse": rmse, "mae": mae, "bias": bias, "corr": corr}


def skill(corrected_rmse: float, raw_rmse: float) -> float:
    if not np.isfinite(corrected_rmse) or not np.isfinite(raw_rmse) or raw_rmse <= 0.0:
        return float("nan")
    return float(1.0 - corrected_rmse / raw_rmse)


def station_metric_table(
    obs: np.ndarray,
    raw_nwp: np.ndarray,
    corrected: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    raw = finite_metrics(raw_nwp, obs, mask)
    corr = finite_metrics(corrected, obs, mask)
    result: dict[str, Any] = {
        "overall": {
            "raw_nwp": raw,
            "corrected": corr,
            "skill_rmse": skill(corr["rmse"], raw["rmse"]),
        },
        "stations": [],
    }
    for idx in range(obs.shape[1]):
        station_mask = mask[:, idx]
        raw_i = finite_metrics(raw_nwp[:, idx], obs[:, idx], station_mask)
        corr_i = finite_metrics(corrected[:, idx], obs[:, idx], station_mask)
        result["stations"].append(
            {
                "station_index": idx,
                "raw_nwp": raw_i,
                "corrected": corr_i,
                "skill_rmse": skill(corr_i["rmse"], raw_i["rmse"]),
            }
        )
    return result


def evaluate(
    model: torch.nn.Module,
    data: dict[str, Any],
    static: dict[str, torch.Tensor],
    config: dict[str, Any],
    station_input_mask: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    n_time = int(data["nwp_input"].shape[0])
    pred_chunks = []
    true_chunks = []
    nwp_chunks = []
    obs_chunks = []
    mask_chunks = []
    with torch.no_grad():
        for start in range(0, n_time, chunk_size):
            end = min(start + chunk_size, n_time)
            batch = make_batch(data, [start], end - start, device)
            nwp_at_stations = bilinear_sample_grid(batch["nwp_target"], static["sample_yx"])
            true_station_residuals = batch["obs_values"] - nwp_at_stations
            features = build_grid_residual_features(
                nwp_input=batch["nwp_input"],
                station_residuals=true_station_residuals,
                u=batch["u"],
                v=batch["v"],
                static=static,
                feature_cfg=config.get("features", {}),
                station_input_mask=station_input_mask,
            )
            pred_residual = model(features)
            pred_at_stations = bilinear_sample_grid(pred_residual, static["sample_yx"])
            pred_chunks.append(pred_at_stations.squeeze(0).cpu().numpy())
            true_chunks.append(true_station_residuals.squeeze(0).cpu().numpy())
            nwp_chunks.append(nwp_at_stations.squeeze(0).cpu().numpy())
            obs_chunks.append(batch["obs_values"].squeeze(0).cpu().numpy())
            mask_chunks.append(batch["obs_mask"].squeeze(0).cpu().numpy().astype(bool))

    pred_residual_at_stations = np.concatenate(pred_chunks, axis=0)
    true_residual_at_stations = np.concatenate(true_chunks, axis=0)
    nwp_at_stations = np.concatenate(nwp_chunks, axis=0)
    obs_values = np.concatenate(obs_chunks, axis=0)
    obs_mask = np.concatenate(mask_chunks, axis=0)
    corrected_at_stations = nwp_at_stations + pred_residual_at_stations
    return {
        "pred_residual_at_stations": pred_residual_at_stations,
        "true_residual_at_stations": true_residual_at_stations,
        "nwp_at_stations": nwp_at_stations,
        "corrected_at_stations": corrected_at_stations,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a full-grid residual checkpoint at the three stations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Optional config override; checkpoint config is used by default.")
    parser.add_argument("--split", default="offline", choices=["offline", "online"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--holdout-station", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else checkpoint.get("config")
    if config is None:
        raise ValueError("No config found in checkpoint; pass --config.")
    device_name = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(device_name, allow_fallback=bool(config.get("allow_device_fallback", True)))
    print_device_info(device)

    data = load_grid_residual_dataset(config, split=args.split, time_limit=args.limit)
    static = static_tensors(data, device)
    n_stations = int(data["obs_values"].shape[1])
    inferred_holdout = None
    if checkpoint.get("best"):
        inferred_holdout = checkpoint["best"].get("holdout_station")
    holdout_station = args.holdout_station if args.holdout_station is not None else inferred_holdout
    station_input_mask, _ = station_masks(n_stations, config, holdout_station, device)
    in_channels = int(
        checkpoint.get(
            "in_channels",
            feature_channel_count(data["nwp_input"].shape[1], n_stations, config.get("features", {})),
        )
    )
    model = build_model(config, in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])

    arrays = evaluate(
        model,
        data,
        static,
        config,
        station_input_mask=station_input_mask,
        device=device,
        chunk_size=int(args.chunk_size),
    )
    metrics = station_metric_table(
        obs=arrays["obs_values"],
        raw_nwp=arrays["nwp_at_stations"],
        corrected=arrays["corrected_at_stations"],
        mask=arrays["obs_mask"],
    )
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "target_channel": data["target_channel"],
            "target_component": data["target_component"],
            "holdout_station": holdout_station,
        }
    )
    print(json.dumps(metrics["overall"], indent=2, allow_nan=True))

    if args.output_dir is None:
        out_dir = Path("outputs") / "grid_residual_evaluation" / f"{Path(args.checkpoint).stem}_{args.split}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "results.json", metrics)
    np.savez_compressed(out_dir / "station_predictions.npz", **arrays)
    print(f"Saved evaluation artifacts to {out_dir}")


if __name__ == "__main__":
    main()
