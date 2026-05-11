from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


def run_inference(
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
    residual_chunks = []
    corrected_chunks = []
    raw_chunks = []
    with torch.no_grad():
        for start in range(0, n_time, chunk_size):
            end = min(start + chunk_size, n_time)
            batch = make_batch(data, [start], end - start, device)
            nwp_at_stations = bilinear_sample_grid(batch["nwp_target"], static["sample_yx"])
            station_residuals = batch["obs_values"] - nwp_at_stations
            features = build_grid_residual_features(
                nwp_input=batch["nwp_input"],
                station_residuals=station_residuals,
                u=batch["u"],
                v=batch["v"],
                static=static,
                feature_cfg=config.get("features", {}),
                station_input_mask=station_input_mask,
            )
            pred_residual = model(features)
            raw_nwp = batch["nwp_target"]
            corrected = raw_nwp + pred_residual
            residual_chunks.append(pred_residual.squeeze(0).squeeze(1).cpu().numpy().astype(np.float32))
            corrected_chunks.append(corrected.squeeze(0).squeeze(1).cpu().numpy().astype(np.float32))
            raw_chunks.append(raw_nwp.squeeze(0).squeeze(1).cpu().numpy().astype(np.float32))

    return {
        "predicted_residual": np.concatenate(residual_chunks, axis=0),
        "corrected_nwp": np.concatenate(corrected_chunks, axis=0),
        "raw_nwp": np.concatenate(raw_chunks, axis=0),
        "uncertainty_proxy": data["distance_features"].min(axis=0).astype(np.float32),
        "station_latlon": data["station_latlon"].astype(np.float32),
        "station_sample_yx": data["sample_yx"].astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer full-grid residual and corrected NWP fields.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Optional config override; checkpoint config is used by default.")
    parser.add_argument("--split", default="online", choices=["offline", "online"])
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
    inferred_holdout = checkpoint.get("best", {}).get("holdout_station") if checkpoint.get("best") else None
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

    arrays = run_inference(
        model,
        data,
        static,
        config,
        station_input_mask=station_input_mask,
        device=device,
        chunk_size=int(args.chunk_size),
    )
    if args.output_dir is None:
        out_dir = Path("outputs") / "grid_residual_inference" / f"{Path(args.checkpoint).stem}_{args.split}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "grid_predictions.npz", **arrays)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "target_channel": data["target_channel"],
        "target_component": data["target_component"],
        "holdout_station": holdout_station,
        "shape": {
            "predicted_residual": list(arrays["predicted_residual"].shape),
            "corrected_nwp": list(arrays["corrected_nwp"].shape),
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved full-grid predictions to {out_dir}")


if __name__ == "__main__":
    main()
