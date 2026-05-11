from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.vector_data_utils import (
    load_nwp_uv140,
    load_vector_dataset,
    latlon_to_xy_km,
)
from model.covariance import safe_cholesky, solve_linear_system
from train.train_vector_offline import build_model, load_config, print_device_info, resolve_device


def grid_coords_from_latlon(lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
    origin = (float(np.nanmean(lat_grid)), float(np.nanmean(lon_grid)))
    return latlon_to_xy_km(lat_grid, lon_grid, origin=origin).reshape(-1, 2).astype(np.float32)


def component_selectors(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    return (
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype),
        torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype),
    )


def cross_transition_component(
    model: torch.nn.Module,
    source_coords: Tensor,
    target_coords: Tensor,
    outputs: dict[str, Tensor],
) -> Tensor:
    """Cross IDE matrix from station residual state to arbitrary target coords."""
    mu_seq = outputs["mu"]
    sigma_seq = outputs["Sigma"]
    A_seq = outputs["A"]
    device = mu_seq.device
    dtype = mu_seq.dtype
    source_coords = source_coords.to(device=device, dtype=dtype)
    target_coords = target_coords.to(device=device, dtype=dtype)
    selectors = component_selectors(device, dtype)
    ell = model.kernel.get_ell().to(device=device, dtype=dtype)
    gamma = model.kernel.gamma_value(device, dtype)
    eye2 = torch.eye(2, device=device, dtype=dtype)
    offsets = target_coords[:, None, :] - source_coords[None, :, :]
    n_time = mu_seq.shape[0]
    n_target = target_coords.shape[0]
    n_source = source_coords.shape[0]
    n_pairs = n_target * n_source

    row_blocks = []
    for i in range(2):
        col_blocks = []
        for j in range(2):
            B = model.kernel.dt * (selectors[i] - gamma * selectors[j])
            shift = mu_seq @ B.T
            B_sigma = torch.matmul(B.unsqueeze(0), sigma_seq)
            D = ell[i, j].pow(2) * eye2 + 2.0 * torch.matmul(B_sigma, B.T)
            D = D + model.kernel.jitter * eye2
            L = safe_cholesky(D)

            offset_prime = offsets.unsqueeze(0) - shift[:, None, None, :]
            flat = offset_prime.reshape(n_time, n_pairs, 2)
            alpha = solve_linear_system(D, flat.transpose(-1, -2)).transpose(-1, -2)
            maha = (flat * alpha).sum(dim=-1).reshape(n_time, n_target, n_source)
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            K = torch.exp(-maha - 0.5 * logdet[:, None, None])
            K = K / K.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
            col_blocks.append(A_seq[:, i, j, None, None] * K)
        row_blocks.append(torch.cat(col_blocks, dim=2))
    return torch.cat(row_blocks, dim=1)


def cross_transition_shared_flow(
    model: torch.nn.Module,
    source_coords: Tensor,
    target_coords: Tensor,
    outputs: dict[str, Tensor],
) -> Tensor:
    flow_mu = outputs["flow_mu"]
    flow_sigma = outputs["flow_Sigma"]
    B_seq = outputs["B"]
    device = flow_mu.device
    dtype = flow_mu.dtype
    source_coords = source_coords.to(device=device, dtype=dtype)
    target_coords = target_coords.to(device=device, dtype=dtype)
    eye2 = torch.eye(2, device=device, dtype=dtype)
    ell = model.kernel.get_ell().to(device=device, dtype=dtype).mean()
    offsets = target_coords[:, None, :] - source_coords[None, :, :]
    n_time = flow_mu.shape[0]
    n_target = target_coords.shape[0]
    n_source = source_coords.shape[0]
    n_pairs = n_target * n_source

    shift = model.kernel.dt * flow_mu
    D = ell.pow(2) * eye2 + 2.0 * flow_sigma
    D = D + model.kernel.jitter * eye2
    L = safe_cholesky(D)
    offset_prime = offsets.unsqueeze(0) - shift[:, None, None, :]
    flat = offset_prime.reshape(n_time, n_pairs, 2)
    alpha = solve_linear_system(D, flat.transpose(-1, -2)).transpose(-1, -2)
    maha = (flat * alpha).sum(dim=-1).reshape(n_time, n_target, n_source)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    K = torch.exp(-maha - 0.5 * logdet[:, None, None])
    K = K / K.sum(dim=2, keepdim=True).clamp_min(1.0e-8)

    row_blocks = []
    for i in range(2):
        col_blocks = []
        for j in range(2):
            col_blocks.append(B_seq[:, i, j, None, None] * K)
        row_blocks.append(torch.cat(col_blocks, dim=2))
    return torch.cat(row_blocks, dim=1)


def apply_cross_modifiers(cross: Tensor, outputs: dict[str, Tensor]) -> Tensor:
    """Optionally apply residual transition scalars that are defined off-station."""
    out = cross
    if "kernel_weight" in outputs:
        out = outputs["kernel_weight"].to(device=out.device, dtype=out.dtype).view(-1, 1, 1) * out
    if "residual_decay" in outputs:
        out = outputs["residual_decay"].to(device=out.device, dtype=out.dtype).view(-1, 1, 1) * out
    return out


def build_cross_transition(
    model: torch.nn.Module,
    source_coords: Tensor,
    target_coords: Tensor,
    outputs: dict[str, Tensor],
    apply_modifiers: bool = False,
) -> Tensor:
    if "flow_mu" in outputs and "flow_Sigma" in outputs and "B" in outputs:
        cross = cross_transition_shared_flow(model, source_coords, target_coords, outputs)
    else:
        cross = cross_transition_component(model, source_coords, target_coords, outputs)
    return apply_cross_modifiers(cross, outputs) if apply_modifiers else cross


def run_station_filter_and_grid_extension(
    model: torch.nn.Module,
    data: dict[str, Any],
    grid_coords: np.ndarray,
    device: torch.device,
    eval_window_size: int,
    eval_stride: int | None,
    apply_modifiers: bool,
) -> dict[str, np.ndarray]:
    if data.get("z_standardizer") is not None:
        raise ValueError("Grid extension currently expects unstandardized residual targets (standardize_z: false).")
    coords = torch.from_numpy(data["coords"]).to(device)
    target_coords = torch.from_numpy(grid_coords).to(device)
    z = torch.from_numpy(data["Z"]).to(device)
    n_time = int(data["X"].shape[0])
    n_grid = int(grid_coords.shape[0])
    if eval_stride is None:
        eval_stride = eval_window_size

    M_sum = np.zeros((n_time, 6, 6), dtype=np.float64)
    counts = np.zeros((n_time, 1), dtype=np.float64)
    control_sum: np.ndarray | None = None

    model.eval()
    with torch.no_grad():
        for start in range(0, n_time, eval_stride):
            end = min(start + eval_window_size, n_time)
            if end <= start:
                continue
            x_chunk = torch.from_numpy(data["X"][start:end]).to(device)
            outputs = model(x_chunk, coords)
            M_sum[start:end] += outputs["M"].detach().cpu().numpy()
            counts[start:end] += 1.0
            if "transition_control" in outputs:
                if control_sum is None:
                    control_sum = np.zeros((n_time, outputs["transition_control"].shape[-1]), dtype=np.float64)
                control_sum[start:end] += outputs["transition_control"].detach().cpu().numpy()

    valid = counts[:, 0] > 0.0
    if not valid.all():
        raise ValueError("Some time steps were not covered; reduce eval stride.")
    M_np = (M_sum / counts[:, :, None]).astype(np.float32)
    control_np = None
    if control_sum is not None:
        control_np = (control_sum / counts).astype(np.float32)

    M_t = torch.from_numpy(M_np).to(device)
    control_t = torch.from_numpy(control_np).to(device) if control_np is not None else None
    with torch.no_grad():
        kf = model.dstm.kalman_filter(
            z=z,
            M_seq=M_t,
            control_seq=control_t,
            reduction="sum",
            return_history=True,
        )
        station_prediction = kf["pred_means"]
        station_analysis = kf["filter_means"]

    grid_prediction_sum = np.zeros((n_time, 2 * n_grid), dtype=np.float32)
    grid_analysis_sum = np.zeros((n_time, 2 * n_grid), dtype=np.float32)
    grid_counts = np.zeros((n_time, 1), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_time, eval_stride):
            end = min(start + eval_window_size, n_time)
            if end <= start:
                continue
            x_chunk = torch.from_numpy(data["X"][start:end]).to(device)
            outputs = model(x_chunk, coords)
            cross = build_cross_transition(
                model,
                source_coords=coords,
                target_coords=target_coords,
                outputs=outputs,
                apply_modifiers=apply_modifiers,
            )
            pred_chunk = torch.bmm(
                cross,
                station_prediction[start:end].unsqueeze(-1),
            ).squeeze(-1)
            analysis_chunk = torch.bmm(
                cross,
                station_analysis[start:end].unsqueeze(-1),
            ).squeeze(-1)
            grid_prediction_sum[start:end] += pred_chunk.detach().cpu().numpy().astype(np.float32)
            grid_analysis_sum[start:end] += analysis_chunk.detach().cpu().numpy().astype(np.float32)
            grid_counts[start:end] += 1.0
    if not (grid_counts[:, 0] > 0.0).all():
        raise ValueError("Some grid residual time steps were not covered; reduce eval stride.")
    grid_prediction = grid_prediction_sum / np.maximum(grid_counts, 1.0)
    grid_analysis = grid_analysis_sum / np.maximum(grid_counts, 1.0)

    return {
        "station_prediction_residual": station_prediction.detach().cpu().numpy().astype(np.float32),
        "station_analysis_residual": station_analysis.detach().cpu().numpy().astype(np.float32),
        "grid_prediction_residual_flat": grid_prediction,
        "grid_analysis_residual_flat": grid_analysis,
        "transition_matrices": M_np,
    }


def split_grid_components(flat_state: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    n_grid = height * width
    u = flat_state[:, :n_grid].reshape(flat_state.shape[0], height, width)
    v = flat_state[:, n_grid:].reshape(flat_state.shape[0], height, width)
    return u.astype(np.float32, copy=False), v.astype(np.float32, copy=False)


def load_raw_uv_grid(config: dict[str, Any], split: str, time_limit: int | None, n_time: int) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = config.get("data", {})
    nwp_path = Path(data_cfg[f"{split}_nwp_path"])
    u_hw_t, v_hw_t = load_nwp_uv140(nwp_path, time_limit=time_limit)
    u = np.moveaxis(u_hw_t, 2, 0)[:n_time].astype(np.float32, copy=False)
    v = np.moveaxis(v_hw_t, 2, 0)[:n_time].astype(np.float32, copy=False)
    return u, v


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend a trained VectorMIDE station residual model to full NWP grid residual estimates."
    )
    parser.add_argument("--config", default="yml_files/VectorMIDE.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["offline", "online"], default="online")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-window-size", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument(
        "--apply-transition-modifiers",
        action="store_true",
        help="Apply kernel_weight/residual_decay to off-station cross kernels. Default uses the base IDE spatial kernel.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device_name = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(device_name, allow_fallback=bool(config.get("allow_device_fallback", True)))
    print_device_info(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = Path(config.get("checkpoint_dir", "checkpoints")) / config.get(
            "offline_checkpoint_name",
            "vector_mide_offline.pt",
        )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:
        print(f"Initialized model parameters not found in checkpoint: {missing}")
    if unexpected:
        print(f"Ignored checkpoint parameters not used by this config: {unexpected}")

    data = load_vector_dataset(config, split=args.split, time_limit=args.limit)
    eval_window_size = args.eval_window_size
    if eval_window_size is None:
        eval_window_size = int(
            min(
                model_config.get("window_size", config.get("window_size", 1008)),
                model_config.get("transformer_max_len", config.get("transformer_max_len", 4096)),
            )
        )

    grid_coords = grid_coords_from_latlon(data["lat_grid"], data["lon_grid"])
    height, width = data["lat_grid"].shape
    outputs = run_station_filter_and_grid_extension(
        model,
        data,
        grid_coords=grid_coords,
        device=device,
        eval_window_size=eval_window_size,
        eval_stride=args.eval_stride,
        apply_modifiers=bool(args.apply_transition_modifiers),
    )
    pred_u_resid, pred_v_resid = split_grid_components(outputs["grid_prediction_residual_flat"], height, width)
    analysis_u_resid, analysis_v_resid = split_grid_components(outputs["grid_analysis_residual_flat"], height, width)
    raw_u, raw_v = load_raw_uv_grid(config, args.split, args.limit, n_time=pred_u_resid.shape[0])

    corrected_prediction_u = raw_u + pred_u_resid
    corrected_prediction_v = raw_v + pred_v_resid
    corrected_analysis_u = raw_u + analysis_u_resid
    corrected_analysis_v = raw_v + analysis_v_resid

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("outputs") / "vector_grid_residual" / f"{Path(ckpt_path).stem}_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "grid_residual_extension.npz",
        raw_u=raw_u,
        raw_v=raw_v,
        prediction_residual_u=pred_u_resid,
        prediction_residual_v=pred_v_resid,
        analysis_residual_u=analysis_u_resid,
        analysis_residual_v=analysis_v_resid,
        corrected_prediction_u=corrected_prediction_u,
        corrected_prediction_v=corrected_prediction_v,
        corrected_analysis_u=corrected_analysis_u,
        corrected_analysis_v=corrected_analysis_v,
        station_prediction_residual=outputs["station_prediction_residual"],
        station_analysis_residual=outputs["station_analysis_residual"],
        measurement_target=data["Y"].astype(np.float32, copy=False),
        station_nwp_baseline=data["nwp_baseline"].astype(np.float32, copy=False),
        coords=data["coords"].astype(np.float32, copy=False),
        grid_coords=grid_coords,
        lat_grid=data["lat_grid"].astype(np.float32, copy=False),
        lon_grid=data["lon_grid"].astype(np.float32, copy=False),
    )
    metadata = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_time": int(pred_u_resid.shape[0]),
        "grid_shape": [int(height), int(width)],
        "state": {
            "prediction": "one-step Kalman prior residual extension, matching the station forecast style",
            "analysis": "filtered same-time residual extension after using available station measurements",
        },
        "apply_transition_modifiers": bool(args.apply_transition_modifiers),
        "output_file": "grid_residual_extension.npz",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved VectorMIDE grid residual extension to {output_dir}")


if __name__ == "__main__":
    main()
