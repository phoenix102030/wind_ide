from __future__ import annotations

import argparse
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
from train.train_vector_offline import build_model, print_device_info, resolve_device, set_seed


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
    lambda_anchor = float(config.get("lambda_anchor", 0.01))
    grad_clip = float(config.get("grad_clip", 1.0))

    metrics = []
    T = data["X"].shape[0]
    for end in range(window_size, T + 1, update_every):
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
                lambda_adv=float(config.get("lambda_adv", 0.1)),
                lambda_smooth=float(config.get("lambda_smooth", 0.001)),
                lambda_reg=float(config.get("lambda_reg", 0.0001)),
            )
            current = trainable_named_parameters(model)
            loss = losses["loss"] + lambda_anchor * anchor_loss(current, anchor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        metrics.append({"end": end, "loss": float(losses["loss"].detach().cpu())})
        print(metrics[-1])

    out_path = Path(config.get("checkpoint_dir", "checkpoints")) / config.get(
        "online_checkpoint_name",
        "vector_mide_online.pt",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        out_path,
    )
    print(f"Saved online checkpoint to {out_path}")


if __name__ == "__main__":
    main()
