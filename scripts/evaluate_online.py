import argparse
from pathlib import Path
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aligned_measurement_nwp import build_aligned_bundle, MeasurementNWPDataset
from src.models.measurement_system import MeasurementWindModel


STATIONS = ["E05", "E06", "ASOW6"]


def auto_device(name):
    if name is not None:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    p.add_argument("--measurement-file", type=str, required=True, help="Online measurement mat")
    p.add_argument("--nwp-file", type=str, required=True, help="Online NWP mat")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-file", type=str, default="outputs/eval_online_results.json")
    return p.parse_args()


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def to_numpy(x):
    return x.detach().cpu().numpy()


def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: [N, 3, 2]
    return dict with per-station and overall metrics for U/V/WS
    """
    results = {}

    # WS
    ws_true = np.sqrt(y_true[..., 0] ** 2 + y_true[..., 1] ** 2)
    ws_pred = np.sqrt(y_pred[..., 0] ** 2 + y_pred[..., 1] ** 2)

    def _metrics(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() == 0:
            return {"mae": np.nan, "rmse": np.nan, "corr": np.nan, "n": 0}
        err = b[mask] - a[mask]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 1 else np.nan
        return {"mae": mae, "rmse": rmse, "corr": corr, "n": int(mask.sum())}

    # per station
    for i, st in enumerate(STATIONS):
        results[st] = {
            "U": _metrics(y_true[:, i, 0], y_pred[:, i, 0]),
            "V": _metrics(y_true[:, i, 1], y_pred[:, i, 1]),
            "WS": _metrics(ws_true[:, i], ws_pred[:, i]),
        }

    # overall
    results["overall"] = {
        "U": _metrics(y_true[..., 0].reshape(-1), y_pred[..., 0].reshape(-1)),
        "V": _metrics(y_true[..., 1].reshape(-1), y_pred[..., 1].reshape(-1)),
        "WS": _metrics(ws_true.reshape(-1), ws_pred.reshape(-1)),
    }

    return results


@torch.no_grad()
def run_model(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        y_pred, mu_v, Sigma_v, K = model(batch)

        all_true.append(to_numpy(batch["y_next"]))
        all_pred.append(to_numpy(y_pred))

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return y_true, y_pred


@torch.no_grad()
def run_persistence(loader, device):
    all_true = []
    all_pred = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        # persistence: y_{t+1} hat = y_t
        all_true.append(to_numpy(batch["y_next"]))
        all_pred.append(to_numpy(batch["y_t"]))

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return y_true, y_pred


def print_metrics(title, metrics):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    for st in STATIONS + ["overall"]:
        print(f"\n[{st}]")
        for var in ["U", "V", "WS"]:
            m = metrics[st][var]
            print(
                f"{var:>2} | "
                f"MAE={m['mae']:.4f} "
                f"RMSE={m['rmse']:.4f} "
                f"Corr={m['corr']:.4f} "
                f"N={m['n']}"
            )


def main():
    args = parse_args()
    device = auto_device(args.device)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    # 用训练时的结构参数重建模型
    model = MeasurementWindModel(
        dt=cfg.get("dt", 1.0),
        hidden_dim=cfg.get("hidden_dim", 32),
        embed_dim=cfg.get("embed_dim", 32),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 2),
        ff_dim=cfg.get("ff_dim", 64),
        dropout=cfg.get("dropout", 0.1),
        init_log_amp=cfg.get("init_log_amp", 0.0),
        init_log_ell_par=cfg.get("init_log_ell_par", 0.5),
        init_log_ell_perp=cfg.get("init_log_ell_perp", 0.0),
        init_log_sigma_eps=cfg.get("init_log_sigma_eps", -2.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    meas_file = str(Path(args.measurement_file).expanduser().resolve())
    nwp_file = str(Path(args.nwp_file).expanduser().resolve())

    bundle = build_aligned_bundle(meas_file, nwp_file)
    dataset = MeasurementNWPDataset(bundle, seq_len=cfg.get("seq_len", 4))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"device: {device}")
    print(f"checkpoint: {ckpt_path}")
    print(f"online samples: {len(dataset)}")

    # model
    y_true_model, y_pred_model = run_model(model, loader, device)
    model_metrics = compute_metrics(y_true_model, y_pred_model)

    # persistence baseline
    y_true_pers, y_pred_pers = run_persistence(loader, device)
    pers_metrics = compute_metrics(y_true_pers, y_pred_pers)

    print_metrics("Model on ONLINE", model_metrics)
    print_metrics("Persistence baseline on ONLINE", pers_metrics)

    out = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "num_samples": len(dataset),
        "model_metrics": model_metrics,
        "persistence_metrics": pers_metrics,
    }

    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nsaved results to: {out_path}")


if __name__ == "__main__":
    main()