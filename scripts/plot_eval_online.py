import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


STATIONS = ["E05", "E06", "ASOW6"]
VARS = ["U", "V", "WS"]


def load_results(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_rmse_comparison(results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = results["model_metrics"]
    pers_metrics = results["persistence_metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, var in zip(axes, VARS):
        labels = STATIONS + ["overall"]
        model_vals = [model_metrics[k][var]["rmse"] for k in labels]
        pers_vals = [pers_metrics[k][var]["rmse"] for k in labels]

        x = np.arange(len(labels))
        w = 0.35

        ax.bar(x - w/2, model_vals, width=w, label="model")
        ax.bar(x + w/2, pers_vals, width=w, label="persistence")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("RMSE")
        ax.set_title(f"{var} RMSE")
        ax.legend()

    plt.savefig(save_dir / "rmse_comparison.png", dpi=200)
    plt.close()


def plot_mae_comparison(results, save_dir):
    save_dir = Path(save_dir)
    model_metrics = results["model_metrics"]
    pers_metrics = results["persistence_metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, var in zip(axes, VARS):
        labels = STATIONS + ["overall"]
        model_vals = [model_metrics[k][var]["mae"] for k in labels]
        pers_vals = [pers_metrics[k][var]["mae"] for k in labels]

        x = np.arange(len(labels))
        w = 0.35

        ax.bar(x - w/2, model_vals, width=w, label="model")
        ax.bar(x + w/2, pers_vals, width=w, label="persistence")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("MAE")
        ax.set_title(f"{var} MAE")
        ax.legend()

    plt.savefig(save_dir / "mae_comparison.png", dpi=200)
    plt.close()


def plot_corr_comparison(results, save_dir):
    save_dir = Path(save_dir)
    model_metrics = results["model_metrics"]
    pers_metrics = results["persistence_metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, var in zip(axes, VARS):
        labels = STATIONS + ["overall"]
        model_vals = [model_metrics[k][var]["corr"] for k in labels]
        pers_vals = [pers_metrics[k][var]["corr"] for k in labels]

        x = np.arange(len(labels))
        w = 0.35

        ax.bar(x - w/2, model_vals, width=w, label="model")
        ax.bar(x + w/2, pers_vals, width=w, label="persistence")

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("Correlation")
        ax.set_title(f"{var} Corr")
        ax.legend()

    plt.savefig(save_dir / "corr_comparison.png", dpi=200)
    plt.close()


def print_readable_summary(results):
    model = results["model_metrics"]["overall"]
    pers = results["persistence_metrics"]["overall"]

    print("\n=== Overall summary ===")
    for var in VARS:
        print(
            f"{var}: "
            f"model RMSE={model[var]['rmse']:.3f}, "
            f"persistence RMSE={pers[var]['rmse']:.3f}, "
            f"model Corr={model[var]['corr']:.3f}, "
            f"persistence Corr={pers[var]['corr']:.3f}"
        )

    print("\nQuick read:")
    print("- Lower RMSE / MAE is better")
    print("- Higher Corr is better")
    print("- Negative Corr usually means prediction direction/trend may be wrong or sample is too small")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json-file", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="outputs/eval_online_plots")
    args = p.parse_args()

    results = load_results(args.json_file)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print_readable_summary(results)
    plot_rmse_comparison(results, save_dir)
    plot_mae_comparison(results, save_dir)
    plot_corr_comparison(results, save_dir)

    print(f"\nSaved plots to: {save_dir}")


if __name__ == "__main__":
    main()