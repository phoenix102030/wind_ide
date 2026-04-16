from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_curves(history, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    train_loss = [h["train_loss_mean"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_min = [h["train_loss_min"] for h in history]
    train_max = [h["train_loss_max"] for h in history]

    epoch_time = [h["epoch_time_sec"] for h in history]

    alpha_mean = [h["alpha_mean"] for h in history]
    ell_par = [h["ell_par_mean"] for h in history]
    ell_perp = [h["ell_perp_mean"] for h in history]
    sigma_eps = [h["sigma_eps"] for h in history]
    grad_norm = [h["grad_norm_mean"] for h in history]

    # 1. train / val loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss_mean")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.fill_between(epochs, train_min, train_max, alpha=0.2, label="train_loss_range")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    # 2. epoch time
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_time, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("Epoch Time")
    plt.tight_layout()
    plt.savefig(save_dir / "epoch_time.png", dpi=200)
    plt.close()

    # 3. parameter trajectory
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, alpha_mean, label="alpha_mean")
    plt.plot(epochs, ell_par, label="ell_par")
    plt.plot(epochs, ell_perp, label="ell_perp")
    plt.plot(epochs, sigma_eps, label="sigma_eps")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Parameter Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "param_curve.png", dpi=200)
    plt.close()

    # 4. grad norm
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, grad_norm, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Mean Gradient Norm")
    plt.tight_layout()
    plt.savefig(save_dir / "grad_norm.png", dpi=200)
    plt.close()