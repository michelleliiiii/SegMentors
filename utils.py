import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def get_device():
    """Return the preferred PyTorch device available on this machine."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Seed each DataLoader worker deterministically from the PyTorch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def mean_dice(pred, target, num_classes, eps=1e-6, exclude_bg=True):
    """Compute mean Dice and per-class Dice from hard segmentation labels."""
    per_class = {}
    start_class = 1 if exclude_bg else 0

    for class_idx in range(start_class, num_classes):
        pred_c = (pred == class_idx).float()
        target_c = (target == class_idx).float()
        inter = (pred_c * target_c).sum(dim=(1, 2))
        denom = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice = (2 * inter + eps) / (denom + eps)
        per_class[class_idx] = dice.mean().item()

    mean_score = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean_score, per_class


def format_seg_metrics(metrics):
    """Format segmentation metrics into a compact printable string."""
    ordered = [
        f"loss={metrics['loss']:.4f}",
        f"mean_dice={metrics['mean_dice']:.4f}",
    ]
    for name in ("ET", "NET", "CC", "ED"):
        key = f"dice_{name}"
        if key in metrics:
            ordered.append(f"{name}={metrics[key]:.4f}")
    return " ".join(ordered)


def save_checkpoint(path, model, optimizer, epoch, metrics, extra=None):
    """Serialize model state, optimizer state, and run metadata to disk."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_loss_figure(history, output_path):
    """Plot and save the pretraining and fine-tuning loss curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    pretrain_epochs = range(1, len(history["pretrain_train_loss"]) + 1)
    axes[0].plot(pretrain_epochs, history["pretrain_train_loss"], label="train")
    axes[0].plot(pretrain_epochs, history["pretrain_val_loss"], label="val")
    axes[0].set_title("Pretraining Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    finetune_epochs = range(1, len(history["finetune_train_loss"]) + 1)
    axes[1].plot(finetune_epochs, history["finetune_train_loss"], label="train")
    axes[1].plot(finetune_epochs, history["finetune_val_loss"], label="val")
    axes[1].set_title("Fine-Tuning Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Segmentation Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
