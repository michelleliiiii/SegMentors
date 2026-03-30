import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils as U


CLASS_NAMES = {
    0: "BG",
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}


def soft_dice_loss(logits, target, num_classes, eps=1e-6, exclude_bg=True):
    """Compute differentiable Dice loss from segmentation logits and labels."""
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    if exclude_bg:
        probs = probs[:, 1:]
        target_one_hot = target_one_hot[:, 1:]

    inter = (probs * target_one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def run_segmentation_epoch(model, loader, optimizer, device, num_classes, w_ce, w_dice):
    """Run one supervised fine-tuning epoch on labeled segmentation data."""
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    sample_count = 0

    for batch in tqdm(loader, desc="Finetune [train]"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        ce = ce_criterion(logits, masks)
        dice = soft_dice_loss(logits, masks, num_classes=num_classes, exclude_bg=True)
        loss = w_ce * ce + w_dice * dice
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / max(1, sample_count)


@torch.no_grad()
def run_segmentation_validation(model, loader, device, num_classes, w_ce, w_dice, save_predictions_dir=None):
    """Evaluate segmentation loss, Dice metrics, and optionally save predictions."""
    model.eval()
    ce_criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_dice = 0.0
    sample_count = 0
    class_dice_sum = {class_idx: 0.0 for class_idx in range(1, num_classes)}

    for batch in tqdm(loader, desc="Segmentation [eval]"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        ce = ce_criterion(logits, masks)
        dice_loss = soft_dice_loss(logits, masks, num_classes=num_classes, exclude_bg=True)
        loss = w_ce * ce + w_dice * dice_loss

        predictions = torch.argmax(logits, dim=1)
        batch_mean_dice, batch_class_dice = U.mean_dice(predictions, masks, num_classes=num_classes, exclude_bg=True)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += batch_mean_dice * batch_size
        sample_count += batch_size

        for class_idx, score in batch_class_dice.items():
            class_dice_sum[class_idx] += score * batch_size

        if save_predictions_dir is not None:
            save_predictions_dir.mkdir(parents=True, exist_ok=True)
            for pred, image_path in zip(predictions.cpu().numpy(), batch["image_path"]):
                stem = Path(image_path).name.replace("__img.npy", "__pred.npy")
                np.save(save_predictions_dir / stem, pred.astype(np.uint8))

    metrics = {
        "loss": total_loss / max(1, sample_count),
        "mean_dice": total_dice / max(1, sample_count),
    }

    for class_idx in range(1, num_classes):
        metrics[f"dice_{CLASS_NAMES.get(class_idx, str(class_idx))}"] = (
            class_dice_sum[class_idx] / max(1, sample_count)
        )

    return metrics
