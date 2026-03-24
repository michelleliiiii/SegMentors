import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from unet2d import UNet2D


CLASS_NAMES = {
    0: "BG",
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def normalize_tensor(x, mode="zscore_per_channel"):
    if mode == "zscore_per_channel":
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    return x


def ensure_chw(array):
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")

    if array.shape[0] in (1, 2, 3, 4) and array.shape[0] < array.shape[-1]:
        return array
    if array.shape[-1] in (1, 2, 3, 4):
        return np.transpose(array, (2, 0, 1))
    return np.transpose(array, (2, 0, 1))


def extract_case_id(path):
    return path.name.split("__")[0]


def load_split_manifest(csv_path):
    records = []
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)
    return records


def build_case_filters(records, split, label_status=None):
    selected = set()
    for row in records:
        if row["split"] != split:
            continue
        if label_status is not None and row.get("label_status") != label_status:
            continue
        selected.add(row["case_id"])
    return selected


class BrainTumorNPYDataset(Dataset):
    def __init__(
        self,
        root="data",
        split="train",
        manifest_csv="ssl_split_manifest.csv",
        label_status=None,
        normalize="zscore_per_channel",
        require_masks=True,
    ):
        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.require_masks = require_masks
        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing folder: {self.image_dir}")
        if self.require_masks and not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing folder: {self.mask_dir}")

        manifest_rows = load_split_manifest(manifest_csv)
        allowed_cases = build_case_filters(manifest_rows, split=split, label_status=label_status)

        samples = []
        for image_path in sorted(self.image_dir.glob("*.npy")):
            case_id = extract_case_id(image_path)
            if allowed_cases and case_id not in allowed_cases:
                continue

            mask_path = self.mask_dir / image_path.name.replace("__img.npy", "__mask.npy")
            if self.require_masks and not mask_path.exists():
                continue

            samples.append(
                {
                    "case_id": case_id,
                    "image_path": image_path,
                    "mask_path": mask_path if mask_path.exists() else None,
                }
            )

        if not samples:
            raise RuntimeError(
                f"No samples found for split={split}, label_status={label_status}, "
                f"image_dir={self.image_dir}"
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.load(sample["image_path"])
        image = torch.from_numpy(ensure_chw(image)).float()
        image = normalize_tensor(image, self.normalize)

        item = {
            "image": image,
            "case_id": sample["case_id"],
            "image_path": str(sample["image_path"]),
        }

        if sample["mask_path"] is not None:
            mask = np.load(sample["mask_path"])
            item["mask"] = torch.from_numpy(mask).long()
            item["mask_path"] = str(sample["mask_path"])

        return item


def build_cross_modal_mask(batch, mask_ratio=0.5, patch_size=16, generator=None):
    if generator is None:
        generator = torch.Generator(device=batch.device if batch.is_cuda else "cpu")

    bsz, channels, height, width = batch.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Patch size {patch_size} must divide image size {(height, width)} "
            "for cross-modal masking."
        )

    patch_h = height // patch_size
    patch_w = width // patch_size
    num_patches = patch_h * patch_w
    num_masked = max(1, int(num_patches * mask_ratio))

    masked_input = batch.clone()
    loss_mask = torch.zeros_like(batch)

    for sample_idx in range(bsz):
        perm = torch.randperm(num_patches, generator=generator, device=batch.device)
        chosen = perm[:num_masked]
        target_modalities = torch.randint(
            low=0,
            high=channels,
            size=(num_masked,),
            generator=generator,
            device=batch.device,
        )

        for patch_index, modality in zip(chosen.tolist(), target_modalities.tolist()):
            row = patch_index // patch_w
            col = patch_index % patch_w
            h0 = row * patch_size
            h1 = h0 + patch_size
            w0 = col * patch_size
            w1 = w0 + patch_size

            masked_input[sample_idx, modality, h0:h1, w0:w1] = 0.0
            loss_mask[sample_idx, modality, h0:h1, w0:w1] = 1.0

    return masked_input, batch, loss_mask


def build_random_channel_dropout_mask(batch, mask_ratio=0.5, patch_size=16, generator=None):
    if generator is None:
        generator = torch.Generator(device=batch.device if batch.is_cuda else "cpu")

    bsz, channels, height, width = batch.shape
    masked_input = batch.clone()
    loss_mask = torch.zeros_like(batch)
    total_pixels = height * width
    masked_pixels = max(1, int(total_pixels * mask_ratio))

    for sample_idx in range(bsz):
        modality = int(torch.randint(0, channels, (1,), generator=generator, device=batch.device).item())
        perm = torch.randperm(total_pixels, generator=generator, device=batch.device)[:masked_pixels]
        rows = torch.div(perm, width, rounding_mode="floor")
        cols = perm % width
        masked_input[sample_idx, modality, rows, cols] = 0.0
        loss_mask[sample_idx, modality, rows, cols] = 1.0

    return masked_input, batch, loss_mask


PRETEXT_TASKS = {
    "cross_modal": build_cross_modal_mask,
    "random_channel_dropout": build_random_channel_dropout_mask,
}


def masked_mae_loss(prediction, target, loss_mask):
    masked_elements = loss_mask.sum().clamp_min(1.0)
    return (torch.abs(prediction - target) * loss_mask).sum() / masked_elements


@torch.no_grad()
def mean_dice(pred, target, num_classes, eps=1e-6, exclude_bg=True):
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


def soft_dice_loss(logits, target, num_classes, eps=1e-6, exclude_bg=True):
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    if exclude_bg:
        probs = probs[:, 1:]
        target_one_hot = target_one_hot[:, 1:]

    inter = (probs * target_one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_pretraining_epoch(model, loader, optimizer, device, masker_fn, args, epoch_seed):
    model.train()
    total_loss = 0.0
    sample_count = 0

    for step, batch in enumerate(tqdm(loader, desc="Pretrain [train]")):
        images = batch["image"].to(device)
        generator = torch.Generator(device=device if device.type != "cpu" else "cpu")
        generator.manual_seed(epoch_seed + step)

        masked_inputs, targets, loss_mask = masker_fn(
            images,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
            generator=generator,
        )

        optimizer.zero_grad(set_to_none=True)
        recon = model(masked_inputs)
        loss = masked_mae_loss(recon, targets, loss_mask)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / max(1, sample_count)


@torch.no_grad()
def run_pretraining_validation(model, loader, device, masker_fn, args, epoch_seed):
    model.eval()
    total_loss = 0.0
    sample_count = 0

    for step, batch in enumerate(tqdm(loader, desc="Pretrain [val]")):
        images = batch["image"].to(device)
        generator = torch.Generator(device=device if device.type != "cpu" else "cpu")
        generator.manual_seed(epoch_seed + step)

        masked_inputs, targets, loss_mask = masker_fn(
            images,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
            generator=generator,
        )

        recon = model(masked_inputs)
        loss = masked_mae_loss(recon, targets, loss_mask)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / max(1, sample_count)


def run_segmentation_epoch(model, loader, optimizer, device, num_classes, w_ce, w_dice):
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
        batch_mean_dice, batch_class_dice = mean_dice(predictions, masks, num_classes=num_classes, exclude_bg=True)

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


def save_checkpoint(path, model, optimizer, epoch, metrics, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def format_seg_metrics(metrics):
    ordered = [
        f"loss={metrics['loss']:.4f}",
        f"mean_dice={metrics['mean_dice']:.4f}",
    ]
    for name in ("ET", "NET", "CC", "ED"):
        key = f"dice_{name}"
        if key in metrics:
            ordered.append(f"{name}={metrics[key]:.4f}")
    return " ".join(ordered)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Self-supervised 2D U-Net training pipeline")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest-csv", default="ssl_split_manifest.csv")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--w-ce", type=float, default=0.5)
    parser.add_argument("--w-dice", type=float, default=0.5)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--pretext-task", choices=sorted(PRETEXT_TASKS.keys()), default="cross_modal")
    parser.add_argument("--output-dir", default="outputs/ssl_unet2d")
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def run_pipeline(args):
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Pretext task: {args.pretext_task}")

    pretrain_train_ds = BrainTumorNPYDataset(
        root=args.data_root,
        split="train",
        manifest_csv=args.manifest_csv,
        label_status=None,
        require_masks=False,
    )
    pretrain_val_ds = BrainTumorNPYDataset(
        root=args.data_root,
        split="val",
        manifest_csv=args.manifest_csv,
        label_status=None,
        require_masks=False,
    )
    finetune_train_ds = BrainTumorNPYDataset(
        root=args.data_root,
        split="train",
        manifest_csv=args.manifest_csv,
        label_status="labeled",
        require_masks=True,
    )
    seg_val_ds = BrainTumorNPYDataset(
        root=args.data_root,
        split="val",
        manifest_csv=args.manifest_csv,
        label_status=None,
        require_masks=True,
    )
    test_ds = BrainTumorNPYDataset(
        root=args.data_root,
        split="test",
        manifest_csv=args.manifest_csv,
        label_status=None,
        require_masks=True,
    )

    print(
        "Dataset sizes:",
        f"pretrain_train={len(pretrain_train_ds)}",
        f"pretrain_val={len(pretrain_val_ds)}",
        f"finetune_train={len(finetune_train_ds)}",
        f"val={len(seg_val_ds)}",
        f"test={len(test_ds)}",
    )

    pretrain_train_loader = make_loader(
        pretrain_train_ds, args.batch_size, True, args.seed, args.num_workers
    )
    pretrain_val_loader = make_loader(
        pretrain_val_ds, args.batch_size, False, args.seed + 1, args.num_workers
    )
    finetune_train_loader = make_loader(
        finetune_train_ds, args.batch_size, True, args.seed + 2, args.num_workers
    )
    seg_val_loader = make_loader(
        seg_val_ds, args.batch_size, False, args.seed + 3, args.num_workers
    )
    test_loader = make_loader(
        test_ds, args.batch_size, False, args.seed + 4, args.num_workers
    )

    masker_fn = PRETEXT_TASKS[args.pretext_task]

    model = UNet2D(
        in_channels=args.in_channels,
        num_classes=args.in_channels,
        base=args.base,
        head_channels=args.in_channels,
    ).to(device)

    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
    best_pretrain_val = float("inf")
    best_pretrain_path = output_dir / "pretrain_best.pt"

    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss = run_pretraining_epoch(
            model,
            pretrain_train_loader,
            pretrain_optimizer,
            device,
            masker_fn,
            args,
            epoch_seed=args.seed * 1000 + epoch,
        )
        val_loss = run_pretraining_validation(
            model,
            pretrain_val_loader,
            device,
            masker_fn,
            args,
            epoch_seed=args.seed * 2000 + epoch,
        )

        print(f"Pretrain epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_pretrain_val:
            best_pretrain_val = val_loss
            save_checkpoint(
                best_pretrain_path,
                model,
                pretrain_optimizer,
                epoch,
                {"train_loss": train_loss, "val_loss": val_loss},
                extra={
                    "stage": "pretrain",
                    "pretext_task": args.pretext_task,
                    "in_channels": args.in_channels,
                    "base": args.base,
                },
            )

    pretrain_state = torch.load(best_pretrain_path, map_location=device)
    model.load_state_dict(pretrain_state["model"])
    model.replace_head(args.num_classes)
    model = model.to(device)

    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    best_val_dice = -1.0
    best_finetune_path = output_dir / "finetune_best.pt"

    print("Starting supervised fine-tuning on labeled training cases only.")
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = run_segmentation_epoch(
            model,
            finetune_train_loader,
            finetune_optimizer,
            device,
            args.num_classes,
            args.w_ce,
            args.w_dice,
        )
        val_metrics = run_segmentation_validation(
            model,
            seg_val_loader,
            device,
            args.num_classes,
            args.w_ce,
            args.w_dice,
        )

        print(f"Finetune epoch {epoch}: train_loss={train_loss:.4f} {format_seg_metrics(val_metrics)}")

        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            save_checkpoint(
                best_finetune_path,
                model,
                finetune_optimizer,
                epoch,
                {"train_loss": train_loss, **val_metrics},
                extra={
                    "stage": "finetune",
                    "in_channels": args.in_channels,
                    "num_classes": args.num_classes,
                    "base": args.base,
                    "class_names": CLASS_NAMES,
                },
            )

    finetune_state = torch.load(best_finetune_path, map_location=device)
    model.load_state_dict(finetune_state["model"])

    prediction_dir = output_dir / "test_predictions"
    test_metrics = run_segmentation_validation(
        model,
        test_loader,
        device,
        args.num_classes,
        args.w_ce,
        args.w_dice,
        save_predictions_dir=prediction_dir,
    )

    print(f"Test metrics: {format_seg_metrics(test_metrics)}")
    print(f"Saved predicted masks to: {prediction_dir}")


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
