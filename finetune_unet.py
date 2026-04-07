import argparse
import csv
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from unet2d import UNet2D


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NPYFolderDataset(Dataset):
    def __init__(self, root="data", split="train", normalize="zscore_per_channel"):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        self.normalize = normalize

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing folders: {self.image_dir} or {self.mask_dir}")

        labeled = None
        if split == "train":
            labeled = set()
            with open("ssl_split_manifest.csv", "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["split"] == "train" and row["label_status"] == "labeled":
                        labeled.add(row["case_id"])

        image_files = sorted(self.image_dir.glob("*.npy"))

        pairs = []
        for path in image_files:
            if not path.name.endswith("__img.npy"):
                continue

            if split == "train":
                case_id = "-".join(path.stem.split("__")[0].split("-")[:4])
                if case_id not in labeled:
                    continue

            mask_name = path.name.replace("__img.npy", "__mask.npy")
            mask_path = self.mask_dir / mask_name

            if mask_path.exists():
                pairs.append((path, mask_path))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        x = np.load(image_path)
        y = np.load(mask_path)

        if x.shape[0] in (1, 2, 3, 4) and x.shape[0] < x.shape[-1]:
            pass
        elif x.shape[-1] in (1, 2, 3, 4):
            x = np.transpose(x, (2, 0, 1))
        else:
            x = np.transpose(x, (2, 0, 1))

        x = torch.from_numpy(x).float()

        if self.normalize == "zscore_per_channel":
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        y = torch.from_numpy(y).long()
        return x, y


class PseudoSoftDataset(Dataset):
    def __init__(
        self,
        root="data",
        split="train",
        normalize="zscore_per_channel",
        manifest_path="ssl_split_manifest.csv",
        pseudo_subdir="pseudo_labels",
        expected_num_classes=5,
    ):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / "images"
        self.pseudo_dir = self.root / split / pseudo_subdir
        self.normalize = normalize
        self.expected_num_classes = expected_num_classes

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {self.image_dir}")
        if not self.pseudo_dir.exists():
            raise FileNotFoundError(f"Missing pseudo folder: {self.pseudo_dir}")

        allowed_cases = set()
        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == "train" and row["label_status"] == "unlabeled":
                    allowed_cases.add(row["case_id"])

        pairs = []
        for path in sorted(self.image_dir.glob("*.npy")):
            if not path.name.endswith("__img.npy"):
                continue

            case_id = "-".join(path.stem.split("__")[0].split("-")[:4])
            if case_id not in allowed_cases:
                continue

            pseudo_name = path.name.replace("__img.npy", "__soft.npy")
            pseudo_path = self.pseudo_dir / pseudo_name

            if pseudo_path.exists():
                pairs.append((path, pseudo_path))

        self.pairs = pairs

        if len(self.pairs) == 0:
            raise RuntimeError("No soft pseudo-label pairs found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, pseudo_path = self.pairs[idx]

        x = np.load(image_path)
        p = np.load(pseudo_path).astype(np.float32)

        if x.shape[0] in (1, 2, 3, 4) and x.shape[0] < x.shape[-1]:
            pass
        elif x.shape[-1] in (1, 2, 3, 4):
            x = np.transpose(x, (2, 0, 1))
        else:
            x = np.transpose(x, (2, 0, 1))

        x = torch.from_numpy(x).float()

        if self.normalize == "zscore_per_channel":
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        if p.ndim != 3:
            raise ValueError(f"Pseudo label must have shape (C,H,W), got {p.shape} for {pseudo_path}")
        if p.shape[0] != self.expected_num_classes:
            raise ValueError(
                f"Expected {self.expected_num_classes} pseudo channels, got {p.shape[0]} for {pseudo_path}"
            )

        p = torch.from_numpy(p).float()
        return x, p


@torch.no_grad()
def mean_dice(pred, target, num_classes, eps=1e-6, exclude_bg=True):
    dices = []

    for c in range(1, num_classes):
        pred_class = (pred == c).float()
        targ_class = (target == c).float()

        inter = (pred_class * targ_class).sum(dim=(1, 2))
        denom = pred_class.sum(dim=(1, 2)) + targ_class.sum(dim=(1, 2))

        dice = (2 * inter + eps) / (denom + eps)
        dices.append(dice)

    return torch.stack(dices, dim=1).mean().item()


@torch.no_grad()
def dice_per_class(pred, target, num_classes, eps=1e-6):
    out = {}

    for c in range(1, num_classes):
        pred_class = (pred == c).float()
        targ_class = (target == c).float()

        inter = (pred_class * targ_class).sum(dim=(1, 2))
        denom = pred_class.sum(dim=(1, 2)) + targ_class.sum(dim=(1, 2))

        dice = (2 * inter + eps) / (denom + eps)
        out[c] = dice.mean().item()

    return out


def hard_dice_loss(logits, target, num_classes, eps=1e-6, exclude_bg=True):
    probs = torch.softmax(logits, dim=1)
    t1h = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    if exclude_bg:
        probs = probs[:, 1:]
        t1h = t1h[:, 1:]

    inter = (probs * t1h).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + t1h.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def soft_cross_entropy_from_probs(logits, soft_targets, confidence_threshold=0.0, fg_boost=1.5):
    log_probs = F.log_softmax(logits, dim=1)
    per_pixel = -(soft_targets * log_probs).sum(dim=1)

    conf, hard = soft_targets.max(dim=1)
    fg_mask = (hard > 0).float()

    weight = torch.ones_like(conf)

    if confidence_threshold > 0.0:
        weight = weight * (conf >= confidence_threshold).float()

    weight = weight * conf
    weight = weight * (1.0 + fg_boost * fg_mask)

    denom = weight.sum().clamp_min(1.0)
    return (per_pixel * weight).sum() / denom


def soft_dice_loss_from_probs(logits, soft_targets, eps=1e-6, exclude_bg=True):
    probs = torch.softmax(logits, dim=1)

    if exclude_bg:
        probs = probs[:, 1:]
        soft_targets = soft_targets[:, 1:]

    inter = (probs * soft_targets).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + soft_targets.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def mask_surface(mask):
    if mask.sum() == 0:
        return mask.astype(bool)

    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    surface = mask.astype(bool) & (~eroded)
    return surface


def hd_95(pred_mask, targ_mask):
    pred_mask = pred_mask.astype(bool)
    targ_mask = targ_mask.astype(bool)

    if pred_mask.sum() == 0 and targ_mask.sum() == 0:
        return 0.0

    if pred_mask.sum() == 0 or targ_mask.sum() == 0:
        return np.nan

    pred_surface = mask_surface(pred_mask)
    targ_surface = mask_surface(targ_mask)

    dt_targ = distance_transform_edt(~targ_surface, sampling=(0.7, 0.7))
    dt_pred = distance_transform_edt(~pred_surface, sampling=(0.7, 0.7))

    pred_to_targ = dt_targ[pred_surface]
    targ_to_pred = dt_pred[targ_surface]

    all_dists = np.concatenate([pred_to_targ, targ_to_pred])

    if all_dists.size == 0:
        return 0.0

    return float(np.percentile(all_dists, 95))


@torch.no_grad()
def mean_hd95(pred, target, num_classes):
    pred_np = pred.detach().cpu().numpy()
    targ_np = target.detach().cpu().numpy()

    vals = []

    for b in range(pred_np.shape[0]):
        for c in range(1, num_classes):
            pred_c = (pred_np[b] == c)
            targ_c = (targ_np[b] == c)

            hd = hd_95(pred_c, targ_c)

            if not np.isnan(hd):
                vals.append(hd)

    if len(vals) == 0:
        return 0.0

    return float(np.mean(vals))


def count_unique_patients(dataset):
    return len({
        "-".join(path.stem.split("__")[0].split("-")[:4])
        for path, _ in dataset.pairs
    })


def main(seed, ckpt):
    set_seed(seed)

    device = get_device()
    print("Device:", device)

    num_classes = 5
    in_channels = 4
    base = 32
    batch_size = 8
    epochs = 8
    lr = 1e-4
    weight_decay = 2.7858130122219456e-4

    w_ce = 0.5
    w_dice = 0.5

    lambda_u = 0.02
    pseudo_conf_thresh = 0.10
    pseudo_fg_boost = 1.5
    pseudo_w_ce = 0.8
    pseudo_w_dice = 0.2

    train = NPYFolderDataset(root="data", split="train", normalize="zscore_per_channel")
    val = NPYFolderDataset(root="data", split="val", normalize="zscore_per_channel")
    pseudo = PseudoSoftDataset(
        root="data",
        split="train",
        normalize="zscore_per_channel",
        manifest_path="ssl_split_manifest.csv",
        pseudo_subdir="pseudo_labels",
        expected_num_classes=num_classes,
    )

    print("Train patients:", count_unique_patients(train))
    print("Val patients:", count_unique_patients(val))
    print("Pseudo unlabeled slices:", len(pseudo))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    pseudo_loader = DataLoader(pseudo, batch_size=batch_size, shuffle=True, num_workers=0)

    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    model.load_state_dict(ckpt["model"])

    ce_criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1e9
    print(f"Loaded checkpoint weights from {ckpt_path}")
    print("Tracking best_val fresh for this fine-tuning run only.")

    for ep in range(epochs):
        model.train()
        running = 0.0
        running_gt = 0.0
        running_pseudo = 0.0

        pseudo_iter = cycle(pseudo_loader)

        for x, y in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs} [train]"):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)

            logits = model(x)
            ce = ce_criterion(logits, y)
            dl = hard_dice_loss(logits, y, num_classes=num_classes, exclude_bg=True)
            gt_loss = w_ce * ce + w_dice * dl

            x_u, p_u = next(pseudo_iter)
            x_u = x_u.to(device)
            p_u = p_u.to(device)

            logits_u = model(x_u)
            pce = soft_cross_entropy_from_probs(
                logits_u,
                p_u,
                confidence_threshold=pseudo_conf_thresh,
                fg_boost=pseudo_fg_boost,
            )
            pdl = soft_dice_loss_from_probs(
                logits_u,
                p_u,
                exclude_bg=True,
            )
            pseudo_loss = pseudo_w_ce * pce + pseudo_w_dice * pdl

            loss = gt_loss + lambda_u * pseudo_loss
            loss.backward()
            optim.step()

            running += loss.item() * x.size(0)
            running_gt += gt_loss.item() * x.size(0)
            running_pseudo += pseudo_loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)
        train_gt_loss = running_gt / len(train_loader.dataset)
        train_pseudo_loss = running_pseudo / len(train_loader.dataset)

        model.eval()
        vloss = 0.0
        vdice = 0.0
        vhd95 = 0.0
        nseen = 0

        vclass1 = 0.0
        vclass2 = 0.0
        vclass3 = 0.0
        vclass4 = 0.0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {ep+1}/{epochs} [val]"):
                x = x.to(device)
                y = y.to(device)

                logits = model(x)

                ce = ce_criterion(logits, y)
                dl = hard_dice_loss(logits, y, num_classes=num_classes, exclude_bg=True)
                loss = w_ce * ce + w_dice * dl

                vloss += loss.item() * x.size(0)

                pred = torch.argmax(logits, dim=1)
                bs = x.size(0)

                class_dice = dice_per_class(pred, y, num_classes=num_classes)

                vdice += mean_dice(pred, y, num_classes=num_classes, exclude_bg=True) * bs
                vhd95 += mean_hd95(pred, y, num_classes=num_classes) * bs
                nseen += bs

                vclass1 += class_dice[1] * bs
                vclass2 += class_dice[2] * bs
                vclass3 += class_dice[3] * bs
                vclass4 += class_dice[4] * bs

        val_loss = vloss / len(val_loader.dataset)
        val_dice = vdice / max(1, nseen)
        val_hd95 = vhd95 / max(1, nseen)

        val_class1_dice = vclass1 / max(1, nseen)
        val_class2_dice = vclass2 / max(1, nseen)
        val_class3_dice = vclass3 / max(1, nseen)
        val_class4_dice = vclass4 / max(1, nseen)

        torch.save(
            {
                "model": model.state_dict(),
                "in_channels": in_channels,
                "num_classes": num_classes,
                "base": base,
                "epoch": ep + 1,
                "val_dice": val_dice,
                "val_hd95": val_hd95,
                "lambda_u": lambda_u,
                "pseudo_conf_thresh": pseudo_conf_thresh,
            },
            f"unet2d_latest_finetuned_{seed}.pt",
        )

        print(
            f"Epoch {ep+1}: "
            f"train_loss={train_loss:.4f} "
            f"train_gt_loss={train_gt_loss:.4f} "
            f"train_pseudo_loss={train_pseudo_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice(excl_bg)={val_dice:.4f} "
            f"val_hd95(excl_bg)={val_hd95:.4f} "
            f"class1_dice={val_class1_dice:.4f} "
            f"class2_dice={val_class2_dice:.4f} "
            f"class3_dice={val_class3_dice:.4f} "
            f"class4_dice={val_class4_dice:.4f}"
        )

        if val_dice > best_val:
            best_val = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "base": base,
                    "best_val_dice": best_val,
                    "best_val_hd95": val_hd95,
                    "epoch": ep + 1,
                    "lambda_u": lambda_u,
                    "pseudo_conf_thresh": pseudo_conf_thresh,
                },
                f"unet2d_best_finetuned_{seed}.pt",
            )
            print(
                f"unet2d_best_finetuned_{seed}.pt saved to directory with "
                f"best val_dice={best_val:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint to fine-tune from")
    args = parser.parse_args()
    main(args.seed, args.ckpt)