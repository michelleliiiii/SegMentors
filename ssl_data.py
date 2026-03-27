from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_case_id(filename: str) -> str:
    parts = filename.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[0]


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    case_info = {}

    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_info[row["case_id"]] = {
                "split": row["split"],
                "label_status": row["label_status"],
            }

    return case_info


class NPYFolderDataset(Dataset):
    def __init__(
        self,
        root="data",
        split="train",
        normalize="zscore_per_channel",
        manifest_path=None,
        label_status=None,
        return_mask=True,
        return_case_id=False,
    ):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        self.normalize = normalize
        self.return_mask = return_mask
        self.return_case_id = return_case_id

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing folder: {self.image_dir}")
        if return_mask and not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing folder: {self.mask_dir}")

        self.case_info = None
        if manifest_path is not None:
            self.case_info = load_manifest(manifest_path)

        image_files = sorted(self.image_dir.glob("*.npy"))
        pairs = []

        for path in image_files:
            if not path.name.endswith("__img.npy"):
                continue

            case_id = extract_case_id(path.name)

            if self.case_info is not None:
                info = self.case_info.get(case_id)
                if info is None:
                    continue
                if info["split"] != split:
                    continue
                if label_status is not None and info["label_status"] != label_status:
                    continue

            if return_mask:
                mask_name = path.name.replace("__img.npy", "__mask.npy")
                mask_path = self.mask_dir / mask_name
                if mask_path.exists():
                    pairs.append((path, mask_path))
            else:
                pairs.append((path, None))

        self.pairs = pairs

        if len(self.pairs) == 0:
            raise ValueError(
                f"No samples found for split={split}, label_status={label_status}, return_mask={return_mask}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        case_id = extract_case_id(image_path.name)

        x = np.load(image_path)

        # enforce (C, H, W)
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

        if not self.return_mask:
            if self.return_case_id:
                return x, case_id
            return x

        y = np.load(mask_path)
        y = torch.from_numpy(y).long()
        if self.return_case_id:
            return x, y, case_id
        return x, y


@torch.no_grad()
def per_class_dice(pred, target, num_classes, eps=1e-6, exclude_bg=True):
    dices = []
    start_class = 1 if exclude_bg else 0

    for c in range(start_class, num_classes):
        pred_c = (pred == c).float()
        targ_c = (target == c).float()

        inter = (pred_c * targ_c).sum(dim=(1, 2))
        denom = pred_c.sum(dim=(1, 2)) + targ_c.sum(dim=(1, 2))

        dice = (2 * inter + eps) / (denom + eps)
        dices.append(dice)

    return torch.stack(dices, dim=1)


@torch.no_grad()
def mean_dice(pred, target, num_classes, eps=1e-6, exclude_bg=True):
    return per_class_dice(
        pred,
        target,
        num_classes=num_classes,
        eps=eps,
        exclude_bg=exclude_bg,
    ).mean().item()


def soft_dice_loss(logits, target, num_classes, eps=1e-6, exclude_bg=True):
    probs = torch.softmax(logits, dim=1)
    t1h = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    if exclude_bg:
        probs = probs[:, 1:]
        t1h = t1h[:, 1:]

    inter = (probs * t1h).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + t1h.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)

    return 1.0 - dice.mean()


def masked_cross_entropy(logits, target, valid_mask):
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    valid_mask = valid_mask.float()
    denom = valid_mask.sum().clamp_min(1.0)
    return (per_pixel * valid_mask).sum() / denom


def masked_soft_cross_entropy(logits, target_probs, valid_mask):
    log_probs = F.log_softmax(logits, dim=1)
    per_pixel = -(target_probs * log_probs).sum(dim=1)
    valid_mask = valid_mask.float()
    denom = valid_mask.sum().clamp_min(1.0)
    return (per_pixel * valid_mask).sum() / denom


def masked_soft_dice_loss(logits, target, valid_mask, num_classes, eps=1e-6, exclude_bg=True):
    probs = torch.softmax(logits, dim=1)
    t1h = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1).float()

    if exclude_bg:
        probs = probs[:, 1:]
        t1h = t1h[:, 1:]

    inter = (probs * t1h * valid_mask).sum(dim=(0, 2, 3))
    denom = (probs * valid_mask).sum(dim=(0, 2, 3)) + (t1h * valid_mask).sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)

    return 1.0 - dice.mean()


def masked_soft_dice_loss_probs(logits, target_probs, valid_mask, eps=1e-6, exclude_bg=True):
    probs = torch.softmax(logits, dim=1)
    valid_mask = valid_mask.unsqueeze(1).float()

    if exclude_bg:
        probs = probs[:, 1:]
        target_probs = target_probs[:, 1:]

    inter = (probs * target_probs * valid_mask).sum(dim=(0, 2, 3))
    denom = (probs * valid_mask).sum(dim=(0, 2, 3)) + (target_probs * valid_mask).sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)

    return 1.0 - dice.mean()


def save_baseline_style_checkpoint(path, model, in_channels, num_classes, base, best_val_dice):
    torch.save(
        {
            "model": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base": base,
            "best_val_dice": best_val_dice,
        },
        path,
    )


def load_baseline_style_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt
