import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import seed_worker


class BrainTumorNPYDataset(Dataset):
    """Load preprocessed MRI slice tensors and optional segmentation masks."""

    def __init__(
        self,
        root="data",
        split="train",
        manifest_csv="ssl_split_manifest_20.csv",
        label_status=None,
        normalize="zscore_per_channel",
        require_masks=True,
    ):
        """Initialize a dataset filtered by split and label status.

        Args:
            root (str): Root directory containing split folders.
            split (str): Dataset split to read, such as ``train`` or ``val``.
            manifest_csv (str): CSV file describing allowed case IDs.
            label_status (str | None): Optional label filter for training cases.
            normalize (str): Image normalization mode.
            require_masks (bool): Whether each sample must have a mask file.
        """
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
        allowed_cases = build_case_filters(
            manifest_rows,
            split=split,
            label_status=label_status,
        )

        samples = []
        for image_path in sorted(self.image_dir.glob("*.npy")):
            case_id = extract_case_id(image_path)
            if allowed_cases and case_id not in allowed_cases:
                continue

            mask_path = self.mask_dir / image_path.name.replace(
                "__img.npy", "__mask.npy"
            )
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
        """Return the number of discovered samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load one sample and return image metadata with an optional mask."""
        sample = self.samples[idx]
        image = np.load(sample["image_path"])
        image = torch.from_numpy(ensure_chw(image)).float()
        image = normalize_tensor(image, self.normalize)

        item = {
            "image": image,
            "case_id": sample["case_id"],
            "image_path": str(sample["image_path"]),
        }

        if self.require_masks and sample["mask_path"] is not None:
            mask = np.load(sample["mask_path"])
            item["mask"] = torch.from_numpy(mask).long()
            item["mask_path"] = str(sample["mask_path"])

        return item


def make_dataloader(dataset, batch_size, shuffle, seed, num_workers):
    """Create a deterministic ``DataLoader`` for the given dataset."""
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


def build_case_filters(records, split, label_status=None):
    """Collect case IDs from the manifest that match the requested subset."""
    selected = set()
    for row in records:
        if row["split"] != split:
            continue
        if label_status is not None and row.get("label_status") != label_status:
            continue
        selected.add(row["case_id"])
    return selected


def extract_case_id(path):
    """Extract the BraTS case ID from a slice filename."""
    return path.name.split("__")[0]


def load_split_manifest(csv_path):
    """Read the split manifest CSV into a list of row dictionaries."""
    records = []
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)
    return records


def normalize_tensor(x, mode="zscore_per_channel"):
    """Normalize an image tensor according to the selected strategy."""
    if mode == "zscore_per_channel":
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    return x


def ensure_chw(array):
    """Convert an image array to channel-first ``(C, H, W)`` format."""
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")

    if array.shape[0] in (1, 2, 3, 4) and array.shape[0] < array.shape[-1]:
        return array
    if array.shape[-1] in (1, 2, 3, 4):
        return np.transpose(array, (2, 0, 1))
    return np.transpose(array, (2, 0, 1))

