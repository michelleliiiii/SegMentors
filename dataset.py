from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_case_id(filename: str) -> str:
    parts = filename.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[0]


def load_manifest(manifest_path: Path) -> dict:
    case_info = {}
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_info[row["case_id"]] = {
                "split": row["split"],
                "label_status": row["label_status"],
            }
    return case_info


def scan_split_pairs(data_root: Path, split: str):
    img_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing image dir: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

    mask_map = {
        m.name.replace("__mask.npy", ""): m
        for m in mask_dir.glob("*.npy")
        if m.is_file()
    }

    pairs = []
    for im in sorted(img_dir.glob("*.npy")):
        if not im.is_file():
            continue
        key = im.name.replace("__img.npy", "")
        m = mask_map.get(key)
        if m is None:
            continue
        case_id = extract_case_id(im.name)
        pairs.append((case_id, im, m))

    return pairs


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    if image.ndim == 2:
        image = image[None, :, :]
    elif image.ndim == 3:
        # Expecting HWC from your preprocessing
        if image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (2, 0, 1))
        else:
            # Already CHW
            pass
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    for c in range(image.shape[0]):
        x = image[c]
        mean = x.mean()
        std = x.std()
        if std > 1e-8:
            image[c] = (x - mean) / std
        else:
            image[c] = x - mean

    return image.astype(np.float32)


def prepare_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float32)

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask.squeeze()

    mask = (mask > 0).astype(np.float32)
    mask = mask[None, :, :]  # 1 x H x W
    return mask


class BraTSSegDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        split: str,
        label_status: str | None = None,
        return_mask: bool = True,
    ):
        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.label_status = label_status
        self.return_mask = return_mask

        case_info = load_manifest(self.manifest_path)
        all_pairs = scan_split_pairs(self.data_root, self.split)

        samples = []
        for case_id, img_path, mask_path in all_pairs:
            if case_id not in case_info:
                continue

            info = case_info[case_id]
            if info["split"] != self.split:
                continue

            if self.label_status is not None and info["label_status"] != self.label_status:
                continue

            samples.append({
                "case_id": case_id,
                "img_path": img_path,
                "mask_path": mask_path,
            })

        if len(samples) == 0:
            raise ValueError(
                f"No samples found for split={self.split}, label_status={self.label_status}"
            )

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = np.load(sample["img_path"])
        image = normalize_image(image)
        image = torch.from_numpy(image)

        out = {
            "image": image,
            "case_id": sample["case_id"],
            "img_path": str(sample["img_path"]),
        }

        if self.return_mask:
            mask = np.load(sample["mask_path"])
            mask = prepare_mask(mask)
            mask = torch.from_numpy(mask)
            out["mask"] = mask
            out["mask_path"] = str(sample["mask_path"])

        return out