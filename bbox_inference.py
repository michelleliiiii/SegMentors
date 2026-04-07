import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from unet2d import UNet2D


MANIFEST_NAME = "ssl_split_manifest.csv"

DATA_ROOT = Path("data")

BBOX_MARGIN = 5
SKIP_EMPTY_PREDICTIONS = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def get_bbox_xyxy_from_mask(mask: np.ndarray, margin: int = 5):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = max(0, int(xs.min()) - margin)
    x_max = min(mask.shape[1] - 1, int(xs.max()) + margin)
    y_min = max(0, int(ys.min()) - margin)
    y_max = min(mask.shape[0] - 1, int(ys.max()) + margin)

    return [x_min, y_min, x_max, y_max]


def read_case_ids_from_manifest(manifest_path: Path, split_name: str):
    case_ids = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_name == "train":
                if row["split"] == "train" and row["label_status"] == "labeled":
                    case_ids.append(row["case_id"])
            elif split_name == "unlabelled_train":
                if row["split"] == "train" and row["label_status"] == "unlabeled":
                    case_ids.append(row["case_id"])
    return sorted(case_ids)


def find_case_image_paths(case_id: str, images_dir: Path):
    return sorted(images_dir.glob(f"{case_id}__*__img.npy"))


def extract_case_id(filename: str):
    return filename.split("__")[0]


def build_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_channels = int(ckpt.get("in_channels", 4))
    num_classes = int(ckpt.get("num_classes", 5))
    base = int(ckpt.get("base", 32))

    model = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, {
        "in_channels": in_channels,
        "num_classes": num_classes,
        "base": base,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "unlabelled_train"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = Path(".").resolve()

    manifest_path = run_dir / MANIFEST_NAME
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path

    out_root = run_dir

    if args.split in {"train", "unlabelled_train"}:
        images_dir = run_dir / DATA_ROOT / "train" / "images"
    else:
        images_dir = run_dir / DATA_ROOT / "val" / "images"

    if args.split in {"train", "unlabelled_train"} and not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    bboxes_csv_path = out_root / "bboxes.csv"

    device = get_device()
    model, model_info = build_model(ckpt_path, device)

    if args.split in {"train", "unlabelled_train"}:
        case_ids = read_case_ids_from_manifest(manifest_path, args.split)
        if not case_ids:
            raise RuntimeError(f"No case IDs found for split={args.split} in {MANIFEST_NAME}")

        all_img_paths = []
        for case_id in case_ids:
            all_img_paths.extend(find_case_image_paths(case_id, images_dir))
    else:
        all_img_paths = sorted(images_dir.glob("*__img.npy"))

    if not all_img_paths:
        raise RuntimeError(f"No image slices found for split={args.split}.")

    bbox_rows = []

    pbar = tqdm(all_img_paths, desc=f"{args.split} slices")
    total_saved = 0
    total_empty = 0

    for img_path in pbar:
        case_id = extract_case_id(img_path.name)

        img = np.load(img_path).astype(np.float32)  # (H, W, C)

        if img.ndim != 3:
            raise ValueError(f"Expected (H,W,C), got {img.shape} for {img_path}")

        mu = img.mean(axis=(0, 1), keepdims=True)
        sd = img.std(axis=(0, 1), keepdims=True) + 1e-8
        img_norm = (img - mu) / sd

        x = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

        with torch.no_grad():
            logits = model(x)
            pred_multiclass = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)

        pred_binary = to_binary_mask(pred_multiclass)
        bbox_xyxy = get_bbox_xyxy_from_mask(pred_binary, margin=BBOX_MARGIN)

        if bbox_xyxy is None:
            total_empty += 1
            if SKIP_EMPTY_PREDICTIONS:
                continue

        x_min = y_min = x_max = y_max = ""
        bbox_str = ""
        if bbox_xyxy is not None:
            x_min, y_min, x_max, y_max = bbox_xyxy
            bbox_str = f"[{x_min},{y_min},{x_max},{y_max}]"

        bbox_rows.append({
            "case_id": case_id,
            "slice_file": img_path.name,
            "bbox_x_min": x_min,
            "bbox_y_min": y_min,
            "bbox_x_max": x_max,
            "bbox_y_max": y_max,
            "bbox_xyxy": bbox_str,
        })
        total_saved += 1

    with open(bboxes_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "slice_file",
                "bbox_x_min",
                "bbox_y_min",
                "bbox_x_max",
                "bbox_y_max",
                "bbox_xyxy",
            ],
        )
        writer.writeheader()
        writer.writerows(bbox_rows)

    print("\nDone.")
    print(f"Seed: {args.seed}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Model info: {model_info}")
    print(f"Images dir: {images_dir}")
    print(f"Saved usable slices: {total_saved}")
    print(f"Skipped empty predictions: {total_empty}")
    print(f"BBoxes CSV: {bboxes_csv_path}")


if __name__ == "__main__":
    main()