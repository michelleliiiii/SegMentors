import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from unet2d import UNet2D



MANIFEST_NAME = "ssl_split_manifest.csv"
CKPT_NAME = "unet2d_best.pt"

DATA_ROOT = Path("data")
OUT_ROOT = Path("Sam_data")

DISPLAY_MODALITY = 0   
BBOX_MARGIN = 5
SKIP_EMPTY_PREDICTIONS = True
COPY_SOURCE_NPY = True



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_to_0_1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def to_uint8_png(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (255.0 * x01).round().astype(np.uint8)


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def get_bbox_xyxy_from_mask(mask: np.ndarray, margin: int = 5):
    """
    Return MedSAM-style bbox: [x_min, y_min, x_max, y_max]
    or None if the mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = max(0, int(xs.min()) - margin)
    x_max = min(mask.shape[1] - 1, int(xs.max()) + margin)
    y_min = max(0, int(ys.min()) - margin)
    y_max = min(mask.shape[0] - 1, int(ys.max()) + margin)

    return [x_min, y_min, x_max, y_max]


def read_unlabeled_train_case_ids(manifest_path: Path):
    case_ids = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train" and row["label_status"] == "unlabeled":
                case_ids.append(row["case_id"])
    return sorted(case_ids)


def find_case_image_paths(case_id: str, images_dir: Path):
    return sorted(images_dir.glob(f"{case_id}__*__img.npy"))


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


def prepare_output_dirs(root: Path):
    dirs = {
        "root": root,
        "images_png": root / "images_png",
        "images_npy": root / "images_npy",
        "pred_binary_masks": root / "pred_binary_masks",
    }
    for d in dirs.values():
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)
    return dirs


def main():
    run_dir = Path(".").resolve()

    manifest_path = run_dir / MANIFEST_NAME
    ckpt_path = run_dir / CKPT_NAME
    train_images_dir = run_dir / DATA_ROOT / "train" / "images"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Missing train images dir: {train_images_dir}")

    out_dirs = prepare_output_dirs(run_dir / OUT_ROOT)
    bboxes_csv_path = out_dirs["root"] / "bboxes.csv"
    records_csv_path = out_dirs["root"] / "records.csv"

    device = get_device()
    model, model_info = build_model(ckpt_path, device)

    unlabeled_case_ids = read_unlabeled_train_case_ids(manifest_path)
    if not unlabeled_case_ids:
        raise RuntimeError("No train,unlabeled case IDs found in ssl_split_manifest.csv")

    all_img_paths = []
    for case_id in unlabeled_case_ids:
        case_imgs = find_case_image_paths(case_id, train_images_dir)
        all_img_paths.extend(case_imgs)

    if not all_img_paths:
        raise RuntimeError("No image slices found for train,unlabeled cases.")

    bbox_rows = []
    record_rows = []

    outer_pbar = tqdm(unlabeled_case_ids, desc="Cases", position=0)
    total_saved = 0
    total_empty = 0

    for case_id in outer_pbar:
        case_img_paths = find_case_image_paths(case_id, train_images_dir)
        if not case_img_paths:
            continue

        inner_pbar = tqdm(case_img_paths, desc=f"{case_id}", leave=False, position=1)

        for img_path in inner_pbar:
            img = np.load(img_path).astype(np.float32)  # (H, W, C)

            if img.ndim != 3:
                raise ValueError(f"Expected (H,W,C), got {img.shape} for {img_path}")

            if DISPLAY_MODALITY >= img.shape[-1]:
                raise ValueError(
                    f"DISPLAY_MODALITY={DISPLAY_MODALITY} but image has only {img.shape[-1]} channels: {img_path}"
                )

            # background/export image for MedSAM
            display_img = normalize_to_0_1(img[..., DISPLAY_MODALITY])
            display_png = to_uint8_png(display_img)

            # model input normalization
            mu = img.mean(axis=(0, 1), keepdims=True)
            sd = img.std(axis=(0, 1), keepdims=True) + 1e-8
            img_norm = (img - mu) / sd

            x = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

            with torch.no_grad():
                logits = model(x)
                pred_multiclass = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)

            pred_binary = to_binary_mask(pred_multiclass)
            bbox_xyxy = get_bbox_xyxy_from_mask(pred_binary, margin=BBOX_MARGIN)

            is_empty = bbox_xyxy is None
            if is_empty and SKIP_EMPTY_PREDICTIONS:
                total_empty += 1
                record_rows.append({
                    "case_id": case_id,
                    "slice_file": img_path.name,
                    "image_png_path": "",
                    "image_npy_path": str(img_path),
                    "pred_binary_mask_path": "",
                    "bbox_xyxy": "",
                    "has_prediction": 0,
                    "is_empty_prediction": 1,
                })
                continue

            stem = img_path.stem.replace("__img", "")
            png_out = out_dirs["images_png"] / f"{stem}.png"
            pred_mask_out = out_dirs["pred_binary_masks"] / f"{stem}__pred_binary.npy"
            copied_npy_out = out_dirs["images_npy"] / img_path.name

            Image.fromarray(display_png).save(png_out)
            np.save(pred_mask_out, pred_binary.astype(np.uint8))

            if COPY_SOURCE_NPY:
                shutil.copy2(img_path, copied_npy_out)

            bbox_str = ""
            x_min = y_min = x_max = y_max = ""
            if bbox_xyxy is not None:
                x_min, y_min, x_max, y_max = bbox_xyxy
                bbox_str = f"[{x_min},{y_min},{x_max},{y_max}]"

                bbox_rows.append({
                    "case_id": case_id,
                    "slice_file": img_path.name,
                    "image_png_path": str(png_out),
                    "pred_binary_mask_path": str(pred_mask_out),
                    "bbox_x_min": x_min,
                    "bbox_y_min": y_min,
                    "bbox_x_max": x_max,
                    "bbox_y_max": y_max,
                    "bbox_xyxy": bbox_str,
                })

            record_rows.append({
                "case_id": case_id,
                "slice_file": img_path.name,
                "image_png_path": str(png_out),
                "image_npy_path": str(copied_npy_out if COPY_SOURCE_NPY else img_path),
                "pred_binary_mask_path": str(pred_mask_out),
                "bbox_xyxy": bbox_str,
                "has_prediction": 1,
                "is_empty_prediction": int(is_empty),
            })

            total_saved += 1

    with open(bboxes_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "slice_file",
                "image_png_path",
                "pred_binary_mask_path",
                "bbox_x_min",
                "bbox_y_min",
                "bbox_x_max",
                "bbox_y_max",
                "bbox_xyxy",
            ],
        )
        writer.writeheader()
        writer.writerows(bbox_rows)

    with open(records_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "slice_file",
                "image_png_path",
                "image_npy_path",
                "pred_binary_mask_path",
                "bbox_xyxy",
                "has_prediction",
                "is_empty_prediction",
            ],
        )
        writer.writeheader()
        writer.writerows(record_rows)

    print("\nDone.")
    print(f"Device: {device}")
    print(f"Model info: {model_info}")
    print(f"Output root: {out_dirs['root']}")
    print(f"Saved usable slices: {total_saved}")
    print(f"Skipped empty predictions: {total_empty}")
    print(f"BBoxes CSV: {bboxes_csv_path}")
    print(f"Records CSV: {records_csv_path}")


if __name__ == "__main__":
    main()