import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet2d import UNet2D
from ssl_data import (
    get_device,
    NPYFolderDataset,
    load_baseline_style_checkpoint,
    per_class_dice,
)


def _class_labels(num_classes, exclude_bg=True):
    start_class = 1 if exclude_bg else 0
    return [f"class_{class_idx}" for class_idx in range(start_class, num_classes)]


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes, exclude_bg=True):
    model.eval()
    class_labels = _class_labels(num_classes, exclude_bg=exclude_bg)
    per_case_rows = []
    total_mean_dice = 0.0
    total_per_class = torch.zeros(len(class_labels), dtype=torch.float64)
    nseen = 0

    for batch in tqdm(loader, desc="[evaluate]"):
        x, y, case_ids = batch
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        batch_per_class = per_class_dice(
            pred,
            y,
            num_classes=num_classes,
            exclude_bg=exclude_bg,
        ).cpu()
        batch_mean = batch_per_class.mean(dim=1)

        bs = x.size(0)
        total_mean_dice += batch_mean.sum().item()
        total_per_class += batch_per_class.sum(dim=0).to(torch.float64)
        nseen += bs

        for idx in range(bs):
            row = {
                "case_id": case_ids[idx],
                "mean_dice": float(batch_mean[idx].item()),
            }
            for class_name, class_dice in zip(class_labels, batch_per_class[idx]):
                row[class_name] = float(class_dice.item())
            per_case_rows.append(row)

    if nseen == 0:
        raise ValueError("Evaluation loader produced zero samples")

    per_class_mean = (total_per_class / nseen).tolist()
    summary = {
        "mean_dice": total_mean_dice / nseen,
        "per_class_dice": {
            class_name: float(score)
            for class_name, score in zip(class_labels, per_class_mean)
        },
        "num_cases": nseen,
    }
    return {
        "summary": summary,
        "per_case_rows": per_case_rows,
    }


def _write_per_case_csv(path, rows, class_labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "mean_dice", *class_labels]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def evaluate_checkpoint(
    ckpt_path,
    split="val",
    manifest_path="ssl_split_manifest.csv",
    batch_size=8,
    exclude_bg=True,
    output_dir=None,
    output_prefix=None,
):
    device = get_device()
    dataset = NPYFolderDataset(
        root="data",
        split=split,
        normalize="zscore_per_channel",
        manifest_path=manifest_path,
        label_status=None,
        return_mask=True,
        return_case_id=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = ckpt["in_channels"]
    num_classes = ckpt["num_classes"]
    base = ckpt["base"]

    model = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    load_baseline_style_checkpoint(ckpt_path, model, device)

    results = evaluate_model(
        model,
        loader,
        device,
        num_classes=num_classes,
        exclude_bg=exclude_bg,
    )
    payload = {
        "checkpoint": str(ckpt_path),
        "split": split,
        **results,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_prefix = output_prefix or f"{Path(ckpt_path).stem}_{split}"
        class_labels = list(results["summary"]["per_class_dice"].keys())
        _write_summary_json(output_dir / f"{output_prefix}_summary.json", payload)
        _write_per_case_csv(
            output_dir / f"{output_prefix}_per_case.csv",
            results["per_case_rows"],
            class_labels,
        )

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="teacher_best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--manifest-path", default="ssl_split_manifest.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    results = evaluate_checkpoint(
        ckpt_path=args.ckpt,
        split=args.split,
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )

    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Split: {results['split']}")
    print(f"Mean Dice (exclude background): {results['summary']['mean_dice']:.4f}")
    print("Per-class Dice:")
    for class_name, score in results["summary"]["per_class_dice"].items():
        print(f"  {class_name}: {score:.4f}")
    if args.output_dir is not None:
        print(f"Saved evaluation artifacts to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
