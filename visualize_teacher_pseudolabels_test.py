import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from ssl_data import NPYFolderDataset, get_device, load_baseline_style_checkpoint
from unet2d import UNet2D


def normalize_for_display(arr):
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    p1 = np.percentile(finite, 1)
    p99 = np.percentile(finite, 99)
    if p99 <= p1:
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, p1, p99)
    return (arr - p1) / (p99 - p1)


def build_label_cmap(num_classes):
    base_colors = [
        (0.0, 0.0, 0.0, 0.0),
        (0.89, 0.10, 0.11, 0.75),
        (0.22, 0.49, 0.72, 0.75),
        (0.30, 0.69, 0.29, 0.75),
        (1.00, 0.50, 0.00, 0.75),
        (0.60, 0.31, 0.64, 0.75),
        (0.65, 0.34, 0.16, 0.75),
        (0.97, 0.51, 0.75, 0.75),
    ]
    if num_classes > len(base_colors):
        raise ValueError(f"Need at least {num_classes} colors, only {len(base_colors)} defined")
    return ListedColormap(base_colors[:num_classes])


@torch.no_grad()
def predict_pseudolabel(model, x, device):
    logits = model(x.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1)
    pseudo = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    conf = probs.max(dim=1).values.squeeze(0).cpu().numpy()
    return pseudo, conf


def main():
    parser = argparse.ArgumentParser(
        description="Visualize teacher pseudolabels for random samples from the test split."
    )
    parser.add_argument("--ckpt", required=True, help="Path to teacher checkpoint")
    parser.add_argument("--manifest-path", default="ssl_split_manifest.csv")
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modality", type=int, default=3, help="Channel index to use as grayscale base")
    parser.add_argument(
        "--all-modalities",
        action="store_true",
        help="Render all input modalities for each sampled case instead of a single chosen modality.",
    )
    parser.add_argument("--output-dir", default="experiment_outputs/teacher_pseudolabel_viz")
    parser.add_argument("--output-name", default="teacher_pseudolabels_test.png")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet2D(
        in_channels=ckpt["in_channels"],
        num_classes=ckpt["num_classes"],
        base=ckpt["base"],
    ).to(device)
    load_baseline_style_checkpoint(args.ckpt, model, device)
    model.eval()

    dataset = NPYFolderDataset(
        root="data",
        split="test",
        normalize="zscore_per_channel",
        manifest_path=args.manifest_path,
        label_status=None,
        return_mask=True,
        return_case_id=True,
    )

    if len(dataset) == 0:
        raise ValueError("Test dataset is empty")
    if args.modality < 0 or args.modality >= ckpt["in_channels"]:
        raise ValueError(f"--modality must be in [0, {ckpt['in_channels'] - 1}]")

    rng = random.Random(args.seed)
    sample_count = min(args.num_samples, len(dataset))
    indices = rng.sample(range(len(dataset)), sample_count)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_modalities:
        rows = sample_count
        cols = ckpt["in_channels"]
    else:
        rows = int(np.ceil(sample_count / 5))
        cols = min(5, sample_count)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    cmap = build_label_cmap(ckpt["num_classes"])

    sampled_cases = []

    modality_names = ["T1", "T1ce", "T2", "FLAIR"]

    for plot_idx, data_idx in enumerate(indices):
        x, y, case_id = dataset[data_idx]
        pseudo, conf = predict_pseudolabel(model, x, device)
        pseudo_masked = np.ma.masked_where(pseudo == 0, pseudo)

        if args.all_modalities:
            for modality_idx in range(ckpt["in_channels"]):
                ax = axes[plot_idx, modality_idx]
                base = normalize_for_display(x[modality_idx].cpu().numpy())
                ax.imshow(base, cmap="gray")
                ax.imshow(
                    pseudo_masked,
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=0,
                    vmax=ckpt["num_classes"] - 1,
                )
                title_prefix = modality_names[modality_idx] if modality_idx < len(modality_names) else f"mod_{modality_idx}"
                ax.set_title(f"{case_id} | {title_prefix}\nmean conf={conf.mean():.3f}", fontsize=9)
                ax.axis("off")
        else:
            r = plot_idx // cols
            c = plot_idx % cols
            ax = axes[r, c]
            base = normalize_for_display(x[args.modality].cpu().numpy())
            ax.imshow(base, cmap="gray")
            ax.imshow(
                pseudo_masked,
                cmap=cmap,
                interpolation="nearest",
                vmin=0,
                vmax=ckpt["num_classes"] - 1,
            )
            title_prefix = modality_names[args.modality] if args.modality < len(modality_names) else f"mod_{args.modality}"
            ax.set_title(f"{case_id} | {title_prefix}\nmean conf={conf.mean():.3f}", fontsize=10)
            ax.axis("off")

        sampled_cases.append(
            {
                "case_id": case_id,
                "dataset_index": data_idx,
                "mean_confidence": float(conf.mean()),
                "foreground_fraction": float((pseudo > 0).mean()),
            }
        )

    if not args.all_modalities:
        for empty_idx in range(sample_count, rows * cols):
            r = empty_idx // cols
            c = empty_idx % cols
            axes[r, c].axis("off")

    if args.all_modalities:
        fig.suptitle("Teacher Pseudolabels on Random Test Samples Across All Modalities", fontsize=16)
    else:
        fig.suptitle("Teacher Pseudolabels on Random Test Samples", fontsize=16)
    fig.tight_layout()
    image_path = output_dir / args.output_name
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    metadata_path = output_dir / f"{Path(args.output_name).stem}_samples.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(Path(args.ckpt).resolve()),
                "split": "test",
                "seed": args.seed,
                "num_samples": sample_count,
                "modality": args.modality,
                "samples": sampled_cases,
            },
            f,
            indent=2,
        )

    print(f"Saved visualization grid to: {image_path.resolve()}")
    print(f"Saved sampled case metadata to: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
