import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from train_unet import NPYFolderDataset, get_device, mean_dice
from unet2d import UNet2D


@torch.no_grad()
def main(seed, ckpt):
    # -------------------------
    # Config
    # -------------------------
    ckpt_path = ckpt
    data_root = "data"
    split = "test"
    batch_size = 8

    device = get_device()
    print("Device:", device)

    ckpt = torch.load(ckpt_path, map_location=device)

    in_channels = ckpt["in_channels"]
    num_classes = ckpt["num_classes"]
    base = ckpt["base"]

    print(f"Checkpoint loaded from: {ckpt_path}")
    print(f"in_channels={in_channels}, num_classes={num_classes}, base={base}")

    model = UNet2D(
        in_channels=in_channels,
        num_classes=num_classes,
        base=base
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    test_ds = NPYFolderDataset(
        root=data_root,
        split=split,
        normalize="zscore_per_channel"
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Number of test slices:", len(test_ds))

    total_dice = 0.0
    total_seen = 0

    for x, y, z in tqdm(test_loader, desc="Testing"):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        bs = x.size(0)
        batch_dice = mean_dice(
            pred,
            y,
            num_classes=num_classes,
            exclude_bg=True
        )

        total_dice += batch_dice * bs
        total_seen += bs

    mean_test_dice = total_dice / max(1, total_seen)

    print(f"\nTest mean Dice (exclude bg): {mean_test_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args.seed, args.ckpt)