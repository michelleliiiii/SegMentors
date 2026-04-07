import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_unet import NPYFolderDataset, get_device
from unet2d import UNet2D


@torch.no_grad()
def dice_per_class(pred, target, num_classes, eps=1e-6, exclude_bg=True):

    start_c = 1 if exclude_bg else 0
    class_dices = []

    for c in range(start_c, num_classes):
        pred_c = (pred == c).float()
        targ_c = (target == c).float()

        inter = (pred_c * targ_c).sum(dim=(1, 2))
        denom = pred_c.sum(dim=(1, 2)) + targ_c.sum(dim=(1, 2))

        dice_c = (2 * inter + eps) / (denom + eps)  
        class_dices.append(dice_c.mean())

    return torch.stack(class_dices)  


@torch.no_grad()
def main(seed, ckpt):
    
    torch.manual_seed(seed)

    data_root = "data"
    split = "test"
    batch_size = 8

    device = get_device()
    print("Device:", device)

    weights = torch.load(ckpt, map_location=device)

    in_channels = weights["in_channels"]
    num_classes = weights["num_classes"]
    base = weights["base"]

    print(f"Checkpoint loaded from: {ckpt}")
    print(f"in_channels={in_channels}, num_classes={num_classes}, base={base}")

    model = UNet2D(
        in_channels=in_channels,
        num_classes=num_classes,
        base=base
    ).to(device)

    model.load_state_dict(weights["model"])
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

    total_seen = 0
    total_class_dice = torch.zeros(num_classes - 1, device=device)

    for x, y in tqdm(test_loader, desc="Testing"):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        bs = x.size(0)
        batch_class_dice = dice_per_class(
            pred,
            y,
            num_classes=num_classes,
            exclude_bg=True
        )

        total_class_dice += batch_class_dice * bs
        total_seen += bs

    mean_class_dice = total_class_dice / max(1, total_seen)
    mean_test_dice = mean_class_dice.mean().item()

    print(f"\nTest mean Dice (exclude bg): {mean_test_dice:.4f}")

    for i, d in enumerate(mean_class_dice, start=1):
        print(f"Test Dice class {i}: {d.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args.seed, args.ckpt)