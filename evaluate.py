import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet2d import UNet2D
from ssl_data import (
    get_device,
    NPYFolderDataset,
    mean_dice,
    load_baseline_style_checkpoint,
)


def main():
    device = get_device()
    print("Device:", device)

    manifest_path = "ssl_split_manifest.csv"

    ckpt_path = "teacher_best.pt"   # change to "student_best.pt" when needed
    split = "test"                  # "val" or "test"

    dataset = NPYFolderDataset(
        root="data",
        split=split,
        normalize="zscore_per_channel",
        manifest_path=manifest_path,
        label_status=None,
        return_mask=True,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = ckpt["in_channels"]
    num_classes = ckpt["num_classes"]
    base = ckpt["base"]

    model = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    load_baseline_style_checkpoint(ckpt_path, model, device)
    model.eval()

    total_dice = 0.0
    nseen = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"[eval {split}]"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            bs = x.size(0)
            total_dice += mean_dice(pred, y, num_classes=num_classes, exclude_bg=True) * bs
            nseen += bs

    mean_val = total_dice / max(1, nseen)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Split: {split}")
    print(f"Mean Dice (exclude background): {mean_val:.4f}")


if __name__ == "__main__":
    main()