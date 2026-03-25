import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet2d import UNet2D
from ssl_data import (
    get_device,
    NPYFolderDataset,
    mean_dice,
    soft_dice_loss,
    save_baseline_style_checkpoint,
)


def train_model(params=None):
    device = get_device()
    print("Device:", device)

    manifest_path = "ssl_split_manifest.csv"

    params = params or {}

    num_classes = 5
    in_channels = 4
    base = 32
    batch_size = params.get("batch_size", 8)
    epochs = params.get("epochs", 33)
    lr = params.get("learning_rate", 4.6549492616937974e-4)
    weight_decay = params.get("weight_decay", 2.7858130122219456e-4)
    w_ce = 0.5
    w_dice = 0.5

    train = NPYFolderDataset(
        root="data",
        split="train",
        normalize="zscore_per_channel",
        manifest_path=manifest_path,
        label_status="labeled",
        return_mask=True,
    )
    val = NPYFolderDataset(
        root="data",
        split="val",
        normalize="zscore_per_channel",
        manifest_path=manifest_path,
        label_status=None,
        return_mask=True,
    )

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0

    for ep in range(epochs):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs} [train-teacher]"):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)

            ce = ce_criterion(logits, y)
            dl = soft_dice_loss(logits, y, num_classes=num_classes, exclude_bg=True)
            loss = w_ce * ce + w_dice * dl

            loss.backward()
            optim.step()

            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)

        model.eval()
        vloss = 0.0
        vdice = 0.0
        nseen = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {ep+1}/{epochs} [val-teacher]"):
                x = x.to(device)
                y = y.to(device)

                logits = model(x)

                ce = ce_criterion(logits, y)
                dl = soft_dice_loss(logits, y, num_classes=num_classes, exclude_bg=True)
                loss = w_ce * ce + w_dice * dl

                vloss += loss.item() * x.size(0)

                pred = torch.argmax(logits, dim=1)
                bs = x.size(0)
                vdice += mean_dice(pred, y, num_classes=num_classes, exclude_bg=True) * bs
                nseen += bs

        val_loss = vloss / len(val_loader.dataset)
        val_dice = vdice / max(1, nseen)

        print(
            f"Epoch {ep+1}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice(excl_bg)={val_dice:.4f}"
        )

        if val_dice > best_val:
            best_val = val_dice
            save_baseline_style_checkpoint(
                "teacher_best.pt",
                model,
                in_channels=in_channels,
                num_classes=num_classes,
                base=base,
                best_val_dice=best_val,
            )
            print("teacher_best.pt saved")

    print(f"Best teacher val Dice: {best_val:.4f}")
    return best_val


def main():
    train_model()


if __name__ == "__main__":
    main()
