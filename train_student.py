import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm

from unet2d import UNet2D
from ssl_data import (
    get_device,
    NPYFolderDataset,
    mean_dice,
    soft_dice_loss,
    masked_cross_entropy,
    masked_soft_dice_loss,
    save_baseline_style_checkpoint,
    load_baseline_style_checkpoint,
)


def main():
    device = get_device()
    print("Device:", device)

    manifest_path = "ssl_split_manifest.csv"

    num_classes = 5
    in_channels = 4
    base = 32
    batch_size_l = 8
    batch_size_u = 8
    epochs = 20
    lr = 1e-3

    # supervised branch weights: same style as baseline
    w_ce_sup = 0.5
    w_dice_sup = 0.5

    # unsupervised branch weights
    w_ce_unsup = 0.5
    w_dice_unsup = 0.5
    lambda_u = 1.0
    conf_thresh = 0.7

    labeled_train = NPYFolderDataset(
        root="data",
        split="train",
        normalize="zscore_per_channel",
        manifest_path=manifest_path,
        label_status="labeled",
        return_mask=True,
    )
    unlabeled_train = NPYFolderDataset(
        root="data",
        split="train",
        normalize="zscore_per_channel",
        manifest_path=manifest_path,
        label_status="unlabeled",
        return_mask=False,
    )
    val = NPYFolderDataset(
        root="data",
        split="val",
        normalize="zscore_per_channel",
        manifest_path=manifest_path,
        label_status=None,
        return_mask=True,
    )

    labeled_loader = DataLoader(labeled_train, batch_size=batch_size_l, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_train, batch_size=batch_size_u, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size_l, shuffle=False, num_workers=0)

    teacher = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    load_baseline_style_checkpoint("teacher_best.pt", teacher, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(student.parameters(), lr=lr)

    best_val = -1.0

    for ep in range(epochs):
        student.train()
        running_total = 0.0
        running_sup = 0.0
        running_unsup = 0.0

        labeled_iter = cycle(labeled_loader)
        unlabeled_iter = cycle(unlabeled_loader)
        steps = max(len(labeled_loader), len(unlabeled_loader))

        for _ in tqdm(range(steps), desc=f"Epoch {ep+1}/{epochs} [train-student]"):
            x_l, y_l = next(labeled_iter)
            x_u = next(unlabeled_iter)

            x_l = x_l.to(device)
            y_l = y_l.to(device)
            x_u = x_u.to(device)

            optim.zero_grad(set_to_none=True)

            # supervised branch
            logits_l = student(x_l)
            sup_ce = ce_criterion(logits_l, y_l)
            sup_dice = soft_dice_loss(logits_l, y_l, num_classes=num_classes, exclude_bg=True)
            sup_loss = w_ce_sup * sup_ce + w_dice_sup * sup_dice

            # teacher pseudo-labels
            with torch.no_grad():
                teacher_logits_u = teacher(x_u)
                teacher_probs_u = torch.softmax(teacher_logits_u, dim=1)
                conf_u, pseudo_y_u = torch.max(teacher_probs_u, dim=1)
                valid_mask = (conf_u >= conf_thresh).float()

            # unsupervised branch
            logits_u = student(x_u)

            if valid_mask.sum().item() > 0:
                unsup_ce = masked_cross_entropy(logits_u, pseudo_y_u, valid_mask)
                unsup_dice = masked_soft_dice_loss(
                    logits_u,
                    pseudo_y_u,
                    valid_mask,
                    num_classes=num_classes,
                    exclude_bg=True,
                )
                unsup_loss = w_ce_unsup * unsup_ce + w_dice_unsup * unsup_dice
            else:
                unsup_loss = torch.zeros((), device=device)

            loss = sup_loss + lambda_u * unsup_loss
            loss.backward()
            optim.step()

            running_total += loss.item()
            running_sup += sup_loss.item()
            running_unsup += unsup_loss.item()

        train_loss = running_total / steps
        train_sup = running_sup / steps
        train_unsup = running_unsup / steps

        student.eval()
        vloss = 0.0
        vdice = 0.0
        nseen = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {ep+1}/{epochs} [val-student]"):
                x = x.to(device)
                y = y.to(device)

                logits = student(x)

                ce = ce_criterion(logits, y)
                dl = soft_dice_loss(logits, y, num_classes=num_classes, exclude_bg=True)
                loss = w_ce_sup * ce + w_dice_sup * dl

                vloss += loss.item() * x.size(0)

                pred = torch.argmax(logits, dim=1)
                bs = x.size(0)
                vdice += mean_dice(pred, y, num_classes=num_classes, exclude_bg=True) * bs
                nseen += bs

        val_loss = vloss / len(val_loader.dataset)
        val_dice = vdice / max(1, nseen)

        print(
            f"Epoch {ep+1}: "
            f"train_total={train_loss:.4f} "
            f"train_sup={train_sup:.4f} "
            f"train_unsup={train_unsup:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice(excl_bg)={val_dice:.4f}"
        )

        if val_dice > best_val:
            best_val = val_dice
            save_baseline_style_checkpoint(
                "student_best.pt",
                student,
                in_channels=in_channels,
                num_classes=num_classes,
                base=base,
                best_val_dice=best_val,
            )
            print("student_best.pt saved")

    print(f"Best student val Dice: {best_val:.4f}")


if __name__ == "__main__":
    main()