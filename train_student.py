import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm
from utils import set_seed

from unet2d import UNet2D
from ssl_data import (
    get_device,
    NPYFolderDataset,
    mean_dice,
    soft_dice_loss,
    masked_cross_entropy,
    masked_soft_dice_loss,
    masked_soft_cross_entropy,
    masked_soft_dice_loss_probs,
    save_baseline_style_checkpoint,
    load_baseline_style_checkpoint,
)


def _compute_lambda_u(epoch_idx, total_epochs, base_lambda_u, schedule):
    if schedule == "constant":
        return base_lambda_u
    if schedule == "ramp":
        ramp_epochs = max(1, total_epochs // 2)
        progress = min(1.0, float(epoch_idx + 1) / float(ramp_epochs))
        return base_lambda_u * progress
    raise ValueError(f"Unsupported lambda schedule: {schedule}")


def train_model(params=None):
    params = params or {}
    seed = params.get("seed", 42)
    set_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = get_device()
    print("Device:", device)
    print("Seed:", seed)

    manifest_path = "ssl_split_manifest.csv"

    num_classes = 5
    in_channels = 4
    base = 32
    batch_size_l = params.get("batch_size_l", 8)
    batch_size_u = params.get("batch_size_u", 8)
    epochs = params.get("epochs", 50)
    lr = params.get("learning_rate", 1e-3)
    weight_decay = params.get("weight_decay", 0.0)

    # supervised branch weights: same style as baseline
    w_ce_sup = 0.5
    w_dice_sup = 0.5

    # unsupervised branch weights
    w_ce_unsup = 0.5
    w_dice_unsup = 0.5
    lambda_u = params.get("lambda_u", 0.4)
    lambda_schedule = params.get("lambda_schedule", "constant")
    conf_thresh = params.get("conf_thresh", 0.7)
    use_conf_mask = params.get("use_conf_mask", True)
    pseudo_label_mode = params.get("pseudo_label_mode", "soft")
    teacher_ckpt_path = params.get("teacher_ckpt_path", "teacher_best.pt")
    save_path = params.get("save_path", "student_best.pt")

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
    load_baseline_style_checkpoint(teacher_ckpt_path, teacher, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0

    for ep in range(epochs):
        student.train()
        running_total = 0.0
        running_sup = 0.0
        running_unsup = 0.0
        lambda_u_epoch = _compute_lambda_u(ep, epochs, lambda_u, lambda_schedule)

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
                conf_u = teacher_probs_u.max(dim=1).values
                if use_conf_mask:
                    valid_mask = (conf_u >= conf_thresh).float()
                else:
                    valid_mask = torch.ones_like(conf_u)

            # unsupervised branch
            logits_u = student(x_u)

            if valid_mask.sum().item() > 0:
                if pseudo_label_mode == "soft":
                    unsup_ce = masked_soft_cross_entropy(logits_u, teacher_probs_u, valid_mask)
                    unsup_dice = masked_soft_dice_loss_probs(
                        logits_u,
                        teacher_probs_u,
                        valid_mask,
                        exclude_bg=True,
                    )
                elif pseudo_label_mode == "hard":
                    teacher_labels_u = torch.argmax(teacher_probs_u, dim=1)
                    unsup_ce = masked_cross_entropy(logits_u, teacher_labels_u, valid_mask)
                    unsup_dice = masked_soft_dice_loss(
                        logits_u,
                        teacher_labels_u,
                        valid_mask,
                        num_classes=num_classes,
                        exclude_bg=True,
                    )
                else:
                    raise ValueError(f"Unsupported pseudo_label_mode: {pseudo_label_mode}")
                unsup_loss = w_ce_unsup * unsup_ce + w_dice_unsup * unsup_dice
            else:
                unsup_loss = torch.zeros((), device=device)

            loss = sup_loss + lambda_u_epoch * unsup_loss
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
            f"lambda_u={lambda_u_epoch:.4f} "
            f"train_total={train_loss:.4f} "
            f"train_sup={train_sup:.4f} "
            f"train_unsup={train_unsup:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice(excl_bg)={val_dice:.4f}"
        )

        if val_dice > best_val:
            best_val = val_dice
            save_baseline_style_checkpoint(
                save_path,
                student,
                in_channels=in_channels,
                num_classes=num_classes,
                base=base,
                best_val_dice=best_val,
            )
            print(f"{save_path} saved")

    print(f"Best student val Dice: {best_val:.4f}")
    return best_val


def main():
    train_model()


if __name__ == "__main__":
    main()
