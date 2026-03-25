import argparse
import contextlib
from pathlib import Path
import sys
import torch

from unet2d import UNet2D
import load_dataset as D
import pretraining as P
import fine_tuning as T
from fine_tuning import CLASS_NAMES
import utils as U

PRETEXT_TASKS = {
    "cross_modal": P.build_cross_modal_mask,
    "random_channel_dropout": P.build_random_channel_dropout_mask,
}


class TeeStream:
    """Write console output to multiple streams at the same time."""

    def __init__(self, *streams):
        """Store the output streams that should receive each message."""
        self.streams = streams

    def write(self, data):
        """Write text data to every configured stream."""
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        """Flush every configured stream."""
        for stream in self.streams:
            stream.flush()


def run_pipeline(args):
    """Execute pretraining, fine-tuning, checkpointing, and test inference."""
    U.set_seed(args.seed)
    device = U.get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "pretrain_train_loss": [],
        "pretrain_val_loss": [],
        "finetune_train_loss": [],
        "finetune_val_loss": [],
    }

    print(f"Device: {device}")
    print(f"Pretext task: {args.pretext_task}")

    pretrain_train_ds = D.BrainTumorNPYDataset(
        root=args.data_root,
        split="train",
        manifest_csv=args.manifest_csv,
        label_status=None,
        require_masks=False,
    )
    pretrain_val_ds = D.BrainTumorNPYDataset(
        root=args.data_root,
        split="val",
        manifest_csv=args.manifest_csv,
        label_status=None,
        require_masks=False,
    )
    finetune_train_ds = D.BrainTumorNPYDataset(
        root=args.data_root,
        split="train",
        manifest_csv=args.manifest_csv,
        label_status="labeled",
        require_masks=True,
    )
    seg_val_ds = D.BrainTumorNPYDataset(
        root=args.data_root,
        split="val",
        manifest_csv=args.manifest_csv,
        label_status=None,
        require_masks=True,
    )
    test_ds = D.BrainTumorNPYDataset(
        root=args.data_root,
        split="test",
        manifest_csv=args.manifest_csv,
        label_status=None,
        require_masks=True,
    )

    print(
        "Dataset sizes:",
        f"pretrain_train={len(pretrain_train_ds)}",
        f"pretrain_val={len(pretrain_val_ds)}",
        f"finetune_train={len(finetune_train_ds)}",
        f"val={len(seg_val_ds)}",
        f"test={len(test_ds)}",
    )

    pretrain_train_loader = D.make_dataloader(
        pretrain_train_ds, args.batch_size, True, args.seed, args.num_workers
    )
    pretrain_val_loader = D.make_dataloader(
        pretrain_val_ds, args.batch_size, False, args.seed + 1, args.num_workers
    )
    finetune_train_loader = D.make_dataloader(
        finetune_train_ds, args.batch_size, True, args.seed + 2, args.num_workers
    )
    seg_val_loader = D.make_dataloader(
        seg_val_ds, args.batch_size, False, args.seed + 3, args.num_workers
    )
    test_loader = D.make_dataloader(
        test_ds, args.batch_size, False, args.seed + 4, args.num_workers
    )

    masker_fn = PRETEXT_TASKS[args.pretext_task]

    model = UNet2D(
        in_channels=args.in_channels,
        num_classes=args.in_channels,
        base=args.base,
        head_channels=args.in_channels,
    ).to(device)

    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
    best_pretrain_val = float("inf")
    best_pretrain_path = output_dir / "pretrain_best.pt"

    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss = P.run_pretraining_epoch(
            model,
            pretrain_train_loader,
            pretrain_optimizer,
            device,
            masker_fn,
            args,
            epoch_seed=args.seed * 1000 + epoch,
        )
        val_loss = P.run_pretraining_validation(
            model,
            pretrain_val_loader,
            device,
            masker_fn,
            args,
            epoch_seed=args.seed * 2000 + epoch,
        )

        print(f"Pretrain epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        history["pretrain_train_loss"].append(train_loss)
        history["pretrain_val_loss"].append(val_loss)

        if val_loss < best_pretrain_val:
            best_pretrain_val = val_loss
            U.save_checkpoint(
                best_pretrain_path,
                model,
                pretrain_optimizer,
                epoch,
                {"train_loss": train_loss, "val_loss": val_loss},
                extra={
                    "stage": "pretrain",
                    "pretext_task": args.pretext_task,
                    "in_channels": args.in_channels,
                    "base": args.base,
                },
            )

    pretrain_state = torch.load(best_pretrain_path, map_location=device)
    model.load_state_dict(pretrain_state["model"])
    model.replace_head(args.num_classes)
    model = model.to(device)

    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    best_val_dice = -1.0
    best_finetune_path = output_dir / "finetune_best.pt"

    print("Starting supervised fine-tuning on labeled training cases only.")
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = T.run_segmentation_epoch(
            model,
            finetune_train_loader,
            finetune_optimizer,
            device,
            args.num_classes,
            args.w_ce,
            args.w_dice,
        )
        val_metrics = T.run_segmentation_validation(
            model,
            seg_val_loader,
            device,
            args.num_classes,
            args.w_ce,
            args.w_dice,
        )

        print(f"Finetune epoch {epoch}: train_loss={train_loss:.4f} {U.format_seg_metrics(val_metrics)}")
        history["finetune_train_loss"].append(train_loss)
        history["finetune_val_loss"].append(val_metrics["loss"])

        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            U.save_checkpoint(
                best_finetune_path,
                model,
                finetune_optimizer,
                epoch,
                {"train_loss": train_loss, **val_metrics},
                extra={
                    "stage": "finetune",
                    "in_channels": args.in_channels,
                    "num_classes": args.num_classes,
                    "base": args.base,
                    "class_names": CLASS_NAMES,
                },
            )

    finetune_state = torch.load(best_finetune_path, map_location=device)
    model.load_state_dict(finetune_state["model"])

    prediction_dir = output_dir / "test_predictions"
    test_metrics = T.run_segmentation_validation(
        model,
        test_loader,
        device,
        args.num_classes,
        args.w_ce,
        args.w_dice,
        save_predictions_dir=prediction_dir,
    )

    print(f"Test metrics: {U.format_seg_metrics(test_metrics)}")
    print(f"Saved predicted masks to: {prediction_dir}")
    loss_figure_path = output_dir / "loss_curves.png"
    U.save_loss_figure(history, loss_figure_path)
    print(f"Saved loss figure to: {loss_figure_path}")


def build_arg_parser():
    """Build the command-line parser for the SSL training workflow."""
    parser = argparse.ArgumentParser(description="Self-supervised 2D U-Net training pipeline")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest-csv", default="ssl_split_manifest_20.csv")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--w-ce", type=float, default=0.5)
    parser.add_argument("--w-dice", type=float, default=0.5)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--pretext-task", choices=sorted(PRETEXT_TASKS.keys()), default="cross_modal")
    parser.add_argument("--output-dir", default="outputs/ssl_unet2d")
    return parser


def parse_args():
    """Parse command-line arguments for training."""
    return build_arg_parser().parse_args()


def main():
    """Parse arguments and start the end-to-end training pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    with log_path.open("a", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print("=" * 80)
            print("Starting training run")
            print(f"Output directory: {output_dir}")
            print(f"Log file: {log_path}")
            print(f"Arguments: {vars(args)}")
            run_pipeline(args)


if __name__ == "__main__":
    main()
