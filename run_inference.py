from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from load_dataset import ensure_chw, normalize_tensor
from unet2d import UNet2D
import utils as U


MASK_CMAP = ListedColormap(
    [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.2, 0.2, 0.7),
        (0.2, 0.9, 0.2, 0.7),
        (1.0, 0.85, 0.2, 0.7),
        (0.2, 0.5, 1.0, 0.7),
    ]
)


def infer_single_npy(model_path, input_npy, output_dir, device=None):
    """Load a checkpoint, infer one input `.npy`, and save the predicted mask.

    Args:
        model_path (str | Path): Path to a trained `.pt` checkpoint file.
        input_npy (str | Path): Path to one input `.npy` MRI slice.
        output_dir (str | Path): Directory where the prediction should be saved.
        device (torch.device | str | None): Device used for inference.

    Returns:
        tuple[np.ndarray, Path]: The predicted mask array and its saved path.
    """
    input_npy = Path(input_npy)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = U.get_device()
    device = torch.device(device)

    checkpoint = torch.load(model_path, map_location=device)
    in_channels = checkpoint.get("in_channels", 4)
    num_classes = checkpoint.get("num_classes", 5)
    base = checkpoint.get("base", 32)

    model = UNet2D(
        in_channels=in_channels,
        num_classes=num_classes,
        base=base,
        head_channels=num_classes,
    )
    model.load_state_dict(checkpoint["model"])

    image = np.load(input_npy)
    image = ensure_chw(image)
    image_tensor = torch.from_numpy(image).float()
    image_tensor = normalize_tensor(image_tensor, mode="zscore_per_channel")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        prediction = (
            torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        )

    pred_path = output_dir / input_npy.name.replace("__img.npy", "__pred.npy")
    np.save(pred_path, prediction)
    return prediction, pred_path


def visualize_prediction(input_npy, predicted_mask, ground_truth_mask, output_path):
    """Visualize one MRI channel with predicted and ground-truth masks overlaid.

    Args:
        input_npy (str | Path): Path to one input `.npy` MRI slice.
        predicted_mask (str | Path | np.ndarray): Predicted segmentation mask.
        ground_truth_mask (str | Path | np.ndarray): Ground-truth segmentation mask.
        output_path (str | Path): Path to save the visualization figure.

    Returns:
        float: Mean Dice score excluding background.
    """
    image = np.load(input_npy)
    image = ensure_chw(image).astype(np.float32)

    if isinstance(predicted_mask, (str, Path)):
        predicted_mask = np.load(predicted_mask)
    if isinstance(ground_truth_mask, (str, Path)):
        ground_truth_mask = np.load(ground_truth_mask)

    display_channel = image[0]
    min_val = float(display_channel.min())
    max_val = float(display_channel.max())
    if max_val > min_val:
        display_channel = (display_channel - min_val) / (max_val - min_val)
    else:
        display_channel = np.zeros_like(display_channel)

    mean_dice, _ = U.mean_dice(
        torch.from_numpy(np.asarray(predicted_mask)).unsqueeze(0),
        torch.from_numpy(np.asarray(ground_truth_mask)).unsqueeze(0),
        num_classes=5,
        exclude_bg=True,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(display_channel, cmap="gray")
    axes[0].imshow(ground_truth_mask, cmap=MASK_CMAP, vmin=0, vmax=4)
    axes[0].set_title("Ground Truth Overlay")
    axes[0].axis("off")

    axes[1].imshow(display_channel, cmap="gray")
    axes[1].imshow(predicted_mask, cmap=MASK_CMAP, vmin=0, vmax=4)
    axes[1].set_title("Prediction Overlay")
    axes[1].text(
        0.02,
        0.98,
        f"Mean Dice: {mean_dice:.4f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.7, "pad": 4},
    )
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return mean_dice
