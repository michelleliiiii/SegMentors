from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def visualize_multimodal_slice(
    image_npy_path,
    mask_npy_path,
    output_path=None,
    modality_names=("T1", "T1ce", "T2", "FLAIR"),
    overlay_modality=3,
    mask_alpha=0.4,
):
    """
    Visualize one 4-channel MRI slice and its segmentation mask.

    Parameters
    ----------
    image_npy_path : str or Path
        Path to .npy file containing one slice with shape (4, H, W)
        or (H, W, 4).
    mask_npy_path : str or Path
        Path to .npy file containing one segmentation mask with shape (H, W).
    output_path : str or Path or None
        If provided, save the visualization image to this path.
    modality_names : tuple[str, ...]
        Names of the 4 modalities.
    overlay_modality : int
        Which modality to use as the base image for mask overlay.
        Default 3 -> FLAIR.
    mask_alpha : float
        Transparency of the segmentation overlay.
    """
    image_npy_path = Path(image_npy_path)
    mask_npy_path = Path(mask_npy_path)

    img = np.load(image_npy_path)
    mask = np.load(mask_npy_path)

    # Accept either (4, H, W) or (H, W, 4)
    if img.ndim != 3:
        raise ValueError(f"Expected image array with 3 dims, got shape {img.shape}")

    if img.shape[0] == 4:
        # already (4, H, W)
        pass
    elif img.shape[-1] == 4:
        img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(
            f"Expected image shape (4,H,W) or (H,W,4), got {img.shape}"
        )

    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (H,W), got {mask.shape}")

    if img.shape[1:] != mask.shape:
        raise ValueError(
            f"Image spatial shape {img.shape[1:]} does not match mask shape {mask.shape}"
        )

    if len(modality_names) != 4:
        raise ValueError("modality_names must contain exactly 4 names.")

    def normalize_for_display(arr):
        """Robust normalization for plotting."""
        arr = arr.astype(np.float32)
        nonzero = arr[arr != 0]
        if nonzero.size == 0:
            return np.zeros_like(arr, dtype=np.float32)

        p1 = np.percentile(nonzero, 1)
        p99 = np.percentile(nonzero, 99)

        if p99 <= p1:
            return np.zeros_like(arr, dtype=np.float32)

        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1)
        return arr

    display_imgs = [normalize_for_display(img[c]) for c in range(4)]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for i in range(4):
        axes[i].imshow(display_imgs[i], cmap="gray")
        axes[i].set_title(modality_names[i], fontsize=11)
        axes[i].axis("off")

    # Overlay panel
    base = display_imgs[overlay_modality]
    axes[4].imshow(base, cmap="gray")
    axes[4].imshow(mask, cmap="jet", alpha=mask_alpha, interpolation="nearest")
    axes[4].set_title(f"{modality_names[overlay_modality]} + Mask", fontsize=11)
    axes[4].axis("off")

    fig.suptitle("Multi-modal Slice Visualization with Mask Overlay", fontsize=14)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    plt.show()


# Example usage
visualize_multimodal_slice(
    image_npy_path="data\BraTS-PED-00001-000\BraTS-PED-00001-000_s85_img.npy",
    mask_npy_path="data\BraTS-PED-00001-000\BraTS-PED-00001-000_s85_seg.npy",
    output_path="output.png"
)