from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np


def normalize_for_display(arr):
    """
    Applies robust normalization to an array for plotting.

    Args:
        arr (np.ndarray): The input image array.

    Returns:
        np.ndarray: The normalized array scaled between 0 and 1.
    """
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

    Args:
        image_npy_path (str or Path): Path to .npy file containing one slice.
        mask_npy_path (str or Path): Path to .npy file containing the mask.
        output_path (str or Path, optional): Path to save the plot.
        modality_names (tuple, optional): Names of the 4 modalities.
        overlay_modality (int, optional): Modality index for mask overlay.
        mask_alpha (float, optional): Transparency of the mask overlay.

    Raises:
        ValueError: If input dimensions or shapes are incompatible.
    """
    image_npy_path = Path(image_npy_path)
    mask_npy_path = Path(mask_npy_path)

    img = np.load(image_npy_path)
    mask = np.load(mask_npy_path)

    # Validate dimensions and reshape if necessary
    if img.ndim != 3:
        raise ValueError(f"Expected image array with 3 dims, got {img.shape}")

    if img.shape[0] == 4:
        pass
    elif img.shape[-1] == 4:
        img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(
            f"Expected image shape (4, H, W) or (H, W, 4), got {img.shape}"
        )

    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (H, W), got {mask.shape}")

    if img.shape[1:] != mask.shape:
        raise ValueError(
            f"Image shape {img.shape[1:]} does not match mask {mask.shape}"
        )

    if len(modality_names) != 4:
        raise ValueError("modality_names must contain exactly 4 names.")

    # Process and display
    display_imgs = [normalize_for_display(img[c]) for c in range(4)]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for i in range(4):
        axes[i].imshow(display_imgs[i], cmap="gray")
        axes[i].set_title(modality_names[i], fontsize=11)
        axes[i].axis("off")

    # Overlay panel
    base = display_imgs[overlay_modality]

    # (background + 4 tumor regions)
    colors = [
        (0, 0, 0, 0),      # 0 = background (transparent)
        (1, 0, 0, 1),      # 1 = red
        (0, 1, 0, 1),      # 2 = green
        (0, 0, 1, 1),      # 3 = blue
        (1, 1, 0, 1),      # 4 = yellow
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

    # overlay
    axes[4].imshow(base, cmap="gray")
    axes[4].imshow(
        mask,
        cmap=cmap,
        norm=norm,
        alpha=mask_alpha,
        interpolation="nearest"
    )
    axes[4].set_title(
        f"{modality_names[overlay_modality]} + Mask", fontsize=11
    )
    axes[4].axis("off")

    fig.suptitle("Multi-modal Slice Visualization with Mask Overlay",
                 fontsize=14)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    plt.show()


def find_good_slices(
    folder_path,
    min_pixels_per_class=50,   # threshold for "good distribution"
    required_classes=(1, 2, 3, 4),
    axis=0                     # slicing axis (0 = z for (Z,H,W))
):
    """
    Iterate through .npy segmentation files and find slices that
    contain all required classes with sufficient pixel counts.
    """

    folder = Path(folder_path)
    files = sorted(folder.glob("*.npy"))

    for f in files:
        seg = np.load(f)  # expected shape (Z, H, W) or similar

        # move slicing axis to front for uniform iteration
        seg = np.moveaxis(seg, axis, 0)

        for i in range(seg.shape[0]):
            slice_ = seg[i]

            # count pixels for each class
            valid = True
            counts = {}

            for c in required_classes:
                count = np.sum(slice_ == c)
                counts[c] = int(count)

                if count < min_pixels_per_class:
                    valid = False
                    break

            if valid:
                print(f"{f.name} | slice {i} | counts: {counts}")
                break  # stop after first good slice per file


if __name__ == "__main__":
    visualize_multimodal_slice(
        image_npy_path=r"self_supervised_learning\SegMentors\data\test\images\BraTS-PED-00034-000__000003__img.npy",
        mask_npy_path=r"self_supervised_learning\SegMentors\data\test\masks\BraTS-PED-00034-000__000003__mask.npy",
        output_path="output.png"
    )
    
    # find_good_slices(
    #     folder_path=r"self_supervised_learning\SegMentors\data\test\masks",
    #     min_pixels_per_class=10
    # )

    # import nibabel as nib

    # nii_path = r"eda\SegMentors\visuals\BraTS-PED-00034-000__000003_nnunet.nii.gz"
    # npy_path = r"eda\SegMentors\visuals\BraTS-PED-00034-000__000003_nnunet.npy"

    # img = nib.load(nii_path)
    # data = img.get_fdata()   # float64

    # # optional: cast to smaller dtype
    # data = data.astype(np.float32)
    # data = np.squeeze(data, axis=None)
    # print(data.shape)

    # np.save(npy_path, data)