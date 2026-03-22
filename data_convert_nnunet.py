from pathlib import Path
import numpy as np
import nibabel as nib


def save_array_as_nifti(array: np.ndarray, out_path: Path, affine: np.ndarray = None) -> None:
    """
    Save a numpy array as .nii.gz using nibabel.

    Parameters
    ----------
    array : np.ndarray
        Array to save. For 2D slices, this should usually be expanded to (H, W, 1).
    out_path : Path
        Output .nii.gz path.
    affine : np.ndarray, optional
        4x4 affine matrix. Identity is used if not provided.
    """
    if affine is None:
        affine = np.eye(4, dtype=np.float32)

    nii = nib.Nifti1Image(array, affine)
    nib.save(nii, str(out_path))


def convert_one_case(
    img_npy_path: Path,
    seg_npy_path: Path,
    case_id: str,
    images_out_dir: Path,
    labels_out_dir: Path,
    affine: np.ndarray = None
) -> None:
    """
    Convert one image-mask pair into nnU-Net style NIfTI files.

    Expected input:
        image: (H, W, 4)
        seg:   (H, W)

    Output:
        imagesTr/case_id_0000.nii.gz
        imagesTr/case_id_0001.nii.gz
        imagesTr/case_id_0002.nii.gz
        imagesTr/case_id_0003.nii.gz
        labelsTr/case_id.nii.gz
    """
    img = np.load(img_npy_path)
    seg = np.load(seg_npy_path)

    if img.ndim != 3:
        raise ValueError(f"Expected image shape (H, W, 4), got {img.shape} for {img_npy_path}")
    if seg.ndim != 2:
        raise ValueError(f"Expected seg shape (H, W), got {seg.shape} for {seg_npy_path}")
    if img.shape[:2] != seg.shape:
        raise ValueError(
            f"Image and segmentation spatial shape mismatch: {img.shape[:2]} vs {seg.shape} "
            f"for {img_npy_path} and {seg_npy_path}"
        )
    if img.shape[2] != 4:
        raise ValueError(f"Expected 4 channels in image, got {img.shape[2]} for {img_npy_path}")

    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    # Save each modality separately for nnU-Net
    for c in range(4):
        modality = img[:, :, c].astype(np.float32)

        # Expand to (H, W, 1) so it is saved as a 3D NIfTI with singleton depth
        modality_3d = modality[:, :, None]

        out_img_path = images_out_dir / f"{case_id}_{c:04d}.nii.gz"
        save_array_as_nifti(modality_3d, out_img_path, affine=affine)

    # Save segmentation as integer labels
    seg = seg.astype(np.uint8)
    seg_3d = seg[:, :, None]

    out_seg_path = labels_out_dir / f"{case_id}.nii.gz"
    save_array_as_nifti(seg_3d, out_seg_path, affine=affine)


def convert_dataset(
    image_dir: str,
    seg_dir: str,
    out_root: str,
    image_suffix: str = "_img.npy",
    seg_suffix: str = "_mask.npy"
) -> None:
    """
    Convert a whole dataset of 2D multi-modal slices to nnU-Net style NIfTI files.

    Matching rule:
        image_dir/imagename_image.npy  <->  seg_dir/imagename_mask.npy

    Output structure:
        out_root/
            imagesTr/
            labelsTr/
    """
    image_dir = Path(image_dir)
    seg_dir = Path(seg_dir)
    out_root = Path(out_root)

    images_out_dir = out_root / "imagesTs"
    labels_out_dir = out_root / "labelsTs"

    image_files = sorted(image_dir.glob(f"*{image_suffix}"))
    if not image_files:
        raise FileNotFoundError(f"No image files ending with {image_suffix} found in {image_dir}")

    converted = 0
    missing_seg = []

    for img_path in image_files:
        img_name = img_path.name

        if not img_name.endswith(image_suffix):
            continue

        # Remove "_image.npy" to get the base case name
        base_name = img_name[:-len(image_suffix)]

        # Match with "imagename_mask.npy"
        seg_path = seg_dir / f"{base_name}{seg_suffix}"

        if not seg_path.exists():
            missing_seg.append(base_name)
            continue

        # nnU-Net case IDs should be simple and consistent
        case_id = base_name.replace(" ", "_")

        convert_one_case(
            img_npy_path=img_path,
            seg_npy_path=seg_path,
            case_id=case_id,
            images_out_dir=images_out_dir,
            labels_out_dir=labels_out_dir
        )
        converted += 1

    print(f"Converted {converted} cases.")
    if missing_seg:
        print(f"Missing segmentation for {len(missing_seg)} image files:")
        for x in missing_seg[:10]:
            print("  ", x)
        if len(missing_seg) > 10:
            print("  ...")


if __name__ == "__main__":
    # Example usage
    convert_dataset(
        image_dir=r"SegMentors\data\test\images",
        seg_dir=r"SegMentors\data\test\masks",
        out_root=r"Dataset001_BrainTumor"
    )