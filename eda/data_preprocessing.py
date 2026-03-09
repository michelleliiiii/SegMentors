import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.utils import resample
from pathlib import Path



def get_patient_list(root_dir):
    """
    Scans the folder structure for BraTS-PED files.

    Args:
        root_dir (str): The base directory containing patient subfolders.

    Returns:
        dict: A dictionary where keys are patient IDs and values are
            dictionaries of file paths mapped by modality ('t1c', 't1n', etc.).
    """
    patient_ids = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    patient_map = {}

    for p_id in patient_ids:
        p_path = os.path.join(root_dir, p_id)
        # Search for each modality using glob patterns
        files = {
            't1c': glob.glob(os.path.join(p_path, "*-t1c.nii.gz")),
            't1n': glob.glob(os.path.join(p_path, "*-t1n.nii.gz")),
            't2': glob.glob(os.path.join(p_path, "*-t2w.nii.gz")),
            't2_flair': glob.glob(os.path.join(p_path, "*-t2f.nii.gz")),
            'seg': glob.glob(os.path.join(p_path, "*-seg.nii.gz"))
        }

        # Verify all files exist
        if all(len(v) > 0 for v in files.values()):
            patient_map[p_id] = {k: v[0] for k, v in files.items()}
        else:
            print(f"Warning: Patient {p_id} is missing modalities. Skipping.")

    return patient_map


def load_patient_data(path_dict):
    """
    Loads NIfTI files into memory and stacks them for a single patient.

    Args:
        path_dict (dict): A dictionary containing paths to patient modalities.

    Returns:
        tuple: (vol_4d, mask_3d)
            - vol_4d (np.ndarray): The 4-channel image stack of shape (H, W, D, 4).
            - mask_3d (np.ndarray): The segmentation mask of shape (H, W, D).
    """
    # Load modalities and stack along a new 4th dimension
    mods = ['t2_flair', 't1c', 't1n', 't2']
    vol_list = [
        nib.load(path_dict[m]).get_fdata().astype(np.float32)
        for m in mods
    ]
    vol_4d = np.stack(vol_list, axis=-1)

    # Load segmentation mask
    mask_3d = nib.load(path_dict['seg']).get_fdata().astype(np.uint8)

    return vol_4d, mask_3d


def get_filtered_indices(mask_3d, neg_ratio=0.1, max_pos=10):
    """
    Selects a balanced set of slice indices for training.

    Args:
        mask_3d (np.ndarray): The 3D segmentation mask volume.
        neg_ratio (float): Ratio of background slices to tumor slices.
        max_pos (int): Maximum number of tumor slices to include.

    Returns:
        tuple: (sorted_indices, tumor_flags)
            - sorted_indices (np.ndarray): Sorted array of selected slice indices.
            - tumor_flags (list): Indicator list (1 for tumor, 0 for background).
    """
    num_slices = mask_3d.shape[2]
    slice_areas = [np.sum(mask_3d[:, :, z] > 0) for z in range(num_slices)]
    tumor_flags = [1 if a > 0 else 0 for a in slice_areas]

    indices = np.arange(num_slices)
    pos_idx = indices[np.array(tumor_flags) == 1]
    neg_idx = indices[np.array(tumor_flags) == 0]

    # Select top 'max_pos' largest tumor slices
    if len(pos_idx) > 0:
        pos_areas = [slice_areas[i] for i in pos_idx]
        top_pos = pos_idx[np.argsort(pos_areas)[-max_pos:]]
    else:
        top_pos = []

    # Sample a small number of background slices
    n_neg = max(1, int(len(top_pos) * neg_ratio))
    neg_sampled = resample(
        neg_idx,
        n_samples=min(n_neg, len(neg_idx)),
        random_state=42
    )

    combined_idx = np.concatenate([top_pos, neg_sampled])
    return np.sort(combined_idx).astype(int), tumor_flags


def run_preprocessing(input_root, output_root, neg_ratio=0.1):
    """
    Orchestrates the preprocessing pipeline for medical imaging data.

    Args:
        input_root (str): Path to raw dataset.
        output_root (str): Path to store preprocessed .npy files.
        neg_ratio (float): The ratio of background slices to include.

    Returns:
        None: Saves files to disk and writes a manifest.csv.
    """
    patients = get_patient_list(input_root)
    manifest = []

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for p_id, paths in patients.items():
        # Sequential processing
        vol_4d, mask_3d = load_patient_data(paths)
        selected_indices, tumor_flags = get_filtered_indices(mask_3d, neg_ratio)

        p_out_dir = os.path.join(output_root, p_id)
        os.makedirs(p_out_dir, exist_ok=True)

        for idx in selected_indices:
            # Prepare filenames
            img_file = f"{p_id}_s{idx}_img.npy"
            seg_file = f"{p_id}_s{idx}_seg.npy"

            # Save 4-channel image and 1-channel mask separately
            np.save(os.path.join(p_out_dir, img_file), vol_4d[:, :, idx, :])
            np.save(os.path.join(p_out_dir, seg_file), mask_3d[:, :, idx])

            # Store relative paths for the manifest
            manifest.append({
                "patient_id": p_id,
                "img_path": os.path.join(p_id, img_file),
                "seg_path": os.path.join(p_id, seg_file),
                "is_tumor": int(tumor_flags[idx])
            })

        print(f"Finished {p_id}: Saved {len(selected_indices)} slices.")

        # Explicit RAM cleanup
        del vol_4d, mask_3d

    # Save manifest for future PCA/UMAP and Training
    manifest_path = os.path.join(output_root, "manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print("Pre-processing complete. Manifest saved.")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    INPUT_ROOT = ROOT / "raw_data/Training"
    OUTPUT_ROOT = ROOT / "preprocessed_data"
    run_preprocessing(INPUT_ROOT, OUTPUT_ROOT, neg_ratio=0.1)