import os
import pandas as pd
from pathlib import Path

import numpy as np


def compute_bg_fg_ratio(folder_path):
    """
    Computes background and foreground pixel ratios for a dataset of masks.

    Args:
        folder_path (str or Path): Path to the directory containing .npy masks.

    Returns:
        list: A list of tuples containing (filename, bg_count, fg_count, ratio).
    """
    total_bg = 0
    total_fg = 0
    results = []
    count = 0

    folder = Path(folder_path)

    for mask_path in folder.rglob("*seg*.npy"):
        seg = np.load(mask_path)

        bg_pixels = np.sum(seg == 0)
        fg_pixels = np.sum(np.isin(seg, [1, 2, 3, 4]))

        total_bg += bg_pixels
        total_fg += fg_pixels

        ratio = bg_pixels / fg_pixels if fg_pixels > 0 else float('inf')

        results.append((mask_path.name, bg_pixels, fg_pixels, ratio))
        count += 1
        print(f"Checked {count} files: {mask_path.name}")

    # Print per-file statistics
    print("\nPer-file statistics:")
    print("-" * 40)
    for name, bg, fg, ratio in results:
        print(f"{name} | BG: {bg} | FG: {fg} | Ratio: {ratio:.2f}")

    # Overall statistics
    print("\nOverall dataset statistics:")
    print("-" * 40)
    print(f"Total background pixels: {total_bg}")
    print(f"Total foreground pixels: {total_fg}")

    if total_fg > 0:
        overall_ratio = total_bg / total_fg
        print(f"Overall BG/FG ratio: {overall_ratio:.2f}")
    else:
        print("No foreground pixels found.")

    return results


def compute_dataset_statistics(input_csv_path, output_csv_path="dataset_statistic.csv"):
    """
    Computes the mean and standard deviation for numerical columns in a CSV.

    Args:
        input_csv_path (str or Path): Path to the source CSV file.
        output_csv_path (str or Path): Path to save the resulting statistics.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics.
    """
    input_path = Path(input_csv_path)

    # Read the CSV
    df = pd.read_csv(input_path)

    # Convert all columns to numeric, forcing errors to NaN (handles "n/a" and strings)
    # The 'apply' with 'to_numeric' handles mixed-type columns safely
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Calculate mean and standard deviation, ignoring NaN/null values automatically
    stats = pd.DataFrame({
        "Mean": df_numeric.mean(),
        "Std_Dev": df_numeric.std()
    })

    # Save the result to a new CSV
    stats.to_csv(output_csv_path)
    print(f"Statistics saved successfully to: {output_csv_path}")

    return stats


if __name__ == "__main__":
    # DATA_FOLDER = "data"
    # compute_bg_fg_ratio(DATA_FOLDER)
    
    INPUT_FILE = "data\original_dataset_metadata.csv"
    compute_dataset_statistics(INPUT_FILE)