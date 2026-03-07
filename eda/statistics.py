import os
import numpy as np

def compute_bg_fg_ratio(folder_path):
    total_bg = 0
    total_fg = 0

    results = []
    count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy") and "seg" in file:
                path = os.path.join(root, file)

                seg = np.load(path)

                bg_pixels = np.sum(seg == 0)
                fg_pixels = np.sum(np.isin(seg, [1,2,3,4]))

                total_bg += bg_pixels
                total_fg += fg_pixels

                ratio = bg_pixels / fg_pixels if fg_pixels > 0 else float('inf')

                results.append((file, bg_pixels, fg_pixels, ratio))
                count += 1
                print(f"Checked {count} files")

    # Print per-file statistics
    print("Per-file statistics:")
    print("--------------------------------------")
    for r in results:
        print(f"{r[0]} | BG: {r[1]} | FG: {r[2]} | BG/FG ratio: {r[3]:.2f}")

    # Overall statistics
    print("\nOverall dataset statistics:")
    print("--------------------------------------")
    print(f"Total background pixels: {total_bg}")
    print(f"Total foreground pixels: {total_fg}")

    if total_fg > 0:
        print(f"Overall BG/FG ratio: {total_bg/total_fg:.2f}")
    else:
        print("No foreground pixels found.")

    return results


if __name__ == "__main__":
    folder = "data"
    compute_bg_fg_ratio(folder)