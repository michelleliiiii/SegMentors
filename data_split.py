import shutil
from pathlib import Path
from random import Random
import numpy as np
import tqdm

SRC_ROOT = Path("preprocessed_dataset/preprocessed_dataset")
OUT_ROOT = Path("data")

VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42
CASE_PREFIX = "BraTS"


def ensure_dirs(root: Path):
    '''
    Expected: output root
    Enforces data structure under root
    Output: creates folders if they don't exist, void return
    '''

    for split in ["train", "val", "test"]:

        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)


def classify_npys(case_dir: Path):
    """
    Expected: directory containing data
    Outputs: lists of image and mask paths
    """
    imgs, masks = [], []

    for p in case_dir.rglob("*.npy"):
        if not p.is_file():
            continue

        name = p.name
        if name.endswith("_img.npy"):
            imgs.append(p)
        elif name.endswith("_seg.npy"):
            masks.append(p)

    return sorted(imgs), sorted(masks)


def make_pairs(imgs, masks):
    '''
    Expected: lists of image and mask paths
    Output: list of pairs where ids match 
    '''
    mask_map = {
        m.name.replace("_seg.npy", ""): m
        for m in masks
    }

    pairs = []
    for im in imgs:
        key = im.name.replace("_img.npy", "")
        m = mask_map.get(key)
        if m is not None:
            pairs.append((im, m))

    return pairs


def copy_pairs(case_id: str, pairs, split: str, out_root: Path):
    '''
    Expected: id, list of image/masks, split type and output dir
    Output: copies files to output dir with names
    '''
    img_out = out_root / split / "images"
    msk_out = out_root / split / "masks"

    for i, (im_p, m_p) in enumerate(pairs):
        im_name = f"{case_id}__{i:06d}__img.npy"
        m_name  = f"{case_id}__{i:06d}__mask.npy"
        shutil.copy2(im_p, img_out / im_name)
        shutil.copy2(m_p,  msk_out / m_name)


def main():
    
    ensure_dirs(OUT_ROOT)
    case_dirs = sorted([p for p in SRC_ROOT.iterdir() if p.is_dir() and p.name.startswith(CASE_PREFIX)])
    
    usable = []
    skipped = []

    print(f"Scanning {len(case_dirs)} case folders...")
    for cd in tqdm.tqdm(case_dirs):
        imgs, masks = classify_npys(cd)
        pairs = make_pairs(imgs, masks)
        if pairs:
            usable.append((cd.name, pairs))
        else:
            skipped.append(cd.name)

    print(f"Usable cases: {len(usable)}")
    
    rng = Random(SEED)
    rng.shuffle(usable)

    n = len(usable)
    n_test = int(round(n * TEST_FRAC))
    n_val  = int(round(n * VAL_FRAC))

    test_set = usable[:n_test]
    val_set  = usable[n_test:n_test + n_val]
    train_set = usable[n_test + n_val:]

    print(f"Split by case: train={len(train_set)} val={len(val_set)} test={len(test_set)}")


    for case_id, pairs in tqdm.tqdm(train_set):
        copy_pairs(case_id, pairs, "train", OUT_ROOT)
    for case_id, pairs in tqdm.tqdm(val_set):
        copy_pairs(case_id, pairs, "val", OUT_ROOT)
    for case_id, pairs in tqdm.tqdm(test_set):
        copy_pairs(case_id, pairs, "test", OUT_ROOT)


    print("FINISHED")


if __name__ == "__main__":
    main()