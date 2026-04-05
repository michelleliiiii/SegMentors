from pathlib import Path
import csv
import math
import argparse

DATA_ROOT = Path("data")
OUT_CSV = Path("ssl_split_manifest.csv")


def extract_case_id(filename: str) -> str:
    parts = filename.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    return parts[0]


def get_case_ids(split: str):
    img_dir = DATA_ROOT / split / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing directory: {img_dir}")

    case_ids = set()
    for p in img_dir.glob("*.npy"):
        if p.is_file():
            case_ids.add(extract_case_id(p.name))

    return sorted(case_ids)


def build_manifest_rows(labeled_frac: float):
    train_cases = get_case_ids("train")
    val_cases = get_case_ids("val")
    test_cases = get_case_ids("test")

    if len(train_cases) == 0:
        raise ValueError("No training cases found in data/train/images")

    if not (0 < labeled_frac <= 1):
        raise ValueError("labeled_frac must be in the range (0, 1]")

    n_labeled = max(1, math.floor(len(train_cases) * labeled_frac))

    labeled_train = train_cases[:n_labeled]
    unlabeled_train = train_cases[n_labeled:]

    rows = []

    for case_id in labeled_train:
        rows.append({
            "case_id": case_id,
            "split": "train",
            "label_status": "labeled"
        })

    for case_id in unlabeled_train:
        rows.append({
            "case_id": case_id,
            "split": "train",
            "label_status": "unlabeled"
        })

    for case_id in val_cases:
        rows.append({
            "case_id": case_id,
            "split": "val",
            "label_status": "labeled"
        })

    for case_id in test_cases:
        rows.append({
            "case_id": case_id,
            "split": "test",
            "label_status": "labeled"
        })

    return rows, labeled_train, unlabeled_train, val_cases, test_cases


def write_manifest(rows):
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "split", "label_status"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labeled_frac",
        type=float,
        required=True,
        help="Fraction of training cases to mark as labeled, e.g. 0.15"
    )
    args = parser.parse_args()

    rows, labeled_train, unlabeled_train, val_cases, test_cases = build_manifest_rows(args.labeled_frac)
    write_manifest(rows)

    print(f"Saved manifest to: {OUT_CSV.resolve()}")
    print(f"Labeled fraction: {args.labeled_frac}")
    print(f"Train labeled:   {len(labeled_train)}")
    print(f"Train unlabeled: {len(unlabeled_train)}")
    print(f"Val labeled:     {len(val_cases)}")
    print(f"Test labeled:    {len(test_cases)}")

    print("\nFirst 10 labeled train cases:")
    for case_id in labeled_train[:10]:
        print(f"  {case_id}")

    print("\nFirst 10 unlabeled train cases:")
    for case_id in unlabeled_train[:10]:
        print(f"  {case_id}")


if __name__ == "__main__":
    main()