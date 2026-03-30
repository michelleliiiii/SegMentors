import argparse
from pathlib import Path

from evaluate import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint on the test split and report mean Dice plus per-class Dice."
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint, e.g. teacher_best.pt")
    parser.add_argument("--manifest-path", default="ssl_split_manifest.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--include-bg",
        action="store_true",
        help="Include background class in Dice computation. Default excludes background.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save a JSON summary and per-case CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional filename prefix for saved artifacts.",
    )
    args = parser.parse_args()

    results = evaluate_checkpoint(
        ckpt_path=args.ckpt,
        split="test",
        manifest_path=args.manifest_path,
        batch_size=args.batch_size,
        exclude_bg=not args.include_bg,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )

    print(f"Checkpoint: {results['checkpoint']}")
    print("Split: test")
    if args.include_bg:
        print(f"Mean Dice (include background): {results['summary']['mean_dice']:.4f}")
    else:
        print(f"Mean Dice (exclude background): {results['summary']['mean_dice']:.4f}")
    print("Per-class Dice:")
    for class_name, score in results["summary"]["per_class_dice"].items():
        print(f"  {class_name}: {score:.4f}")

    if args.output_dir is not None:
        print(f"Saved evaluation artifacts to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
