"""Fine-tune one model from COCO weights.

Usage:
    python src/train.py --model yolo11s
    python src/train.py --model rtdetr-l --batch 2 --device cpu
"""

import argparse
import sys

from common import (
    DATA_YAML,
    EPOCHS,
    IMGSZ,
    MODELS,
    PATIENCE,
    RUNS_DIR,
    SEED,
    best_weights,
    default_batch,
    load_model,
    pick_device,
    total_ram_gb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument(
        "--batch", type=int, help="default: per-model value in common.MODELS, lower on Macs under 12 GB RAM"
    )
    parser.add_argument("--device", help="cpu, mps, or a CUDA index such as 0 (default: auto)")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_YAML.exists():
        sys.exit(f"{DATA_YAML} not found. Run `python src/download.py` first.")

    device = args.device or pick_device()
    batch = args.batch or default_batch(args.model, device)
    if not args.batch and batch != MODELS[args.model]["batch"]:
        print(f"Only {total_ram_gb():.0f} GB RAM: using batch {batch} instead of {MODELS[args.model]['batch']}.")
    print(f"Training {args.model} on {device}: epochs={args.epochs} imgsz={args.imgsz} batch={batch}")

    model = load_model(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        patience=args.patience,
        seed=SEED,
        deterministic=True,
        project=str(RUNS_DIR),
        name=args.model,
        exist_ok=True,
        plots=True,
    )

    print(f"\nBest weights: {best_weights(args.model)}")
    print(f"Training curves: {RUNS_DIR / args.model / 'results.png'}")


if __name__ == "__main__":
    main()
