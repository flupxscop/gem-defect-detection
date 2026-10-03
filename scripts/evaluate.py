"""Evaluate trained weights on whole test images.

Usage:
    python scripts/evaluate.py runs/yolo11s/weights/best.pt --imgsz 640
    python scripts/evaluate.py runs/yolo11s-tiles/weights/best.pt --data data/dataset-tiles/data_fixed.yaml --imgsz 1280

Prints precision, recall, mAP50 and mAP50-95 so runs trained on different
resolutions or tilings can be compared on the same 16 test photos.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import DATA_YAML, MODELS, load_model, pick_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", choices=list(MODELS), help="architecture (default: guessed from the path)")
    parser.add_argument("--device", help="cpu, mps, or a CUDA index (default: auto)")
    args = parser.parse_args()

    name = args.model or ("rtdetr-l" if "rtdetr" in str(args.weights) else "yolo11s")
    metrics = load_model(name, args.weights).val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=1,
        device=args.device or pick_device(),
        plots=False,
        verbose=False,
    )
    box = metrics.box
    print(
        f"{args.weights}  imgsz={args.imgsz}  precision={box.mp:.3f}  recall={box.mr:.3f}  "
        f"mAP50={box.map50:.3f}  mAP50-95={box.map:.3f}"
    )


if __name__ == "__main__":
    main()
