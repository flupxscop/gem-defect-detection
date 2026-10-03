"""Download the dataset from Roboflow and prepare it for training.

Usage:
    python src/download.py                    # keep only the Inclusion class (default)
    python src/download.py --classes all      # keep every class in the export
    python src/download.py --classes Inclusion Diamond
"""

import argparse
import os
import random
import shutil
import sys

import yaml
from dotenv import load_dotenv

from common import DATA_DIR, DATASET_DIR, ROOT, SEED

REQUIRED_ENV = ["ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT", "ROBOFLOW_VERSION"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_FRACTION = 0.15
# The source dataset also labels the whole stone ("Diamond"). Those big, easy boxes would
# inflate recall/mAP, so by default only inclusions are kept.
DEFAULT_CLASSES = ["Inclusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--classes", nargs="+", default=DEFAULT_CLASSES, help="class names to keep (case-insensitive), or 'all'"
    )
    parser.add_argument("--version", type=int, help="Roboflow version (default: ROBOFLOW_VERSION from .env)")
    parser.add_argument("--out", help="dataset folder name under data/ (default: dataset)")
    return parser.parse_args()


def read_config() -> dict:
    """Read Roboflow settings from .env and stop if any are missing."""
    load_dotenv(ROOT / ".env")
    config = {key: os.getenv(key, "").strip() for key in REQUIRED_ENV}
    missing = [key for key, value in config.items() if not value]
    if missing:
        sys.exit(
            "Missing settings in .env: " + ", ".join(missing) + "\nCopy .env.example to .env and fill in every value."
        )
    if not config["ROBOFLOW_VERSION"].isdigit():
        sys.exit("ROBOFLOW_VERSION must be a number, e.g. 1")
    return config


def download(config: dict) -> None:
    """Download the dataset in YOLOv11 format into data/dataset."""
    from roboflow import Roboflow

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=config["ROBOFLOW_API_KEY"])
    project = rf.workspace(config["ROBOFLOW_WORKSPACE"]).project(config["ROBOFLOW_PROJECT"])
    version = project.version(int(config["ROBOFLOW_VERSION"]))
    version.download("yolov11", location=str(DATASET_DIR), overwrite=True)


def polygon_line_to_box(line: str) -> str:
    """Convert one YOLO label line to `cls cx cy w h`.

    Lines that are already boxes (5 values) are returned unchanged.
    Polygon lines (`cls x1 y1 x2 y2 ...`) become their bounding box.
    """
    parts = line.split()
    if len(parts) <= 5:
        return " ".join(parts)
    cls, coords = parts[0], [float(v) for v in parts[1:]]
    xs, ys = coords[0::2], coords[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def convert_polygons() -> int:
    """Rewrite every polygon label as a bounding box. Returns lines converted."""
    converted = 0
    for label_file in DATASET_DIR.glob("*/labels/*.txt"):
        lines = [line for line in label_file.read_text().splitlines() if line.strip()]
        new_lines = [polygon_line_to_box(line) for line in lines]
        converted += sum(len(line.split()) > 5 for line in lines)
        label_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return converted


def list_images(split: str) -> list:
    folder = DATASET_DIR / split / "images"
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def ensure_valid_split() -> None:
    """Carve 15% of train into valid when the export has no valid split."""
    if list_images("valid"):
        return
    train_images = list_images("train")
    if len(train_images) < 2:
        sys.exit("Not enough training images to create a valid split.")

    rng = random.Random(SEED)
    n_valid = max(1, round(len(train_images) * VALID_FRACTION))
    picked = rng.sample(train_images, n_valid)

    for sub in ("images", "labels"):
        (DATASET_DIR / "valid" / sub).mkdir(parents=True, exist_ok=True)
    for image in picked:
        shutil.move(str(image), DATASET_DIR / "valid" / "images" / image.name)
        label = DATASET_DIR / "train" / "labels" / f"{image.stem}.txt"
        if label.exists():
            shutil.move(str(label), DATASET_DIR / "valid" / "labels" / label.name)
    print(f"No valid split in export: moved {n_valid} of {len(train_images)} train images to valid.")


def read_names() -> list:
    names = yaml.safe_load((DATASET_DIR / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def filter_classes(names: list, keep: list) -> list:
    """Drop label lines for classes not in `keep` and renumber the rest. Returns kept names."""
    if [k.lower() for k in keep] == ["all"]:
        return names
    wanted = {k.lower() for k in keep}
    kept = [n for n in names if n.lower() in wanted]
    unknown = wanted - {n.lower() for n in names}
    if unknown or not kept:
        sys.exit(f"Unknown class(es): {', '.join(sorted(unknown)) or ', '.join(keep)}. Available: {', '.join(names)}")
    remap = {names.index(n): i for i, n in enumerate(kept)}

    dropped = 0
    for label_file in DATASET_DIR.glob("*/labels/*.txt"):
        out = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            old_id = int(parts[0])
            if old_id in remap:
                out.append(" ".join([str(remap[old_id]), *parts[1:]]))
            else:
                dropped += 1
        label_file.write_text("\n".join(out) + ("\n" if out else ""))
    if dropped:
        print(f"Dropped {dropped} labels of classes: {', '.join(n for n in names if n not in kept)}.")
    return kept


def write_data_yaml(names: list) -> dict:
    """Write data_fixed.yaml with absolute paths, adding test when present."""
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
    }
    if list_images("test"):
        data["test"] = "test/images"
    data["nc"] = len(names)
    data["names"] = names

    (DATASET_DIR / "data_fixed.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return data


def main() -> None:
    global DATASET_DIR
    args = parse_args()
    if args.out:
        DATASET_DIR = DATA_DIR / args.out
    config = read_config()
    if args.version:
        config["ROBOFLOW_VERSION"] = str(args.version)
    download(config)
    converted = convert_polygons()
    if converted:
        print(f"Converted {converted} polygon labels to bounding boxes.")
    names = filter_classes(read_names(), args.classes)
    ensure_valid_split()
    data = write_data_yaml(names)

    print(f"\nDataset ready: {DATASET_DIR / 'data_fixed.yaml'}")
    for split in ("train", "valid", "test"):
        print(f"  {split:<5} {len(list_images(split)):>4} images")
    print(f"  classes ({data['nc']}): {', '.join(map(str, data['names']))}")


if __name__ == "__main__":
    main()
