"""Cut a full-resolution dataset into overlapping tiles for small-object training.

Usage:
    python src/tile.py --source data/dataset-full --scale 0.5 --tile 640

Inclusions are tiny relative to the whole photo, so training on whole images
shrunk to 640 px throws most of their pixels away. Each image is resized by
--scale and cut into --tile squares covering it with overlap; boxes are clipped
to each tile. Train and valid become tiles; test keeps whole images, so the
final evaluation runs on full photos at imgsz = scale × original width.
"""

import argparse
import math
import shutil
from pathlib import Path

import yaml
from PIL import Image

from common import DATA_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Keep a clipped box only if at least this much of it is inside the tile.
MIN_VISIBLE = 0.4
MIN_OVERLAP = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DATA_DIR / "dataset-full")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "dataset-tiles")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--tile", type=int, default=640)
    return parser.parse_args()


def tile_origins(length: int, tile: int) -> list[int]:
    """Evenly spaced tile starts that cover [0, length), overlapping by at least MIN_OVERLAP."""
    if length <= tile:
        return [0]
    count = math.ceil((length - MIN_OVERLAP) / (tile - MIN_OVERLAP))
    step = (length - tile) / (count - 1)
    return [round(i * step) for i in range(count)]


def read_boxes(label: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
                boxes.append((c, x1, y1, x1 + w * width, y1 + h * height))
    return boxes


def clip_to_tile(box, x0: int, y0: int, tile: int) -> str | None:
    c, x1, y1, x2, y2 = box
    cx1, cy1 = max(x1, x0), max(y1, y0)
    cx2, cy2 = min(x2, x0 + tile), min(y2, y0 + tile)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    if (cx2 - cx1) * (cy2 - cy1) < MIN_VISIBLE * (x2 - x1) * (y2 - y1):
        return None
    cx, cy = (cx1 + cx2) / 2 - x0, (cy1 + cy2) / 2 - y0
    return f"{c} {cx / tile:.6f} {cy / tile:.6f} {(cx2 - cx1) / tile:.6f} {(cy2 - cy1) / tile:.6f}"


def tile_split(source: Path, out: Path, scale: float, tile: int) -> tuple[int, int]:
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    tiles = boxes_kept = 0
    for image_path in sorted(p for p in (source / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS):
        image = Image.open(image_path).convert("RGB")
        width, height = round(image.width * scale), round(image.height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        boxes = read_boxes(source / "labels" / f"{image_path.stem}.txt", width, height)

        for y0 in tile_origins(height, tile):
            for x0 in tile_origins(width, tile):
                name = f"{image_path.stem}_{x0}_{y0}"
                image.crop((x0, y0, x0 + tile, y0 + tile)).save(out / "images" / f"{name}.jpg", quality=95)
                lines = [line for b in boxes if (line := clip_to_tile(b, x0, y0, tile))]
                (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                tiles += 1
                boxes_kept += len(lines)
    return tiles, boxes_kept


def main() -> None:
    args = parse_args()
    source_yaml = yaml.safe_load((args.source / "data_fixed.yaml").read_text())
    if args.out.exists():
        shutil.rmtree(args.out)

    for split in ("train", "valid"):
        tiles, boxes = tile_split(args.source / split, args.out / split, args.scale, args.tile)
        print(f"{split:5} {tiles:4} tiles, {boxes} boxes")

    data = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": str((args.source / "test" / "images").resolve()),
        "nc": source_yaml["nc"],
        "names": source_yaml["names"],
    }
    (args.out / "data_fixed.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    sample = next((args.source / "test" / "images").iterdir())
    full_width = Image.open(sample).width
    print(f"Dataset: {args.out / 'data_fixed.yaml'}")
    print(f"Evaluate whole test images at imgsz {round(full_width * args.scale)}")


if __name__ == "__main__":
    main()
