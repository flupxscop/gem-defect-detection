"""Evaluate every trained model on the same split and write the comparison.

Usage:
    python src/compare.py
    python src/compare.py --samples 4 --conf 0.25

Outputs in results/:
    comparison.csv, comparison.json, comparison.md, comparison.png, side_by_side.png
"""

import argparse
import csv
import json
import random
import sys

import yaml

from common import DATA_YAML, IMGSZ, MODELS, RESULTS_DIR, RUNS_DIR, SEED, best_weights, load_model, pick_device

# Keep in sync with web/src/components/ResultPanel.tsx.
MODEL_COLORS = {"yolo11s": "#2a78d6", "rtdetr-l": "#eb6834"}
GT_COLOR = "#008300"
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# Low threshold for mAP (Ultralytics default); same value for every model.
EVAL_CONF = 0.001
RECALL_TIE = 0.02
MAX_LABELS = 15
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FIELDS = [
    "model",
    "architecture",
    "split",
    "images",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "ms_per_image",
    "device",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=0.25, help="confidence for side_by_side.png")
    parser.add_argument("--samples", type=int, default=4, help="images in side_by_side.png")
    parser.add_argument("--device", help="cpu, mps, or a CUDA index (default: auto)")
    return parser.parse_args()


def eval_split(data: dict) -> str:
    return "test" if data.get("test") else "val"


def count_images(split: str) -> int:
    folder = DATA_YAML.parent / ("test" if split == "test" else "valid") / "images"
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def evaluate(name: str, split: str, imgsz: int, device: str) -> dict:
    """Validate one model; batch=1 so ms/image is comparable across models."""
    model = load_model(name, best_weights(name))
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        conf=EVAL_CONF,
        batch=1,
        device=device,
        plots=False,
        project=str(RUNS_DIR / "eval"),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    return {
        "model": name,
        "architecture": MODELS[name]["architecture"],
        "split": split,
        "images": count_images(split),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "ms_per_image": round(float(metrics.speed["inference"]), 2),
        "device": device,
    }


def recommend(rows: list) -> tuple:
    """Highest recall wins; if another model is within RECALL_TIE of it, the faster one wins."""
    best_recall = max(r["recall"] for r in rows)
    contenders = [r for r in rows if best_recall - r["recall"] <= RECALL_TIE]
    pick = min(contenders, key=lambda r: r["ms_per_image"])
    if len(contenders) > 1:
        reason = f"recall within {RECALL_TIE} of the best, so the faster model wins"
    else:
        reason = "highest recall"
    return pick["model"], reason


def write_tables(rows: list) -> None:
    with open(RESULTS_DIR / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # The web app bundles the JSON copy.
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")


def write_markdown(rows: list, pick: str, reason: str) -> None:
    header = "| Model | Architecture | Precision | Recall | mAP50 | mAP50-95 | ms / image |"
    lines = [
        f"# Model comparison ({rows[0]['split']} split, {rows[0]['images']} images, device: {rows[0]['device']})",
        "",
        header,
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {MODELS[r['model']]['label']} | {r['architecture']} | {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {r['mAP50']:.3f} | {r['mAP50-95']:.3f} | {r['ms_per_image']:.1f} |"
        )
    lines += ["", f"**Recommended:** {MODELS[pick]['label']} ({reason})."]
    (RESULTS_DIR / "comparison.md").write_text("\n".join(lines) + "\n")


def style_axis(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_comparison(rows: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["precision", "recall", "mAP50", "mAP50-95"]
    colors = [MODEL_COLORS.get(r["model"], "#4a3aa7") for r in rows]
    fig, (ax_acc, ax_ms) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]}, facecolor=SURFACE
    )
    # 2px surface-colored edge keeps a gap between adjacent bars.
    bar_style = {"edgecolor": SURFACE, "linewidth": 2}

    width = 0.8 / len(rows)
    for i, (r, color) in enumerate(zip(rows, colors, strict=True)):
        xs = [j - 0.4 + width * (i + 0.5) for j in range(len(metrics))]
        values = [r[m] for m in metrics]
        ax_acc.bar(xs, values, width, color=color, **bar_style)
        for x, v in zip(xs, values, strict=True):
            ax_acc.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)
    ax_acc.set_ylim(0, 1.08)
    ax_acc.set_xticks(range(len(metrics)), ["Precision", "Recall", "mAP50", "mAP50-95"])
    ax_acc.set_title("Accuracy (higher is better)", loc="left", color=TEXT_PRIMARY, fontsize=11)
    style_axis(ax_acc)

    ms = [r["ms_per_image"] for r in rows]
    top = max(ms) or 1
    ax_ms.bar(range(len(rows)), ms, 0.6, color=colors, **bar_style)
    for i, v in enumerate(ms):
        ax_ms.text(i, v + top * 0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)
    ax_ms.set_ylim(0, top * 1.2)
    ax_ms.set_xticks(range(len(rows)), [MODELS[r["model"]]["label"] for r in rows])
    ax_ms.set_title("ms / image (lower is better)", loc="left", color=TEXT_PRIMARY, fontsize=11)
    style_axis(ax_ms)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(
        handles,
        [MODELS[r["model"]]["label"] for r in rows],
        loc="upper right",
        frameon=False,
        ncol=len(rows),
        labelcolor=TEXT_PRIMARY,
    )
    fig.suptitle(
        f"YOLO11s vs RT-DETR-L on the {rows[0]['split']} split",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / "comparison.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def read_gt_boxes(image_path, width: int, height: int) -> list:
    label = label_path(image_path)
    boxes = []
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = map(float, parts)
            boxes.append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


def label_path(image_path):
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def pick_samples(image_dir, n: int) -> list:
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    labelled = [p for p in images if label_path(p).exists() and label_path(p).read_text().strip()]
    pool = labelled or images
    return random.Random(SEED).sample(pool, min(n, len(pool)))


def plot_side_by_side(names: list, split: str, data: dict, conf: float, imgsz: int, device: str, n: int) -> None:
    """Grid of sample images: ground truth next to each model's predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    image_dir = DATA_YAML.parent / data["test" if split == "test" else "val"]
    samples = pick_samples(image_dir, n)
    if not samples:
        print("No images found for side_by_side.png; skipped.")
        return

    models = {name: load_model(name, best_weights(name)) for name in names}
    columns = ["Ground truth"] + [MODELS[n]["label"] for n in names]
    fig, axes = plt.subplots(
        len(samples), len(columns), figsize=(3.6 * len(columns), 3.6 * len(samples)), squeeze=False, facecolor=SURFACE
    )

    def draw(ax, image, boxes, color, title):
        ax.imshow(image)
        # Confidence labels only while they stay readable.
        show_labels = len(boxes) <= MAX_LABELS
        for x1, y1, x2, y2, text in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            if text and show_labels:
                ax.text(
                    x1,
                    y1 - 2,
                    text,
                    color="white",
                    fontsize=7,
                    va="bottom",
                    clip_on=True,
                    bbox={"facecolor": color, "edgecolor": "none", "pad": 1},
                )
        ax.set_title(title, fontsize=9, color=TEXT_PRIMARY)
        ax.axis("off")

    for row, path in enumerate(samples):
        image = Image.open(path).convert("RGB")
        gt = [(*b, "") for b in read_gt_boxes(path, *image.size)]
        draw(axes[row][0], image, gt, GT_COLOR, f"Ground truth · {len(gt)} boxes")
        for col, name in enumerate(names, start=1):
            result = models[name].predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            boxes = [
                (*xyxy, f"{c:.2f}")
                for xyxy, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True)
            ]
            draw(
                axes[row][col],
                image,
                boxes,
                MODEL_COLORS.get(name, "#4a3aa7"),
                f"{MODELS[name]['label']} · {len(boxes)} boxes",
            )
        axes[row][0].text(
            -0.04,
            0.5,
            path.stem[:28],
            transform=axes[row][0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=7,
            color=TEXT_SECONDARY,
        )

    fig.suptitle(f"Predictions at conf ≥ {conf} ({split} split)", color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "side_by_side.png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not DATA_YAML.exists():
        sys.exit(f"{DATA_YAML} not found. Run `python src/download.py` first.")

    names = [n for n in MODELS if best_weights(n).exists()]
    missing = [n for n in MODELS if n not in names]
    if missing:
        print(f"Skipping untrained models: {', '.join(missing)}")
    if not names:
        sys.exit("No trained models found. Run `python src/train.py --model <name>` first.")

    data = yaml.safe_load(DATA_YAML.read_text())
    split = eval_split(data)
    device = args.device or pick_device()
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for name in names:
        print(f"Evaluating {name} on {split}...")
        rows.append(evaluate(name, split, args.imgsz, device))

    pick, reason = recommend(rows)
    write_tables(rows)
    write_markdown(rows, pick, reason)
    plot_comparison(rows)
    plot_side_by_side(names, split, data, args.conf, args.imgsz, device, args.samples)

    print("\n" + (RESULTS_DIR / "comparison.md").read_text())
    print(f"Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
