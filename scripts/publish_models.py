"""Publish the exported ONNX models to a Hugging Face model repo.

Usage:
    python scripts/publish_models.py --repo <user>/gemscan-models

The repo acts as the model registry: the Docker build downloads the files
from a pinned revision, so serving code and weights are versioned separately.
Prints the commit hash to pass as MODEL_REVISION.
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
METRICS = ROOT / "results" / "comparison.csv"
DATASET_URL = "https://universe.roboflow.com/diamond-classification-data/diamond-inclusion"


def model_card(manifest: dict) -> str:
    rows = list(csv.DictReader(METRICS.open())) if METRICS.exists() else []
    table = "\n".join(
        f"| {manifest[r['model']]['label']} | {float(r['precision']):.3f} | {float(r['recall']):.3f} "
        f"| {float(r['mAP50']):.3f} | {float(r['mAP50-95']):.3f} |"
        for r in rows
        if r["model"] in manifest
    )
    return f"""---
license: cc-by-4.0
library_name: onnx
pipeline_tag: object-detection
tags: [object-detection, onnx, yolo11, rt-detr, gemstone, quality-control]
---

# GemScan inclusion detectors (ONNX)

YOLO11s and RT-DETR-L fine-tuned to detect inclusions in diamond photos, exported to ONNX
(opset 17, static 1×3×640×640 input, FP32). `manifest.json` lists each file with its
architecture, input size and class names.

Test split (16 images, 117 inclusions):

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
{table}

Trained on [Diamond Inclusion]({DATASET_URL}) (CC BY 4.0) by Diamond Classification Data.
Pre/post-processing that reproduces Ultralytics exactly is in `app/inference.py` of the source repository.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="e.g. your-name/gemscan-models")
    args = parser.parse_args()

    manifest_path = MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        sys.exit("models/manifest.json not found. Run `python src/export.py` first.")
    manifest = json.loads(manifest_path.read_text())

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(model_card(manifest))
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=args.repo)
    commit = api.upload_folder(
        folder_path=MODELS_DIR,
        repo_id=args.repo,
        allow_patterns=["*.onnx", "manifest.json"],
        commit_message="Update ONNX models",
    )
    print(f"Published to https://huggingface.co/{args.repo}")
    print(f"MODEL_REVISION={commit.oid}")


if __name__ == "__main__":
    main()
