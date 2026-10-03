"""Shared paths, device selection and the model registry used by every script."""

import os

# Must be set before torch is imported so unsupported MPS ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data_fixed.yaml"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

SEED = 0
IMGSZ = 640
EPOCHS = 60
PATIENCE = 20

# Register models here; every script reads this table. `batch` is tuned for a 16 GB Mac and
# `batch_low_mem` for 8 GB Macs, where batch 16 at 640 px runs out of memory and swaps.
MODELS = {
    "yolo11s": {
        "weights": "yolo11s.pt",
        "family": "yolo",
        "label": "YOLO11s",
        "architecture": "CNN, one-stage",
        "batch": 16,
        "batch_low_mem": 4,
    },
    "rtdetr-l": {
        "weights": "rtdetr-l.pt",
        "family": "rtdetr",
        "label": "RT-DETR-L",
        "architecture": "Transformer (DETR-based)",
        "batch": 4,
        "batch_low_mem": 2,
    },
}


def pick_device() -> str:
    """Return the best available device: CUDA -> MPS -> CPU."""
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


LOW_MEM_GB = 12


def total_ram_gb() -> float:
    """Physical RAM in GB (Apple Silicon shares it with the GPU)."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3


def default_batch(name: str, device: str) -> int:
    """Per-model batch size, reduced on low-RAM machines unless training on CUDA."""
    spec = MODELS[name]
    if device not in ("cpu", "mps") or total_ram_gb() >= LOW_MEM_GB:
        return spec["batch"]
    return spec["batch_low_mem"]


def best_weights(name: str) -> Path:
    return RUNS_DIR / name / "weights" / "best.pt"


def load_model(name: str, weights: str | Path | None = None):
    """Load a registered model with the right Ultralytics class.

    With no `weights`, loads the COCO-pretrained checkpoint from MODELS.
    """
    from ultralytics import RTDETR, YOLO

    spec = MODELS[name]
    cls = RTDETR if spec["family"] == "rtdetr" else YOLO
    return cls(str(weights or spec["weights"]))
