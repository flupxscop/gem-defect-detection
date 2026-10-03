import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.inference import Detection, ModelInfo  # noqa: E402
from app.main import create_app  # noqa: E402
from app.settings import Settings  # noqa: E402


class FakeDetector:
    """Returns one fixed box in input-image pixels."""

    def __init__(self, name: str):
        self.info = ModelInfo(name, name.upper(), "test", "yolo", 640, ["Inclusion"])
        self.seen_shapes = []

    def predict(self, image: np.ndarray, conf: float):
        self.seen_shapes.append(image.shape)
        return [Detection("Inclusion", 0.9, (10.0, 20.0, 30.0, 40.0))] if conf <= 0.9 else []

    def warmup(self):
        pass


@pytest.fixture
def detectors():
    return {"yolo11s": FakeDetector("yolo11s"), "rtdetr-l": FakeDetector("rtdetr-l")}


@pytest.fixture
def client(detectors, tmp_path):
    settings = Settings(static_dir=tmp_path / "no-web", metrics_csv=tmp_path / "none.csv", max_upload_bytes=1024 * 1024)
    with TestClient(create_app(settings, detectors)) as test_client:
        yield test_client


def encode(image: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@pytest.fixture
def png_bytes():
    return encode(Image.new("RGB", (64, 48), "white"))
