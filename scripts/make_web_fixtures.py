"""Generate fixtures that pin the browser pipeline (web/src/detect) to the Python one.

Usage:
    python scripts/make_web_fixtures.py

Writes web/src/detect/fixtures.json, used by web/src/detect/parity.test.ts.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.inference import ModelInfo, RtDetrDetector, YoloDetector  # noqa: E402

OUT = ROOT / "web" / "src" / "detect" / "fixtures.json"


def detector(cls, imgsz):
    d = cls.__new__(cls)  # pre/post-processing only, no ONNX session
    d.info = ModelInfo("m", "M", "test", "yolo", imgsz, ["a", "b"])
    return d


def main() -> None:
    rng = np.random.default_rng(0)

    image = rng.integers(0, 256, (7, 11, 3), dtype=np.uint8)
    resize = [
        {"width": w, "height": h, "expected": cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR).tolist()}
        for w, h in [(5, 9), (17, 4), (22, 14)]
    ]

    letterbox_tensor, _ = detector(YoloDetector, 16).preprocess(image)
    stretch_tensor, _ = detector(RtDetrDetector, 16).preprocess(image)

    # YOLO output: [4 + 2 classes, 60 anchors], with clusters of overlapping boxes.
    anchors = 60
    yolo = np.zeros((6, anchors), np.float32)
    centres = rng.uniform(50, 590, (anchors // 6, 2)).repeat(6, axis=0)
    yolo[:2] = (centres + rng.normal(0, 4, centres.shape)).T
    yolo[2:4] = rng.uniform(10, 60, (2, anchors))
    yolo[4:6] = rng.uniform(0, 1, (2, anchors))
    boxes, scores, classes = detector(YoloDetector, 640).decode(yolo, 0.25)

    rtdetr = rng.uniform(0, 1, (40, 6)).astype(np.float32)  # [cx, cy, w, h, score, class]
    rtdetr[:, 5] = rng.integers(0, 2, 40)
    rt_boxes, rt_scores, rt_classes = detector(RtDetrDetector, 640).decode(rtdetr, 0.25)

    fixtures = {
        "image": {"width": 11, "height": 7, "rgb": image.tolist()},
        "resize": resize,
        "letterbox": {"size": 16, "tensor": letterbox_tensor.ravel().round(6).tolist()},
        "stretch": {"size": 16, "tensor": stretch_tensor.ravel().round(6).tolist()},
        "yolo": {
            "output": yolo.ravel().tolist(),
            "classes": 2,
            "conf": 0.25,
            "expected": [
                {"box": b.tolist(), "score": float(s), "class": int(c)}
                for b, s, c in zip(boxes, scores, classes, strict=True)
            ],
        },
        "rtdetr": {
            "output": rtdetr.ravel().tolist(),
            "conf": 0.25,
            "expected": [
                {"box": b.tolist(), "score": float(s), "class": int(c)}
                for b, s, c in zip(rt_boxes, rt_scores, rt_classes, strict=True)
            ],
        },
    }
    OUT.write_text(json.dumps(fixtures))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
