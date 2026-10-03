"""Check that the server's ONNX pipeline reproduces Ultralytics.

Usage:
    python scripts/check_parity.py

1. Prediction parity: for every test/valid image, plus a resized non-square
   copy to exercise resizing and letterboxing, compare app.inference with
   Ultralytics running the same ONNX file. Boxes are matched greedily by IoU;
   the check fails below 99% agreement.
2. Accuracy: compare test-split mAP of the exported ONNX model with the
   original PyTorch weights.

Note: Ultralytics pads .pt inputs to the smallest multiple of 32 while a static
ONNX graph needs a full square, so .pt and .onnx predictions on non-square
images legitimately differ. That is why (1) uses the ONNX file as reference.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from app.inference import load_detectors  # noqa: E402
from common import DATA_YAML, DATASET_DIR, best_weights, load_model  # noqa: E402

CONF = 0.25
MIN_IOU = 0.9
MAX_CONF_DIFF = 0.02
REQUIRED_AGREEMENT = 0.99


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def matched(ref_boxes, ref_conf, onnx_boxes, onnx_conf) -> int:
    if len(ref_boxes) == 0 or len(onnx_boxes) == 0:
        return 0
    overlaps = iou(ref_boxes, onnx_boxes)
    used, count = set(), 0
    for i in np.argsort(-ref_conf):
        for j in np.argsort(-overlaps[i]):
            if j in used or overlaps[i, j] < MIN_IOU:
                break
            if abs(ref_conf[i] - onnx_conf[j]) <= MAX_CONF_DIFF:
                used.add(j)
                count += 1
                break
    return count


def images() -> list[Image.Image]:
    paths = sorted((DATASET_DIR / "test" / "images").iterdir()) + sorted((DATASET_DIR / "valid" / "images").iterdir())
    out = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        out.append(image)
        out.append(image.resize((1280, 960), Image.Resampling.BILINEAR))
    return out


def main() -> None:
    detectors = load_detectors(ROOT / "models", threads=4)
    if not detectors:
        sys.exit("No ONNX models in models/. Run `python src/export.py` first.")

    samples = images()
    failed = False
    print(f"Prediction parity ({len(samples)} images, conf={CONF})")
    for name, detector in detectors.items():
        reference = load_model(name, ROOT / "models" / f"{name}.onnx")
        total_ref = total_onnx = total_matched = 0
        for image in samples:
            ref = reference.predict(image, conf=CONF, imgsz=detector.info.imgsz, device="cpu", verbose=False)[0]
            ref_boxes, ref_conf = ref.boxes.xyxy.numpy(), ref.boxes.conf.numpy()
            dets = detector.predict(np.asarray(image), CONF)
            onnx_boxes = np.array([d.box for d in dets]).reshape(-1, 4)
            onnx_conf = np.array([d.confidence for d in dets])

            total_ref += len(ref_boxes)
            total_onnx += len(onnx_boxes)
            total_matched += matched(ref_boxes, ref_conf, onnx_boxes, onnx_conf)

        agreement = total_matched / max(total_ref, total_onnx, 1)
        status = "OK" if agreement >= REQUIRED_AGREEMENT else "FAIL"
        failed |= status == "FAIL"
        print(
            f"  {name:9} {status}  ultralytics={total_ref}  server={total_onnx}  "
            f"matched={total_matched}  agreement={agreement:.1%}"
        )

    print("Test-split accuracy, PyTorch weights vs exported ONNX")
    for name in detectors:
        scores = {}
        for kind, weights in (("pt", best_weights(name)), ("onnx", ROOT / "models" / f"{name}.onnx")):
            box = (
                load_model(name, weights)
                .val(data=str(DATA_YAML), split="test", imgsz=640, batch=1, device="cpu", plots=False, verbose=False)
                .box
            )
            scores[kind] = (box.mr, box.map50)
        (pt_r, pt_map), (ox_r, ox_map) = scores["pt"], scores["onnx"]
        print(f"  {name:9} recall {pt_r:.3f} -> {ox_r:.3f}   mAP50 {pt_map:.3f} -> {ox_map:.3f}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
