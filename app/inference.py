"""ONNX Runtime detectors for the YOLO11 and RT-DETR exports.

Pre- and post-processing mirror Ultralytics so results match `model.predict`:
YOLO letterboxes the image and needs NMS; RT-DETR stretches it to a square,
and its export already selects the top 300 queries, so it needs no NMS. Resizing uses OpenCV's bilinear
filter like Ultralytics; Pillow's antialiased resize shifts scores by ~0.05.

Images are RGB uint8 arrays of shape (height, width, 3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

LETTERBOX_FILL = (114, 114, 114)
NMS_IOU = 0.7
MAX_DETECTIONS = 300
CLASS_OFFSET = 7680  # shifts boxes per class so one NMS pass never merges classes


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in input-image pixels


@dataclass(frozen=True)
class ModelInfo:
    name: str
    label: str
    architecture: str
    family: str
    imgsz: int
    classes: list[str]


class Detector:
    def __init__(self, path: Path, info: ModelInfo, threads: int):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.info = info

    def predict(self, image: np.ndarray, conf: float) -> list[Detection]:
        tensor, restore = self.preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0][0]
        boxes, scores, classes = self.decode(output, conf)
        boxes = restore(boxes)
        return [
            Detection(self.info.classes[c], float(s), tuple(float(v) for v in b))
            for b, s, c in zip(boxes, scores, classes, strict=True)
        ]

    def warmup(self) -> None:
        size = self.info.imgsz
        self.session.run(None, {self.input_name: np.zeros((1, 3, size, size), np.float32)})

    def preprocess(self, image: np.ndarray):
        raise NotImplementedError

    def decode(self, output: np.ndarray, conf: float):
        raise NotImplementedError


class YoloDetector(Detector):
    def preprocess(self, image):
        size = self.info.imgsz
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        new_w, new_h = round(w * scale), round(h * scale)
        if (new_w, new_h) != (w, h):
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=LETTERBOX_FILL)

        def restore(boxes: np.ndarray) -> np.ndarray:
            boxes = (boxes - [left, top, left, top]) / scale
            return boxes.clip(0, [w, h, w, h])

        return to_tensor(image), restore

    def decode(self, output, conf):
        output = output.T  # (anchors, 4 + classes)
        scores_all = output[:, 4:]
        classes = scores_all.argmax(1)
        scores = scores_all[np.arange(len(classes)), classes]
        keep = scores > conf
        boxes, scores, classes = xywh_to_xyxy(output[keep, :4]), scores[keep], classes[keep]

        kept = nms(boxes + (classes * CLASS_OFFSET)[:, None], scores, NMS_IOU)[:MAX_DETECTIONS]
        return boxes[kept], scores[kept], classes[kept]


class RtDetrDetector(Detector):
    def preprocess(self, image):
        size = self.info.imgsz
        h, w = image.shape[:2]
        if (w, h) != (size, size):
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

        def restore(boxes: np.ndarray) -> np.ndarray:
            return boxes * [w, h, w, h]

        return to_tensor(image), restore

    def decode(self, output, conf):
        # Rows are [cx, cy, w, h, score, class_index], boxes normalised to 0-1.
        scores, classes = output[:, 4], output[:, 5].astype(int)
        keep = scores > conf
        order = np.argsort(-scores[keep])
        boxes = xywh_to_xyxy(output[keep, :4])[order]
        return boxes, scores[keep][order], classes[keep][order]


DETECTORS = {"yolo": YoloDetector, "rtdetr": RtDetrDetector}


def load_detectors(models_dir: Path, threads: int) -> dict[str, Detector]:
    """Load every model listed in models_dir/manifest.json."""
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        return {}

    detectors = {}
    for name, entry in json.loads(manifest_path.read_text()).items():
        path = models_dir / entry["file"]
        if not path.exists():
            continue
        info = ModelInfo(
            name=name,
            label=entry["label"],
            architecture=entry["architecture"],
            family=entry["family"],
            imgsz=entry["imgsz"],
            classes=entry["classes"],
        )
        detectors[name] = DETECTORS[info.family](path, info, threads)
    return detectors


def to_tensor(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xy, wh = boxes[:, :2], boxes[:, 2:4] / 2
    return np.concatenate([xy - wh, xy + wh], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns indices sorted by score."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=int)
