import numpy as np

from app.inference import ModelInfo, RtDetrDetector, YoloDetector, nms, xywh_to_xyxy


def make(cls):
    detector = cls.__new__(cls)  # skip loading an ONNX session
    detector.info = ModelInfo("m", "M", "test", "yolo", 640, ["Inclusion"])
    return detector


def test_xywh_to_xyxy():
    assert xywh_to_xyxy(np.array([[50.0, 40.0, 20.0, 10.0]])).tolist() == [[40.0, 35.0, 60.0, 45.0]]


def test_nms_drops_overlaps_and_keeps_separate_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_nms_handles_no_boxes():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).tolist() == []


def test_yolo_letterbox_round_trip():
    detector = make(YoloDetector)
    tensor, restore = detector.preprocess(np.zeros((480, 1280, 3), np.uint8))

    assert tensor.shape == (1, 3, 640, 640)
    # A box in letterboxed space maps back to original pixels.
    scale = 640 / 1280
    top = (640 - 480 * scale) / 2
    box = np.array([[100 * scale, top + 50 * scale, 300 * scale, top + 150 * scale]])
    assert np.allclose(restore(box), [[100, 50, 300, 150]], atol=1)


def test_yolo_decode_filters_by_confidence():
    detector = make(YoloDetector)
    output = np.zeros((5, 3), np.float32)  # 4 box values + 1 class, 3 anchors
    output[:4, 0] = [100, 100, 20, 20]
    output[4, 0] = 0.8
    output[:4, 1] = [300, 300, 20, 20]
    output[4, 1] = 0.1

    boxes, scores, _ = detector.decode(output, conf=0.25)
    assert boxes.tolist() == [[90, 90, 110, 110]]
    assert scores.tolist() == [np.float32(0.8)]


def test_rtdetr_boxes_are_normalised_and_scaled_to_image():
    detector = make(RtDetrDetector)
    _, restore = detector.preprocess(np.zeros((300, 600, 3), np.uint8))
    output = np.array([[0.5, 0.5, 0.2, 0.2, 0.9], [0.1, 0.1, 0.1, 0.1, 0.05]], np.float32)

    boxes, scores, _ = detector.decode(output, conf=0.25)
    assert np.allclose(restore(boxes), [[240, 120, 360, 180]])
    assert len(scores) == 1
