import { RawDetection } from "./types";

export const NMS_IOU = 0.7;
export const MAX_DETECTIONS = 300;

type Box = [number, number, number, number];

/**
 * YOLO output is [1, 4 + classes, anchors] with boxes as centre x/y, width, height in input pixels.
 * Returns boxes in input pixels after per-class NMS, highest score first.
 */
export function decodeYolo(output: Float32Array, numClasses: number, conf: number): RawDetection[] {
  const anchors = output.length / (4 + numClasses);
  const candidates: RawDetection[] = [];
  for (let a = 0; a < anchors; a++) {
    let best = 0;
    let score = output[4 * anchors + a];
    for (let c = 1; c < numClasses; c++) {
      const s = output[(4 + c) * anchors + a];
      if (s > score) {
        score = s;
        best = c;
      }
    }
    if (score <= conf) continue;
    candidates.push({
      classIndex: best,
      confidence: score,
      box: xywhToXyxy(output[a], output[anchors + a], output[2 * anchors + a], output[3 * anchors + a]),
    });
  }
  return nms(candidates, NMS_IOU).slice(0, MAX_DETECTIONS);
}

/**
 * RT-DETR's export selects the top queries in-graph: output is [1, queries, 6] with rows
 * [cx, cy, w, h, score, class_index] and boxes normalised to 0-1. No NMS needed.
 */
export function decodeRtDetr(output: Float32Array, conf: number): RawDetection[] {
  const stride = 6;
  const detections: RawDetection[] = [];
  for (let q = 0; q < output.length / stride; q++) {
    const row = q * stride;
    const score = output[row + 4];
    if (score <= conf) continue;
    detections.push({
      classIndex: output[row + 5],
      confidence: score,
      box: xywhToXyxy(output[row], output[row + 1], output[row + 2], output[row + 3]),
    });
  }
  return detections.sort((a, b) => b.confidence - a.confidence);
}

/** Greedy NMS within each class. Input order doesn't matter; output is sorted by score. */
export function nms(detections: RawDetection[], iouThreshold: number): RawDetection[] {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: RawDetection[] = [];
  for (const d of sorted) {
    if (kept.every((k) => k.classIndex !== d.classIndex || iou(k.box, d.box) <= iouThreshold)) kept.push(d);
  }
  return kept;
}

export function iou(a: Box, b: Box): number {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / (union + 1e-9);
}

function xywhToXyxy(cx: number, cy: number, w: number, h: number): Box {
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}
