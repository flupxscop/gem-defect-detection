# Model comparison (test split, 16 images, device: mps)

| Model | Architecture | Precision | Recall | mAP50 | mAP50-95 | ms / image |
|---|---|---:|---:|---:|---:|---:|
| YOLO11s | CNN, one-stage | 0.390 | 0.213 | 0.187 | 0.059 | 15.1 |
| RT-DETR-L | Transformer (DETR-based) | 0.357 | 0.282 | 0.195 | 0.067 | 83.3 |

**Recommended:** RT-DETR-L (highest recall).
