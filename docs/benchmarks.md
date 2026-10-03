# Benchmarks

All numbers below were measured, not estimated. Reproduce them with the commands in each section.

## Environments

| Name | Hardware | Notes |
|---|---|---|
| **Local container** | Apple M2 (8 GB), Docker Desktop, container limited to `--cpus 2 --memory 3g` | linux/arm64, mirrors the 2-vCPU free tier |
| **Hugging Face Space** | Free "CPU basic" tier (2 vCPU, 16 GB) | linux/x86_64, the production deployment |

## Serving: load test

```bash
docker run -d --name gemscan --cpus 2 --memory 3g -p 7860:7860 gemscan
python scripts/benchmark.py --url http://localhost:7860 --model yolo11s --concurrency 1 4 16 --requests 60
```

Each request uploads one 640×640 sample JPEG (30–45 KB). Latency is measured at the client; model time comes
from the `Server-Timing` header. Requests are processed one at a time (`MAX_CONCURRENCY=1`) and queue beyond
that, so throughput is flat and latency grows with queue depth. No request failed.

**Local container (2 CPUs)**

| Model | Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Model time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s | 1 | 6.7 | 146 | 159 | 165 | 143 |
| YOLO11s | 4 | 6.8 | 579 | 659 | 678 | 147 |
| YOLO11s | 16 | 6.8 | 2322 | 2337 | 2339 | 145 |
| RT-DETR-L | 1 | 1.5 | 650 | 690 | 707 | 648 |
| RT-DETR-L | 4 | 1.5 | 2625 | 2724 | 2736 | 658 |
| RT-DETR-L | 16 | 1.5 | 10450 | 10533 | 10541 | 654 |

**Hugging Face Space (production)**

HF_SPACE_RESULTS

## Container footprint

| Metric | Value |
|---|---|
| Cold start (container start → `/api/health` ok, both models loaded and warmed) | 2.4 s |
| Memory, idle | 430 MB |
| Memory, after load test | 683 MB |
| Python dependencies in the image | 291 MB (no PyTorch) |
| Model files | 169 MB (YOLO11s 38 MB, RT-DETR-L 131 MB) |

## Why ONNX Runtime instead of PyTorch for serving

Measured in-process on the M2 with 2 threads, both models loaded:

| | PyTorch + Ultralytics | ONNX Runtime (`app/inference.py`) |
|---|---:|---:|
| Import, load and first prediction | 2.6 s | **1.1 s** |
| Peak memory | 908 MB | **466 MB** |
| Largest dependency | `torch` wheel, 555 MB (x86_64) | `onnxruntime`, 59 MB installed |
| YOLO11s predict | **88 ms** | 144 ms |
| RT-DETR-L predict | **486 ms** | 645 ms |

PyTorch is faster per image on Apple Silicon, whose CPU kernels it optimises well. The free host only gives a
small CPU and its instances sleep when idle, so startup time, memory and image size matter more there; the
production latency on x86 is in the Hugging Face table above.

## Correctness of the ONNX pipeline

`python scripts/check_parity.py` compares the server pipeline with Ultralytics on the same ONNX files,
over 98 images (every test and valid image, plus a resized 1280×960 copy of each to exercise resizing and
letterboxing):

| Model | Ultralytics boxes | Server boxes | Matched (IoU ≥ 0.9, Δconf ≤ 0.02) |
|---|---:|---:|---:|
| YOLO11s | 130 | 130 | 100% |
| RT-DETR-L | 1511 | 1511 | 100% |

Test-split accuracy of the exported ONNX vs the PyTorch weights:

| Model | Recall (pt → onnx) | mAP50 (pt → onnx) |
|---|---|---|
| YOLO11s | 0.213 → 0.205 | 0.187 → 0.174 |
| RT-DETR-L | 0.282 → 0.282 | 0.195 → 0.195 |

The YOLO difference comes from the evaluation protocol, not the model: Ultralytics validates `.pt` weights on
rectangular batches padded to 672 px, while a static ONNX graph is evaluated at 640 px. On the same 640 px
input the predictions are identical (table above).

## Things tried and rejected

| Change | Result | Decision |
|---|---|---|
| Two parallel inferences with 1 thread each (`MAX_CONCURRENCY=2 ORT_THREADS=1`) | Throughput +4–7%, but single-request latency 146 → 267 ms (YOLO) and 650 → 1199 ms (RT-DETR) | Keep one inference at a time with all threads: an interactive demo cares about latency |
| Dynamic INT8 quantization (`MatMul`/`Gemm`) | RT-DETR-L 645 → 597 ms (−7%), YOLO11s unchanged; both convolution-bound | Not worth the accuracy risk |
| Pillow instead of OpenCV for resizing | Pillow's antialiasing shifted scores by ~0.05 on resized images, so only ~70% of boxes matched Ultralytics | Use OpenCV `INTER_LINEAR`, as Ultralytics does |

## Other optimisations in place

- **Client-side downscaling:** the browser resizes photos to ≤ 1280 px (EXIF rotation applied) before upload, so a
  4–12 MB phone photo is sent as roughly 150–300 KB.
- **JPEG draft decoding:** the server decodes large JPEGs at reduced scale, so API clients that send full-size photos
  don't pay for a full-resolution decode.
- **Warm-up at startup:** the first real request doesn't pay for graph initialisation.
- **Load shedding:** beyond 16 queued requests the server returns `503` with `Retry-After` instead of piling up.
- **Static assets:** Vite's fingerprinted bundles are served with `Cache-Control: immutable` and gzip (51 KB of JS).
