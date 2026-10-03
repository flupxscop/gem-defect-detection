# Benchmarks

All numbers below were measured, not estimated. Reproduce them with the commands in each section.

## Environments

| Name | Hardware | Notes |
|---|---|---|
| **Browser (production)** | Apple M2 (8 GB), Chromium | The live demo runs both models in the visitor's browser with onnxruntime-web 1.30 |
| **Local container** | Apple M2 (8 GB), Docker Desktop, container limited to `--cpus 2 --memory 3g` | linux/arm64, the self-hosted server |

## Browser inference (live demo)

Measured on https://gemscan.nanthawatchan28.workers.dev (Cloudflare), https://chantarontw-gemscan.static.hf.space
(Hugging Face) and the local production build (`vite preview`).

| Metric | WebGPU | WASM (4 threads) |
|---|---:|---:|
| YOLO11s, median per image | 62 ms | 174 ms |
| RT-DETR-L, median per image | 236 ms | 770 ms |

| Load | Time |
|---|---:|
| Page interactive | 0.36 s |
| First visit: YOLO11s ready (38 MB download) | 5.9 s |
| First visit: both models ready (169 MB) | 10.9 s |
| Return visit: both models ready (Cache Storage, no download) | 4.8 s |
| Return visit: click a sample → both results on screen | 0.7 s |
| First visit on Cloudflare, also downloading the 26 MB runtime from jsDelivr: both models ready | 12.1 s |

Download times depend on the visitor's connection; model files are cached in Cache Storage keyed by the pinned
model-repo commit, so they are fetched once. WASM threads need cross-origin isolation, which the Space enables
with COOP/COEP headers; without it onnxruntime-web falls back to one thread.

**Parity with the server.** On the four sample images at confidence ≥ 0.25 the browser produced the same boxes
as the Python server: YOLO11s 3/3 and RT-DETR-L 62/62 matched (IoU ≥ 0.9, Δconf ≤ 0.02), on both WebGPU and WASM.
`web/src/detect/parity.test.ts` pins the TypeScript pre- and post-processing to fixtures generated from the
Python code (`scripts/make_web_fixtures.py`), and runs in CI.

## Self-hosted server: load test

```bash
docker run -d --name gemscan --cpus 2 --memory 3g -p 7860:7860 gemscan
python scripts/benchmark.py --url http://localhost:7860 --model yolo11s --concurrency 1 4 16 --requests 60
```

Each request uploads one 640×640 sample JPEG (30–45 KB). Latency is measured at the client; model time comes
from the `Server-Timing` header. Requests are processed one at a time (`MAX_CONCURRENCY=1`) and queue beyond
that, so throughput is flat and latency grows with queue depth. No request failed.

| Model | Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Model time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s | 1 | 6.8 | 146 | 156 | 176 | 142 |
| YOLO11s | 4 | 6.9 | 576 | 604 | 605 | 144 |
| YOLO11s | 16 | 6.8 | 2322 | 2337 | 2339 | 145 |
| RT-DETR-L | 1 | 1.6 | 639 | 657 | 752 | 638 |
| RT-DETR-L | 4 | 1.5 | 2594 | 2682 | 2688 | 649 |
| RT-DETR-L | 16 | 1.5 | 10450 | 10533 | 10541 | 654 |

### Thread pool sized from the container's CPU quota

`docker run --cpus 2` (and Kubernetes CPU limits) set a cgroup quota but `os.cpu_count()` still reports every
host core. Sizing ONNX Runtime's thread pool from it oversubscribes the quota and the kernel throttles the
container. `app/settings.py` reads `/sys/fs/cgroup/cpu.max` instead:

| Same container, `--cpus 2` | p50 YOLO11s | p50 RT-DETR-L |
|---|---:|---:|
| 8 threads (host core count) | 810 ms | 2898 ms |
| 2 threads (cgroup quota) | **146 ms** | **639 ms** |

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

PyTorch is faster per image on Apple Silicon, whose CPU kernels it optimises well. For a small CPU host that
scales to zero, startup time, memory and image size matter more, and ONNX is also what makes in-browser
inference possible.

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

## Accuracy experiment: higher resolution

The originals are 2584×1936, while the dataset version used for training is resized to 640×640, so the median
inclusion is only 16 px on its short side. To test whether resolution limits accuracy, YOLO11s was retrained on
the full-resolution export (`src/download.py --version 2 --out dataset-full`), shrunk to half size and cut into
640 px tiles (`src/tile.py`, 6 tiles per image, median inclusion 32 px), then evaluated on whole test images at
imgsz 1280 (`scripts/evaluate.py`).

| Model (16 test images, 117 inclusions) | AP @ IoU 0.5 | AP @ IoU 0.3 | AP @ IoU 0.1 | Max recall @ IoU 0.1 |
|---|---:|---:|---:|---:|
| YOLO11s, whole images at 640 | 0.176 | 0.325 | 0.421 | 0.91 |
| YOLO11s, half-resolution tiles, evaluated at 1280 | 0.161 | 0.292 | 0.403 | 0.84 |

Higher resolution did not help (training early-stopped at epoch 47, best at 27). The gap between IoU 0.5 and
0.1 shows the models find most labelled inclusions but draw boxes that disagree with the labels, which mix
one-box-per-flaw with large boxes around groups of flaws. Label consistency, not model capacity or resolution,
is what limits mAP on this dataset.

