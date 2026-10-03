"""HTTP API and static host for the gemstone inclusion demo.

uvicorn app.main:app --port 8000
"""

import asyncio
import csv
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .inference import Detector, load_detectors
from .schemas import Detection, Health, ImageSize, ModelSummary, Prediction
from .settings import Settings

log = logging.getLogger("gemscan")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
METRIC_COLUMNS = {"images", "precision", "recall", "mAP50", "mAP50-95", "ms_per_image"}
# Large JPEGs are decoded at a reduced scale (DCT scaling); the models only see 640 px.
DECODE_TARGET = 1280


class Busy(Exception):
    pass


class InferenceGate:
    """Caps concurrent inferences and sheds load once the wait queue is full."""

    def __init__(self, concurrency: int, max_queue: int):
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_queue = max_queue
        self._waiting = 0

    @asynccontextmanager
    async def slot(self):
        if self._waiting >= self._max_queue:
            raise Busy
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        try:
            yield
        finally:
            self._semaphore.release()


def create_app(settings: Settings | None = None, detectors: dict[str, Detector] | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if detectors is None:
            app.state.detectors = load_detectors(settings.models_dir, settings.ort_threads)
            for detector in app.state.detectors.values():
                detector.warmup()
        else:
            app.state.detectors = detectors
        app.state.gate = InferenceGate(settings.max_concurrency, settings.max_queue)
        log.info("models loaded: %s", ", ".join(app.state.detectors) or "none")
        yield

    app = FastAPI(
        title="Gemstone Inclusion Detection",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/assets/"):
            # Vite fingerprints asset filenames, so they can be cached forever.
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        if timing := getattr(request.state, "server_timing", None):
            response.headers["Server-Timing"] = timing
        return response

    @app.get("/api/health", response_model=Health)
    def health(request: Request) -> Health:
        loaded = list(request.app.state.detectors)
        return Health(status="ok" if loaded else "no_models", models=loaded)

    @app.get("/api/models", response_model=list[ModelSummary])
    def models(request: Request) -> list[ModelSummary]:
        metrics = read_metrics(settings)
        return [
            ModelSummary(
                name=name,
                label=detector.info.label,
                architecture=detector.info.architecture,
                metrics=metrics.get(name),
            )
            for name, detector in request.app.state.detectors.items()
        ]

    @app.post("/api/predict", response_model=list[Prediction], response_model_by_alias=True)
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        model: Literal["yolo11s", "rtdetr-l", "both"] = Form("both"),
        conf: float = Form(0.25, ge=0, le=1),
    ):
        """Detect inclusions. Returns one prediction per model, two for `both`."""
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, "Only JPG, PNG and WEBP images are supported")

        available = request.app.state.detectors
        names = list(available) if model == "both" else [model]
        if missing := [n for n in names if n not in available]:
            raise HTTPException(503, f"Model not available: {', '.join(missing)}")

        started = time.perf_counter()
        data = await read_upload(file, settings.max_upload_bytes)
        image, scale = await run_in_threadpool(decode_image, data)
        decode_ms = (time.perf_counter() - started) * 1000

        try:
            async with request.app.state.gate.slot():
                results = [await run_in_threadpool(run_model, available[n], image, scale, conf) for n in names]
        except Busy:
            raise HTTPException(503, "Server is busy, please retry shortly", headers={"Retry-After": "2"}) from None

        request.state.server_timing = ", ".join(
            [f"decode;dur={decode_ms:.1f}"] + [f"{p.model};dur={p.inference_ms:.1f}" for p in results]
        )
        log.info(
            "predict models=%s size=%dx%d detections=%s inference_ms=%s",
            ",".join(names),
            results[0].image.width,
            results[0].image.height,
            [len(p.detections) for p in results],
            [p.inference_ms for p in results],
        )
        return results

    if settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")

    return app


def read_metrics(settings: Settings) -> dict[str, dict]:
    if not settings.metrics_csv.exists():
        return {}
    with settings.metrics_csv.open() as f:
        rows = {}
        for row in csv.DictReader(f):
            name = row.pop("model")
            rows[name] = {k: float(v) if k in METRIC_COLUMNS else v for k, v in row.items()}
        return rows


async def read_upload(upload: UploadFile, limit: int) -> bytes:
    chunks, size = [], 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"File is larger than {limit // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def decode_image(data: bytes) -> tuple[np.ndarray, float]:
    """Decode to an upright RGB array, plus the factor that maps it back to original pixels."""
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(415, "Only JPG, PNG and WEBP images are supported")
        full_width = image.width
        if image.format == "JPEG":
            image.draft("RGB", (DECODE_TARGET, DECODE_TARGET))
        scale = full_width / image.width
        # Browsers apply EXIF rotation when displaying, so boxes must use the same frame.
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image), scale
    except (UnidentifiedImageError, OSError) as err:
        raise HTTPException(415, "File is not a readable image") from err


def run_model(detector: Detector, image: np.ndarray, scale: float, conf: float) -> Prediction:
    started = time.perf_counter()
    detections = detector.predict(image, conf)
    elapsed_ms = (time.perf_counter() - started) * 1000
    height, width = image.shape[:2]
    return Prediction(
        model=detector.info.name,
        inference_ms=round(elapsed_ms, 1),
        image=ImageSize(width=round(width * scale), height=round(height * scale)),
        detections=[
            Detection(class_name=d.label, confidence=round(d.confidence, 4), box=[round(v * scale, 1) for v in d.box])
            for d in detections
        ],
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
