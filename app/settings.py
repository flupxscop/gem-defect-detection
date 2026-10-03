import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    models_dir: Path = field(default_factory=lambda: _path("MODELS_DIR", ROOT / "models"))
    metrics_csv: Path = field(default_factory=lambda: _path("METRICS_CSV", ROOT / "results" / "comparison.csv"))
    static_dir: Path = field(default_factory=lambda: _path("STATIC_DIR", ROOT / "web" / "dist"))
    cors_origins: list[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    )
    max_upload_bytes: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 10) * 1024 * 1024)
    # ONNX Runtime threads per inference. Defaults to all cores, since requests
    # are serialised by max_concurrency on small CPU hosts.
    ort_threads: int = field(default_factory=lambda: _int("ORT_THREADS", os.cpu_count() or 1))
    max_concurrency: int = field(default_factory=lambda: _int("MAX_CONCURRENCY", 1))
    max_queue: int = field(default_factory=lambda: _int("MAX_QUEUE", 16))
