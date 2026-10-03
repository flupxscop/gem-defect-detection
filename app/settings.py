import math
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def available_cpus(cpu_max: Path = CGROUP_CPU_MAX) -> int:
    """CPUs this process may use, honouring container quotas.

    `docker run --cpus 2` and Kubernetes CPU limits set a cgroup quota but leave
    os.cpu_count() at the host's core count. Sizing the thread pool from the host
    oversubscribes the quota and makes inference several times slower.
    """
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = cpu_max.read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return cpus


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
    # ONNX Runtime threads per inference. Defaults to every available CPU, since
    # requests are serialised by max_concurrency.
    ort_threads: int = field(default_factory=lambda: _int("ORT_THREADS", available_cpus()))
    max_concurrency: int = field(default_factory=lambda: _int("MAX_CONCURRENCY", 1))
    max_queue: int = field(default_factory=lambda: _int("MAX_QUEUE", 16))
