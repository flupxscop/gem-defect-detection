"""Load-test the /api/predict endpoint.

Usage:
    python scripts/benchmark.py --url http://localhost:7860 --model yolo11s --concurrency 1 4 8
    python scripts/benchmark.py --url https://<space>.hf.space --requests 50

Sends the sample images round-robin and reports client-side latency
percentiles, throughput and error counts per concurrency level, plus the
server-side inference time from the Server-Timing header.
"""

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx

SAMPLES = sorted((Path(__file__).resolve().parent.parent / "web" / "public" / "samples").glob("*.jpg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--model", default="yolo11s", choices=["yolo11s", "rtdetr-l", "both"])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--requests", type=int, default=100, help="requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * len(ordered)) - 1))
    return ordered[index]


def server_ms(header: str) -> float:
    """Sum of the model durations in a Server-Timing header, excluding decode."""
    total = 0.0
    for part in header.split(","):
        name, _, duration = part.strip().partition(";dur=")
        if name != "decode" and duration:
            total += float(duration)
    return total


async def run_level(client: httpx.AsyncClient, url: str, model: str, concurrency: int, total: int) -> dict:
    images = [p.read_bytes() for p in SAMPLES]
    latencies, inference, statuses = [], [], {}
    counter = iter(range(total))

    async def worker():
        for i in counter:
            files = {"file": ("stone.jpg", images[i % len(images)], "image/jpeg")}
            started = time.perf_counter()
            response = await client.post(f"{url}/api/predict", files=files, data={"model": model})
            latencies.append((time.perf_counter() - started) * 1000)
            statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
            if response.status_code == 200:
                inference.append(server_ms(response.headers.get("server-timing", "")))

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - started

    ok = statuses.get(200, 0)
    return {
        "concurrency": concurrency,
        "requests": total,
        "rps": ok / elapsed,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "inference": statistics.mean(inference) if inference else 0.0,
        "errors": total - ok,
        "statuses": statuses,
    }


async def main() -> None:
    args = parse_args()
    url = args.url.rstrip("/")
    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(args.warmup):
            files = {"file": ("stone.jpg", SAMPLES[0].read_bytes(), "image/jpeg")}
            (await client.post(f"{url}/api/predict", files=files, data={"model": args.model})).raise_for_status()

        print(f"model={args.model}  url={url}  requests/level={args.requests}\n")
        print("| Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Model time (ms) | Errors |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for level in args.concurrency:
            r = await run_level(client, url, args.model, level, args.requests)
            errors = f"{r['errors']}" + (f" {r['statuses']}" if r["errors"] else "")
            print(
                f"| {r['concurrency']} | {r['rps']:.1f} | {r['p50']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} "
                f"| {r['inference']:.0f} | {errors} |"
            )


if __name__ == "__main__":
    asyncio.run(main())
