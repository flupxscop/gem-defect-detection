"""Download the published ONNX models into models/ for local development.

Usage:
    python scripts/fetch_models.py
    python scripts/fetch_models.py --repo <user>/gemscan-models --revision <sha>
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="ChantaroNtw/gemscan-models")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    path = snapshot_download(
        args.repo,
        revision=args.revision,
        allow_patterns=["*.onnx", "manifest.json"],
        local_dir=ROOT / "models",
    )
    print(f"Models in {path}")


if __name__ == "__main__":
    main()
