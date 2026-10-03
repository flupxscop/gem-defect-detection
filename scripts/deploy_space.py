"""Deploy the web app to a free Hugging Face static Space.

Usage:
    (cd web && npm ci && npm run build)
    python scripts/deploy_space.py --space <user>/gemscan

The app runs the models in the visitor's browser (onnxruntime-web), so the
Space only hosts static files; the models are fetched from the model repo
pinned in web/src/config.ts.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "web" / "dist"

SPACE_CARD = """---
title: GemScan
emoji: 💎
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Gemstone inclusion detection, YOLO11 vs RT-DETR
custom_headers:
  cross-origin-opener-policy: same-origin
  cross-origin-embedder-policy: require-corp
  cross-origin-resource-policy: cross-origin
---

Detects inclusions in diamond photos with YOLO11s and RT-DETR-L, running entirely in your browser
(onnxruntime-web, WebGPU with a WASM fallback). Source: {source}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="e.g. your-name/gemscan")
    parser.add_argument("--source-url", default="https://github.com/flupxscop/gem-defect-detection")
    args = parser.parse_args()

    if not (DIST / "index.html").exists():
        sys.exit("web/dist not found. Run `npm run build` in web/ first.")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(SPACE_CARD.format(source=args.source_url))
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=args.space, repo_type="space")

    commit = api.upload_folder(
        folder_path=DIST,
        repo_id=args.space,
        repo_type="space",
        delete_patterns=["assets/*"],  # drop bundles from previous builds
        commit_message="Deploy",
    )
    host = args.space.replace("/", "-").lower()
    print(f"Deployed {commit.oid[:7]}: https://huggingface.co/spaces/{args.space}")
    print(f"App URL: https://{host}.static.hf.space")


if __name__ == "__main__":
    main()
