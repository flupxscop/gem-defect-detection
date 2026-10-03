"""Deploy the app to a Hugging Face Docker Space (free CPU tier).

Usage:
    python scripts/deploy_space.py --space <user>/gemscan --model-repo <user>/gemscan-models --revision <sha>

Uploads only source files; the Space builds the Dockerfile and downloads the
models from the model repo at the pinned revision.
"""

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = [
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "app/*.py",
    "results/comparison.csv",
    "web/index.html",
    "web/package.json",
    "web/package-lock.json",
    "web/tsconfig.json",
    "web/vite.config.ts",
    "web/src/**",
    "web/public/**",
]

SPACE_CARD = """---
title: GemScan
emoji: 💎
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Gemstone inclusion detection, YOLO11 vs RT-DETR
---

Live demo for {source}. Models: [{model_repo}](https://huggingface.co/{model_repo}).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="e.g. your-name/gemscan")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--revision", required=True, help="model repo commit to serve")
    parser.add_argument("--source-url", default="the GemScan project")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_variable(args.space, "MODEL_REPO", args.model_repo)
    api.add_space_variable(args.space, "MODEL_REVISION", args.revision)

    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(SPACE_CARD.format(source=args.source_url, model_repo=args.model_repo))
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=args.space, repo_type="space")

    commit = api.upload_folder(
        folder_path=ROOT,
        repo_id=args.space,
        repo_type="space",
        allow_patterns=SOURCE_FILES,
        delete_patterns=["app/**", "web/src/**", "web/public/**"],
        commit_message="Deploy",
    )
    print(f"Deployed {commit.oid[:7]}: https://huggingface.co/spaces/{args.space}")
    print(f"App URL: https://{args.space.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
