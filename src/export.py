"""Export trained weights to ONNX for the inference server.

Usage:
    python src/export.py

Writes models/<name>.onnx and models/manifest.json, which is everything the
server needs. The server image does not ship PyTorch.
"""

import json
import shutil
import sys

from common import IMGSZ, MODELS, ROOT, best_weights, load_model

MODELS_DIR = ROOT / "models"
OPSET = 17


def main() -> None:
    trained = [name for name in MODELS if best_weights(name).exists()]
    if not trained:
        sys.exit("No trained models found. Run `python src/train.py --model <name>` first.")

    MODELS_DIR.mkdir(exist_ok=True)
    manifest = {}
    for name in trained:
        model = load_model(name, best_weights(name))
        exported = model.export(format="onnx", imgsz=IMGSZ, opset=OPSET, simplify=True, dynamic=False)
        target = MODELS_DIR / f"{name}.onnx"
        shutil.copy(exported, target)

        spec = MODELS[name]
        manifest[name] = {
            "file": target.name,
            "label": spec["label"],
            "architecture": spec["architecture"],
            "family": spec["family"],
            "imgsz": IMGSZ,
            "classes": [model.names[i] for i in sorted(model.names)],
        }
        print(f"{name}: {target} ({target.stat().st_size / 1e6:.1f} MB)")

    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {MODELS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
