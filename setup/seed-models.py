#!/usr/bin/env python3
"""Import the optional Desktop/models TorchScript examples once into the user library."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from backend.model_runtime import ModelRegistry

source = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT.parent / "models"
if not source.exists():
    source = ROOT / "models" / "examples"
registry = ModelRegistry(ROOT / "models" / "user")
existing = {item.get("sha256") for item in registry.list()}
specs = {
    "radar_occupancy_e3_finetuned.pt": [{"sensor": "radar", "representation": "raw_adc", "frames": 10, "shape": [1, 10, 2, 1, 3], "fit": "left_pad_latest", "normalization": {"kind": "none"}}],
    "fusion_occupancy_e2.pt": [
        {"sensor": "radar", "representation": "raw_adc", "frames": 10, "shape": [1, 10, 2, 1, 3], "fit": "left_pad_latest", "normalization": {"kind": "none"}},
        {"sensor": "csi", "representation": "iq", "samples": 256, "shape": [1, 256, 52], "fit": "left_pad_latest", "normalization": {"kind": "none"}},
    ],
}
for filename, inputs in specs.items():
    artifact = source / filename
    if not artifact.exists():
        continue
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest in existing:
        continue
    metadata = {"schema": "thoth-model/v1", "name": artifact.stem.replace("_", " ").title(), "version": "provided", "inputs": inputs, "output": {"kind": "logits", "path": []}, "class_names": ["empty", "occupied"]}
    item = registry.add(artifact, metadata, source="provided-example")
    print(f"Imported {filename} as disabled model {item['id']}")
