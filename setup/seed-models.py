#!/usr/bin/env python3
"""Import the bundled E2 occupancy TorchScript exports into the user library.

The exports are self-describing: meta.json inside each archive carries the
normalization statistics, the tuned minute_threshold and the top-2 window
aggregation rule. They are registered as execution="minute" models so the
collector runs them once per finished capture minute (see
backend/model_runtime.py::ModelRegistry.run_minute) and are enabled by
default so occupancy is scored after every collected minute.

The minute verdict is a majority vote over per-window predictions; the
aggregation rule only shapes the reported confidence. Presence labels are
generic — archives exported as absent/present work the same as
empty/occupied.
"""
from pathlib import Path
import hashlib
import json
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

RADAR_INPUT = {"sensor": "radar", "representation": "e2_maps", "frames": 50,
               "shape": [1, 50, 2, 24, 24], "fit": "left_pad_latest",
               "normalization": {"kind": "none"}}
CSI_INPUT = {"sensor": "csi", "representation": "e2_grid", "samples": 128,
             "shape": [1, 128, 52], "fit": "left_pad_latest",
             "normalization": {"kind": "none"}}
specs = {
    "radar_occupancy.pt": [dict(RADAR_INPUT)],
    "fusion_occupancy.pt": [dict(RADAR_INPUT), dict(CSI_INPUT)],
}


def _embedded_meta(artifact: Path) -> dict:
    try:
        import torch
        extra = {"meta.json": ""}
        torch.jit.load(str(artifact), map_location="cpu", _extra_files=extra)
        return json.loads(extra["meta.json"]) if extra["meta.json"] else {}
    except Exception:
        return {}


for filename, inputs in specs.items():
    artifact = source / filename
    if not artifact.exists():
        continue
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest in existing:
        continue
    embedded = _embedded_meta(artifact)
    metadata = {
        "schema": "whispy-model/v1",
        "name": artifact.stem.replace("_", " ").title(),
        "version": str(embedded.get("modality") or "e2"),
        "inputs": inputs,
        "output": {"kind": "logits", "path": []},
        "class_names": ["empty", "occupied"],
        "execution": "minute",
        "aggregation": {
            "kind": str(embedded.get("minute_aggregation") or "top2"),
            "threshold": embedded.get("minute_threshold"),
        },
    }
    item = registry.add(artifact, metadata, source="provided-example")
    registry.set_enabled(item["id"], True)
    print(f"Imported {filename} as enabled minute-level model {item['id']}")
