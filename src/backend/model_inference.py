"""Run local Thoth model predictions for captured minute folders."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import Config

THOTH_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(getattr(Config, "MODELS_DIR", THOTH_ROOT / "models")).expanduser()
DATA_TYPES = {"csi", "radar", "image", "video"}


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _metadata_candidates(model_path: Path) -> Iterable[Path]:
    yield model_path.with_suffix("")
    yield model_path.with_suffix(".json")
    yield model_path.with_suffix(".metadata.json")
    yield model_path.parent / f"{model_path.name}.json"
    yield model_path.parent / "metadata.json"


def _load_metadata(model_path: Path, folder_data_type: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    metadata_path: Optional[Path] = None
    for candidate in _metadata_candidates(model_path):
        if candidate.exists():
            metadata = _read_json(candidate)
            metadata_path = candidate
            break
    metadata.setdefault("model_name", model_path.stem)
    metadata.setdefault("data_type", folder_data_type)
    metadata.setdefault("labels", [])
    if metadata_path:
        metadata["metadata_path"] = str(metadata_path)
    return metadata

def _labels_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Return the required classification labels from model metadata."""
    value = metadata.get("labels")
    if isinstance(value, list):
        labels = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        labels = [item.strip() for item in value.split(",") if item.strip()]
    else:
        labels = []

    # Backward-compatible read only; new deployments must write labels.
    if not labels:
        for legacy_key in ("classes", "class_names"):
            legacy = metadata.get(legacy_key)
            if isinstance(legacy, list):
                labels = [str(item).strip() for item in legacy if str(item).strip()]
                break
    return labels


def discover_models(models_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return .pth models stored under dedicated data-type folders."""
    models_root = Path(models_root or MODELS_ROOT)
    discovered: List[Dict[str, Any]] = []
    if not models_root.exists():
        return discovered

    for data_type_dir in sorted(item for item in models_root.iterdir() if item.is_dir()):
        folder_type = data_type_dir.name.lower()
        if folder_type not in DATA_TYPES:
            continue
        for model_path in sorted(data_type_dir.rglob("*.pth")):
            metadata = _load_metadata(model_path, folder_type)
            discovered.append({
                "path": model_path,
                "model_name": str(metadata.get("model_name") or model_path.stem),
                "data_type": folder_type,
                "metadata": metadata,
            })
    return discovered


def _capture_paths(minute_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    outputs = manifest.get("outputs") if isinstance(manifest, dict) else {}
    outputs = outputs if isinstance(outputs, dict) else {}

    radar_files = outputs.get("radar", {}).get("files") if isinstance(outputs.get("radar"), dict) else None
    radar_path = Path(radar_files[0]) if isinstance(radar_files, list) and radar_files else None
    if radar_path is None:
        radar_path = next(iter(sorted(minute_dir.glob("mmw_radar_raw_*.bin"))), None)

    return {
        "csi": (
            Path(outputs.get("wifi_csi", {}).get("timestamped_path"))
            if isinstance(outputs.get("wifi_csi"), dict) and outputs.get("wifi_csi", {}).get("timestamped_path")
            else minute_dir / "wifi_csi_timestamped.csv"
        ),
        "radar": radar_path,
        "image": next(iter(sorted(minute_dir.glob("*.jpg"))), None)
        or next(iter(sorted(minute_dir.glob("*.jpeg"))), None)
        or next(iter(sorted(minute_dir.glob("*.png"))), None),
        "video": (
            Path(outputs.get("video", {}).get("path"))
            if isinstance(outputs.get("video"), dict) and outputs.get("video", {}).get("path")
            else minute_dir / "usb_camera.mp4"
        ),
    }


def _parse_csi_payload(raw: str) -> List[float]:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        values = json.loads(raw[start : end + 1])
    except Exception:
        return []
    return [float(value) for value in values if isinstance(value, (int, float))]


def _load_csi(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            for row in reader:
                raw = str(row.get("raw_csi_line") or row.get("line") or row.get("data") or "")
                values = _parse_csi_payload(raw)
                if values:
                    rows.append(values)
        else:
            handle.seek(0)
            for line in handle:
                values = _parse_csi_payload(line)
                if values:
                    rows.append(values)
    return rows


def _resize_flat(values: List[float], size: int) -> List[float]:
    if len(values) == size:
        return values
    if len(values) > size:
        return values[:size]
    return values + [0.0] * (size - len(values))


def _shape_from_metadata(metadata: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    raw = metadata.get("input_shape") or metadata.get("shape") or metadata.get("input")
    if not isinstance(raw, list):
        return None
    shape: List[int] = []
    for item in raw:
        try:
            dim = int(item)
        except Exception:
            return None
        if dim <= 0:
            return None
        shape.append(dim)
    return tuple(shape) if shape else None


def _tensor_for_data(data_type: str, path: Path, metadata: Dict[str, Any]):
    import numpy as np
    import torch

    shape = _shape_from_metadata(metadata)
    if data_type == "csi":
        rows = _load_csi(path)
        if not rows:
            raise ValueError(f"No CSI samples found in {path}")
        width = max(len(row) for row in rows)
        values = np.asarray([_resize_flat(row, width) for row in rows], dtype=np.float32)
    elif data_type == "radar":
        values = np.fromfile(path, dtype=np.uint8).astype(np.float32)
        if values.size == 0:
            raise ValueError(f"No radar bytes found in {path}")
        values = values / 255.0
    elif data_type == "image":
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Pillow is required for image model inference") from exc
        image = Image.open(path).convert("RGB")
        target = metadata.get("image_size") or metadata.get("size")
        if isinstance(target, list) and len(target) >= 2:
            image = image.resize((int(target[0]), int(target[1])))
        values = np.asarray(image, dtype=np.float32) / 255.0
        values = np.transpose(values, (2, 0, 1))
    elif data_type == "video":
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for video model inference") from exc
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError(f"No video frame found in {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target = metadata.get("image_size") or metadata.get("size")
        if isinstance(target, list) and len(target) >= 2:
            frame = cv2.resize(frame, (int(target[0]), int(target[1])))
        values = np.asarray(frame, dtype=np.float32) / 255.0
        values = np.transpose(values, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    flat = values.reshape(-1)
    if shape:
        target_size = int(np.prod(shape))
        flat = np.asarray(_resize_flat(flat.astype(float).tolist(), target_size), dtype=np.float32)
        values = flat.reshape(shape)

    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    if not metadata.get("input_includes_batch", False):
        tensor = tensor.unsqueeze(0)
    return tensor


def _load_torch_model(model_path: Path):
    import torch

    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model
    except Exception:
        try:
            loaded = torch.load(str(model_path), map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(str(model_path), map_location="cpu")
        if isinstance(loaded, dict) and callable(loaded.get("model")):
            model = loaded["model"]
        else:
            model = loaded
        if not callable(model):
            raise TypeError(
                "Model file did not contain a callable model. Save a TorchScript model "
                "or a torch-saved nn.Module for local inference."
            )
        if hasattr(model, "eval"):
            model.eval()
        return model


def _prediction_from_output(output: Any, labels: List[Any]) -> Dict[str, Any]:
    import torch

    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        output = torch.as_tensor(output)
    output = output.detach().cpu()
    if output.ndim > 1:
        output = output[0]
    probabilities = torch.softmax(output.float(), dim=0) if output.numel() > 1 else output.float()
    index = int(torch.argmax(probabilities).item()) if probabilities.numel() else 0
    confidence = _safe_float(probabilities[index].item()) if probabilities.numel() else None
    label = str(labels[index]) if index < len(labels) else str(index)
    return {
        "prediction": label,
        "class_index": index,
        "confidence": confidence,
        "probability": confidence,
        "scores": [_safe_float(value) for value in probabilities.tolist()],
    }


def predict_minute(minute_dir: Path, labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run all matching local models for one completed minute and write predictions.json."""
    minute_dir = Path(minute_dir)
    manifest = _read_json(minute_dir / "manifest.json")
    capture_paths = _capture_paths(minute_dir, manifest)
    timeline: List[Dict[str, Any]] = []
    deployed_models: List[Dict[str, Any]] = []
    discovered = discover_models()

    for model in discovered:
        model_path: Path = model["path"]
        data_type = str(model["data_type"])
        metadata = model["metadata"]
        data_path = capture_paths.get(data_type)
        entry: Dict[str, Any] = {
            "minute": minute_dir.name,
            "model_name": model["model_name"],
            "model_path": str(model_path),
            "data_type": data_type,
            "data_path": str(data_path) if data_path else None,
            "metadata_path": metadata.get("metadata_path"),
            "generated_at": iso_now(),
        }

        if not data_path or not data_path.exists():
            entry.update({"status": "skipped", "error": f"No {data_type} data found for this minute."})
            timeline.append(entry)
            deployed_models.append(entry)
            continue

        model_labels = _labels_from_metadata(metadata)
        if not model_labels:
            entry.update({
                "status": "skipped",
                "error": "Model metadata must include a non-empty labels list for classification deployment.",
            })
            timeline.append(entry)
            deployed_models.append(entry)
            continue

        try:
            tensor = _tensor_for_data(data_type, data_path, metadata)
            model_obj = _load_torch_model(model_path)
            import torch

            with torch.no_grad():
                output = model_obj(tensor)
            entry.update(_prediction_from_output(output, model_labels))
            entry["labels"] = [str(item) for item in model_labels]
            entry["classes"] = [str(item) for item in model_labels]
            entry["status"] = "ok"
        except Exception as exc:
            entry.update({
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
            })

        timeline.append(entry)
        deployed_models.append(entry)

    result = {
        "minute": minute_dir.name,
        "generated_at": iso_now(),
        "source": "backend.model_inference",
        "models_root": str(MODELS_ROOT),
        "labels": labels or manifest.get("labels") or [],
        "deployed_models": deployed_models,
        "timeline": timeline,
    }
    (minute_dir / "predictions.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
