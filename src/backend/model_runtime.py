"""User-managed TorchScript classification models for Thoth captures.

The registry deliberately stores no executable Python.  Models must be self-contained
TorchScript archives and every tensor transformation is described by thoth-model/v1
metadata.  PyTorch is imported lazily so capture and dashboard features still start
on systems where the optional runtime has not been installed yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

MODEL_SCHEMA = "thoth-model/v1"
MANIFEST_SCHEMA = "thoth-minute-manifest/v7"
SUPPORTED_EXTENSIONS = {".pt", ".pth"}
SUPPORTED_SENSORS = {"radar", "csi"}
_REGISTRY_LOCK = threading.RLock()


class ModelValidationError(ValueError):
    """Raised when an artifact or its metadata violates the public contract."""


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on installation profile
        raise ModelValidationError("CPU PyTorch is required to validate or run TorchScript models") from exc
    return torch


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _clean_names(value: object) -> list[str]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value:
        name = " ".join(str(item or "").split())
        if name and name not in result:
            result.append(name)
    return result


def _positive_int(value: object, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{label} must be a positive integer") from exc
    if number < 1:
        raise ModelValidationError(f"{label} must be a positive integer")
    return number


def normalize_metadata(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ModelValidationError("metadata must be a JSON object")
    if metadata.get("schema") != MODEL_SCHEMA:
        raise ModelValidationError(f"metadata.schema must be {MODEL_SCHEMA}")
    name = " ".join(str(metadata.get("name") or metadata.get("model_name") or "").split())
    version = " ".join(str(metadata.get("version") or "").split())
    if not name or not version:
        raise ModelValidationError("model name and version are required")

    raw_inputs = metadata.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ModelValidationError("inputs must contain at least one ordered radar or CSI input")
    inputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_inputs):
        if not isinstance(raw, dict):
            raise ModelValidationError(f"inputs[{index}] must be an object")
        sensor = str(raw.get("sensor") or raw.get("kind") or "").strip().lower()
        if sensor not in SUPPORTED_SENSORS or sensor in seen:
            raise ModelValidationError("v1 inputs must contain radar and/or csi at most once")
        seen.add(sensor)
        representation = str(raw.get("representation") or "").strip().lower()
        allowed = {"raw_adc", "fft_power"} if sensor == "radar" else {"iq", "magnitude_phase"}
        if representation not in allowed:
            raise ModelValidationError(f"unsupported {sensor} representation: {representation}")
        shape = raw.get("shape") or raw.get("expected_shape")
        if not isinstance(shape, list) or not shape:
            raise ModelValidationError(f"inputs[{index}].shape must be a non-empty integer list")
        normalized_shape = [_positive_int(item, f"inputs[{index}].shape") for item in shape]
        count_key = "frames" if sensor == "radar" else "samples"
        count = _positive_int(raw.get(count_key), f"inputs[{index}].{count_key}")
        fit = str(raw.get("fit") or raw.get("padding_truncation") or "left_pad_latest").lower()
        if fit not in {"left_pad_latest", "right_pad_earliest"}:
            raise ModelValidationError("padding/truncation must be left_pad_latest or right_pad_earliest")
        normalization = raw.get("normalization") or {"kind": "none"}
        if isinstance(normalization, str):
            normalization = {"kind": normalization}
        if not isinstance(normalization, dict):
            raise ModelValidationError("normalization must be an object")
        norm_kind = str(normalization.get("kind") or "none").lower()
        if norm_kind not in {"none", "zscore", "minmax"}:
            raise ModelValidationError(f"unsupported normalization: {norm_kind}")
        item = {
            "sensor": sensor,
            "representation": representation,
            count_key: count,
            "shape": normalized_shape,
            "fit": fit,
            "normalization": {**normalization, "kind": norm_kind},
        }
        if sensor == "csi":
            receivers = raw.get("receivers", raw.get("receiver_selection", []))
            subcarriers = raw.get("subcarriers", raw.get("subcarrier_selection", []))
            item["receivers"] = [int(value) for value in receivers] if isinstance(receivers, list) else []
            item["subcarriers"] = [int(value) for value in subcarriers] if isinstance(subcarriers, list) else []
        inputs.append(item)

    output = metadata.get("output")
    if not isinstance(output, dict):
        output = {"kind": metadata.get("output_kind")}
    output_kind = str(output.get("kind") or "").lower()
    if output_kind not in {"logits", "probabilities"}:
        raise ModelValidationError("output.kind must be logits or probabilities")
    path = output.get("path", [])
    if path is None:
        path = []
    if not isinstance(path, list) or any(not isinstance(item, (str, int)) for item in path):
        raise ModelValidationError("output.path must be a list of dict keys and/or tuple indexes")
    return {
        "schema": MODEL_SCHEMA,
        "name": name,
        "version": version,
        "inputs": inputs,
        "output": {"kind": output_kind, "path": path},
        "class_names": _clean_names(metadata.get("class_names") or metadata.get("labels")),
    }


def _select_output(value: Any, path: Sequence[str | int]) -> Any:
    for selector in path:
        try:
            value = value[selector]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelValidationError(f"model output does not contain selector {selector!r}") from exc
    return value


def _embedded_names(model: Any) -> list[str]:
    for attribute in ("class_names", "labels"):
        try:
            names = _clean_names(getattr(model, attribute))
        except Exception:
            names = []
        if names:
            return names
    return []


def validate_torchscript(path: Path, metadata: object) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ModelValidationError("model filename must end in .pt or .pth")
    normalized = normalize_metadata(metadata)
    torch = _torch()
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
    except Exception as exc:
        raise ModelValidationError(f"artifact is not a loadable self-contained TorchScript model: {exc}") from exc
    tensors = [torch.zeros(tuple(item["shape"]), dtype=torch.float32) for item in normalized["inputs"]]
    try:
        with torch.inference_mode():
            output = model(*tensors)
        output = _select_output(output, normalized["output"]["path"])
        if not torch.is_tensor(output):
            raise TypeError("selected output is not a tensor")
        values = output.detach().cpu().float().reshape(-1)
    except Exception as exc:
        raise ModelValidationError(f"dry-run inference failed: {exc}") from exc
    if values.numel() < 1 or not bool(torch.isfinite(values).all()):
        raise ModelValidationError("selected output must contain at least one finite class value")
    names = _embedded_names(model) or normalized["class_names"]
    if not names:
        raise ModelValidationError("class_names are required when the TorchScript archive has no labels attribute")
    if values.numel() == 1 and len(names) == 2:
        normalized["binary_output"] = True
    elif len(names) != values.numel():
        raise ModelValidationError(f"class_names has {len(names)} entries but dry-run output has {values.numel()}")
    normalized["class_names"] = names
    normalized["output_dimension"] = int(values.numel())
    return normalized


def _fit(values: np.ndarray, shape: Sequence[int], mode: str) -> np.ndarray:
    size = math.prod(shape)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size >= size:
        flat = flat[-size:] if mode == "left_pad_latest" else flat[:size]
    else:
        padding = np.zeros(size - flat.size, dtype=np.float32)
        flat = np.concatenate((padding, flat)) if mode == "left_pad_latest" else np.concatenate((flat, padding))
    return flat.reshape(tuple(shape))


def _normalize(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    kind = spec.get("kind", "none")
    if kind == "zscore":
        mean = float(spec.get("mean", 0.0))
        std = float(spec.get("std", 1.0))
        if std == 0:
            raise ModelValidationError("zscore std cannot be zero")
        return (values - mean) / std
    if kind == "minmax":
        minimum = float(spec.get("min", 0.0))
        maximum = float(spec.get("max", 1.0))
        if maximum <= minimum:
            raise ModelValidationError("minmax max must be greater than min")
        return (values - minimum) / (maximum - minimum)
    return values


def radar_tensor(frames: Sequence[bytes], spec: dict[str, Any]) -> np.ndarray:
    requested = int(spec["frames"])
    selected = list(frames[-requested:])
    rows: list[np.ndarray] = []
    for packet in selected:
        payload = packet[12:] if len(packet) >= 12 and int.from_bytes(packet[:4], "little") == 0 else packet
        adc = np.frombuffer(payload[: len(payload) - (len(payload) % 2)], dtype="<i2").astype(np.float32)
        if adc.size == 0:
            raise ModelValidationError("radar input contains a malformed or empty frame")
        if spec["representation"] == "fft_power":
            adc = np.abs(np.fft.rfft(adc)).astype(np.float32) ** 2
        rows.append(adc)
    values = np.concatenate(rows) if rows else np.asarray([], dtype=np.float32)
    return _normalize(_fit(values, spec["shape"], spec["fit"]), spec["normalization"])


def _csi_pairs(line: str) -> np.ndarray:
    matches = re.findall(r"\[([^\]]*)\]", line)
    payload = matches[-1] if matches else line
    values = np.asarray([float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", payload)], dtype=np.float32)
    values = values[: values.size - (values.size % 2)]
    return values.reshape(-1, 2) if values.size else np.empty((0, 2), dtype=np.float32)


def csi_tensor(samples: Sequence[str | tuple[int, str]], spec: dict[str, Any]) -> np.ndarray:
    receivers = set(spec.get("receivers") or [])
    selected: list[np.ndarray] = []
    for value in samples:
        receiver, line = value if isinstance(value, tuple) else (0, value)
        if receivers and int(receiver) not in receivers:
            continue
        pairs = _csi_pairs(str(line))
        if not len(pairs):
            raise ModelValidationError("CSI input contains a malformed sample")
        carriers = spec.get("subcarriers") or []
        if carriers:
            valid = [index for index in carriers if 0 <= index < len(pairs)]
            if len(valid) != len(carriers):
                raise ModelValidationError("CSI subcarrier selection exceeds the available sample")
            pairs = pairs[valid]
        if spec["representation"] == "magnitude_phase" and len(pairs):
            # ESP CSI serializes imaginary then real components.
            pairs = np.stack((np.hypot(pairs[:, 1], pairs[:, 0]), np.arctan2(pairs[:, 0], pairs[:, 1])), axis=1)
        selected.append(pairs.reshape(-1))
    requested = int(spec["samples"])
    selected = selected[-requested:]
    values = np.concatenate(selected) if selected else np.asarray([], dtype=np.float32)
    return _normalize(_fit(values, spec["shape"], spec["fit"]), spec["normalization"])


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.path = self.root / "registry.json"

    def list(self) -> list[dict[str, Any]]:
        with _REGISTRY_LOCK:
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return []
            return [dict(item) for item in value.get("models", []) if isinstance(item, dict)]

    def _save(self, models: list[dict[str, Any]]) -> None:
        _atomic_json(self.path, {"schema": "thoth-model-registry/v1", "models": models})

    def add(self, uploaded: Path, metadata: object, *, source: str = "local") -> dict[str, Any]:
        normalized = validate_torchscript(uploaded, metadata)
        raw = Path(uploaded).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        model_id = str(uuid.uuid4())
        suffix = Path(uploaded).suffix.lower()
        self.artifacts.mkdir(parents=True, exist_ok=True)
        destination = self.artifacts / f"{model_id}{suffix}"
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, destination)
        item = {
            "id": model_id,
            "filename": destination.name,
            "sha256": digest,
            "bytes": len(raw),
            "enabled": False,
            "source": source,
            "metadata": normalized,
            "last_error": None,
        }
        with _REGISTRY_LOCK:
            models = self.list()
            models.append(item)
            self._save(models)
        return item

    def set_enabled(self, model_id: str, enabled: bool) -> dict[str, Any]:
        with _REGISTRY_LOCK:
            models = self.list()
            match = next((item for item in models if item.get("id") == model_id), None)
            if match is None:
                raise KeyError(model_id)
            match["enabled"] = bool(enabled)
            match["last_error"] = None
            self._save(models)
            return dict(match)

    def delete(self, model_id: str) -> None:
        with _REGISTRY_LOCK:
            models = self.list()
            match = next((item for item in models if item.get("id") == model_id), None)
            if match is None:
                raise KeyError(model_id)
            (self.artifacts / str(match.get("filename"))).unlink(missing_ok=True)
            self._save([item for item in models if item.get("id") != model_id])

    def _set_last_error(self, model_id: str, message: str | None) -> None:
        with _REGISTRY_LOCK:
            models = self.list()
            match = next((model for model in models if model.get("id") == model_id), None)
            if match is not None and match.get("last_error") != message:
                match["last_error"] = message
                self._save(models)

    def run_enabled(self, radar_frames: Sequence[bytes], csi_samples: Sequence[str | tuple[int, str]], chunk_index: int, timestamp: str) -> list[dict[str, Any]]:
        torch = _torch()
        results: list[dict[str, Any]] = []
        for item in self.list():
            if not item.get("enabled"):
                continue
            metadata = item.get("metadata") or {}
            base = {"model_id": item.get("id"), "model_name": metadata.get("name"), "model_version": metadata.get("version"), "chunk_index": int(chunk_index), "timestamp": timestamp}
            try:
                tensors = []
                missing = []
                for spec in metadata.get("inputs") or []:
                    if spec.get("sensor") == "radar":
                        if not radar_frames:
                            missing.append(f"radar requires {spec.get('frames')} frames; received 0")
                        array = radar_tensor(radar_frames, spec)
                    else:
                        receivers = set(spec.get("receivers") or [])
                        available = sum(1 for value in csi_samples if not receivers or (isinstance(value, tuple) and int(value[0]) in receivers) or (not isinstance(value, tuple) and 0 in receivers))
                        if available == 0:
                            missing.append(f"csi requires {spec.get('samples')} selected samples; received 0")
                        array = csi_tensor(csi_samples, spec)
                    tensors.append(torch.from_numpy(np.ascontiguousarray(array)).float())
                if missing:
                    results.append({**base, "status": "skipped", "reason": "; ".join(missing)})
                    continue
                model = torch.jit.load(str(self.artifacts / str(item["filename"])), map_location="cpu")
                model.eval()
                with torch.inference_mode():
                    output = _select_output(model(*tensors), metadata.get("output", {}).get("path") or [])
                    values = output.detach().cpu().float().reshape(-1)
                    if metadata.get("binary_output"):
                        positive = torch.sigmoid(values[0]) if metadata["output"]["kind"] == "logits" else values[0].clamp(0, 1)
                        probabilities = torch.stack((1 - positive, positive))
                    else:
                        probabilities = torch.softmax(values, dim=0) if metadata["output"]["kind"] == "logits" else values
                    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
                index = int(torch.argmax(probabilities).item())
                names = metadata["class_names"]
                confidence = float(probabilities[index].item())
                results.append({**base, "status": "ok", "class": names[index], "confidence": confidence, "confidence_saturated": confidence >= 0.999, "scores": {name: float(probabilities[i].item()) for i, name in enumerate(names)}})
                self._set_last_error(str(item.get("id")), None)
            except Exception as exc:
                message = str(exc)
                results.append({**base, "status": "error", "error": message})
                self._set_last_error(str(item.get("id")), message)
        return results


def append_manifest_predictions(path: Path, results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Atomically append result timelines without altering human labels."""
    with _REGISTRY_LOCK:
        try:
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            manifest = {}
        manifest["schema"] = MANIFEST_SCHEMA
        predictions = manifest.get("model_predictions")
        if not isinstance(predictions, list):
            predictions = []
        by_id = {str(item.get("model_id")): item for item in predictions if isinstance(item, dict)}
        for result in results:
            model_id = str(result.get("model_id"))
            timeline = by_id.setdefault(model_id, {"model_id": model_id, "model_name": result.get("model_name"), "model_version": result.get("model_version"), "timeline": []})
            timeline["timeline"].append(dict(result))
        manifest["model_predictions"] = list(by_id.values())
        _atomic_json(Path(path), manifest)
        return manifest
