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
        allowed = {"raw_adc", "fft_power", "e2_maps"} if sensor == "radar" else {"iq", "magnitude_phase", "e2_grid"}
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
    execution = str(metadata.get("execution") or "chunk").strip().lower()
    if execution not in {"chunk", "minute"}:
        raise ModelValidationError("execution must be chunk or minute")
    aggregation = metadata.get("aggregation")
    if aggregation is not None and not isinstance(aggregation, dict):
        raise ModelValidationError("aggregation must be an object")
    return {
        "schema": MODEL_SCHEMA,
        "name": name,
        "version": version,
        "inputs": inputs,
        "output": {"kind": output_kind, "path": path},
        "class_names": _clean_names(metadata.get("class_names") or metadata.get("labels")),
        "execution": execution,
        "aggregation": dict(aggregation) if isinstance(aggregation, dict) else {},
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


# ---------------------------------------------------------------------------
# E2 occupancy preprocessing ("e2_maps" / "e2_grid" representations)
#
# Ported from the radar repo's E2/preprocess.py so the exported occupancy
# models receive exactly the tensors they were trained on:
#   radar: uint12 payload -> Hann-windowed range FFT -> range-Doppler and
#          range-azimuth maps -> log1p -> bilinear resize to 24x24 ->
#          non-overlapping 50-frame windows -> per-channel z-score.
#   csi:   CSI_DATA lines -> 52-subcarrier amplitude -> per-window linear
#          interpolation onto a 128-step grid -> log1p -> per-subcarrier
#          z-score; windows with <2 samples stay zeroed.
# Normalization statistics come from the archive's embedded meta.json.
# ---------------------------------------------------------------------------

E2_WINDOW_FRAMES = 50
E2_MAP_SIZE = 24
E2_CSI_STEPS = 128
E2_SUBCARRIERS = 52
E2_RANGE_BINS = 64
E2_AZ_BINS = 16
E2_FRAME_PAYLOAD_BYTES = 64 * 128 * 3 * 12 // 8  # 36864
_E2_HANN_R = np.hanning(128).astype(np.float32)
_E2_HANN_D = np.hanning(64).astype(np.float32)
_E2_CSI_MASK = np.array(
    [False] * 6 + [True] * 26 + [False] + [True] * 26 + [False] * 5,
    dtype=bool,
)
_E2_CSI_RE = re.compile(r"\[([^\]]+)\]")


def _e2_read_uint12(blob: bytes) -> np.ndarray:
    """Vectorized uint12 decode: packed bytes -> float32 samples."""
    data = np.frombuffer(blob, dtype=np.uint8)
    triplets = data.reshape(-1, 3).astype(np.uint16)
    a, b, c = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    out = np.empty(a.shape[0] * 2, dtype=np.float32)
    out[0::2] = (a << 4) + (b >> 4)
    out[1::2] = ((b % 16) << 8) + c
    return out


def _e2_frames_to_maps(payloads: Sequence[bytes]) -> np.ndarray:
    """(N) raw 36864-byte payloads -> (N, 2, 24, 24) log1p maps."""
    from scipy import fft as sfft
    from scipy.ndimage import zoom

    n = len(payloads)
    adc = _e2_read_uint12(b"".join(payloads)).reshape(n, 64, 128, 3)
    adc *= _E2_HANN_R[None, None, :, None] * _E2_HANN_D[None, :, None, None]
    R = sfft.fft(adc, axis=2, workers=-1)                            # range
    RD = sfft.fftshift(sfft.fft(R, axis=1, workers=-1), axes=1)      # doppler
    rd = np.abs(RD[:, :, :E2_RANGE_BINS, :]).mean(axis=3)            # (N,64,64)
    RA = sfft.fft(R[:, :, :E2_RANGE_BINS, :], n=E2_AZ_BINS, axis=3, workers=-1)
    ra = np.abs(RA).mean(axis=1).transpose(0, 2, 1)                  # (N,16,64)
    s = E2_MAP_SIZE
    rd = zoom(np.log1p(rd), (1, s / rd.shape[1], s / rd.shape[2]), order=1)
    ra = zoom(np.log1p(ra), (1, s / ra.shape[1], s / ra.shape[2]), order=1)
    return np.stack([rd, ra], axis=1).astype(np.float32)


def e2_radar_windows(
    frames: Sequence[bytes],
    times: Sequence[float],
    norm: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Raw wire packets -> normalized (nW, 50, 2, 24, 24) windows.

    Returns (windows, win_t0, win_t1, win_max_gap); windows is None when
    fewer than E2_WINDOW_FRAMES valid frames exist. Malformed packets are
    dropped before windowing so window boundaries stay aligned with
    `times`. win_max_gap is the largest inter-frame timestamp gap inside
    each window — large gaps mark FIFO-overflow holes whose Doppler
    content is corrupted.
    """
    payloads: list[bytes] = []
    kept_times: list[float] = []
    for index, packet in enumerate(frames):
        payload = bytes(packet[12:]) if len(packet) >= 12 else bytes(packet)
        if len(payload) != E2_FRAME_PAYLOAD_BYTES:
            continue
        payloads.append(payload)
        kept_times.append(float(times[index]) if index < len(times) else 0.0)
    n_win = len(payloads) // E2_WINDOW_FRAMES
    if n_win == 0:
        return None, np.empty(0), np.empty(0), np.empty(0)
    maps = _e2_frames_to_maps(payloads[: n_win * E2_WINDOW_FRAMES])
    windows = maps.reshape(n_win, E2_WINDOW_FRAMES, 2, E2_MAP_SIZE, E2_MAP_SIZE)
    mean = np.asarray(norm.get("radar_mean") or [0.0, 0.0], dtype=np.float32)
    std = np.asarray(norm.get("radar_std") or [1.0, 1.0], dtype=np.float32)
    windows = (windows - mean.reshape(1, 1, 2, 1, 1)) / std.reshape(1, 1, 2, 1, 1)
    kept = np.asarray(kept_times[: n_win * E2_WINDOW_FRAMES], dtype=np.float64)
    kept = kept.reshape(n_win, E2_WINDOW_FRAMES)
    win_t0 = kept[:, 0]
    win_t1 = kept[:, -1]
    win_max_gap = np.diff(kept, axis=1).max(axis=1)
    return windows.astype(np.float32), win_t0, win_t1, win_max_gap


def _e2_parse_csi_amplitude(line: str) -> np.ndarray | None:
    """One CSI_DATA line -> float32 amplitude vector of 52 subcarriers."""
    match = _E2_CSI_RE.search(line)
    if not match:
        return None
    tokens = [t for t in match.group(1).split(",") if t.strip()]
    if len(tokens) != 128:
        return None
    try:
        values = np.asarray(tokens, dtype=np.float64)
    except ValueError:
        return None
    imag = values[0::2][_E2_CSI_MASK]
    real = values[1::2][_E2_CSI_MASK]
    return np.hypot(real, imag).astype(np.float32)


def e2_csi_windows(
    csi_samples: Sequence[tuple[int, float, str]],
    win_t0: np.ndarray,
    win_t1: np.ndarray,
    norm: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """(receiver, t, line) samples -> normalized (nW, 128, 52) grids.

    The receiver with the most samples is used (E2 rule). Returns
    (windows, valid); windows with <2 in-window samples stay zeroed.
    """
    n_win = len(win_t0)
    out = np.zeros((n_win, E2_CSI_STEPS, E2_SUBCARRIERS), dtype=np.float32)
    valid = np.zeros(n_win, dtype=bool)
    by_receiver: dict[int, list[tuple[float, str]]] = {}
    for receiver, t, line in csi_samples:
        by_receiver.setdefault(int(receiver), []).append((float(t), line))
    if by_receiver:
        best = max(by_receiver, key=lambda r: len(by_receiver[r]))
        parsed = [(t, _e2_parse_csi_amplitude(line)) for t, line in by_receiver[best]]
        parsed = [(t, v) for t, v in parsed if v is not None]
        if parsed:
            ts = np.asarray([t for t, _ in parsed], dtype=np.float64)
            amp = np.stack([v for _, v in parsed])
            order = np.argsort(ts, kind="stable")
            ts, amp = ts[order], amp[order]
            for k in range(n_win):
                if win_t1[k] <= win_t0[k]:
                    continue
                mask = (ts >= win_t0[k]) & (ts <= win_t1[k])
                count = int(mask.sum())
                if count == 0:
                    continue
                if count == 1:
                    out[k] = np.repeat(amp[mask], E2_CSI_STEPS, axis=0)
                else:
                    grid = np.linspace(win_t0[k], win_t1[k], E2_CSI_STEPS)
                    for j in range(E2_SUBCARRIERS):
                        out[k, :, j] = np.interp(grid, ts[mask], amp[mask, j])
                valid[k] = count >= 2
    np.log1p(out, out=out)
    mean = np.asarray(norm.get("csi_mean") or [0.0] * E2_SUBCARRIERS, dtype=np.float32)
    std = np.asarray(norm.get("csi_std") or [1.0] * E2_SUBCARRIERS, dtype=np.float32)
    out = (out - mean) / std
    out[~valid] = 0.0
    return out.astype(np.float32), valid


def _load_with_meta(path: Path) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    extra = {"meta.json": ""}
    try:
        model = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
    except TypeError:
        model = torch.jit.load(str(path), map_location="cpu")
        extra = {"meta.json": ""}
    model.eval()
    try:
        meta = json.loads(extra["meta.json"]) if extra.get("meta.json") else {}
    except Exception:
        meta = {}
    return model, meta if isinstance(meta, dict) else {}


# Loaded TorchScript modules are expensive to deserialize, so keep them in a
# small cache keyed by path + mtime. A re-seeded or re-uploaded artifact gets a
# new mtime and is reloaded automatically on the next call.
_MODEL_CACHE: dict[str, tuple[float, Any, dict[str, Any]]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _cached_load(path: Path) -> tuple[Any, dict[str, Any]]:
    """Return the loaded (model, meta) for *path*, reloading only on change."""
    key = str(path)
    try:
        mtime = os.path.getmtime(key)
    except OSError:
        mtime = -1.0
    with _MODEL_CACHE_LOCK:
        hit = _MODEL_CACHE.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1], hit[2]
    model, meta = _load_with_meta(path)
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = (mtime, model, meta)
    return model, meta


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

    def get(self, model_id: str) -> dict[str, Any] | None:
        match = next((item for item in self.list() if item.get("id") == model_id), None)
        return dict(match) if match is not None else None

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
            "ha_link": None,
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

    def set_ha_link(self, model_id: str, ha_link: object) -> dict[str, Any]:
        """Persist a per-model Home Assistant device link.

        ha_link may be None/falsey to clear the link, or a dict with:
          enabled   - whether this model drives the linked entity
          entity_id - HA entity to control (light./switch./fan./input_number./number.)
          scope     - 'chunk' or 'minute' (minute also covers partial-minute)
          mode      - 'categorical' (on/off) or 'numeric' (proportional value)
        """
        with _REGISTRY_LOCK:
            models = self.list()
            match = next((item for item in models if item.get("id") == model_id), None)
            if match is None:
                raise KeyError(model_id)
            if not ha_link:
                match["ha_link"] = None
                self._save(models)
                return dict(match)
            if not isinstance(ha_link, dict):
                raise ModelValidationError("ha_link must be an object")
            entity_id = str(ha_link.get("entity_id") or "").strip().lower()
            scope = str(ha_link.get("scope") or "minute").strip().lower()
            mode = str(ha_link.get("mode") or "categorical").strip().lower()
            if scope not in {"chunk", "minute"}:
                raise ModelValidationError("ha_link.scope must be 'chunk' or 'minute'")
            if mode not in {"categorical", "numeric"}:
                raise ModelValidationError("ha_link.mode must be 'categorical' or 'numeric'")
            if "." not in entity_id:
                raise ModelValidationError("ha_link.entity_id must look like 'domain.name'")
            match["ha_link"] = {
                "enabled": bool(ha_link.get("enabled")),
                "entity_id": entity_id,
                "scope": scope,
                "mode": mode,
            }
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
            if str(metadata.get("execution") or "chunk") == "minute":
                continue  # minute-level models run once via run_minute()
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
                model, _ = _cached_load(self.artifacts / str(item["filename"]))
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

    def run_minute(
        self,
        radar_frames: Sequence[bytes],
        radar_times: Sequence[float],
        csi_samples: Sequence[tuple[int, float, str]],
        timestamp: str,
    ) -> list[dict[str, Any]]:
        """Run every enabled minute-execution model once over the whole minute.

        radar_frames are raw wire packets (12-byte header + uint12 payload),
        radar_times their capture seconds on the same clock as the CSI sample
        times in csi_samples ((receiver, t, line) tuples). Each model produces
        a single timeline entry: per-window probabilities are aggregated with
        the archive's minute_aggregation rule (default top-2 mean) and compared
        against minute_threshold — matching the E2 evaluation protocol.
        """
        torch = _torch()
        results: list[dict[str, Any]] = []
        for item in self.list():
            if not item.get("enabled"):
                continue
            metadata = item.get("metadata") or {}
            if str(metadata.get("execution") or "chunk") != "minute":
                continue
            base = {
                "model_id": item.get("id"),
                "model_name": metadata.get("name"),
                "model_version": metadata.get("version"),
                "chunk_index": -1,
                "scope": "minute",
                "timestamp": timestamp,
            }
            try:
                model, embedded = _cached_load(self.artifacts / str(item["filename"]))
                norm = embedded.get("norm") if isinstance(embedded.get("norm"), dict) else {}
                aggregation = embedded.get("minute_aggregation") or \
                    (metadata.get("aggregation") or {}).get("kind") or "top2"
                threshold = float(
                    embedded.get("minute_threshold")
                    or (metadata.get("aggregation") or {}).get("threshold")
                    or embedded.get("threshold")
                    or 0.5
                )
                tensors: list[Any] = []
                missing: list[str] = []
                win_t0 = win_t1 = None
                win_max_gap = np.zeros(0)
                csi_valid = np.zeros(0, dtype=bool)
                for spec in metadata.get("inputs") or []:
                    sensor = spec.get("sensor")
                    representation = spec.get("representation")
                    if sensor == "radar" and representation == "e2_maps":
                        windows, win_t0, win_t1, win_max_gap = e2_radar_windows(radar_frames, radar_times, norm)
                        if windows is None:
                            missing.append(
                                f"radar requires {E2_WINDOW_FRAMES} valid frames; "
                                f"received {len(radar_frames)}"
                            )
                        else:
                            tensors.append(torch.from_numpy(windows))
                    elif sensor == "csi" and representation == "e2_grid":
                        if win_t0 is None:
                            missing.append("csi e2_grid requires radar e2_maps windows first")
                            continue
                        grid, csi_valid = e2_csi_windows(csi_samples, win_t0, win_t1, norm)
                        if not bool(csi_valid.any()):
                            missing.append("csi produced no window with >=2 samples this minute")
                        tensors.append(torch.from_numpy(grid))
                    elif sensor == "radar":
                        if not radar_frames:
                            missing.append(f"radar requires {spec.get('frames')} frames; received 0")
                        tensors.append(torch.from_numpy(np.ascontiguousarray(radar_tensor(radar_frames, spec))).float())
                    else:
                        flat = [(rx, line) for rx, _t, line in csi_samples]
                        tensors.append(torch.from_numpy(np.ascontiguousarray(csi_tensor(flat, spec))).float())
                if missing:
                    results.append({**base, "status": "skipped", "reason": "; ".join(missing)})
                    continue
                with torch.inference_mode():
                    output = _select_output(model(*tensors), metadata.get("output", {}).get("path") or [])
                    values = output.detach().cpu().float().reshape(-1)
                names = metadata["class_names"]
                if values.numel() == 1 or (metadata.get("binary_output") and len(names) == 2):
                    # one score per window? binary models emit (nW,) or (nW,1)
                    probs = torch.sigmoid(values) if metadata["output"]["kind"] == "logits" else values.clamp(0, 1)
                else:
                    probs = torch.softmax(values, dim=0) if metadata["output"]["kind"] == "logits" else values
                probs_np = probs.numpy().reshape(-1)
                # FIFO-overflow holes corrupt a window's Doppler content and
                # spike its occupancy score. Vote only over windows whose
                # frame timing is intact (<=2.5x the nominal 100 ms period);
                # if too few survive, use the least-corrupted windows instead.
                excluded = 0
                pool = probs_np
                if win_max_gap.size == probs_np.size and probs_np.size:
                    clean = win_max_gap <= 0.25
                    if int(clean.sum()) >= 2:
                        pool = probs_np[clean]
                        excluded = int((~clean).sum())
                    elif probs_np.size >= 2:
                        pool = probs_np[np.argsort(win_max_gap)[:2]]
                        excluded = int(probs_np.size - 2)
                if str(aggregation) == "top2" and pool.size >= 2:
                    minute_prob = float(np.sort(pool)[-2:].mean())
                else:
                    minute_prob = float(pool.mean()) if pool.size else 0.0
                if len(names) == 2:
                    # predict the most probable class (argmax) so the reported
                    # class always matches the higher score; `threshold` stays
                    # as the confidence gate surfaced to consumers.
                    index = 1 if minute_prob >= 0.5 else 0
                    scores = {names[0]: 1.0 - minute_prob, names[1]: minute_prob}
                    confidence = minute_prob if index == 1 else 1.0 - minute_prob
                else:
                    index = int(np.argmax(probs_np)) if probs_np.size else 0
                    scores = {name: float(probs_np[i]) for i, name in enumerate(names[: probs_np.size])}
                    confidence = float(probs_np[index]) if probs_np.size else 0.0
                results.append({
                    **base,
                    "status": "ok",
                    "class": names[index],
                    "confidence": confidence,
                    "confidence_saturated": confidence >= 0.999,
                    "scores": scores,
                    "threshold": threshold,
                    "aggregation": str(aggregation),
                    "window_count": int(probs_np.size),
                    "windows_excluded": excluded,
                    "csi_windows_valid": int(csi_valid.sum()) if csi_valid.size else 0,
                    "window_probabilities": [round(float(p), 4) for p in probs_np],
                    "window_max_gaps": [round(float(g), 3) for g in win_max_gap],
                })
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
