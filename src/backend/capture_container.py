"""Versioned, synchronized NPZ storage for one Thoth capture minute.

The container intentionally uses only numeric NumPy arrays.  Variable-length
payloads are represented as a flat uint8 array plus int64 offsets so readers
can keep ``allow_pickle=False``.  A compact JSON metadata document describes
the device, settings, labels and per-second sensor coverage.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


CONTAINER_FILENAME = "capture.npz"
CONTAINER_SCHEMA = "thoth-capture-npz/v1"
NANOSECONDS = 1_000_000_000


def _iso_ns(value: object) -> int:
    if not value:
        return 0
    try:
        parsed = dt.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
        return int(parsed.timestamp() * NANOSECONDS)
    except (TypeError, ValueError, OverflowError):
        return 0


def _json_array(value: object) -> np.ndarray:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def _pack_blobs(blobs: Iterable[bytes]) -> tuple[np.ndarray, np.ndarray]:
    values = [bytes(value) for value in blobs]
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    if values:
        offsets[1:] = np.cumsum([len(value) for value in values], dtype=np.int64)
    joined = b"".join(values)
    return np.frombuffer(joined, dtype=np.uint8).copy(), offsets


def _blob_at(data: np.ndarray, offsets: np.ndarray, index: int) -> bytes:
    if index < 0 or index + 1 >= len(offsets):
        return b""
    start, finish = int(offsets[index]), int(offsets[index + 1])
    return data[start:finish].tobytes()


def _split_radar_packets(data: bytes) -> Iterator[bytes]:
    """Yield complete DreamHat frames, retaining their 12-byte wire header."""
    offset = 0
    while offset + 12 <= len(data):
        if int.from_bytes(data[offset:offset + 4], "little") != 0:
            break
        payload_size = int.from_bytes(data[offset + 8:offset + 12], "little")
        finish = offset + 12 + payload_size
        if payload_size < 0 or finish > len(data):
            break
        yield data[offset:finish]
        offset = finish


def _path_for(minute_dir: Path, value: object) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = minute_dir / candidate.name
    return candidate if candidate.exists() and candidate.is_file() else None


def _second_index(
    monotonic_ns: int,
    unix_ns: int,
    origin_monotonic_ns: int,
    origin_unix_ns: int,
    fallback: int,
    second_count: int,
) -> int:
    if monotonic_ns > 0 and origin_monotonic_ns > 0:
        value = (monotonic_ns - origin_monotonic_ns) // NANOSECONDS
    elif unix_ns > 0 and origin_unix_ns > 0:
        value = (unix_ns - origin_unix_ns) // NANOSECONDS
    else:
        value = fallback
    return max(0, min(second_count - 1, int(value)))


def _manifest_without_host_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    """Copy a manifest while replacing absolute capture paths with basenames."""
    def clean(value: object, key: str = "") -> object:
        if isinstance(value, dict):
            return {str(item_key): clean(item_value, str(item_key)) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [clean(item, key) for item in value]
        if isinstance(value, str) and key in {
            "path", "bin_path", "camera_path", "xy_tracking", "file", "files"
        }:
            return Path(value).name
        return value
    return clean(manifest)  # type: ignore[return-value]


def build_capture_container(
    minute_dir: Path,
    manifest: dict[str, Any],
    *,
    remove_fragments: bool = True,
) -> dict[str, Any]:
    """Build and validate ``capture.npz`` from the live minute fragments.

    Source files are removed only after the finished archive has been opened
    successfully with pickle disabled.  Metadata and derived visualization
    artifacts remain as sidecars for fast timeline scans and old clients.
    """
    minute_dir = Path(minute_dir)
    minute_dir.mkdir(parents=True, exist_ok=True)
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    duration = max(0.001, float(manifest.get("duration_seconds") or 60.0))
    second_count = max(1, int(math.ceil(duration)))
    origin_unix_ns = _iso_ns(manifest.get("capture_started") or manifest.get("scheduled_start"))
    origin_monotonic_ns = int(manifest.get("capture_started_monotonic_ns") or 0)
    second_unix_ns = origin_unix_ns + np.arange(second_count, dtype=np.int64) * NANOSECONDS
    second_mono_ns = (
        origin_monotonic_ns + np.arange(second_count, dtype=np.int64) * NANOSECONDS
        if origin_monotonic_ns > 0 else np.zeros(second_count, dtype=np.int64)
    )

    fragment_paths: set[Path] = set()
    radar_payloads: list[bytes] = []
    radar_unix: list[int] = []
    radar_mono: list[int] = []
    radar_seconds: list[int] = []
    radar_sequences: list[int] = []
    radar = outputs.get("radar") if isinstance(outputs.get("radar"), dict) else {}
    chunks = radar.get("chunks") if isinstance(radar.get("chunks"), list) else []
    for fallback_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        chunk_index = int(chunk.get("chunk_index") or fallback_index)
        path = _path_for(minute_dir, chunk.get("bin_path"))
        if path is None:
            continue
        fragment_paths.add(path)
        packets = list(_split_radar_packets(path.read_bytes()))
        raw_mono = chunk.get("frame_monotonic_ns")
        frame_mono = [int(value) for value in raw_mono] if isinstance(raw_mono, list) else []
        start_unix = _iso_ns(chunk.get("started"))
        finish_unix = _iso_ns(chunk.get("finished_capture") or chunk.get("finished"))
        for frame_index, packet in enumerate(packets):
            mono_ns = frame_mono[frame_index] if frame_index < len(frame_mono) else 0
            if len(packets) > 1 and finish_unix > start_unix:
                unix_ns = start_unix + ((finish_unix - start_unix) * frame_index // (len(packets) - 1))
            else:
                unix_ns = start_unix or origin_unix_ns + chunk_index * NANOSECONDS
            second = _second_index(
                mono_ns, unix_ns, origin_monotonic_ns, origin_unix_ns,
                chunk_index, second_count,
            )
            radar_payloads.append(packet)
            radar_unix.append(unix_ns)
            radar_mono.append(mono_ns)
            radar_seconds.append(second)
            radar_sequences.append(int.from_bytes(packet[4:8], "little") if len(packet) >= 8 else -1)

    camera_by_second: list[tuple[bytes, int, int]] = [(b"", 0, 0) for _ in range(second_count)]
    camera = outputs.get("camera") if isinstance(outputs.get("camera"), dict) else {}
    camera_entries = camera.get("frames") if isinstance(camera.get("frames"), list) else []
    if not camera_entries:
        camera_entries = [
            {
                "path": chunk.get("camera_path"),
                "captured_at": chunk.get("camera_captured_at") or chunk.get("started"),
                "monotonic_ns": chunk.get("camera_monotonic_ns"),
                "second_index": chunk.get("chunk_index"),
            }
            for chunk in chunks if isinstance(chunk, dict) and chunk.get("camera_path")
        ]
    for fallback_index, frame in enumerate(camera_entries):
        if not isinstance(frame, dict):
            continue
        path = _path_for(minute_dir, frame.get("path") or frame.get("camera_path"))
        if path is None:
            continue
        fragment_paths.add(path)
        unix_ns = _iso_ns(frame.get("captured_at") or frame.get("timestamp"))
        mono_ns = int(frame.get("monotonic_ns") or frame.get("camera_monotonic_ns") or 0)
        second = _second_index(
            mono_ns, unix_ns, origin_monotonic_ns, origin_unix_ns,
            int(frame.get("second_index") or fallback_index), second_count,
        )
        # At most one camera frame belongs to a synchronized second.
        if not camera_by_second[second][0]:
            camera_by_second[second] = (path.read_bytes(), unix_ns, mono_ns)

    csi_payloads: list[bytes] = []
    csi_unix: list[int] = []
    csi_mono: list[int] = []
    csi_seconds: list[int] = []
    csi_receivers: list[int] = []
    csi = outputs.get("wifi_csi") if isinstance(outputs.get("wifi_csi"), dict) else {}
    receivers = csi.get("receivers") if isinstance(csi.get("receivers"), list) else []
    if not receivers and csi.get("path"):
        receivers = [csi]
    receiver_metadata: list[dict[str, Any]] = []
    for receiver_index, receiver in enumerate(receivers):
        if not isinstance(receiver, dict):
            continue
        receiver_metadata.append({
            "device_id": receiver.get("device_id") or f"csi-{receiver_index + 1}",
            "port": receiver.get("device"),
            "baud": receiver.get("baud") or csi.get("baud"),
        })
        path = _path_for(minute_dir, receiver.get("path") or receiver.get("file"))
        if path is None:
            continue
        fragment_paths.add(path)
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw = str(row.get("raw_csi_line") or row.get("data") or "").strip()
                    if not raw:
                        continue
                    unix_ns = _iso_ns(row.get("host_timestamp"))
                    try:
                        mono_ns = int(row.get("monotonic_ns") or 0)
                    except (TypeError, ValueError):
                        mono_ns = 0
                    second = _second_index(
                        mono_ns, unix_ns, origin_monotonic_ns, origin_unix_ns,
                        0, second_count,
                    )
                    csi_payloads.append(raw.encode("utf-8"))
                    csi_unix.append(unix_ns)
                    csi_mono.append(mono_ns)
                    csi_seconds.append(second)
                    csi_receivers.append(receiver_index)
        except OSError:
            continue

    sense_payloads: list[bytes] = []
    sense_unix: list[int] = []
    sense_mono: list[int] = []
    sense_seconds: list[int] = []
    sense = outputs.get("sense_hat") if isinstance(outputs.get("sense_hat"), dict) else {}
    sense_paths = sense.get("files") if isinstance(sense.get("files"), list) else []
    if sense.get("path"):
        sense_paths = [*sense_paths, sense["path"]]
    for value in sense_paths:
        path = _path_for(minute_dir, value)
        if path is None:
            continue
        fragment_paths.add(path)
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                unix_ns = _iso_ns(row.get("host_timestamp"))
                mono_ns = int(row.get("monotonic_ns") or 0)
                sense_payloads.append(line.encode("utf-8"))
                sense_unix.append(unix_ns)
                sense_mono.append(mono_ns)
                sense_seconds.append(_second_index(
                    mono_ns, unix_ns, origin_monotonic_ns, origin_unix_ns,
                    0, second_count,
                ))
        except (OSError, TypeError, ValueError):
            continue

    camera_payload_data, camera_offsets = _pack_blobs(item[0] for item in camera_by_second)
    radar_payload_data, radar_offsets = _pack_blobs(radar_payloads)
    csi_payload_data, csi_offsets = _pack_blobs(csi_payloads)
    sense_payload_data, sense_offsets = _pack_blobs(sense_payloads)

    per_second = []
    for index in range(second_count):
        per_second.append({
            "second_index": index,
            "start_unix_ns": int(second_unix_ns[index]),
            "start_monotonic_ns": int(second_mono_ns[index]),
            "camera_frames": int(bool(camera_by_second[index][0])),
            "radar_samples": radar_seconds.count(index),
            "csi_samples": csi_seconds.count(index),
            "sense_hat_samples": sense_seconds.count(index),
        })
    metadata = {
        "schema": CONTAINER_SCHEMA,
        "collection_unit": "minute",
        "sample_unit": "synchronized-second",
        "timebase": {
            "unix": "UTC nanoseconds since Unix epoch",
            "monotonic": "device CLOCK_MONOTONIC nanoseconds",
            "capture_started_unix_ns": origin_unix_ns,
            "capture_started_monotonic_ns": origin_monotonic_ns,
        },
        "device": {
            "device_id": manifest.get("device_id"),
            "device_name": manifest.get("device_name"),
            "host": manifest.get("host"),
        },
        "labels": manifest.get("labels") or [],
        "capture_settings": manifest.get("capture_settings") or {},
        "sensor_configuration": {
            "enabled": manifest.get("sensors_enabled") or {},
            "csi_receivers": receiver_metadata,
            "radar": {key: radar.get(key) for key in ("config_dir", "type", "average_sampling_rate_hz") if key in radar},
            "camera": {key: camera.get(key) for key in ("device", "type") if key in camera},
        },
        "manifest": _manifest_without_host_paths(manifest),
        "seconds": per_second,
    }

    arrays: dict[str, np.ndarray] = {
        "metadata_json": _json_array(metadata),
        "second_start_unix_ns": second_unix_ns,
        "second_start_monotonic_ns": second_mono_ns,
        "camera_present": np.asarray([bool(item[0]) for item in camera_by_second], dtype=np.uint8),
        "camera_capture_unix_ns": np.asarray([item[1] for item in camera_by_second], dtype=np.int64),
        "camera_capture_monotonic_ns": np.asarray([item[2] for item in camera_by_second], dtype=np.int64),
        "camera_jpeg_bytes": camera_payload_data,
        "camera_jpeg_offsets": camera_offsets,
        "radar_sample_unix_ns": np.asarray(radar_unix, dtype=np.int64),
        "radar_sample_monotonic_ns": np.asarray(radar_mono, dtype=np.int64),
        "radar_sample_second_index": np.asarray(radar_seconds, dtype=np.int16),
        "radar_sample_sequence": np.asarray(radar_sequences, dtype=np.int64),
        "radar_sample_bytes": radar_payload_data,
        "radar_sample_offsets": radar_offsets,
        "csi_sample_unix_ns": np.asarray(csi_unix, dtype=np.int64),
        "csi_sample_monotonic_ns": np.asarray(csi_mono, dtype=np.int64),
        "csi_sample_second_index": np.asarray(csi_seconds, dtype=np.int16),
        "csi_sample_receiver_index": np.asarray(csi_receivers, dtype=np.int16),
        "csi_sample_bytes": csi_payload_data,
        "csi_sample_offsets": csi_offsets,
        "sense_sample_unix_ns": np.asarray(sense_unix, dtype=np.int64),
        "sense_sample_monotonic_ns": np.asarray(sense_mono, dtype=np.int64),
        "sense_sample_second_index": np.asarray(sense_seconds, dtype=np.int16),
        "sense_sample_bytes": sense_payload_data,
        "sense_sample_offsets": sense_offsets,
    }
    destination = minute_dir / CONTAINER_FILENAME
    temporary = minute_dir / f".{CONTAINER_FILENAME}.{os.getpid()}.tmp"
    try:
        with temporary.open("wb") as handle:
            # Payloads such as JPEG and radar frames are already compact; ZIP
            # storage avoids wasting CPU during the minute-boundary rollover.
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        with np.load(destination, allow_pickle=False) as archive:
            if str(read_metadata_from_archive(archive).get("schema")) != CONTAINER_SCHEMA:
                raise ValueError("capture container validation failed")
    finally:
        temporary.unlink(missing_ok=True)

    if remove_fragments:
        for fragment in fragment_paths:
            if fragment.parent == minute_dir and fragment != destination:
                fragment.unlink(missing_ok=True)

    return {
        "schema": CONTAINER_SCHEMA,
        "filename": destination.name,
        "bytes": destination.stat().st_size,
        "second_count": second_count,
        "camera_frames": int(sum(bool(item[0]) for item in camera_by_second)),
        "radar_samples": len(radar_payloads),
        "csi_samples": len(csi_payloads),
        "sense_hat_samples": len(sense_payloads),
    }


def read_metadata_from_archive(archive: Any) -> dict[str, Any]:
    raw = archive["metadata_json"].astype(np.uint8, copy=False).tobytes()
    value = json.loads(raw.decode("utf-8"))
    return value if isinstance(value, dict) else {}


def read_capture_metadata(path: Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return read_metadata_from_archive(archive)


def read_camera_frame(path: Path, second_index: int) -> bytes | None:
    with np.load(Path(path), allow_pickle=False) as archive:
        present = archive["camera_present"]
        if second_index < 0 or second_index >= len(present) or not bool(present[second_index]):
            return None
        return _blob_at(archive["camera_jpeg_bytes"], archive["camera_jpeg_offsets"], second_index)


def first_camera_frame(path: Path) -> bytes | None:
    with np.load(Path(path), allow_pickle=False) as archive:
        present = archive["camera_present"]
        indexes = np.flatnonzero(present)
        if not len(indexes):
            return None
        return _blob_at(archive["camera_jpeg_bytes"], archive["camera_jpeg_offsets"], int(indexes[0]))


def radar_bytes(path: Path, second_index: int | None = None) -> bytes:
    with np.load(Path(path), allow_pickle=False) as archive:
        seconds = archive["radar_sample_second_index"]
        payload = archive["radar_sample_bytes"]
        offsets = archive["radar_sample_offsets"]
        indexes = range(len(seconds)) if second_index is None else np.flatnonzero(seconds == int(second_index))
        return b"".join(_blob_at(payload, offsets, int(index)) for index in indexes)


def iter_csi_lines(path: Path, second_index: int | None = None) -> Iterator[str]:
    with np.load(Path(path), allow_pickle=False) as archive:
        seconds = archive["csi_sample_second_index"]
        payload = archive["csi_sample_bytes"]
        offsets = archive["csi_sample_offsets"]
        indexes = range(len(seconds)) if second_index is None else np.flatnonzero(seconds == int(second_index))
        for index in indexes:
            yield _blob_at(payload, offsets, int(index)).decode("utf-8", errors="replace")


def csi_average_series(path: Path, limit: int = 2400) -> list[float]:
    series: list[float] = []
    for line in iter_csi_lines(path):
        payloads = re.findall(r"\[([^\]]*)\]", line)
        payload = payloads[-1] if payloads else line
        values = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", payload)]
        magnitudes = [math.hypot(values[index + 1], values[index]) for index in range(0, len(values) - 1, 2)]
        if magnitudes:
            series.append(sum(magnitudes) / len(magnitudes))
    return series[-limit:]


def update_capture_metadata(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Atomically update JSON metadata while preserving every sensor array."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.metadata.tmp")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
            metadata = read_metadata_from_archive(archive)
        for key, value in updates.items():
            metadata[key] = value
        manifest = metadata.get("manifest") if isinstance(metadata.get("manifest"), dict) else {}
        if "labels" in updates:
            manifest["labels"] = updates["labels"]
            manifest["primary_label"] = updates["labels"][0] if updates["labels"] else None
        if "capture_settings" in updates:
            manifest["capture_settings"] = updates["capture_settings"]
        metadata["manifest"] = manifest
        arrays["metadata_json"] = _json_array(metadata)
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)
