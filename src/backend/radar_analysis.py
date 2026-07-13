"""Radar chunk analysis helpers for XY localization payloads."""

from __future__ import annotations

import csv
import json
import math
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
TRACK_EXAMPLE_DIR = MMW_RELEASE / "example_2_advanced"
ROOM_CONFIG = TRACK_EXAMPLE_DIR / "config" / "room_config.json"
PROCESSING_CONFIG = TRACK_EXAMPLE_DIR / "config" / "processing_config_advanced.json"
RADAR_CONFIG_DIR = MMW_RELEASE / "radar_config" / "config_3rx_3m"

for path in (MMW_RELEASE, TRACK_EXAMPLE_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:  # pragma: no cover - import availability depends on device packages
    from signal_proc import SigProc
    from utility.helper import calculate_frame_size, find_register_config_in_directory, find_setting_in_directory, parse_full_frame, parse_radar_cfg, read_uint12, split_samples
except Exception:  # pragma: no cover
    SigProc = None  # type: ignore[assignment]
    calculate_frame_size = None  # type: ignore[assignment]
    find_register_config_in_directory = None  # type: ignore[assignment]
    find_setting_in_directory = None  # type: ignore[assignment]
    parse_full_frame = None  # type: ignore[assignment]
    parse_radar_cfg = None  # type: ignore[assignment]
    read_uint12 = None  # type: ignore[assignment]
    split_samples = None  # type: ignore[assignment]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_room_config() -> Dict[str, Any]:
    room = load_json(ROOM_CONFIG)
    if room:
        return room
    return {
        "width_m": 3.73,
        "depth_m": 5.0,
        "height_m": 2.5,
        "sensor_wall": "Back",
        "sensor_position_m": 2.02,
        "sensor_height_m": 1.0,
        "floor_anchored_targets": False,
        "max_object_height_m": 2.2,
        "max_object_width_m": 0.95,
        "max_object_depth_m": 0.8,
        "max_lying_length_m": 2.2,
    }


def load_processing_config() -> Dict[str, Any]:
    return load_json(PROCESSING_CONFIG)


def load_radar_config() -> Dict[str, Any]:
    if find_setting_in_directory is None:
        return {}
    try:
        setting_file = find_setting_in_directory(str(RADAR_CONFIG_DIR))
        setting = load_json(Path(setting_file))
        if parse_radar_cfg is None:
            return setting
        return parse_radar_cfg(setting)
    except Exception:
        return {}


def iter_radar_frames(bin_path: Path) -> Iterable[tuple[int, np.ndarray]]:
    if parse_full_frame is None or read_uint12 is None or split_samples is None:
        return []
    try:
        raw = bin_path.read_bytes()
    except Exception:
        return []

    offset = 0
    total = len(raw)
    radar_cfg = load_radar_config()
    num_chirps = int(radar_cfg.get("num_chirps_per_frame") or 0)
    num_samples = int(radar_cfg.get("num_samples_per_chirp") or 0)
    num_antennas = int(radar_cfg.get("num_antennas") or 0)
    if not (num_chirps and num_samples and num_antennas):
        return []

    while offset < total:
        if offset + 12 > total:
            break
        _version, _seq, payload_length = struct.unpack_from("<III", raw, offset)
        frame_end = offset + 12 + payload_length
        if payload_length <= 0 or frame_end > total:
            break
        frame = parse_full_frame(raw[offset:frame_end])
        if not frame:
            break
        _version, seq, data_len, frame_bytes = frame
        offset = frame_end
        try:
            adc_data = read_uint12(frame_bytes)
            split = split_samples(adc_data, 1, num_chirps, num_samples, num_antennas)
            yield seq, np.transpose(split[0, :, :, :], (2, 0, 1))
        except Exception:
            continue


def decode_radar_frame(full_frame: bytes, radar_cfg: Optional[Dict[str, Any]] = None) -> Optional[tuple[int, np.ndarray]]:
    """Decode one frame obtained directly from BGT60TR13C.frame_buffer."""
    if parse_full_frame is None or read_uint12 is None or split_samples is None:
        return None
    config = radar_cfg or load_radar_config()
    try:
        parsed = parse_full_frame(full_frame)
        if not parsed:
            return None
        _version, seq, _data_len, frame_bytes = parsed
        adc_data = read_uint12(frame_bytes)
        split = split_samples(
            adc_data,
            1,
            int(config["num_chirps_per_frame"]),
            int(config["num_samples_per_chirp"]),
            int(config["num_antennas"]),
        )
        return int(seq), np.transpose(split[0, :, :, :], (2, 0, 1))
    except Exception:
        return None


def create_signal_processor() -> Any:
    if SigProc is None:
        raise RuntimeError("Radar tracking dependencies are unavailable.")
    radar_config = load_radar_config()
    if not radar_config:
        raise RuntimeError("Radar configuration could not be loaded.")
    return SigProc(str(PROCESSING_CONFIG), radar_config)


def occupancy_label(detected_frames: int, evaluated_frames: int, threshold_percent: float) -> str:
    threshold = min(100.0, max(0.0, float(threshold_percent)))
    return "occupied" if evaluated_frames > 0 and detected_frames * 100.0 >= threshold * evaluated_frames else "empty"


class StreamingChunkAnalyzer:
    """Incrementally analyze live frames while retaining one tracker across chunks."""

    FIELDNAMES = [
        "chunk_index", "frame_index", "seq", "detected", "target_count", "occupied",
        "primary_target_id", "x_m", "y_m", "z_m", "width_m", "depth_m", "height_m",
        "pose", "snr_db", "score", "noise_floor_db", "threshold_db", "peak_power_db",
        "motion_points", "targets_json", "shadow_points_json",
    ]

    def __init__(
        self,
        processor: Any,
        csv_path: Path,
        chunk_index: int,
        chunk_seconds: float,
        room: Dict[str, Any],
        radar_detection_threshold_db: float,
        occupancy_threshold_percent: float,
    ):
        self.processor = processor
        self.processor.threshold_db = min(40.0, max(0.0, float(radar_detection_threshold_db)))
        self.radar_config = getattr(processor, "radar_config", None) or load_radar_config()
        self.csv_path = csv_path
        self.csv_temporary = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
        self.chunk_index = chunk_index
        self.chunk_seconds = chunk_seconds
        self.room = room
        self.occupancy_threshold_percent = min(100.0, max(0.0, float(occupancy_threshold_percent)))
        self.evaluated_frames = 0
        self.detected_frames = 0
        self.chunk_points: list[np.ndarray] = []
        self.chunk_weights: list[np.ndarray] = []
        self.last_targets: list[Dict[str, Any]] = []
        self.last_detection: Dict[str, Any] = {}
        self.last_position = [0.0, 0.0]
        self.last_score = 0.0
        self.processing_seconds = 0.0
        self.max_processing_seconds = 0.0
        self.max_queue_lag_ms = 0.0
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.csv_temporary, "w", encoding="utf-8", newline="", buffering=64 * 1024)
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()

    def process(self, full_frame: bytes) -> None:
        processing_started = time.perf_counter()
        decoded = decode_radar_frame(full_frame, self.radar_config)
        if decoded is None:
            return
        seq, frame = decoded
        try:
            targets = self.processor.update(frame)
        except Exception:
            return
        detection = dict(self.processor.last_detection or {})
        shadow = dict(self.processor.last_motion_shadow or {})
        shadow_points = np.asarray(shadow.get("points") if shadow.get("points") is not None else np.empty((0, 3)), dtype=float)
        shadow_intensity = np.asarray(shadow.get("intensity") if shadow.get("intensity") is not None else np.empty(0), dtype=float)
        frame_index = self.evaluated_frames
        self.evaluated_frames += 1
        if detection.get("detected"):
            self.detected_frames += 1
        self.last_targets = [_serialize_target(target, self.room) for target in targets]
        self.last_detection = detection
        if self.last_targets:
            lead = self.last_targets[0]
            self.last_position = [float(lead["position"][0]), float(lead["position"][1])]
            self.last_score = float(lead.get("snr_db") or detection.get("cfar_peak_db") or 0.0)

        world_points = np.asarray([
            _clip_point_to_room(world_from_local(point, self.room), self.room) for point in shadow_points
        ], dtype=float) if shadow_points.size else np.empty((0, 3), dtype=float)
        if world_points.size:
            self.chunk_points.append(world_points)
            weights = shadow_intensity if shadow_intensity.size == len(world_points) else np.ones(len(world_points))
            self.chunk_weights.append(np.clip(weights, 0.02, 1.0))
            if not self.last_targets:
                self.last_position = [float(world_points[0][0]), float(world_points[0][1])]
                self.last_score = float(np.max(weights) if weights.size else 0.0)

        primary = self.last_targets[0] if self.last_targets else None
        self.writer.writerow({
            "chunk_index": self.chunk_index,
            "frame_index": frame_index,
            "seq": seq,
            "detected": bool(detection.get("detected")),
            "target_count": len(targets),
            "occupied": bool(targets),
            "primary_target_id": primary["id"] if primary else "",
            "x_m": primary["position"][0] if primary else "",
            "y_m": primary["position"][1] if primary else "",
            "z_m": primary["position"][2] if primary else "",
            "width_m": primary["size"][0] if primary else "",
            "depth_m": primary["size"][1] if primary else "",
            "height_m": primary["size"][2] if primary else "",
            "pose": primary["pose"] if primary else "",
            "snr_db": primary["snr_db"] if primary else "",
            "score": self.last_score,
            "noise_floor_db": detection.get("noise_floor_db", ""),
            "threshold_db": detection.get("threshold_db", self.processor.threshold_db),
            "peak_power_db": detection.get("cfar_peak_db", ""),
            "motion_points": int(detection.get("motion_points") or 0),
            "targets_json": json.dumps(self.last_targets, separators=(",", ":")),
            "shadow_points_json": json.dumps(shadow_points.round(4).tolist(), separators=(",", ":")),
        })
        elapsed = time.perf_counter() - processing_started
        self.processing_seconds += elapsed
        self.max_processing_seconds = max(self.max_processing_seconds, elapsed)

    def finish(self) -> Dict[str, Any]:
        finalization_started = time.perf_counter()
        self.handle.flush()
        self.handle.close()
        self.csv_temporary.replace(self.csv_path)
        points = np.vstack(self.chunk_points) if self.chunk_points else np.empty((0, 3), dtype=float)
        weights = np.concatenate(self.chunk_weights) if self.chunk_weights else np.empty(0, dtype=float)
        x_axis, y_axis, z = _bin_points(points, self.room, weights=weights, resolution=72)
        ratio = self.detected_frames / self.evaluated_frames if self.evaluated_frames else 0.0
        label = occupancy_label(self.detected_frames, self.evaluated_frames, self.occupancy_threshold_percent)
        primary = self.last_targets[0] if self.last_targets else None
        frame = {
            "name": f"chunk-{self.chunk_index:02d}", "index": self.chunk_index,
            "x": x_axis, "y": y_axis, "z": z, "location": self.last_position,
            "score": self.last_score, "detected": label == "occupied",
            "snr_db": float(primary.get("snr_db")) if primary else float(self.last_detection.get("cfar_peak_db") or 0.0),
            "threshold_db": float(self.last_detection.get("threshold_db") or self.processor.threshold_db),
            "peak_power_db": self.last_detection.get("cfar_peak_db"),
            "noise_floor_db": self.last_detection.get("noise_floor_db"),
            "targets": self.last_targets, "motion_points": int(self.last_detection.get("motion_points") or 0),
        }
        payload = {
            "plot": "xy-tracking", "title": "X-Y localization", "x_label": "X (m)", "y_label": "Y (m)",
            "x": x_axis, "y": y_axis, "z": z, "frames": [frame], "frame_count": self.evaluated_frames,
            "sample_count": 1, "frame_interval_ms": max(50, int(round(self.chunk_seconds * 1000))),
            "updated": datetime.now(timezone.utc).isoformat(),
            "occupancy": {"label": label, "detected_frames": self.detected_frames, "evaluated_frames": self.evaluated_frames,
                          "ratio": ratio, "threshold_percent": self.occupancy_threshold_percent, "chunk_seconds": self.chunk_seconds},
            "location": self.last_position, "score": self.last_score, "detected": label == "occupied",
            "snr_db": frame["snr_db"], "threshold_db": frame["threshold_db"], "peak_power_db": frame["peak_power_db"],
            "noise_floor_db": frame["noise_floor_db"], "targets": self.last_targets,
            "motion_points": frame["motion_points"], "chunk_index": self.chunk_index,
            "chunk_seconds": self.chunk_seconds, "room": self.room,
        }
        payload["performance"] = {
            "average_processing_ms": round(self.processing_seconds * 1000 / max(1, self.evaluated_frames), 3),
            "max_processing_ms": round(self.max_processing_seconds * 1000, 3),
            "max_queue_lag_ms": round(self.max_queue_lag_ms, 3),
            "finalization_ms": round((time.perf_counter() - finalization_started) * 1000, 3),
        }
        return payload


def _sensor_pose(room: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = float(room.get("width_m", 3.73))
    depth = float(room.get("depth_m", 5.0))
    position = float(room.get("sensor_position_m", width / 2.0))
    height = float(room.get("sensor_height_m", 1.0))
    wall = str(room.get("sensor_wall", "Back"))
    if wall == "Back":
        return np.array([position, 0.0, height]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if wall == "Front":
        return np.array([position, depth, height]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
    if wall == "Left":
        return np.array([0.0, position, height]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    return np.array([width, position, height]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])


def world_from_local(point: Iterable[float], room: Dict[str, Any]) -> np.ndarray:
    origin, lateral_axis, forward_axis = _sensor_pose(room)
    local = np.asarray(list(point), dtype=float)
    if local.size < 3:
        local = np.pad(local, (0, 3 - local.size))
    return origin + lateral_axis * local[0] + forward_axis * local[1] + np.array([0.0, 0.0, local[2]])


def _clip_point_to_room(point: np.ndarray, room: Dict[str, Any]) -> np.ndarray:
    clipped = point.copy()
    clipped[0] = float(np.clip(clipped[0], 0.0, float(room.get("width_m", 1.0))))
    clipped[1] = float(np.clip(clipped[1], 0.0, float(room.get("depth_m", 1.0))))
    clipped[2] = float(np.clip(clipped[2], 0.0, float(room.get("height_m", 1.0))))
    return clipped


def _empty_xy_grid(room: Dict[str, Any], resolution: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = float(room.get("width_m", 3.73))
    depth = float(room.get("depth_m", 5.0))
    cols = max(24, resolution)
    rows = max(24, int(round(resolution * depth / max(width, 1e-6))))
    x_axis = np.linspace(0.0, width, cols).tolist()
    y_axis = np.linspace(0.0, depth, rows).tolist()
    return np.asarray(x_axis, dtype=float), np.asarray(y_axis, dtype=float), np.zeros((rows, cols), dtype=float)


def _bin_points(points: np.ndarray, room: Dict[str, Any], weights: Optional[np.ndarray] = None, resolution: int = 64) -> tuple[list[float], list[float], list[list[float]]]:
    x_axis, y_axis, grid = _empty_xy_grid(room, resolution=resolution)
    if points.size:
        clipped = np.asarray([_clip_point_to_room(point, room) for point in points], dtype=float)
        xs = clipped[:, 0]
        ys = clipped[:, 1]
        if weights is None:
            weights = np.ones(len(points), dtype=float)
        hist, y_edges, x_edges = np.histogram2d(
            ys,
            xs,
            bins=[len(y_axis), len(x_axis)],
            range=[[0.0, float(room.get("depth_m", 5.0))], [0.0, float(room.get("width_m", 3.73))]],
            weights=weights,
        )
        grid += hist
    return x_axis.tolist(), y_axis.tolist(), grid.tolist()


def _serialize_target(target: Dict[str, Any], room: Dict[str, Any]) -> Dict[str, Any]:
    local = np.array([
        float(target.get("lateral_m", 0.0)),
        float(target.get("forward_m", 0.0)),
        float(target.get("vertical_m", 0.0)),
    ])
    world = _clip_point_to_room(world_from_local(local, room), room)
    size = np.array([
        float(target.get("width_m", 0.2)),
        float(target.get("depth_m", 0.2)),
        float(target.get("height_m", 0.4)),
    ])
    world_size = np.array([
        size[1] if str(room.get("sensor_wall", "Back")) in {"Left", "Right"} else size[0],
        size[0] if str(room.get("sensor_wall", "Back")) in {"Left", "Right"} else size[1],
        size[2],
    ])
    return {
        "id": int(target.get("id", 0)),
        "pose": target.get("pose"),
        "presence_mode": target.get("presence_mode"),
        "range_m": round(float(target.get("range_m", 0.0)), 3),
        "position": [round(float(value), 3) for value in world],
        "size": [round(float(value), 3) for value in world_size],
        "snr_db": round(float(target.get("snr_db", 0.0)), 2),
        "confidence": round(float(target.get("confidence", target.get("snr_db", 0.0))), 2),
    }


def analyze_radar_chunk(
    bin_path: Path,
    csv_path: Path,
    chunk_index: int,
    chunk_seconds: float,
    room: Optional[Dict[str, Any]] = None,
    *,
    frames: Optional[Iterable[bytes]] = None,
    processor: Any = None,
    radar_detection_threshold_db: float = 8.0,
    occupancy_threshold_percent: float = 50.0,
) -> Dict[str, Any]:
    if SigProc is None:
        raise RuntimeError("Radar tracking dependencies are unavailable.")

    room = room or load_room_config()
    radar_config = load_radar_config()
    if not radar_config:
        raise RuntimeError("Radar configuration could not be loaded.")

    proc = processor or SigProc(str(PROCESSING_CONFIG), radar_config)
    proc.threshold_db = min(40.0, max(0.0, float(radar_detection_threshold_db)))
    frame_period_s = float(getattr(proc, "frame_period_s", 0.0) or 0.0)

    fieldnames = [
        "chunk_index",
        "frame_index",
        "seq",
        "detected",
        "target_count",
        "occupied",
        "primary_target_id",
        "x_m",
        "y_m",
        "z_m",
        "width_m",
        "depth_m",
        "height_m",
        "pose",
        "snr_db",
        "score",
        "noise_floor_db",
        "threshold_db",
        "peak_power_db",
        "motion_points",
        "targets_json",
        "shadow_points_json",
    ]

    evaluated_frames = 0
    detected_frames = 0
    chunk_points: list[np.ndarray] = []
    chunk_weights: list[np.ndarray] = []
    frame_rows: list[Dict[str, Any]] = []
    frame_payloads: list[Dict[str, Any]] = []
    last_targets: list[Dict[str, Any]] = []
    last_detection: Dict[str, Any] = {}
    last_position = [0.0, 0.0]
    last_score = 0.0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    with open(csv_temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        source_frames: Iterable[tuple[int, np.ndarray]]
        if frames is None:
            source_frames = iter_radar_frames(bin_path)
        else:
            source_frames = (
                decoded for decoded in (decode_radar_frame(raw, radar_config) for raw in frames) if decoded is not None
            )
        for frame_index, (seq, frame) in enumerate(source_frames):
            try:
                targets = proc.update(frame)
            except Exception:
                continue

            detection = dict(proc.last_detection or {})
            shadow = dict(proc.last_motion_shadow or {})
            shadow_points_value = shadow.get("points")
            shadow_intensity_value = shadow.get("intensity")
            shadow_points = np.asarray(
                shadow_points_value if shadow_points_value is not None else np.empty((0, 3), dtype=float),
                dtype=float,
            )
            shadow_intensity = np.asarray(
                shadow_intensity_value if shadow_intensity_value is not None else np.empty(0, dtype=float),
                dtype=float,
            )
            evaluated_frames += 1
            if detection.get("detected"):
                detected_frames += 1
            last_targets = [_serialize_target(target, room) for target in targets]
            last_detection = detection
            if last_targets:
                lead = last_targets[0]
                last_position = [float(lead["position"][0]), float(lead["position"][1])]
                last_score = float(lead.get("snr_db") or detection.get("cfar_peak_db") or 0.0)
            elif shadow_points.size:
                world_points = np.asarray([_clip_point_to_room(world_from_local(point, room), room) for point in shadow_points], dtype=float)
                chunk_points.append(world_points)
                chunk_weights.append(np.clip(shadow_intensity, 0.02, 1.0))
                last_position = [float(world_points[0][0]), float(world_points[0][1])]
                last_score = float(np.max(shadow_intensity) if shadow_intensity.size else 0.0)

            row_targets = last_targets or []
            row = {
                "chunk_index": chunk_index,
                "frame_index": frame_index,
                "seq": seq,
                "detected": bool(detection.get("detected")),
                "target_count": len(targets),
                "occupied": bool(targets),
                "primary_target_id": row_targets[0]["id"] if row_targets else "",
                "x_m": row_targets[0]["position"][0] if row_targets else "",
                "y_m": row_targets[0]["position"][1] if row_targets else "",
                "z_m": row_targets[0]["position"][2] if row_targets else "",
                "width_m": row_targets[0]["size"][0] if row_targets else "",
                "depth_m": row_targets[0]["size"][1] if row_targets else "",
                "height_m": row_targets[0]["size"][2] if row_targets else "",
                "pose": row_targets[0]["pose"] if row_targets else "",
                "snr_db": row_targets[0]["snr_db"] if row_targets else "",
                "score": float(last_score),
                "noise_floor_db": detection.get("noise_floor_db", ""),
                "threshold_db": detection.get("threshold_db", proc.threshold_db),
                "peak_power_db": detection.get("cfar_peak_db", ""),
                "motion_points": int(detection.get("motion_points") or 0),
                "targets_json": json.dumps(last_targets, separators=(",", ":")) if last_targets else "[]",
                "shadow_points_json": json.dumps(
                    [
                        [round(float(point[0]), 4), round(float(point[1]), 4), round(float(point[2]), 4)]
                        for point in shadow_points
                    ],
                    separators=(",", ":"),
                ),
            }
            writer.writerow(row)
            frame_rows.append(row)

            world_points = np.asarray([
                _clip_point_to_room(world_from_local(point, room), room) for point in shadow_points
            ], dtype=float) if shadow_points.size else np.empty((0, 3), dtype=float)
            if world_points.size:
                chunk_points.append(world_points)
                chunk_weights.append(np.clip(shadow_intensity if shadow_intensity.size else np.ones(len(world_points)), 0.02, 1.0))

            frame_payloads.append({
                "frame_index": frame_index,
                "seq": seq,
                "detected": bool(detection.get("detected")),
                "targets": last_targets,
                "location": last_position,
                "score": float(last_score),
                "snr_db": float(row_targets[0]["snr_db"]) if row_targets else float(detection.get("cfar_peak_db") or 0.0),
                "threshold_db": float(detection.get("threshold_db") or proc.threshold_db),
                "noise_floor_db": detection.get("noise_floor_db"),
                "peak_power_db": detection.get("cfar_peak_db"),
                "motion_points": int(detection.get("motion_points") or 0),
            })
    csv_temporary.replace(csv_path)

    if chunk_points:
        points = np.vstack(chunk_points)
        weights = np.concatenate(chunk_weights) if chunk_weights else np.ones(len(points), dtype=float)
    else:
        points = np.empty((0, 3), dtype=float)
        weights = np.empty(0, dtype=float)

    x_axis, y_axis, z = _bin_points(points, room, weights=weights, resolution=72)
    occupancy_ratio = detected_frames / evaluated_frames if evaluated_frames else 0.0
    occupancy_threshold_percent = min(100.0, max(0.0, float(occupancy_threshold_percent)))
    occupied = occupancy_label(detected_frames, evaluated_frames, occupancy_threshold_percent) == "occupied"
    primary = last_targets[0] if last_targets else None

    payload = {
        "plot": "xy-tracking",
        "title": "X-Y localization",
        "x_label": "X (m)",
        "y_label": "Y (m)",
        "x": x_axis,
        "y": y_axis,
        "z": z,
        "frames": [{
            "name": f"chunk-{chunk_index:02d}",
            "index": chunk_index,
            "x": x_axis,
            "y": y_axis,
            "z": z,
            "location": last_position,
            "score": float(last_score),
            "detected": occupied,
            "snr_db": float(primary.get("snr_db")) if primary else float(last_detection.get("cfar_peak_db") or 0.0),
            "threshold_db": float(last_detection.get("threshold_db") or proc.threshold_db),
            "peak_power_db": last_detection.get("cfar_peak_db"),
            "noise_floor_db": last_detection.get("noise_floor_db"),
            "targets": last_targets,
            "motion_points": int(last_detection.get("motion_points") or 0),
        }],
        "frame_count": max(1, evaluated_frames),
        "sample_count": 1,
        "frame_interval_ms": max(50, int(round(max(chunk_seconds, frame_period_s or 0.12) * 1000))),
        "updated": datetime.now(timezone.utc).isoformat(),
        "occupancy": {
            "label": "occupied" if occupied else "empty",
            "detected_frames": detected_frames,
            "evaluated_frames": evaluated_frames,
            "ratio": occupancy_ratio,
            "threshold_percent": occupancy_threshold_percent,
            "chunk_seconds": float(chunk_seconds),
        },
        "location": last_position,
        "score": float(last_score),
        "detected": occupied,
        "snr_db": float(primary.get("snr_db")) if primary else float(last_detection.get("cfar_peak_db") or 0.0),
        "threshold_db": float(last_detection.get("threshold_db") or proc.threshold_db),
        "peak_power_db": last_detection.get("cfar_peak_db"),
        "noise_floor_db": last_detection.get("noise_floor_db"),
        "targets": last_targets,
        "motion_points": int(last_detection.get("motion_points") or 0),
        "chunk_index": chunk_index,
        "chunk_seconds": float(chunk_seconds),
        "frame_rows": frame_rows,
        "frame_payloads": frame_payloads,
        "room": room,
    }
    return payload


def compile_minute_xy_payload(chunk_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    room = load_room_config()
    latest = None
    total_detected = 0
    total_evaluated = 0

    for chunk in chunk_payloads:
        if not isinstance(chunk, dict):
            continue
        frames.extend(chunk.get("frames") or [])
        latest = chunk
        occupancy = chunk.get("occupancy") or {}
        total_detected += int(occupancy.get("detected_frames") or 0)
        total_evaluated += int(occupancy.get("evaluated_frames") or 0)

    if latest is None:
        latest = {
            "x": [],
            "y": [],
            "z": [],
            "location": [0.0, 0.0],
            "score": 0.0,
            "detected": False,
            "snr_db": 0.0,
            "threshold_db": 0.0,
            "peak_power_db": 0.0,
            "noise_floor_db": 0.0,
            "targets": [],
            "motion_points": 0,
        }

    occupancy_ratio = total_detected / total_evaluated if total_evaluated else 0.0
    threshold_percent = float((latest.get("occupancy") or {}).get("threshold_percent", 50.0))
    x_axis = latest.get("x") or []
    y_axis = latest.get("y") or []
    z = latest.get("z") or []

    return {
        "plot": "xy-tracking",
        "title": "X-Y localization",
        "x_label": "X (m)",
        "y_label": "Y (m)",
        "x": x_axis,
        "y": y_axis,
        "z": z,
        "frames": frames or [{
            "name": "chunk-00",
            "index": 0,
            "x": x_axis,
            "y": y_axis,
            "z": z,
            "location": latest.get("location") or [0.0, 0.0],
            "score": float(latest.get("score") or 0.0),
            "detected": bool(latest.get("detected")),
            "snr_db": float(latest.get("snr_db") or 0.0),
            "threshold_db": float(latest.get("threshold_db") or 0.0),
            "peak_power_db": latest.get("peak_power_db"),
            "noise_floor_db": latest.get("noise_floor_db"),
            "targets": latest.get("targets") or [],
            "motion_points": int(latest.get("motion_points") or 0),
        }],
        "frame_count": len(frames) if frames else 1,
        "sample_count": len(chunk_payloads),
        "frame_interval_ms": max(50, int(round(sum(float(chunk.get("chunk_seconds") or 10.0) for chunk in chunk_payloads) * 1000 / max(1, len(frames) or len(chunk_payloads) or 1)))),
        "updated": datetime.now(timezone.utc).isoformat(),
        "occupancy": {
            "label": occupancy_label(total_detected, total_evaluated, threshold_percent),
            "detected_frames": total_detected,
            "evaluated_frames": total_evaluated,
            "ratio": occupancy_ratio,
            "threshold_percent": threshold_percent,
            "chunk_seconds": chunk_payloads[0].get("chunk_seconds") if chunk_payloads else 10.0,
        },
        "location": latest.get("location") or [0.0, 0.0],
        "score": float(latest.get("score") or 0.0),
        "detected": bool(latest.get("detected")),
        "snr_db": float(latest.get("snr_db") or 0.0),
        "threshold_db": float(latest.get("threshold_db") or 0.0),
        "peak_power_db": latest.get("peak_power_db"),
        "noise_floor_db": latest.get("noise_floor_db"),
        "targets": latest.get("targets") or [],
        "motion_points": int(latest.get("motion_points") or 0),
        "chunk_count": len(chunk_payloads),
        "room": room,
    }
