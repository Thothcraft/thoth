"""Radar chunk analysis helpers for XY localization payloads."""

from __future__ import annotations

import csv
import json
import math
import os
import struct
import threading
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
TARGET_IDENTITY_PATH = THOTH_ROOT / "config" / "radar_target_identity.json"
LIVE_OCCUPANCY_PATH = THOTH_ROOT / "config" / "radar_occupancy.json"

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
    if not room:
        room = {
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
    if not isinstance(room.get("radar_cones"), list) or not room.get("radar_cones"):
        room["radar_cones"] = [{
            "id": "radar-1", "name": "Radar 1", "enabled": True,
            "wall": room.get("sensor_wall", "Back"),
            "position_m": room.get("sensor_position_m", 2.0),
            "height_m": room.get("sensor_height_m", 1.0),
            "horizontal_deg": 40.0, "vertical_deg": 65.0,
            "range_m": 15.0, "azimuth_deg": 0.0,
        }]
    for key in ("doors", "windows", "furniture", "zones"):
        if not isinstance(room.get(key), list):
            room[key] = []
    room.pop("sleep_anchor", None)
    return room


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


def validate_region_thresholds(yellow_threshold_percent: float, green_threshold_percent: float) -> tuple[float, float]:
    """Return validated red/yellow/green boundaries as percentages."""
    yellow = float(yellow_threshold_percent)
    green = float(green_threshold_percent)
    if not 0.0 <= yellow < green <= 100.0:
        raise ValueError("thresholds must satisfy 0 <= yellow < green <= 100")
    return yellow, green


def occupancy_region(
    detected_frames: int,
    evaluated_frames: int,
    yellow_threshold_percent: float,
    green_threshold_percent: float,
) -> str:
    """Classify a completed chunk from its evaluated-frame detection ratio."""
    yellow, green = validate_region_thresholds(yellow_threshold_percent, green_threshold_percent)
    if evaluated_frames <= 0:
        return "red"
    ratio_percent = max(0, detected_frames) * 100.0 / evaluated_frames
    if ratio_percent >= green:
        return "green"
    if ratio_percent >= yellow:
        return "yellow"
    return "red"


def occupancy_label(detected_frames: int, evaluated_frames: int, threshold_percent: float) -> str:
    threshold = min(100.0, max(0.0, float(threshold_percent)))
    return "occupied" if evaluated_frames > 0 and detected_frames * 100.0 >= threshold * evaluated_frames else "empty"


class PersistentTargetIdentity:
    """Associate processor-local tracks with stable IDs across capture minutes."""

    def __init__(self, path: Path = TARGET_IDENTITY_PATH, mode: str = "balanced"):
        self.path = path
        self.mode = mode if mode in {"responsive", "balanced", "precision"} else "balanced"
        self.match_distance_m = {"responsive": 1.5, "balanced": 1.0, "precision": 0.65}[self.mode]
        self.local_to_global: Dict[int, int] = {}
        state = load_json(path)
        self.next_id = max(1, int(state.get("next_id") or 1))
        self.tracks: Dict[int, Dict[str, Any]] = {
            int(key): value for key, value in (state.get("tracks") or {}).items() if isinstance(value, dict)
        }

    def assign(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now = time.time()
        self.tracks = {
            key: value for key, value in self.tracks.items()
            if now - float(value.get("last_seen") or 0) <= 300.0
        }
        used: set[int] = set()
        assigned: List[Dict[str, Any]] = []
        for source in targets:
            target = dict(source)
            local_id = int(target.get("id") or 0)
            position = np.asarray(target.get("position") or [0.0, 0.0, 0.0], dtype=float)
            global_id = self.local_to_global.get(local_id)
            if global_id not in self.tracks or global_id in used:
                global_id = None
            if global_id is None:
                candidates = []
                for candidate_id, previous in self.tracks.items():
                    if candidate_id in used:
                        continue
                    previous_position = np.asarray(previous.get("position") or [0.0, 0.0, 0.0], dtype=float)
                    distance = float(np.linalg.norm(position[:2] - previous_position[:2]))
                    if distance <= self.match_distance_m:
                        candidates.append((distance, candidate_id))
                if candidates:
                    global_id = min(candidates)[1]
                else:
                    global_id = self.next_id
                    self.next_id += 1
                self.local_to_global[local_id] = global_id
            previous = self.tracks.get(global_id) or {}
            previous_position = np.asarray(previous.get("position") or position, dtype=float)
            residual = float(np.linalg.norm(position[:2] - previous_position[:2]))
            base_error = {"responsive": 0.28, "balanced": 0.18, "precision": 0.10}[self.mode]
            error_m = round(min(1.5, max(0.05, base_error + residual * 0.35)), 3)
            target.update({
                "source_id": local_id,
                "id": global_id,
                "position_error_m": error_m,
                "position_bounds": {
                    "x": [round(float(position[0] - error_m), 3), round(float(position[0] + error_m), 3)],
                    "y": [round(float(position[1] - error_m), 3), round(float(position[1] + error_m), 3)],
                },
            })
            self.tracks[global_id] = {
                "position": [round(float(value), 3) for value in position],
                "position_error_m": error_m,
                "last_seen": now,
            }
            used.add(global_id)
            assigned.append(target)
        return assigned

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            temporary.write_text(json.dumps({
                "next_id": self.next_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "tracks": {str(key): value for key, value in self.tracks.items()},
            }, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


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
        yellow_threshold_percent: float = 20.0,
        green_threshold_percent: float = 60.0,
        identity: Optional[PersistentTargetIdentity] = None,
        live_state_path: Optional[Path] = LIVE_OCCUPANCY_PATH,
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
        self.yellow_threshold_percent, self.green_threshold_percent = validate_region_thresholds(
            yellow_threshold_percent, green_threshold_percent
        )
        self.identity = identity
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
        self.playback_frames: List[Dict[str, Any]] = []
        self.live_state_path = live_state_path
        self.last_live_publish = 0.0
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
        serialized_targets = [_serialize_target(target, self.room) for target in targets]
        self.last_targets = self.identity.assign(serialized_targets) if self.identity else serialized_targets
        # Live visualization and chunk labeling share this exact signal.
        person_detected = bool(self.last_targets)
        detection["signal_detected"] = bool(detection.get("detected"))
        detection["detected"] = person_detected
        if person_detected:
            self.detected_frames += 1
        self.last_detection = detection
        if self.last_targets:
            lead = self.last_targets[0]
            self.last_position = [float(lead["position"][0]), float(lead["position"][1])]
            self.last_score = float(lead.get("snr_db") or detection.get("cfar_peak_db") or 0.0)

        world_points = _world_points_from_local(shadow_points, self.room)
        if world_points.size:
            self.chunk_points.append(world_points)
            weights = shadow_intensity if shadow_intensity.size == len(world_points) else np.ones(len(world_points))
            self.chunk_weights.append(np.clip(weights, 0.02, 1.0))
            if not self.last_targets:
                self.last_position = [float(world_points[0][0]), float(world_points[0][1])]
                self.last_score = float(np.max(weights) if weights.size else 0.0)

        primary = self.last_targets[0] if self.last_targets else None
        self.playback_frames.append({
            "name": f"chunk-{self.chunk_index:02d}-frame-{frame_index:04d}",
            "index": frame_index,
            "chunk_index": self.chunk_index,
            "seq": seq,
            "location": list(self.last_position),
            "score": self.last_score,
            "detected": bool(detection.get("detected")),
            "snr_db": float(primary.get("snr_db")) if primary else float(detection.get("cfar_peak_db") or 0.0),
            "threshold_db": float(detection.get("threshold_db") or self.processor.threshold_db),
            "targets": self.last_targets,
        })
        self._write_live_state(world_points)
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

    def _write_live_state(self, world_points: np.ndarray) -> None:
        """Publish the analyzed frame without blocking capture or the dashboard."""
        if self.live_state_path is None:
            return
        now = time.monotonic()
        if now - self.last_live_publish < 0.2:
            return
        self.last_live_publish = now
        temporary = self.live_state_path.with_name(
            f".{self.live_state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        cone = next((
            item for item in (self.room.get("radar_cones") or [])
            if isinstance(item, dict) and item.get("enabled") is not False
        ), {})
        try:
            self.live_state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps({
                "updated_at": time.time(),
                "occupied": occupancy_region(self.detected_frames, self.evaluated_frames, self.yellow_threshold_percent, self.green_threshold_percent) == "green",
                "classification": occupancy_region(self.detected_frames, self.evaluated_frames, self.yellow_threshold_percent, self.green_threshold_percent),
                "person_detected": bool(self.last_targets),
                "detected_frames": self.detected_frames,
                "evaluated_frames": self.evaluated_frames,
                "ratio": self.detected_frames / self.evaluated_frames if self.evaluated_frames else 0.0,
                "threshold_percent": self.occupancy_threshold_percent,
                "yellow_threshold_percent": self.yellow_threshold_percent,
                "green_threshold_percent": self.green_threshold_percent,
                "target_count": len(self.last_targets),
                "targets": self.last_targets,
                "shadow": np.asarray(world_points, dtype=float).round(3).tolist(),
                "room": self.room,
                "fov": {
                    "horizontal_deg": cone.get("horizontal_deg", 40.0),
                    "vertical_deg": cone.get("vertical_deg", 65.0),
                    "range_m": cone.get("range_m", 15.0),
                },
                "chunk_index": self.chunk_index,
                "frame_index": self.evaluated_frames - 1,
            }, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, self.live_state_path)
        except OSError:
            pass
        finally:
            temporary.unlink(missing_ok=True)

    def finish(self) -> Dict[str, Any]:
        finalization_started = time.perf_counter()
        self.handle.flush()
        self.handle.close()
        self.csv_temporary.replace(self.csv_path)
        if self.identity:
            self.identity.save()
        points = np.vstack(self.chunk_points) if self.chunk_points else np.empty((0, 3), dtype=float)
        weights = np.concatenate(self.chunk_weights) if self.chunk_weights else np.empty(0, dtype=float)
        x_axis, y_axis, z = _bin_points(points, self.room, weights=weights, resolution=72)
        ratio = self.detected_frames / self.evaluated_frames if self.evaluated_frames else 0.0
        classification = occupancy_region(
            self.detected_frames, self.evaluated_frames,
            self.yellow_threshold_percent, self.green_threshold_percent,
        )
        label = "occupied" if classification == "green" else "empty"
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
            "x": x_axis, "y": y_axis, "z": z, "frames": self.playback_frames or [frame], "frame_count": self.evaluated_frames,
            "sample_count": self.evaluated_frames, "frame_interval_ms": max(50, int(round(self.chunk_seconds * 1000 / max(1, self.evaluated_frames)))),
            "updated": datetime.now(timezone.utc).isoformat(),
            "occupancy": {"label": label, "classification": classification, "detected_frames": self.detected_frames, "evaluated_frames": self.evaluated_frames,
                          "ratio": ratio, "threshold_percent": self.occupancy_threshold_percent,
                          "yellow_threshold_percent": self.yellow_threshold_percent,
                          "green_threshold_percent": self.green_threshold_percent,
                          "chunk_seconds": self.chunk_seconds},
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
    cones = room.get("radar_cones") if isinstance(room.get("radar_cones"), list) else []
    primary = next((cone for cone in cones if isinstance(cone, dict) and cone.get("enabled") is not False), {})
    position = float(primary.get("position_m", room.get("sensor_position_m", width / 2.0)))
    height = float(primary.get("height_m", room.get("sensor_height_m", 1.0)))
    wall = str(primary.get("wall", room.get("sensor_wall", "Back")))
    if wall == "Back":
        origin, lateral, forward = np.array([position, 0.0, height]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    elif wall == "Front":
        origin, lateral, forward = np.array([position, depth, height]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
    elif wall == "Left":
        origin, lateral, forward = np.array([0.0, position, height]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    else:
        origin, lateral, forward = np.array([width, position, height]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    angle = math.radians(float(primary.get("azimuth_deg", 0.0)))
    rotated_forward = math.cos(angle) * forward + math.sin(angle) * lateral
    rotated_lateral = math.cos(angle) * lateral - math.sin(angle) * forward
    return origin, rotated_lateral, rotated_forward


def world_from_local(point: Iterable[float], room: Dict[str, Any]) -> np.ndarray:
    origin, lateral_axis, forward_axis = _sensor_pose(room)
    local = np.asarray(list(point), dtype=float)
    if local.size < 3:
        local = np.pad(local, (0, 3 - local.size))
    return origin + lateral_axis * local[0] + forward_axis * local[1] + np.array([0.0, 0.0, local[2]])


def _world_points_from_local(points: np.ndarray, room: Dict[str, Any]) -> np.ndarray:
    local = np.asarray(points, dtype=float)
    if not local.size:
        return np.empty((0, 3), dtype=float)
    local = local.reshape((-1, local.shape[-1]))
    if local.shape[1] < 3:
        local = np.pad(local, ((0, 0), (0, 3 - local.shape[1])))
    origin, lateral_axis, forward_axis = _sensor_pose(room)
    world = (
        origin[None, :]
        + local[:, 0, None] * lateral_axis[None, :]
        + local[:, 1, None] * forward_axis[None, :]
    )
    world[:, 2] += local[:, 2]
    world[:, 0] = np.clip(world[:, 0], 0.0, float(room.get("width_m", 1.0)))
    world[:, 1] = np.clip(world[:, 1], 0.0, float(room.get("depth_m", 1.0)))
    world[:, 2] = np.clip(world[:, 2], 0.0, float(room.get("height_m", 1.0)))
    return world


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
        clipped = np.asarray(points, dtype=float).copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(room.get("width_m", 1.0)))
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(room.get("depth_m", 1.0)))
        clipped[:, 2] = np.clip(clipped[:, 2], 0.0, float(room.get("height_m", 1.0)))
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
    cones = room.get("radar_cones") if isinstance(room.get("radar_cones"), list) else []
    primary_cone = next((cone for cone in cones if isinstance(cone, dict) and cone.get("enabled") is not False), {})
    zones = []
    for zone in room.get("zones") or []:
        if not isinstance(zone, dict):
            continue
        x, y = float(zone.get("x") or 0), float(zone.get("y") or 0)
        width, depth = float(zone.get("width") or 1), float(zone.get("depth") or 1)
        if x <= float(world[0]) <= x + width and y <= float(world[1]) <= y + depth:
            label = str(zone.get("label") or zone.get("id") or "zone").strip()
            if label and label not in zones:
                zones.append(label)
    return {
        "id": int(target.get("id", 0)),
        "cone_id": primary_cone.get("id", "radar-1"),
        "pose": target.get("pose"),
        "presence_mode": target.get("presence_mode"),
        "range_m": round(float(target.get("range_m", 0.0)), 3),
        "position": [round(float(value), 3) for value in world],
        "size": [round(float(value), 3) for value in world_size],
        "snr_db": round(float(target.get("snr_db", 0.0)), 2),
        "confidence": round(float(target.get("confidence", target.get("snr_db", 0.0))), 2),
        "zones": zones,
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
    yellow_threshold_percent: float = 20.0,
    green_threshold_percent: float = 60.0,
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
    yellow_threshold_percent, green_threshold_percent = validate_region_thresholds(yellow_threshold_percent, green_threshold_percent)
    classification = occupancy_region(detected_frames, evaluated_frames, yellow_threshold_percent, green_threshold_percent)
    occupied = classification == "green"
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
            "classification": classification,
            "detected_frames": detected_frames,
            "evaluated_frames": evaluated_frames,
            "ratio": occupancy_ratio,
            "threshold_percent": occupancy_threshold_percent,
            "yellow_threshold_percent": yellow_threshold_percent,
            "green_threshold_percent": green_threshold_percent,
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
    yellow_threshold = float((latest.get("occupancy") or {}).get("yellow_threshold_percent", 20.0))
    green_threshold = float((latest.get("occupancy") or {}).get("green_threshold_percent", 60.0))
    classification = occupancy_region(total_detected, total_evaluated, yellow_threshold, green_threshold)
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
            "label": "occupied" if classification == "green" else "empty",
            "classification": classification,
            "detected_frames": total_detected,
            "evaluated_frames": total_evaluated,
            "ratio": occupancy_ratio,
            "threshold_percent": threshold_percent,
            "yellow_threshold_percent": yellow_threshold,
            "green_threshold_percent": green_threshold,
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
