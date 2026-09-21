"""Per-minute collection runtime: shared context + worker threads.

Extracted from ``minute_collector.main`` so the capture orchestration is a set
of module-level functions operating on a single :class:`CollectorContext`
instead of deeply-nested closures. ``minute_collector.main`` builds the context
and calls :func:`run`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.config import Config  # type: ignore
    from backend.capture_container import build_capture_container  # type: ignore
    from backend.model_runtime import ModelRegistry, WindowedAnalyzer, E2_WINDOW_FRAMES, is_occupancy_result  # type: ignore
    from backend.live_features import compute_live_features  # type: ignore
    from backend.capture_hardware import (  # type: ignore
        RADAR_CFG,
        RADAR_FRAMES_PER_CHUNK,
        _minute_csi_samples,
        _minute_radar_frames,
        acquire_radar_lock,
        chown_to_invoking_user,
        collect_csi,
        collect_sensehat,
        csi_capture_stats,
        find_camera,
        find_csi_ports,
        iso_now,
        release_radar_lock,
        settle_radar_gpio,
        start_radar_capture,
        stop_radar_capture,
    )
    from backend.collector_manifest import (  # type: ignore
        DATA_ROOT,
        annotate_chunk_result,
        compact_manifest,
        load_processing_settings,
        minute_start,
        normalize_labels,
        output_dir_for_minute,
        sleep_until,
        summarize_minute_results,
        write_json_atomic,
    )
    from backend.radar_analysis import (  # type: ignore
        PersistentTargetIdentity,
        StreamingChunkAnalyzer,
        compile_minute_xy_payload,
        create_signal_processor,
        load_room_config,
        occupancy_label,
    )
    from backend.home_assistant import publish_model_occupancy, control_linked_device  # type: ignore
else:
    from .config import Config
    from .capture_container import build_capture_container
    from .model_runtime import ModelRegistry, WindowedAnalyzer, E2_WINDOW_FRAMES, is_occupancy_result
    from .live_features import compute_live_features
    from .capture_hardware import (
        RADAR_CFG,
        RADAR_FRAMES_PER_CHUNK,
        _minute_csi_samples,
        _minute_radar_frames,
        acquire_radar_lock,
        chown_to_invoking_user,
        collect_csi,
        collect_sensehat,
        csi_capture_stats,
        find_camera,
        find_csi_ports,
        iso_now,
        release_radar_lock,
        settle_radar_gpio,
        start_radar_capture,
        stop_radar_capture,
    )
    from .collector_manifest import (
        DATA_ROOT,
        annotate_chunk_result,
        compact_manifest,
        load_processing_settings,
        minute_start,
        normalize_labels,
        output_dir_for_minute,
        sleep_until,
        summarize_minute_results,
        write_json_atomic,
    )
    from .radar_analysis import (
        PersistentTargetIdentity,
        StreamingChunkAnalyzer,
        compile_minute_xy_payload,
        create_signal_processor,
        load_room_config,
        occupancy_label,
    )
    from .home_assistant import publish_model_occupancy, control_linked_device

THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
CSI_HEADER = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"
# A minute contains at most about sixty 10-frame chunks. The dedicated live
# worker now owns freshness, so archival jobs can be buffered for the whole
# minute instead of discarding a valid saved chunk during a transient CPU spike.
MAX_PENDING_ANALYSIS_CHUNKS = 64
LIVE_VISUALIZATION_INTERVAL_SECONDS = 0.05
MAX_PENDING_ANALYSIS_FRAMES_PER_CHUNK = 1

sys.path.insert(0, str(MMW_RELEASE))


class CollectorContext:
    """Mutable shared state for one minute's collection.

    All worker threads read/write through this object, replacing the closure
    variables that previously lived in ``main``.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.live_only = bool(getattr(args, "live_only", False))
        self.preset_labels = normalize_labels(args.label or [])
        self.initial_settings = load_processing_settings()
        self.device_identity: dict[str, Any] = {}
        try:
            value = json.loads((Path(Config.CONFIG_DIR).expanduser() / "device_config.json").read_text(encoding="utf-8"))
            self.device_identity = value if isinstance(value, dict) else {}
        except (FileNotFoundError, OSError, ValueError):
            pass
        self.chunk_seconds = 1.0
        self.expected_chunks = max(1, int(math.floor(float(args.duration))))
        self.target_start = minute_start(args.start_now, args.scheduled_start)
        if self.live_only:
            self.folder_name = "live"
            self.output_dir = DATA_ROOT / "live"
            shutil.rmtree(self.output_dir, ignore_errors=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.folder_name = self.target_start.strftime("%Y%m%d_%H%M")
            self.output_dir = output_dir_for_minute(self.folder_name, self.preset_labels)
            self.output_dir.mkdir(parents=True, exist_ok=False)

        self.manifest: dict[str, Any] = {
            "schema": "thoth-minute-manifest/v7",
            "collection_unit": "minute",
            "sample_unit": "one-second synchronized sensor window",
            "folder_minute": self.folder_name,
            "scheduled_start": self.target_start.isoformat(timespec="seconds"),
            "duration_seconds": args.duration,
            "chunk_seconds": self.chunk_seconds,
            "chunk_frames": RADAR_FRAMES_PER_CHUNK,
            "expected_chunks": self.expected_chunks,
            "preset_labels": self.preset_labels,
            "labels": self.preset_labels or ["collecting"],
            "device_id": self.device_identity.get("device_id"),
            "device_name": self.device_identity.get("device_name"),
            "capture_settings": self.initial_settings,
            "outputs": {},
            "assets": [],
            "errors": [],
            "warnings": [],
            "primary_label": self.preset_labels[0] if self.preset_labels else "collecting",
            "relative_path": str(self.output_dir.relative_to(DATA_ROOT)),
            "sensors_enabled": {
                "usb_camera": not args.no_camera,
                "dreamhat_radar": not args.no_radar,
                "esp32_csi": not args.no_csi,
                "sense_hat": not args.no_sensehat,
            },
        }

        self.csi_ports: list[str] = []
        self.csi_candidates: list[str] = []
        if not args.no_csi:
            self.csi_ports, self.csi_candidates = find_csi_ports(args.csi_port, args.csi_baud, args.csi_detect_seconds)
            if not self.csi_ports:
                self.manifest["errors"].append("No ESP32 CSI serial device found.")
        self.manifest["csi_receiver_count"] = len(self.csi_ports)
        self.manifest["sensor_labels"] = {
            "esp32_csi": f"csix{len(self.csi_ports)}" if len(self.csi_ports) > 1 else "csi",
        }

        # Timing (populated when capture actually starts).
        self.capture_started_monotonic = 0.0
        self.capture_started_monotonic_ns = 0
        self.stop_at = math.inf
        self.capture_started = ""

        # Sensor handles + worker threads.
        self.csi_stop = threading.Event()
        self.csi_threads: list[threading.Thread] = []
        self.sense_thread: threading.Thread | None = None
        self.camera: str | None = None
        self.camera_thread: threading.Thread | None = None
        self.camera_frames: list[dict[str, Any]] = []
        try:
            self.camera_fps = max(0.2, min(30.0, float(self.initial_settings.get("camera_fps") or 1.0)))
        except (TypeError, ValueError):
            self.camera_fps = 1.0
        if self.live_only:
            self.camera_fps = max(self.camera_fps, 5.0)
        self.radar_reader_thread: threading.Thread | None = None
        self.radar: Any | None = None
        self.radar_lock: Any | None = None
        self.radar_analysis_thread: threading.Thread | None = None
        self.radar_live_thread: threading.Thread | None = None
        self.radar_upload_thread: threading.Thread | None = None
        self.model_thread: threading.Thread | None = None
        self.partial_minute_thread: threading.Thread | None = None
        self.live_features_thread: threading.Thread | None = None

        # Queues + shared buffers.
        self.analysis_queue: queue.Queue[Any] = queue.Queue(maxsize=MAX_PENDING_ANALYSIS_CHUNKS)
        self.live_analysis_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.live_queue_key: dict[str, Any] = {"stream": self.folder_name}
        self.upload_queue: queue.Queue[Any] = queue.Queue()
        self.model_queue: queue.Queue[Any] = queue.Queue()
        self.partial_minute_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.radar_chunk_results: list[dict[str, Any]] = []
        self.radar_frame_count = 0
        self.radar_first_frame_at: float | None = None
        self.radar_last_frame_at: float | None = None
        self.publish_lock = threading.Lock()
        self.model_registry = ModelRegistry(THOTH_ROOT / "models" / "user")
        self.windowed_analyzer = WindowedAnalyzer(self.model_registry)
        self.radar_model_history: list[bytes] = []
        self.room_config = load_room_config()
        self.live_radar_buffer: deque[bytes] = deque(maxlen=256)
        self.live_features_stop = threading.Event()


def enqueue_latest_chunk_frame(
    analysis_queue: queue.Queue[Any],
    entry: dict[str, Any],
    frame: bytes,
    captured_at: float,
    *metadata: Any,
) -> bool:
    """Queue a frame while retaining the newest pending samples for its chunk."""
    replaced = False
    with analysis_queue.mutex:
        matching_indexes = [
            index
            for index, item in enumerate(analysis_queue.queue)
            if isinstance(item, tuple)
            and len(item) >= 2
            and item[0] == "frame"
            and item[1] is entry
        ]
        if len(matching_indexes) >= MAX_PENDING_ANALYSIS_FRAMES_PER_CHUNK:
            del analysis_queue.queue[matching_indexes[0]]
            analysis_queue.unfinished_tasks = max(0, analysis_queue.unfinished_tasks - 1)
            analysis_queue.not_full.notify()
            replaced = True
    analysis_queue.put_nowait(("frame", entry, frame, captured_at, *metadata))
    return replaced


def enqueue_analysis_chunk(
    analysis_queue: queue.Queue[Any],
    job: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Queue one exact 10-frame archival job."""
    if len(job) < 4 or job[0] != "chunk" or len(job[3]) != RADAR_FRAMES_PER_CHUNK:
        raise ValueError(
            f"radar analysis chunks require exactly {RADAR_FRAMES_PER_CHUNK} frames"
        )
    dropped: list[dict[str, Any]] = []
    while True:
        try:
            analysis_queue.put_nowait(job)
            return dropped
        except queue.Full:
            try:
                stale = analysis_queue.get_nowait()
            except queue.Empty:
                continue
            try:
                if (
                    isinstance(stale, tuple)
                    and len(stale) >= 2
                    and stale[0] == "chunk"
                    and isinstance(stale[1], dict)
                ):
                    dropped.append(stale[1])
            finally:
                analysis_queue.task_done()


def live_chunk_statistics(analyzer: StreamingChunkAnalyzer) -> dict[str, Any]:
    """Build the partial chunk result published while analysis is running."""
    evaluated = analyzer.evaluated_frames
    detected = analyzer.detected_frames
    label = occupancy_label(
        detected, evaluated, (100.0 / evaluated) if evaluated else 100.0,
    )
    classification = "green" if label == "occupied" else "red"
    return {
        "status": "collecting",
        "detected_frames": detected,
        "evaluated_frames": evaluated,
        "ratio": detected / evaluated if evaluated else 0.0,
        "classification": classification,
        "occupied": label == "occupied",
        "location": list(analyzer.last_position),
        "score": analyzer.last_score,
        "people_count": len(analyzer.last_targets),
        "targets": analyzer.last_targets,
    }


def current_csi_samples(ctx: CollectorContext) -> list[tuple[int, str]]:
    samples: list[tuple[int, str]] = []
    wifi = ctx.manifest.get("outputs", {}).get("wifi_csi", {})
    receivers = wifi.get("receivers") if isinstance(wifi, dict) else None
    if not isinstance(receivers, list):
        receivers = [wifi] if isinstance(wifi, dict) else []
    for receiver_index, receiver in enumerate(receivers):
        try:
            path = Path(str(receiver.get("path") or ""))
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if "CSI_DATA" in line:
                        samples.append((receiver_index, line.rstrip()))
        except (AttributeError, OSError):
            continue
    return samples


def fire_model_device_links(ctx: CollectorContext, results: list[dict[str, Any]], scope: str) -> None:
    """Drive each model's configured Home Assistant device link."""
    for prediction in results or []:
        if not isinstance(prediction, dict) or prediction.get("status") != "ok":
            continue
        model_id = str(prediction.get("model_id") or "")
        if not model_id:
            continue
        try:
            item = ctx.model_registry.get(model_id)
        except Exception:
            item = None
        ha_link = (item or {}).get("ha_link")
        if not ha_link:
            continue
        try:
            control_linked_device(
                prediction,
                ha_link,
                ctx.folder_name,
                scope=scope,
                chunk_index=prediction.get("chunk_index"),
            )
        except Exception as exc:
            logging.getLogger(__name__).error("Linked device control failed: %s", exc)


def effective_preset_labels(ctx: CollectorContext) -> list[str]:
    """Use the latest labels so additions and removals affect this minute."""
    return normalize_labels(load_processing_settings().get("labels"))


def refresh_manifest_labels(ctx: CollectorContext) -> list[str]:
    active_labels = effective_preset_labels(ctx)
    previous_preset_labels = normalize_labels(ctx.manifest.get("preset_labels"))
    existing = normalize_labels(ctx.manifest.get("labels"))
    existing = [
        label for label in existing
        if label != "collecting" and label not in previous_preset_labels
    ]
    ctx.manifest["preset_labels"] = active_labels
    ctx.manifest["labels"] = list(dict.fromkeys([*active_labels, *existing])) or ["collecting"]
    ctx.manifest["primary_label"] = ctx.manifest["labels"][0]
    return active_labels


def merge_home_assistant_status(ctx: CollectorContext) -> None:
    status_path = ctx.output_dir / ".home_assistant_status.json"
    try:
        statuses = json.loads(status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return
    chunks = (((ctx.manifest.get("outputs") or {}).get("radar") or {}).get("chunks") or [])
    minute_status = statuses.get("minute") if isinstance(statuses, dict) else None
    if isinstance(minute_status, dict):
        ctx.manifest["home_assistant"] = minute_status
    for entry in chunks:
        status = statuses.get(str(entry.get("chunk_index"))) if isinstance(statuses, dict) else None
        if isinstance(status, dict):
            entry["home_assistant"] = status


def write_live_manifest(ctx: CollectorContext) -> None:
    refresh_manifest_labels(ctx)
    merge_home_assistant_status(ctx)
    assets: list[dict[str, Any]] = []
    for entry in ctx.radar_chunk_results:
        index = int(entry.get("chunk_index") or 0)
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        occupancy = result.get("occupancy") if isinstance(result.get("occupancy"), dict) else {}
        common = {
            "second_index": index,
            "started_at": entry.get("started"),
            "finished_at": entry.get("finished_capture"),
            "duration_seconds": entry.get("chunk_seconds"),
            "labels": result.get("labels") or entry.get("labels") or [],
            "prediction": occupancy.get("label") or entry.get("status"),
            "properties": {
                "frame_count": entry.get("chunk_frames"),
                "detected_frames": occupancy.get("detected_frames"),
                "evaluated_frames": occupancy.get("evaluated_frames"),
                "ratio": occupancy.get("ratio"),
                "people_count": result.get("people_count", entry.get("people_count", 0)),
                "location": result.get("location"),
                "location_score": result.get("score"),
                "targets": result.get("targets") or [],
                "xy_map": result.get("xy_map") or {},
            },
        }
        radar_name = Path(str(entry.get("bin_path") or "")).name
        if radar_name:
            assets.append({**common, "sensor": "radar", "filename": radar_name, "content_type": "application/octet-stream"})
    for frame in ctx.camera_frames:
        camera_name = Path(str(frame.get("path") or "")).name
        if camera_name:
            assets.append({
                "sensor": "camera",
                "second_index": frame.get("second_index"),
                "started_at": frame.get("captured_at"),
                "filename": camera_name,
                "content_type": "image/jpeg",
            })
    csi_output = (ctx.manifest.get("outputs") or {}).get("wifi_csi")
    if isinstance(csi_output, dict):
        receivers = csi_output.get("receivers") if isinstance(csi_output.get("receivers"), list) else []
        if not receivers and csi_output.get("path"):
            receivers = [csi_output]
        for receiver in receivers:
            if not isinstance(receiver, dict) or not receiver.get("path"):
                continue
            assets.append({
                "sensor": "wifi_csi",
                "filename": Path(str(receiver["path"])).name,
                "content_type": "text/csv",
                "started_at": receiver.get("started") or csi_output.get("started"),
                "duration_seconds": ctx.args.duration,
                "coverage": "continuous minute stream",
                "labels": ctx.manifest.get("labels") or [],
                "properties": {
                    "baud": receiver.get("baud", csi_output.get("baud")),
                    "device": receiver.get("device"),
                    "device_id": receiver.get("device_id"),
                    "average_sampling_rate_hz": receiver.get("average_sampling_rate_hz"),
                },
            })
    ctx.manifest["assets"] = assets
    # Per-minute progress is expressed in seconds, not chunks: each radar batch
    # maps to one synchronized second.
    captured_seconds = len(ctx.radar_chunk_results)
    ctx.manifest["progress"] = {
        "unit": "second",
        "captured_seconds": captured_seconds,
        "total_seconds": ctx.expected_chunks,
        "percent": round(100.0 * captured_seconds / max(1, ctx.expected_chunks), 1),
    }
    snapshot = dict(ctx.manifest)
    snapshot["capture_started"] = ctx.capture_started
    snapshot["status"] = "collecting"
    outputs = dict(ctx.manifest.get("outputs") or {})
    radar_output = outputs.get("radar")
    if isinstance(radar_output, dict):
        radar_snapshot = dict(radar_output)
        radar_snapshot["chunks"] = [
            {key: value for key, value in entry.items() if key != "result"}
            for entry in (radar_output.get("chunks") or [])
        ]
        outputs["radar"] = radar_snapshot
    snapshot["outputs"] = outputs
    write_json_atomic(ctx.output_dir / "manifest.json", snapshot)


def publish_radar_results(ctx: CollectorContext) -> None:
    # Built-in semantic chunk publishing is removed in manifest v7.
    return


def upload_live_chunk(ctx: CollectorContext, index: int) -> None:
    # Live chunk API carried heuristic occupancy fields; model timelines are
    # persisted locally and uploaded with the completed manifest.
    return


def enqueue_home_assistant(ctx: CollectorContext, entry: dict[str, Any], occupancy: dict[str, Any], result: dict[str, Any], scope: str = "chunk") -> None:
    # Home Assistant updates come only from publish_model_occupancy.
    return


def run_model_worker(ctx: CollectorContext) -> None:
    while True:
        job = ctx.model_queue.get()
        try:
            if job is None:
                return
            chunk_index, frames, timestamp = job
            try:
                results = ctx.windowed_analyzer.push(frames, current_csi_samples(ctx), timestamp)
            except Exception as exc:
                logging.getLogger(__name__).error("User model runtime unavailable: %s", exc)
                results = []
            if results:
                fire_model_device_links(ctx, results, "chunk")
                occupancy_results = [item for item in results if item.get("status") == "ok" and is_occupancy_result(item)]
                if occupancy_results:
                    selected = max(occupancy_results, key=lambda item: float(item.get("confidence") or 0.0))
                    publish_model_occupancy(selected, ctx.folder_name, chunk_index=chunk_index)
                with ctx.publish_lock:
                    timelines = ctx.manifest.setdefault("model_predictions", [])
                    by_id = {str(item.get("model_id")): item for item in timelines if isinstance(item, dict)}
                    for prediction in results:
                        model_id = str(prediction.get("model_id"))
                        timeline = by_id.setdefault(model_id, {
                            "model_id": model_id,
                            "model_name": prediction.get("model_name"),
                            "model_version": prediction.get("model_version"),
                            "timeline": [],
                        })
                        timeline["timeline"].append(prediction)
                    ctx.manifest["model_predictions"] = list(by_id.values())
                    write_live_manifest(ctx)
        finally:
            ctx.model_queue.task_done()


def run_partial_minute_worker(ctx: CollectorContext) -> None:
    """Run the minute-level model on the windows collected so far."""
    while True:
        job = ctx.partial_minute_queue.get()
        try:
            if job is None:
                return
            pframes, ptimes, ptimestamp = job
            try:
                presults = ctx.model_registry.run_minute(
                    list(pframes), list(ptimes), _minute_csi_samples(ctx.manifest), ptimestamp
                )
            except Exception as exc:
                logging.getLogger(__name__).error("Partial minute inference failed: %s", exc)
                presults = []
            if presults:
                fire_model_device_links(ctx, presults, "partial_minute")
            pocc = [
                item for item in presults
                if item.get("status") == "ok" and is_occupancy_result(item)
            ]
            if pocc:
                selected = dict(max(pocc, key=lambda item: float(item.get("confidence") or 0.0)))
                selected["scope"] = "partial_minute"
                publish_model_occupancy(selected, ctx.folder_name, chunk_index=None)
            if presults:
                with ctx.publish_lock:
                    timelines = ctx.manifest.setdefault("model_predictions", [])
                    by_id = {str(item.get("model_id")): item for item in timelines if isinstance(item, dict)}
                    for prediction in presults:
                        prediction = dict(prediction)
                        prediction["scope"] = "partial_minute"
                        model_id = str(prediction.get("model_id"))
                        timeline = by_id.setdefault(model_id, {
                            "model_id": model_id,
                            "model_name": prediction.get("model_name"),
                            "model_version": prediction.get("model_version"),
                            "timeline": [],
                        })
                        timeline["timeline"].append(prediction)
                    ctx.manifest["model_predictions"] = list(by_id.values())
                    write_live_manifest(ctx)
        finally:
            ctx.partial_minute_queue.task_done()


def run_upload_worker(ctx: CollectorContext) -> None:
    while True:
        index = ctx.upload_queue.get()
        try:
            if index is None:
                return
            upload_live_chunk(ctx, int(index))
        finally:
            ctx.upload_queue.task_done()


def run_camera_worker(ctx: CollectorContext) -> None:
    """Stream camera frames at the configured rate (camera_fps per second)."""
    if not ctx.camera:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        ctx.manifest["warnings"].append("ffmpeg was not found in PATH; camera capture disabled.")
        return
    seq_pattern = ctx.output_dir / "camera_%05d.jpg"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "v4l2",
        "-i", ctx.camera,
        "-vf", f"fps={ctx.camera_fps}",
        "-q:v", "2",
        str(seq_pattern),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        ctx.manifest["warnings"].append(f"Camera stream failed to start: {exc}")
        return
    seen = 0
    last_manifest_write = 0.0
    try:
        while time.monotonic() < ctx.stop_at:
            files = sorted(ctx.output_dir.glob("camera_*.jpg"))
            # skip the file ffmpeg may still be writing
            complete = files[:-1] if len(files) > 1 else []
            for idx in range(seen, len(complete)):
                image_path = complete[idx]
                second_index = min(ctx.expected_chunks - 1, int(idx / ctx.camera_fps))
                frame = {
                    "second_index": second_index,
                    "frame_index": idx,
                    "captured_at": iso_now(),
                    "monotonic_ns": time.monotonic_ns(),
                    "path": str(image_path),
                }
                with ctx.publish_lock:
                    ctx.camera_frames.append(frame)
                    if second_index < len(ctx.radar_chunk_results):
                        ctx.radar_chunk_results[second_index].update({
                            "camera_path": str(image_path),
                            "camera_captured_at": frame["captured_at"],
                            "camera_monotonic_ns": frame["monotonic_ns"],
                        })
                seen = idx + 1
            now = time.monotonic()
            if seen and now - last_manifest_write >= 1.0:
                with ctx.publish_lock:
                    ctx.manifest["outputs"].setdefault("camera", {})["frames"] = ctx.camera_frames
                    write_live_manifest(ctx)
                last_manifest_write = now
            time.sleep(0.4)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
        # register any frames written just before shutdown
        files = sorted(ctx.output_dir.glob("camera_*.jpg"))
        for idx in range(seen, len(files)):
            image_path = files[idx]
            second_index = min(ctx.expected_chunks - 1, int(idx / ctx.camera_fps))
            ctx.camera_frames.append({
                "second_index": second_index,
                "frame_index": idx,
                "captured_at": iso_now(),
                "monotonic_ns": time.monotonic_ns(),
                "path": str(image_path),
            })
        if ctx.camera_frames:
            ctx.manifest["outputs"].setdefault("camera", {})["frames"] = ctx.camera_frames


def run_analysis_worker(ctx: CollectorContext) -> None:
    try:
        processor = create_signal_processor()
    except Exception as init_exc:
        while True:
            failed_job = ctx.analysis_queue.get()
            try:
                if failed_job is None:
                    return
                failed_entry = failed_job[1]
                failed_entry.update({
                    "status": "empty",
                    "classification": "red",
                    "occupied": False,
                    "data_quality": "analysis_error",
                    "error": str(init_exc),
                    "finished": iso_now(),
                })
                with ctx.publish_lock:
                    write_live_manifest(ctx)
            finally:
                ctx.analysis_queue.task_done()
    identity: PersistentTargetIdentity | None = None
    while True:
        job = ctx.analysis_queue.get()
        entry: dict[str, Any] | None = None
        analyzer: StreamingChunkAnalyzer | None = None
        try:
            if job is None:
                return
            _kind, entry, settings_snapshot, frames, captured_at = job
            mode = str(settings_snapshot.get("system_mode") or "balanced")
            if identity is None:
                identity = PersistentTargetIdentity(mode=mode)
            if hasattr(processor, "set_system_mode"):
                processor.set_system_mode(mode)
            analyzer = StreamingChunkAnalyzer(
                processor,
                None,
                int(entry["chunk_index"]),
                float(entry["chunk_seconds"]),
                ctx.room_config,
                0.45,
                0.0,
                0.0,
                0.01,
                identity=identity,
                live_state_path=None,
                radar_detection_threshold_db=float(
                    settings_snapshot.get("radar_detection_threshold_db") or 8.0
                ),
            )
            entry["status"] = "analyzing"
            with ctx.publish_lock:
                write_live_manifest(ctx)
            analyzer.max_queue_lag_ms = max(
                0.0, (time.monotonic() - float(captured_at)) * 1000
            )
            for frame in frames:
                analyzer.process(frame)
            result = analyzer.finish()
            result["bin_path"] = entry.get("bin_path")
            result["camera_path"] = entry.get("camera_path")
            # Frame offsets follow the captured 10-frame bins, even when a
            # stale analysis job was deferred to keep the live view current.
            previous_frames = int(entry["chunk_index"]) * RADAR_FRAMES_PER_CHUNK
            chunk_labels = normalize_labels(settings_snapshot.get("labels")) or ctx.preset_labels
            annotate_chunk_result(
                result, settings_snapshot, ctx.room_config, chunk_labels,
                ctx.folder_name, ctx.expected_chunks, previous_frames,
            )
            entry["result"] = result
            occupancy = result.get("occupancy", {})
            label = occupancy.get("label") or "empty"
            if int(occupancy.get("evaluated_frames") or 0) != RADAR_FRAMES_PER_CHUNK:
                raise RuntimeError(
                    f"analyzed {occupancy.get('evaluated_frames', 0)} of "
                    f"{RADAR_FRAMES_PER_CHUNK} radar frames"
                )
            entry.update({
                "status": label,
                "detected_frames": occupancy.get("detected_frames", 0),
                "evaluated_frames": occupancy.get("evaluated_frames", 0),
                "ratio": occupancy.get("ratio", 0.0),
                "occupied": label == "occupied",
                "location": result.get("location"),
                "score": result.get("score"),
                "people_count": result.get("people_count", 0),
                "targets": result.get("targets") or [],
                "labels": result.get("labels") or [],
                "activity_labels": result.get("activity_labels") or [],
                "activity": result.get("activity"),
                "join": result.get("join"),
                "performance": result.get("performance"),
                "finished": iso_now(),
            })
            if settings_snapshot.get("auto_occupancy_label_enabled"):
                ctx.manifest["auto_occupancy_label"] = occupancy
            publish_radar_results(ctx)
            enqueue_home_assistant(ctx, entry, occupancy, result)
            ctx.upload_queue.put(int(entry["chunk_index"]))
        except Exception as exc:
            if entry is not None:
                entry["error"] = str(exc)
                entry.update({
                    "status": "empty",
                    "classification": "red",
                    "occupied": False,
                    "data_quality": "analysis_error",
                    "finished": iso_now(),
                })
                with ctx.publish_lock:
                    write_live_manifest(ctx)
                ctx.upload_queue.put(int(entry["chunk_index"]))
            if analyzer is not None:
                try:
                    analyzer.handle.close()
                except Exception:
                    pass
        finally:
            ctx.analysis_queue.task_done()


def run_live_features_worker(ctx: CollectorContext) -> None:
    """Push per-second signal features (radar SNR, CSI amp/variance, STFT)
    to the local internal endpoint, which relays them to the brain's
    live-chunks feed. ``chunk_index`` is reused as a per-second key so the
    live view shows a per-second signal timeline rather than predictions.
    """
    second_index = 0
    while not ctx.live_features_stop.is_set() and not ctx.csi_stop.is_set():
        try:
            radar_snapshot = list(ctx.live_radar_buffer)
            csi_lines = [line for _, line in current_csi_samples(ctx)[-256:]]
            features = compute_live_features(
                radar_frames=radar_snapshot,
                csi_lines=csi_lines,
            )
            if features:
                payload = json.dumps({
                    "minute": ctx.folder_name,
                    "chunk_index": second_index,
                    "chunk_frames": len(radar_snapshot),
                    "status": "collecting",
                    "features": features,
                    "captured_at": iso_now(),
                }, separators=(",", ":")).encode("utf-8")
                request = urllib.request.Request(
                    "http://127.0.0.1:5000/api/internal/capture-chunk",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3.0):
                    pass
            second_index += 1
        except Exception as exc:
            print(f"Live features push deferred: {exc}", file=sys.stderr)
        ctx.live_features_stop.wait(1.0)


def run_live_analysis_worker(ctx: CollectorContext) -> None:
    """Analyze only the newest captured frame for the live Presence view."""
    try:
        processor = create_signal_processor()
    except Exception as exc:
        print(f"Live radar visualization unavailable: {exc}", file=sys.stderr)
        while True:
            failed_job = ctx.live_analysis_queue.get()
            try:
                if failed_job is None:
                    return
            finally:
                ctx.live_analysis_queue.task_done()

    analyzer: StreamingChunkAnalyzer | None = None
    analyzer_chunk_index: int | None = None
    # Shared across per-chunk analyzer recreations so the published
    # sensor_hz measures the continuous live rate, not a per-chunk reset.
    live_frame_times: deque[float] = deque(maxlen=60)
    try:
        while True:
            job = ctx.live_analysis_queue.get()
            try:
                if job is None:
                    return
                _, _, frame, captured_at, chunk_index, settings_snapshot = job
                chunk_index = int(chunk_index)
                if analyzer is None or analyzer_chunk_index != chunk_index:
                    if analyzer is not None:
                        analyzer.handle.close()
                    mode = str(settings_snapshot.get("system_mode") or "balanced")
                    if hasattr(processor, "set_system_mode"):
                        processor.set_system_mode(mode)
                    analyzer = StreamingChunkAnalyzer(
                        processor,
                        None,
                        chunk_index,
                        1.0,
                        ctx.room_config,
                        0.45,
                        0.0,
                        0.0,
                        0.01,
                        identity=None,
                        radar_detection_threshold_db=float(
                            settings_snapshot.get("radar_detection_threshold_db") or 8.0
                        ),
                        live_example2_only=True,
                    )
                    analyzer.frame_times = live_frame_times
                    analyzer_chunk_index = chunk_index
                analyzer.max_queue_lag_ms = max(
                    0.0, (time.monotonic() - float(captured_at)) * 1000
                )
                analyzer.process(frame)
            except Exception as exc:
                print(f"Live radar frame deferred: {exc}", file=sys.stderr)
            finally:
                ctx.live_analysis_queue.task_done()
    finally:
        if analyzer is not None:
            analyzer.handle.close()


def run_radar_reader(ctx: CollectorContext) -> None:
    radar = ctx.radar
    frames: list[bytes] = []
    frame_times: list[float] = []
    minute_frames: list[bytes] = []
    minute_times: list[float] = []
    last_window_count = 0
    last_live_enqueue = 0.0
    live_settings = ctx.initial_settings
    # Resolved once per minute: registry.list() hits disk and must not run
    # per frame inside the hot reader loop.
    maximum_model_frames = max(
        (
            int(spec.get("frames") or 0)
            for model in ctx.model_registry.list()
            if model.get("enabled")
            and str((model.get("metadata") or {}).get("execution") or "chunk") != "minute"
            for spec in (model.get("metadata") or {}).get("inputs", [])
            if spec.get("sensor") == "radar"
        ),
        default=RADAR_FRAMES_PER_CHUNK,
    )
    while time.monotonic() < ctx.stop_at:
        remaining = ctx.stop_at - time.monotonic()
        try:
            full_frame = bytes(radar.frame_buffer.get(timeout=min(0.1, max(0.01, remaining))))
        except queue.Empty:
            continue
        captured_at = time.monotonic()
        ctx.live_radar_buffer.append(full_frame)
        if not frames:
            live_settings = load_processing_settings()
        if ctx.live_only:
            # Live-only mode: every frame goes to the live worker, which
            # keeps only the newest pending frame — full sensor rate, no
            # chunking, no model queues.
            try:
                enqueue_latest_chunk_frame(
                    ctx.live_analysis_queue,
                    ctx.live_queue_key,
                    full_frame,
                    captured_at,
                    0,
                    live_settings,
                )
            except queue.Full:
                pass
            ctx.radar_frame_count += 1
            ctx.radar_first_frame_at = captured_at if ctx.radar_first_frame_at is None else ctx.radar_first_frame_at
            ctx.radar_last_frame_at = captured_at
            continue
        if captured_at - last_live_enqueue >= LIVE_VISUALIZATION_INTERVAL_SECONDS:
            try:
                enqueue_latest_chunk_frame(
                    ctx.live_analysis_queue,
                    ctx.live_queue_key,
                    full_frame,
                    captured_at,
                    len(ctx.radar_chunk_results),
                    live_settings,
                )
                last_live_enqueue = captured_at
            except queue.Full:
                pass
        ctx.radar_frame_count += 1
        ctx.radar_first_frame_at = captured_at if ctx.radar_first_frame_at is None else ctx.radar_first_frame_at
        ctx.radar_last_frame_at = captured_at
        frames.append(full_frame)
        minute_frames.append(full_frame)
        minute_times.append(captured_at)
        # A minute-level model needs 50-frame windows; run it as soon as a
        # new window completes instead of waiting for the whole minute.
        if len(minute_frames) // E2_WINDOW_FRAMES > last_window_count:
            last_window_count = len(minute_frames) // E2_WINDOW_FRAMES
            # maxsize-1 queue: drop any stale partial so the newest job
            # (with the most windows) is always the one that runs.
            try:
                ctx.partial_minute_queue.get_nowait()
                ctx.partial_minute_queue.task_done()
            except queue.Empty:
                pass
            try:
                ctx.partial_minute_queue.put_nowait((
                    tuple(minute_frames), tuple(minute_times), iso_now(),
                ))
            except queue.Full:
                pass
        ctx.radar_model_history.append(full_frame)
        if len(ctx.radar_model_history) > maximum_model_frames:
            del ctx.radar_model_history[:-maximum_model_frames]
        frame_times.append(captured_at)
        if len(frames) < RADAR_FRAMES_PER_CHUNK:
            continue

        chunk_index = len(ctx.radar_chunk_results)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        radar_path = ctx.output_dir / f"radar_{chunk_index:03d}_{timestamp}.bin"
        radar_path.write_bytes(b"".join(frames))
        duration = max(0.001, frame_times[-1] - frame_times[0])
        chunk_entry: dict[str, Any] = {
            "chunk_index": chunk_index,
            "bin_path": str(radar_path),
            "started": dt.datetime.fromtimestamp(
                time.time() - duration
            ).astimezone().isoformat(timespec="milliseconds"),
            "finished_capture": iso_now(),
            "chunk_seconds": duration,
            "chunk_frames": RADAR_FRAMES_PER_CHUNK,
            "status": "stored",
            "frame_sequence_start": int.from_bytes(frames[0][4:8], "little"),
            "frame_sequence_end": int.from_bytes(frames[-1][4:8], "little"),
            "frame_monotonic_ns": [int(value * 1_000_000_000) for value in frame_times],
        }
        ctx.radar_chunk_results.append(chunk_entry)
        ctx.manifest["outputs"]["radar"]["chunks"].append(chunk_entry)
        ctx.manifest["expected_chunks"] = max(ctx.expected_chunks, len(ctx.radar_chunk_results))
        with ctx.publish_lock:
            write_live_manifest(ctx)
        ctx.upload_queue.put(chunk_index)
        # Feed only this chunk's new frames; the WindowedAnalyzer keeps the
        # full-minute buffer and gives each model its own trailing window.
        ctx.model_queue.put((chunk_index, tuple(frames), iso_now()))
        # Queue the chunk for XY localization / occupancy analysis so the
        # per-chunk location, targets and xy_map land in the collected data.
        enqueue_analysis_chunk(
            ctx.analysis_queue,
            ("chunk", chunk_entry, dict(ctx.initial_settings), tuple(frames), captured_at),
        )
        frames = []
        frame_times = []

    if frames:
        ctx.manifest["warnings"].append(
            f"Discarded {len(frames)} radar frames at the minute boundary; "
            f"chunks require exactly {RADAR_FRAMES_PER_CHUNK} frames."
        )


def run(ctx: CollectorContext) -> int:
    """Orchestrate one minute of collection: spawn workers, wait, finalize."""
    args = ctx.args
    manifest = ctx.manifest
    output_dir = ctx.output_dir
    live_only = ctx.live_only

    if live_only:
        # SIGTERM from the supervisor should unwind finally blocks so the
        # radar GPIO/SPI handoff settles before the process exits.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    print(f"Output folder: {output_dir}")
    if live_only:
        print("Live-only mode: streaming sensors until terminated")
    else:
        print(f"Waiting for real-clock minute: {ctx.target_start.isoformat(timespec='seconds')}")
        sleep_until(ctx.target_start)

    ctx.capture_started_monotonic = time.monotonic()
    ctx.capture_started_monotonic_ns = time.monotonic_ns()
    ctx.stop_at = math.inf if live_only else ctx.capture_started_monotonic + args.duration
    ctx.capture_started = iso_now()
    manifest["capture_started_monotonic_ns"] = ctx.capture_started_monotonic_ns
    print(f"Capture started: {ctx.capture_started}")

    try:
        if not live_only:
            ctx.model_thread = threading.Thread(target=run_model_worker, args=(ctx,), name="UserModelInference", daemon=True)
            ctx.model_thread.start()
            ctx.radar_analysis_thread = threading.Thread(
                target=run_analysis_worker, args=(ctx,), name="RadarChunkAnalysis", daemon=True
            )
            ctx.radar_analysis_thread.start()
        ctx.radar_live_thread = threading.Thread(
            target=run_live_analysis_worker, args=(ctx,), name="RadarLive", daemon=True
        )
        ctx.radar_live_thread.start()
        ctx.live_features_thread = threading.Thread(
            target=run_live_features_worker, args=(ctx,), name="LiveFeatures", daemon=True
        )
        ctx.live_features_thread.start()
        if not live_only:
            ctx.partial_minute_thread = threading.Thread(
                target=run_partial_minute_worker, args=(ctx,), name="PartialMinuteInference", daemon=True
            )
            ctx.partial_minute_thread.start()
        if not args.no_sensehat:
            sense_file = output_dir / "sense_hat.jsonl"
            manifest["outputs"]["sense_hat"] = {
                "type": "json-lines",
                "path": str(sense_file),
                "files": [str(sense_file)],
                "started": iso_now(),
            }
            ctx.sense_thread = threading.Thread(
                target=collect_sensehat,
                args=(sense_file, ctx.csi_stop, manifest["errors"]),
                name="SenseHat",
                daemon=True,
            )
            ctx.sense_thread.start()

        if not args.no_csi and ctx.csi_ports:
            csi_started = iso_now()
            receivers: list[dict[str, Any]] = []
            for index, csi_port in enumerate(ctx.csi_ports, start=1):
                csi_file = output_dir / (
                    "wifi_csi.csv" if len(ctx.csi_ports) == 1 else f"wifi_csi_{index:02d}.csv"
                )
                csi_thread = threading.Thread(
                    target=collect_csi,
                    args=(csi_port, args.csi_baud, csi_file, ctx.csi_stop),
                    name=f"CSIReceiver-{index}",
                    daemon=True,
                )
                csi_thread.start()
                ctx.csi_threads.append(csi_thread)
                receivers.append({
                    "path": str(csi_file),
                    "type": "csv",
                    "device": csi_port,
                    "device_id": str(
                        (load_processing_settings().get("csi_device_ids") or {}).get(csi_port)
                        or f"csi-{index}"
                    ),
                    "baud": args.csi_baud,
                    "started": csi_started,
                })
            manifest["outputs"]["wifi_csi"] = {
                # Primary fields preserve compatibility with v4 readers.
                **receivers[0],
                "type": "csv",
                "baud": args.csi_baud,
                "receiver_count": len(receivers),
                "display_name": f"csix{len(receivers)}" if len(receivers) > 1 else "csi",
                "receivers": receivers,
                "detected_candidates": ctx.csi_candidates,
                "started": csi_started,
                "source": "USB-connected ESP32 receiver printing ESP-NOW CSI_DATA lines",
            }
        elif not args.no_csi and not ctx.csi_ports:
            csi_file = output_dir / "wifi_csi.csv"
            with open(csi_file, "w", encoding="utf-8", newline="") as output_fd:
                csv.writer(output_fd).writerow(["host_timestamp", "monotonic_ns", "serial_port", "raw_csi_line"])
            manifest["outputs"]["wifi_csi"] = {
                "path": str(csi_file),
                "type": "csv",
                "device": None,
                "baud": args.csi_baud,
                "detected_candidates": ctx.csi_candidates,
                "started": iso_now(),
                "source": "CSI receiver unavailable; header-only minute artifact",
                "data_quality": "sensor_missing",
            }
            manifest["warnings"].append("No ESP32 CSI serial device found; skipping CSI for this minute.")

        if not args.no_camera:
            ctx.camera = find_camera(args.camera)
            if ctx.camera is None:
                manifest["warnings"].append("No /dev/video* USB camera device found; skipping chunk images.")
            else:
                manifest["outputs"]["camera"] = {
                    "type": "one-jpeg-per-synchronized-second",
                    "device": ctx.camera,
                    "frames": ctx.camera_frames,
                }
                ctx.camera_thread = threading.Thread(
                    target=run_camera_worker,
                    args=(ctx,),
                    name="ChunkCamera",
                    daemon=True,
                )
                ctx.camera_thread.start()

        if not args.no_radar:
            manifest["outputs"]["radar"] = {
                "type": "chunked-bin",
                "config_dir": str(RADAR_CFG),
                "chunk_frames": RADAR_FRAMES_PER_CHUNK,
                "chunks": [],
                "note": "The hardware reader rotates one binary file for every 10 complete radar frames.",
            }
            with ctx.publish_lock:
                write_live_manifest(ctx)
            if not live_only:
                ctx.radar_upload_thread = threading.Thread(target=run_upload_worker, args=(ctx,), name="RadarUpload", daemon=True)
                ctx.radar_upload_thread.start()
            try:
                ctx.radar_lock = acquire_radar_lock()
                ctx.radar = start_radar_capture()
            except Exception as exc:
                manifest["warnings"].append(f"Radar failed to start: {exc}")
                release_radar_lock(ctx.radar_lock)
                ctx.radar_lock = None
            if ctx.radar is not None:
                ctx.radar_reader_thread = threading.Thread(
                    target=run_radar_reader,
                    args=(ctx,),
                    name="RadarReader",
                    daemon=True,
                )
                ctx.radar_reader_thread.start()

        last_live_prune = 0.0
        while True:
            remaining = ctx.stop_at - time.monotonic()
            if remaining <= 0:
                break
            if live_only and time.monotonic() - last_live_prune >= 10.0:
                last_live_prune = time.monotonic()
                # Live mode only needs the freshest camera frame — prune the
                # image2 backlog so long sessions don't fill the disk.
                try:
                    stale_frames = sorted(output_dir.glob("camera_*.jpg"))
                    for stale in stale_frames[:-30]:
                        stale.unlink(missing_ok=True)
                except OSError:
                    pass
            time.sleep(min(remaining, 0.25))
        if ctx.radar_reader_thread is not None:
            ctx.radar_reader_thread.join(timeout=2)
        stop_radar_capture(ctx.radar)
        ctx.radar = None
        if ctx.radar_lock is not None:
            settle_radar_gpio()
        release_radar_lock(ctx.radar_lock)
        ctx.radar_lock = None

    except KeyboardInterrupt:
        manifest["errors"].append("Interrupted by user.")
    except Exception as exc:
        manifest["errors"].append(str(exc))
    finally:
        stop_radar_capture(ctx.radar)
        release_radar_lock(ctx.radar_lock)
        ctx.csi_stop.set()
        for csi_thread in ctx.csi_threads:
            csi_thread.join(timeout=5)
        if ctx.sense_thread is not None:
            ctx.sense_thread.join(timeout=5)
        if ctx.radar_analysis_thread is not None:
            ctx.analysis_queue.put(None)
            ctx.radar_analysis_thread.join(timeout=90.0)
            if ctx.radar_analysis_thread.is_alive():
                manifest["errors"].append("Radar analysis exceeded its shutdown deadline.")
        ctx.live_features_stop.set()
        if ctx.radar_live_thread is not None:
            ctx.live_analysis_queue.put(None)
            ctx.radar_live_thread.join(timeout=5.0)
        if ctx.camera_thread is not None:
            ctx.camera_thread.join(timeout=30.0)
        if ctx.radar_upload_thread is not None:
            ctx.upload_queue.put(None)
            ctx.radar_upload_thread.join(timeout=15.0)
        if not live_only and not ctx.radar_chunk_results and ctx.model_registry.list():
            # Make sensor outages visible in every enabled model timeline.
            try:
                skipped = ctx.model_registry.run_enabled([], current_csi_samples(ctx), 0, iso_now())
                if skipped:
                    with ctx.publish_lock:
                        timelines = manifest.setdefault("model_predictions", [])
                        by_id = {str(item.get("model_id")): item for item in timelines if isinstance(item, dict)}
                        for prediction in skipped:
                            model_id = str(prediction.get("model_id"))
                            timeline = by_id.setdefault(model_id, {"model_id": model_id, "model_name": prediction.get("model_name"), "model_version": prediction.get("model_version"), "timeline": []})
                            timeline["timeline"].append(prediction)
                        manifest["model_predictions"] = list(by_id.values())
                        write_live_manifest(ctx)
            except Exception as exc:
                manifest["errors"].append(f"Unable to record model sensor status: {exc}")
        if ctx.model_thread is not None:
            ctx.model_queue.put(None)
            ctx.model_thread.join(timeout=90.0)
            if ctx.model_thread.is_alive():
                manifest["errors"].append("User model inference exceeded its shutdown deadline.")
        try:
            ctx.partial_minute_queue.get_nowait()
            ctx.partial_minute_queue.task_done()
        except queue.Empty:
            pass
        try:
            ctx.partial_minute_queue.put_nowait(None)
        except queue.Full:
            pass
        current_labels = effective_preset_labels(ctx)
        manifest["preset_labels"] = current_labels
        manifest["labels"] = current_labels
        manifest["primary_label"] = current_labels[0] if current_labels else None
        if not live_only and not any(output_dir.glob("radar_*.bin")) and not args.no_radar:
            manifest["errors"].append("Radar produced no complete 10-frame chunks for this minute.")

        radar_files = sorted(str(path) for path in output_dir.glob("radar_*.bin"))
        if radar_files:
            manifest["outputs"].setdefault("radar", {})["files"] = radar_files
        radar_span = (
            ctx.radar_last_frame_at - ctx.radar_first_frame_at
            if ctx.radar_first_frame_at is not None and ctx.radar_last_frame_at is not None
            else 0.0
        )
        radar_rate = (
            (ctx.radar_frame_count - 1) / radar_span
            if ctx.radar_frame_count > 1 and radar_span > 0
            else 0.0
        )
        if "radar" in manifest["outputs"]:
            manifest["outputs"]["radar"].update({
                "sample_count": ctx.radar_frame_count,
                "average_sampling_rate_hz": round(radar_rate, 3),
                "observed_span_seconds": round(radar_span, 3),
            })
        camera_files = sorted(str(path) for path in output_dir.glob("camera_*.jpg"))
        if camera_files:
            manifest["outputs"].setdefault("camera", {})["files"] = camera_files
        if ctx.camera is not None and len(camera_files) != ctx.expected_chunks:
            manifest["warnings"].append(
                f"Camera captured {len(camera_files)} of {ctx.expected_chunks} synchronized second images."
            )

        wifi_csi = manifest["outputs"].get("wifi_csi")
        if isinstance(wifi_csi, dict):
            receivers = wifi_csi.get("receivers") if isinstance(wifi_csi.get("receivers"), list) else [wifi_csi]
            for receiver in receivers:
                if not isinstance(receiver, dict) or not receiver.get("path"):
                    continue
                path = Path(str(receiver["path"]))
                receiver["device_id"] = str(
                    (load_processing_settings().get("csi_device_ids") or {}).get(str(receiver.get("device")))
                    or receiver.get("device_id")
                    or "csi"
                )
                if not path.exists() and not args.no_csi:
                    with open(path, "w", encoding="utf-8", newline="") as output_fd:
                        csv.writer(output_fd).writerow(["host_timestamp", "monotonic_ns", "serial_port", "raw_csi_line"])
                    receiver["data_quality"] = "capture_error"
                    manifest["errors"].append(
                        f"CSI capture failed for {receiver.get('device')}; wrote a header-only minute CSV."
                    )
                if path.exists():
                    receiver["bytes"] = path.stat().st_size
                    receiver.update(csi_capture_stats(path))
                if path.exists() and receiver.get("sample_count", 0) == 0:
                    manifest["warnings"].append(
                        f"ESP32 CSI receiver {receiver.get('device')} produced no CSI_DATA samples this minute."
                    )
            if receivers:
                primary = receivers[0]
                for key in ("bytes", "sample_count", "average_sampling_rate_hz", "observed_span_seconds"):
                    if key in primary:
                        wifi_csi[key] = primary[key]
            wifi_csi["receivers"] = receivers
        manifest["sampling_rates_hz"] = {
            "radar": round(radar_rate, 3),
            "wifi_csi": {
                str(receiver.get("device_id") or receiver.get("device") or index): receiver.get("average_sampling_rate_hz", 0.0)
                for index, receiver in enumerate(
                    (wifi_csi.get("receivers") if isinstance(wifi_csi, dict) and isinstance(wifi_csi.get("receivers"), list) else []),
                    start=1,
                )
                if isinstance(receiver, dict)
            },
        }
        manifest["expected_chunks"] = len(ctx.radar_chunk_results)
        manifest["progress"] = {
            "unit": "second",
            "captured_seconds": len(ctx.radar_chunk_results),
            "total_seconds": ctx.expected_chunks,
            "percent": round(100.0 * len(ctx.radar_chunk_results) / max(1, ctx.expected_chunks), 1),
        }

        # Surface silent capture loss: a healthy radar minute yields ~10 fps,
        # so anything under 2 fps means most frames were dropped (e.g. FIFO
        # starvation) even though no hard error was raised.
        if not args.no_radar and radar_rate < 2.0:
            manifest["degraded"] = True
            manifest["warnings"].append(
                f"Radar yield degraded: {ctx.radar_frame_count} frames at {radar_rate:.2f} fps "
                "(expected ~10 fps); check FIFO/GSR errors in the collector log."
            )

        manifest["capture_started"] = ctx.capture_started
        manifest["capture_finished"] = iso_now()
        manifest["host"] = os.uname().nodename
        manifest["status"] = "success" if not manifest["errors"] else "partial" if manifest["warnings"] else "error"
        manifest_file = output_dir / "manifest.json"
        merge_home_assistant_status(ctx)

        # Minute-level occupancy models (E2 exports): one verdict per minute.
        # Runs while the radar_*.bin / wifi_csi_*.csv fragments are still on
        # disk so results are also embedded in the container manifest.
        minute_results: list[dict[str, Any]] = []
        if not live_only:
            try:
                minute_frames, minute_frame_times = _minute_radar_frames(output_dir, manifest)
                minute_results = ctx.model_registry.run_minute(
                    minute_frames,
                    minute_frame_times,
                    _minute_csi_samples(manifest),
                    iso_now(),
                )
            except Exception as exc:
                minute_results = []
                manifest["errors"].append(f"Minute-level model inference failed: {exc}")
        if minute_results:
            fire_model_device_links(ctx, minute_results, "minute")
            occupancy_results = [
                item for item in minute_results
                if item.get("status") == "ok" and is_occupancy_result(item)
            ]
            if occupancy_results:
                selected = max(occupancy_results, key=lambda item: float(item.get("confidence") or 0.0))
                publish_model_occupancy(selected, ctx.folder_name, chunk_index=None)
            with ctx.publish_lock:
                timelines = manifest.setdefault("model_predictions", [])
                by_id = {str(item.get("model_id")): item for item in timelines if isinstance(item, dict)}
                for prediction in minute_results:
                    model_id = str(prediction.get("model_id"))
                    timeline = by_id.setdefault(model_id, {
                        "model_id": model_id,
                        "model_name": prediction.get("model_name"),
                        "model_version": prediction.get("model_version"),
                        "timeline": [],
                    })
                    timeline["timeline"].append(prediction)
                manifest["model_predictions"] = list(by_id.values())

        # Voted minute verdict for chunk-execution models: each model ran on its
        # own sliding window through the minute; votes aggregate to one verdict.
        try:
            voted = ctx.windowed_analyzer.minute_predictions()
        except Exception:
            voted = []
        if voted:
            manifest["minute_predictions"] = voted
            fire_model_device_links(ctx, voted, "minute")
            occupancy_votes = [
                item for item in voted
                if item.get("status") == "ok" and is_occupancy_result(item)
            ]
            if occupancy_votes:
                selected = max(occupancy_votes, key=lambda item: float(item.get("confidence") or 0.0))
                publish_model_occupancy(selected, ctx.folder_name, chunk_index=None)

        if live_only:
            manifest["container"] = {"skipped": "live-only mode"}
        else:
            try:
                manifest["container"] = build_capture_container(output_dir, manifest, remove_fragments=True)
                manifest["assets"] = [
                    {
                        "sensor": "synchronized_capture",
                        "filename": "capture.npz",
                        "content_type": "application/x-npz",
                        "coverage": "all synchronized sensor samples",
                        "labels": manifest.get("labels") or [],
                    },
                    *([{
                        "sensor": "radar_tracking",
                        "filename": "xy-tracking.json",
                        "content_type": "application/json",
                        "labels": manifest.get("labels") or [],
                    }] if (output_dir / "xy-tracking.json").exists() else []),
                ]
            except Exception as exc:
                manifest["errors"].append(f"Synchronized container finalization failed: {exc}")
        manifest["status"] = "success" if not manifest["errors"] else "partial" if manifest["warnings"] else "error"
        ctx.manifest = compact_manifest(manifest)
        write_json_atomic(manifest_file, ctx.manifest)

        chown_to_invoking_user(output_dir)

    print(f"Capture finished: {ctx.manifest['capture_finished']}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    if ctx.manifest["warnings"]:
        print("Warnings:")
        for warning in ctx.manifest["warnings"]:
            print(f"  - {warning}")
    if ctx.manifest["errors"]:
        print("Errors:")
        for error in ctx.manifest["errors"]:
            print(f"  - {error}")
        return 1
    return 0
