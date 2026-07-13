#!/usr/bin/env python3
"""Capture one synchronized minute of Thoth sensor data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
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
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.config import Config  # type: ignore
    from backend.radar_analysis import StreamingChunkAnalyzer, compile_minute_xy_payload, create_signal_processor, load_room_config  # type: ignore
else:
    from .config import Config
    from .radar_analysis import StreamingChunkAnalyzer, compile_minute_xy_payload, create_signal_processor, load_room_config

THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
RADAR_CFG = MMW_RELEASE / "radar_config" / "config_3rx_3m"
DATA_ROOT = Path(Config.CAPTURE_DATA_DIR).expanduser()
CAPTURE_SETTINGS_PATH = Path(Config.CONFIG_DIR).expanduser() / "capture_settings.json"
CSI_HEADER = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"

sys.path.insert(0, str(MMW_RELEASE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one synchronized Thoth sensor minute into the configured data folder."
    )
    parser.add_argument("--duration", type=float, default=59.5, help="Capture duration in seconds.")
    parser.add_argument("--camera", default=None, help="USB camera device, for example /dev/video0.")
    parser.add_argument("--csi-port", default="auto", help="ESP32 CSI receiver serial port, or 'auto'.")
    parser.add_argument("--csi-baud", type=int, default=115200, help="ESP32 CSI receiver baud rate.")
    parser.add_argument(
        "--csi-detect-seconds",
        type=float,
        default=0.0,
        help="Optional seconds to probe serial ports for CSI_DATA in auto mode. Default avoids pre-opening the ESP32.",
    )
    parser.add_argument("--no-camera", action="store_true", help="Skip USB camera capture.")
    parser.add_argument("--no-radar", action="store_true", help="Skip mmWave radar capture.")
    parser.add_argument("--no-csi", action="store_true", help="Skip ESP32 CSI serial capture.")
    parser.add_argument("--no-sensehat", action="store_true", help="Skip Sense HAT capture.")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=10.0,
        help="Radar capture and analysis chunk size in seconds.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Preset one or more labels for the captured minute. Repeat the flag to add multiple labels.",
    )
    parser.add_argument(
        "--start-now",
        action="store_true",
        help="Start immediately and name the folder from the current real-clock minute.",
    )
    return parser.parse_args()


def normalize_labels(labels: object) -> list[str]:
    if isinstance(labels, str):
        items = labels.split(",")
    elif isinstance(labels, list):
        items = labels
    else:
        items = []

    cleaned: list[str] = []
    for item in items:
        label = str(item or "").strip().replace("/", "_").replace("\\", "_")
        label = " ".join(label.split())
        if label and label not in cleaned:
            cleaned.append(label)
    return cleaned


def output_dir_for_minute(folder_name: str, labels: list[str]) -> Path:
    root = DATA_ROOT
    if labels:
        return root / labels[0] / folder_name
    return root / folder_name


def minute_start(start_now: bool) -> dt.datetime:
    now = dt.datetime.now().astimezone()
    current_minute = now.replace(second=0, microsecond=0)
    if start_now:
        return current_minute
    if now.second == 0 and now.microsecond < 250_000:
        return current_minute
    return current_minute + dt.timedelta(minutes=1)


def sleep_until(target: dt.datetime) -> None:
    while True:
        remaining = target.timestamp() - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary, "w", encoding="utf-8") as fd:
        json.dump(payload, fd, indent=2)
    temporary.replace(path)


def load_processing_settings() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "radar_detection_threshold_db": 8.0,
        "occupancy_threshold_percent": 50.0,
        "auto_occupancy_label_enabled": True,
        "revision": 0,
        "updated_at": None,
    }
    try:
        loaded = json.loads(CAPTURE_SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            defaults.update({key: loaded[key] for key in defaults if key in loaded})
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Unable to load processing settings: {exc}", file=sys.stderr)
    defaults["radar_detection_threshold_db"] = min(40.0, max(0.0, float(defaults["radar_detection_threshold_db"])))
    defaults["occupancy_threshold_percent"] = min(100.0, max(0.0, float(defaults["occupancy_threshold_percent"])))
    defaults["auto_occupancy_label_enabled"] = bool(defaults["auto_occupancy_label_enabled"])
    return defaults


def find_camera(requested: str | None) -> str | None:
    if requested:
        return requested

    devices = sorted(
        glob.glob("/dev/video*"),
        key=lambda path: int(Path(path).name.removeprefix("video")),
    )
    usb_devices = []
    for device in devices:
        sysfs_device = Path("/sys/class/video4linux") / Path(device).name / "device"
        try:
            resolved = sysfs_device.resolve(strict=True)
        except OSError:
            continue
        if any((parent / "idVendor").exists() for parent in (resolved, *resolved.parents)):
            usb_devices.append(device)

    v4l2_ctl = shutil.which("v4l2-ctl")
    if v4l2_ctl:
        for device in usb_devices:
            try:
                probe = subprocess.run(
                    [v4l2_ctl, "--device", device, "--all"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            is_usb_camera = "Driver name      : uvcvideo" in probe.stdout or "Bus info         : usb-" in probe.stdout
            is_capture_node = "Video Capture" in probe.stdout and "Video Output" not in probe.stdout
            if probe.returncode == 0 and is_usb_camera and is_capture_node:
                return device
    return None


def serial_candidates() -> list[str]:
    try:
        import serial.tools.list_ports as list_ports

        ports = [port.device for port in list_ports.comports()]
    except Exception:
        ports = []

    globbed = glob.glob("/dev/serial/by-id/*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    candidates = []
    for port in ports + globbed:
        resolved = str(Path(port).resolve()) if port.startswith("/dev/serial/by-id/") else port
        if resolved not in candidates:
            candidates.append(resolved)
    return sorted(candidates)


def open_serial_without_reset(port: str, baud: int, timeout: float):
    import serial

    connection = serial.Serial()
    connection.port = port
    connection.baudrate = baud
    connection.timeout = timeout
    # ESP32-C6 USB Serial/JTAG reboots when DTR is deasserted as the port opens.
    # Keep DTR asserted and RTS deasserted so rotating minute files does not
    # reset the CSI receiver firmware.
    connection.dtr = True
    connection.rts = False
    connection.open()
    return connection


def probe_csi_port(port: str, baud: int, timeout_s: float) -> bool:
    try:
        import serial

        deadline = time.monotonic() + timeout_s
        with open_serial_without_reset(port, baud, 0.1) as ser:
            while time.monotonic() < deadline:
                line = ser.readline()
                if not line:
                    continue
                text = line.decode("utf-8", errors="ignore").strip()
                if text.startswith("CSI_DATA,"):
                    return True
    except Exception:
        return False
    return False


def find_csi_port(requested: str, baud: int, detect_seconds: float) -> tuple[str | None, list[str]]:
    candidates = serial_candidates()
    if requested and requested != "auto":
        return requested, candidates

    try:
        import serial.tools.list_ports as list_ports

        for port in list_ports.comports():
            manufacturer = (port.manufacturer or "").lower()
            description = (port.description or "").lower()
            if port.vid == 0x303A or "espressif" in manufacturer or "espressif" in description:
                return port.device, candidates
    except Exception:
        pass

    if detect_seconds <= 0:
        return (candidates[0] if candidates else None), candidates

    per_port_timeout = max(0.2, detect_seconds / max(1, len(candidates)))
    for port in candidates:
        if probe_csi_port(port, baud, per_port_timeout):
            return port, candidates
    return (candidates[0] if candidates else None), candidates


def collect_csi(
    port: str,
    baud: int,
    raw_file: Path,
    timestamped_file: Path,
    all_serial_file: Path,
    stop_event: threading.Event,
) -> None:
    try:
        import serial
    except Exception as exc:
        with open(timestamped_file.with_suffix(".error.json"), "w", encoding="utf-8") as fd:
            json.dump({"timestamp": iso_now(), "error": f"pyserial import failed: {exc}"}, fd, indent=2)
        return

    try:
        with (
            open_serial_without_reset(port, baud, 0.05) as ser,
            open(raw_file, "w", encoding="utf-8", buffering=1) as raw_fd,
            open(timestamped_file, "w", encoding="utf-8", newline="", buffering=1) as ts_fd,
            open(all_serial_file, "w", encoding="utf-8", buffering=1) as all_fd,
        ):
            raw_fd.write(CSI_HEADER + "\n")
            writer = csv.writer(ts_fd)
            writer.writerow(["host_timestamp", "monotonic_ns", "serial_port", "raw_csi_line"])

            if ser.in_waiting:
                ser.read(ser.in_waiting)

            while not stop_event.is_set():
                line = ser.readline()
                if not line:
                    continue
                host_timestamp = iso_now()
                monotonic_ns = time.monotonic_ns()
                text = line.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                all_fd.write(json.dumps({
                    "host_timestamp": host_timestamp,
                    "monotonic_ns": monotonic_ns,
                    "serial_port": port,
                    "line": text,
                }, separators=(",", ":")) + "\n")
                if text.startswith("CSI_DATA,"):
                    raw_fd.write(text + "\n")
                    writer.writerow([host_timestamp, monotonic_ns, port, text])
    except Exception as exc:
        with open(timestamped_file.with_suffix(".error.json"), "w", encoding="utf-8") as fd:
            json.dump({"timestamp": iso_now(), "port": port, "baud": baud, "error": str(exc)}, fd, indent=2)


def collect_sensehat(output_file: Path, stop_event: threading.Event, interval: float = 0.2) -> None:
    try:
        from sense_hat import SenseHat
    except Exception as exc:
        output_file.with_suffix(".error.json").write_text(
            json.dumps({"timestamp": iso_now(), "error": f"sense_hat import failed: {exc}"}, indent=2),
            encoding="utf-8",
        )
        return

    try:
        sense = SenseHat()
        with open(output_file, "w", encoding="utf-8", buffering=1) as fd:
            while not stop_event.is_set():
                row = {
                    "host_timestamp": iso_now(),
                    "monotonic_ns": time.monotonic_ns(),
                    "temperature_c": sense.get_temperature(),
                    "humidity_percent": sense.get_humidity(),
                    "pressure_mbar": sense.get_pressure(),
                    "acceleration": sense.get_accelerometer_raw(),
                    "gyroscope": sense.get_gyroscope_raw(),
                    "compass": sense.get_compass_raw(),
                    "orientation": sense.get_orientation(),
                }
                fd.write(json.dumps(row, separators=(",", ":")) + "\n")
                time.sleep(max(0.05, interval))
    except Exception as exc:
        output_file.with_suffix(".error.json").write_text(
            json.dumps({"timestamp": iso_now(), "error": str(exc)}, indent=2),
            encoding="utf-8",
        )


def start_video_capture(camera: str, output_file: Path, duration: float) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH.")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "v4l2",
        "-i",
        camera,
        "-t",
        f"{duration:.3f}",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(output_file),
    ]
    log_file = output_file.with_suffix(".ffmpeg.log")
    log_fd = open(log_file, "wb")
    proc = subprocess.Popen(cmd, stdout=log_fd, stderr=subprocess.STDOUT)
    proc._dreamhat_log_fd = log_fd  # type: ignore[attr-defined]
    return proc


def stop_video_capture(proc: subprocess.Popen | None, timeout: float = 10.0) -> int | None:
    if proc is None:
        return None
    if proc.poll() is None:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
    log_fd = getattr(proc, "_dreamhat_log_fd", None)
    if log_fd is not None:
        log_fd.close()
    return proc.returncode


def start_radar_capture(output_prefix: Path) -> Any:
    if not Path("/dev/spidev0.0").exists():
        raise RuntimeError("/dev/spidev0.0 is missing; enable SPI and reboot the Raspberry Pi.")

    from utility.BGT60TR13C import BGT60TR13C, RET_VAL_OK
    from utility.helper import calculate_frame_size, find_register_config_in_directory, find_setting_in_directory

    bgt60tr13c = None
    try:
        bgt60tr13c = BGT60TR13C(spi_speed=50_000_000, save_to_file=str(output_prefix))
        if bgt60tr13c.check_chip_id() != RET_VAL_OK:
            raise RuntimeError("BGT60TR13C chip ID check failed.")

        reg_file = find_register_config_in_directory(str(RADAR_CFG))
        setting_file = find_setting_in_directory(str(RADAR_CFG))
        bgt60tr13c.load_register_config_file(reg_file)

        with open(setting_file, "r", encoding="utf-8") as fd:
            setting_data = json.load(fd)

        frame_size = calculate_frame_size(setting_data)
        bgt60tr13c.set_fifo_parameters(frame_size, 4096, 2048)
        if bgt60tr13c.start() != RET_VAL_OK:
            raise RuntimeError("BGT60TR13C failed to start.")

        return bgt60tr13c
    except Exception:
        stop_radar_capture(bgt60tr13c)
        raise


def stop_radar_capture(radar: Any | None) -> None:
    if radar is not None:
        radar.stop()


def chown_to_invoking_user(path: Path) -> None:
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if not sudo_uid or not sudo_gid:
        return
    uid = int(sudo_uid)
    gid = int(sudo_gid)
    for root, dirs, files in os.walk(path):
        os.chown(root, uid, gid)
        for name in dirs:
            os.chown(os.path.join(root, name), uid, gid)
        for name in files:
            os.chown(os.path.join(root, name), uid, gid)


def main() -> int:
    args = parse_args()
    preset_labels = normalize_labels(args.label or [])
    chunk_seconds = max(1.0, float(args.chunk_seconds))
    target_start = minute_start(args.start_now)
    folder_name = target_start.strftime("%Y%m%d_%H%M")
    output_dir = output_dir_for_minute(folder_name, preset_labels)
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "folder_minute": folder_name,
        "scheduled_start": target_start.isoformat(timespec="seconds"),
        "duration_seconds": args.duration,
        "chunk_seconds": chunk_seconds,
        "expected_chunks": max(1, int(math.ceil(args.duration / chunk_seconds))),
        "labels": preset_labels,
        "outputs": {},
        "errors": [],
        "warnings": [],
        "primary_label": preset_labels[0] if preset_labels else None,
        "relative_path": str(output_dir.relative_to(DATA_ROOT)),
        "sensors_enabled": {
            "usb_camera": not args.no_camera,
            "dreamhat_radar": not args.no_radar,
            "esp32_csi": not args.no_csi,
            "sense_hat": not args.no_sensehat,
        },
    }

    csi_port = None
    csi_candidates: list[str] = []
    if not args.no_csi:
        csi_port, csi_candidates = find_csi_port(args.csi_port, args.csi_baud, args.csi_detect_seconds)
        if csi_port is None:
            manifest["errors"].append("No ESP32 CSI serial device found.")

    print(f"Output folder: {output_dir}")
    print(f"Waiting for real-clock minute: {target_start.isoformat(timespec='seconds')}")
    sleep_until(target_start)

    stop_at = time.monotonic() + args.duration
    capture_started = iso_now()
    print(f"Capture started: {capture_started}")

    video_proc: subprocess.Popen | None = None
    csi_stop = threading.Event()
    csi_thread: threading.Thread | None = None
    sense_stop = threading.Event()
    sense_thread: threading.Thread | None = None
    radar_analysis_thread: threading.Thread | None = None
    radar_upload_thread: threading.Thread | None = None
    # Hold at most one scheduled minute of frames. This remains bounded while
    # ensuring signal processing can never back-pressure the SPI reader.
    analysis_queue: queue.Queue[Any] = queue.Queue(maxsize=768)
    upload_queue: queue.Queue[Any] = queue.Queue(maxsize=6)
    radar_chunk_results: list[dict[str, Any]] = []
    publish_lock = threading.Lock()
    room_config = load_room_config()

    def merge_home_assistant_status() -> None:
        status_path = output_dir / ".home_assistant_status.json"
        try:
            statuses = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return
        chunks = (((manifest.get("outputs") or {}).get("radar") or {}).get("chunks") or [])
        for entry in chunks:
            status = statuses.get(str(entry.get("chunk_index"))) if isinstance(statuses, dict) else None
            if isinstance(status, dict):
                entry["home_assistant"] = status

    def write_live_manifest() -> None:
        merge_home_assistant_status()
        snapshot = dict(manifest)
        snapshot["capture_started"] = capture_started
        snapshot["status"] = "collecting"
        outputs = dict(manifest.get("outputs") or {})
        radar_output = outputs.get("radar")
        if isinstance(radar_output, dict):
            radar_snapshot = dict(radar_output)
            radar_snapshot["chunks"] = [
                {key: value for key, value in entry.items() if key != "result"}
                for entry in (radar_output.get("chunks") or [])
            ]
            outputs["radar"] = radar_snapshot
        snapshot["outputs"] = outputs
        write_json_atomic(output_dir / "manifest.json", snapshot)

    def publish_radar_results() -> None:
        with publish_lock:
            completed = [entry["result"] for entry in radar_chunk_results if isinstance(entry.get("result"), dict)]
            if not completed:
                return
            xy_payload = compile_minute_xy_payload(completed)
            timeline = [
                {
                    "model_name": "xy-tracking",
                    "chunk_index": chunk.get("chunk_index"),
                    "chunk_seconds": chunk.get("chunk_seconds"),
                    "prediction": chunk.get("occupancy", {}).get("label"),
                    "occupied": chunk.get("occupancy", {}).get("label") == "occupied",
                    "location": chunk.get("location"),
                    "score": chunk.get("score"),
                    "target_count": len(chunk.get("targets") or []),
                    "targets": chunk.get("targets") or [],
                    "detected_frames": chunk.get("occupancy", {}).get("detected_frames", 0),
                    "evaluated_frames": chunk.get("occupancy", {}).get("evaluated_frames", 0),
                    "ratio": chunk.get("occupancy", {}).get("ratio", 0.0),
                    "bin_path": chunk.get("bin_path"),
                    "csv_path": chunk.get("csv_path"),
                }
                for chunk in sorted(completed, key=lambda item: int(item.get("chunk_index", 0)))
            ]
            write_json_atomic(output_dir / "xy-tracking.json", xy_payload)
            write_json_atomic(output_dir / "predictions.json", {
                "generated_at": iso_now(),
                "chunk_seconds": chunk_seconds,
                "labels": preset_labels,
                "timeline": timeline,
                "summary": xy_payload.get("occupancy", {}),
            })
            write_live_manifest()

    def upload_live_chunk(index: int) -> None:
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:5000/api/captures/{folder_name}/upload?incremental=1&chunk={index}",
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120):
                pass
        except Exception as exc:
            print(f"Live chunk {index} upload deferred: {exc}", file=sys.stderr)

    def enqueue_home_assistant(entry: dict[str, Any], occupancy: dict[str, Any], result: dict[str, Any]) -> None:
        payload = json.dumps({
            "minute": folder_name,
            "chunk_index": entry["chunk_index"],
            "occupancy": occupancy,
            "location": result.get("location"),
            "confidence": result.get("score"),
            "timestamp": entry["finished"],
        }).encode("utf-8")
        try:
            request = urllib.request.Request(
                "http://127.0.0.1:5000/api/internal/home-assistant/publish",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=1.0):
                pass
            entry["home_assistant"] = {"success": True, "status": "queued", "updated_at": iso_now()}
        except Exception as exc:
            entry["home_assistant"] = {"success": False, "status": "queue_error", "error": str(exc), "updated_at": iso_now()}

    def run_upload_worker() -> None:
        while True:
            index = upload_queue.get()
            try:
                if index is None:
                    return
                upload_live_chunk(int(index))
            finally:
                upload_queue.task_done()

    def run_analysis_worker() -> None:
        try:
            processor = create_signal_processor()
        except Exception as init_exc:
            failed_entry: dict[str, Any] | None = None
            while True:
                failed_job = analysis_queue.get()
                try:
                    if failed_job is None:
                        return
                    if failed_job[0] == "start":
                        failed_entry = failed_job[1]
                    elif failed_job[0] == "end" and failed_entry is not None:
                        failed_entry.update({"status": "error", "error": str(init_exc), "finished": iso_now()})
                        with publish_lock:
                            write_live_manifest()
                finally:
                    analysis_queue.task_done()
        analyzer: StreamingChunkAnalyzer | None = None
        entry: dict[str, Any] | None = None
        settings_snapshot: dict[str, Any] = {}
        while True:
            job = analysis_queue.get()
            try:
                if job is None:
                    return
                kind = job[0]
                if kind == "start":
                    entry, settings_snapshot = job[1], job[2]
                    analyzer = StreamingChunkAnalyzer(
                        processor, Path(entry["csv_path"]), int(entry["chunk_index"]),
                        float(entry["chunk_seconds"]), room_config,
                        float(settings_snapshot["radar_detection_threshold_db"]),
                        float(settings_snapshot["occupancy_threshold_percent"]),
                    )
                    continue
                if kind == "frame":
                    if analyzer is not None:
                        analyzer.max_queue_lag_ms = max(analyzer.max_queue_lag_ms, (time.monotonic() - float(job[2])) * 1000)
                        analyzer.process(job[1])
                    continue
                if kind != "end" or analyzer is None or entry is None:
                    continue
                entry["status"] = "analyzing"
                with publish_lock:
                    write_live_manifest()
                result = analyzer.finish()
                result["bin_path"] = entry.get("bin_path")
                result["csv_path"] = entry.get("csv_path")
                entry["result"] = result
                occupancy = result.get("occupancy", {})
                label = occupancy.get("label") or "empty"
                entry.update({
                    "status": "error" if entry.get("capture_error") else label,
                    "detected_frames": occupancy.get("detected_frames", 0),
                    "evaluated_frames": occupancy.get("evaluated_frames", 0),
                    "ratio": occupancy.get("ratio", 0.0),
                    "occupied": label == "occupied",
                    "location": result.get("location"),
                    "score": result.get("score"),
                    "performance": result.get("performance"),
                    "finished": iso_now(),
                })
                if settings_snapshot.get("auto_occupancy_label_enabled"):
                    manifest["auto_occupancy_label"] = occupancy
                publish_radar_results()
                enqueue_home_assistant(entry, occupancy, result)
                try:
                    upload_queue.put_nowait(int(entry["chunk_index"]))
                except queue.Full:
                    entry["cloud_upload"] = {"status": "deferred", "error": "upload queue is full"}
                analyzer = None
                entry = None
            except Exception as exc:
                if entry is not None:
                    entry["error"] = str(exc)
                    entry.update({"status": "error", "finished": iso_now()})
                    with publish_lock:
                        write_live_manifest()
                if analyzer is not None:
                    try:
                        analyzer.handle.close()
                    except Exception:
                        pass
                analyzer = None
                entry = None
            finally:
                analysis_queue.task_done()

    try:
        if not args.no_csi and csi_port is not None:
            csi_raw_file = output_dir / "wifi_csi_raw.csv"
            csi_timestamped_file = output_dir / "wifi_csi_timestamped.csv"
            csi_all_serial_file = output_dir / "wifi_csi_serial_all.jsonl"
            csi_started = iso_now()
            csi_thread = threading.Thread(
                target=collect_csi,
                args=(csi_port, args.csi_baud, csi_raw_file, csi_timestamped_file, csi_all_serial_file, csi_stop),
                daemon=True,
            )
            csi_thread.start()
            manifest["outputs"]["wifi_csi"] = {
                "raw_path": str(csi_raw_file),
                "timestamped_path": str(csi_timestamped_file),
                "all_serial_path": str(csi_all_serial_file),
                "type": "csv/jsonl",
                "device": csi_port,
                "baud": args.csi_baud,
                "detected_candidates": csi_candidates,
                "started": csi_started,
                "source": "USB-connected ESP32 receiver printing ESP-NOW CSI_DATA lines",
            }
        elif not args.no_csi and csi_port is None:
            manifest["warnings"].append("No ESP32 CSI serial device found; skipping CSI for this minute.")

        if not args.no_camera:
            camera = find_camera(args.camera)
            if camera is None:
                manifest["warnings"].append("No /dev/video* USB camera device found; skipping video for this minute.")
            else:
                video_file = output_dir / "usb_camera.mp4"
                video_started = iso_now()
                video_proc = start_video_capture(camera, video_file, args.duration)
                manifest["outputs"]["video"] = {
                    "path": str(video_file),
                    "type": "mp4",
                    "device": camera,
                    "started": video_started,
                }

        if not args.no_sensehat:
            sense_file = output_dir / "sense_hat.jsonl"
            sense_started = iso_now()
            sense_thread = threading.Thread(
                target=collect_sensehat,
                args=(sense_file, sense_stop),
                daemon=True,
            )
            sense_thread.start()
            manifest["outputs"]["sense_hat"] = {
                "path": str(sense_file),
                "type": "jsonl",
                "source": "Sense HAT GPIO / I2C",
                "started": sense_started,
            }

        if not args.no_radar:
            manifest["outputs"]["radar"] = {
                "type": "chunked-bin+csv",
                "config_dir": str(RADAR_CFG),
                "chunk_seconds": chunk_seconds,
                "chunks": [],
                "note": "Radar is captured and analyzed in 10-second windows so each chunk gets its own .bin and .csv pair.",
            }
            with publish_lock:
                write_live_manifest()
            radar_analysis_thread = threading.Thread(target=run_analysis_worker, name="RadarAnalysis", daemon=True)
            radar_upload_thread = threading.Thread(target=run_upload_worker, name="RadarUpload", daemon=True)
            radar_analysis_thread.start()
            radar_upload_thread.start()
            for chunk_index in range(6):
                remaining = stop_at - time.monotonic()
                if remaining <= 0:
                    break
                current_chunk_seconds = min(chunk_seconds, remaining)
                radar_prefix = output_dir / f"mmw_radar_raw_{chunk_index:02d}"
                radar_started = iso_now()
                settings_snapshot = load_processing_settings()
                radar = None
                csv_path = output_dir / f"mmw_radar_xy_{chunk_index:02d}.csv"
                chunk_entry: dict[str, Any] = {
                    "chunk_index": chunk_index,
                    "bin_path": "",
                    "csv_path": str(csv_path),
                    "started": radar_started,
                    "chunk_seconds": current_chunk_seconds,
                    "status": "collecting",
                    "settings": settings_snapshot,
                }
                radar_chunk_results.append(chunk_entry)
                manifest["outputs"]["radar"]["chunks"].append(chunk_entry)
                with publish_lock:
                    write_live_manifest()
                analysis_queue.put(("start", chunk_entry, settings_snapshot))
                try:
                    radar = start_radar_capture(radar_prefix)
                    chunk_stop = time.monotonic() + current_chunk_seconds
                    while True:
                        remaining_chunk = chunk_stop - time.monotonic()
                        if remaining_chunk <= 0:
                            break
                        try:
                            analysis_queue.put(("frame", bytes(radar.frame_buffer.get(timeout=min(remaining_chunk, 0.1))), time.monotonic()))
                        except queue.Empty:
                            pass
                except Exception as exc:
                    manifest["warnings"].append(f"Radar chunk {chunk_index} failed to start: {exc}")
                    chunk_entry["error"] = str(exc)
                finally:
                    stop_radar_capture(radar)

                if radar is not None:
                    while True:
                        try:
                            analysis_queue.put(("frame", bytes(radar.frame_buffer.get_nowait()), time.monotonic()))
                        except queue.Empty:
                            break

                radar_files = sorted(output_dir.glob(f"mmw_radar_raw_{chunk_index:02d}_*.bin"))
                if not radar_files:
                    chunk_entry.update({"status": "error", "capture_error": True, "error": chunk_entry.get("error") or "Radar binary was not created"})
                    with publish_lock:
                        write_live_manifest()
                else:
                    radar_path = radar_files[-1]
                    chunk_entry["bin_path"] = str(radar_path)
                    chunk_entry["status"] = "stored"
                with publish_lock:
                    write_live_manifest()
                analysis_queue.put(("end", chunk_entry))

        while True:
            remaining = stop_at - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.25))

    except KeyboardInterrupt:
        manifest["errors"].append("Interrupted by user.")
    except Exception as exc:
        manifest["errors"].append(str(exc))
    finally:
        csi_stop.set()
        if csi_thread is not None:
            csi_thread.join(timeout=5)
        sense_stop.set()
        if sense_thread is not None:
            sense_thread.join(timeout=5)
        if radar_analysis_thread is not None:
            analysis_queue.put(None)
            radar_analysis_thread.join()
        if radar_upload_thread is not None:
            try:
                upload_queue.put_nowait(None)
            except queue.Full:
                pass
            radar_upload_thread.join(timeout=0.2)
        completed_chunks: list[dict[str, Any]] = []
        for chunk in radar_chunk_results:
            if not isinstance(chunk, dict):
                continue
            result = chunk.pop("result", None)
            error = chunk.get("error")
            if isinstance(result, dict):
                completed_chunks.append(result)
                chunk.update({
                    "status": "error" if chunk.get("capture_error") else result.get("occupancy", {}).get("label", "empty"),
                    "detected_frames": result.get("occupancy", {}).get("detected_frames", 0),
                    "evaluated_frames": result.get("occupancy", {}).get("evaluated_frames", 0),
                    "occupied": result.get("occupancy", {}).get("label") == "occupied",
                    "location": result.get("location"),
                    "score": result.get("score"),
                    "targets": result.get("targets", []),
                    "finished": iso_now(),
                })
                if not error:
                    chunk.pop("error", None)
            elif error:
                chunk.update({
                    "status": "error",
                    "error": error,
                    "finished": iso_now(),
                })

        if completed_chunks:
            xy_payload = compile_minute_xy_payload(completed_chunks)
            xy_payload_path = output_dir / "xy-tracking.json"
            write_json_atomic(xy_payload_path, xy_payload)
            manifest["outputs"]["xy_tracking"] = {
                "path": str(xy_payload_path),
                "type": "json",
                "chunk_count": len(completed_chunks),
            }
            manifest["outputs"]["predictions"] = {
                "path": str(output_dir / "predictions.json"),
                "type": "json",
                "chunk_count": len(completed_chunks),
            }
            predictions_path = output_dir / "predictions.json"
            prediction_payload = {
                "generated_at": iso_now(),
                "chunk_seconds": chunk_seconds,
                "labels": preset_labels,
                "timeline": [
                    {
                        "model_name": "xy-tracking",
                        "chunk_index": chunk.get("chunk_index"),
                        "chunk_seconds": chunk.get("chunk_seconds"),
                        "prediction": chunk.get("occupancy", {}).get("label"),
                        "occupied": chunk.get("occupancy", {}).get("label") == "occupied",
                        "location": chunk.get("location"),
                        "score": chunk.get("score"),
                        "target_count": len(chunk.get("targets") or []),
                        "targets": chunk.get("targets") or [],
                        "detected_frames": chunk.get("occupancy", {}).get("detected_frames", 0),
                        "evaluated_frames": chunk.get("occupancy", {}).get("evaluated_frames", 0),
                        "ratio": chunk.get("occupancy", {}).get("ratio", 0.0),
                        "bin_path": chunk.get("bin_path"),
                        "csv_path": chunk.get("csv_path"),
                    }
                    for chunk in completed_chunks
                ],
                "summary": xy_payload.get("occupancy", {}),
            }
            write_json_atomic(predictions_path, prediction_payload)

        video_returncode = stop_video_capture(video_proc)
        if video_returncode is not None:
            manifest["outputs"].setdefault("video", {})["ffmpeg_returncode"] = video_returncode
            video_path = Path(manifest["outputs"]["video"]["path"])
            if video_returncode != 0 or not video_path.exists() or video_path.stat().st_size == 0:
                manifest["warnings"].append(f"Camera capture failed with ffmpeg exit code {video_returncode}.")

        radar_files = sorted(str(path) for path in output_dir.glob("mmw_radar_raw_*.bin"))
        if radar_files:
            manifest["outputs"].setdefault("radar", {})["files"] = radar_files
        radar_csv_files = sorted(str(path) for path in output_dir.glob("mmw_radar_xy_*.csv"))
        if radar_csv_files:
            manifest["outputs"].setdefault("radar", {})["csv_files"] = radar_csv_files

        wifi_csi = manifest["outputs"].get("wifi_csi")
        if isinstance(wifi_csi, dict):
            for key in ("raw_path", "timestamped_path", "all_serial_path"):
                path = Path(wifi_csi[key])
                if path.exists():
                    wifi_csi[f"{key}_bytes"] = path.stat().st_size
            raw_path = Path(wifi_csi["raw_path"])
            if raw_path.exists() and raw_path.stat().st_size <= len(CSI_HEADER) + 1:
                manifest["warnings"].append("ESP32 CSI receiver was detected but produced no CSI_DATA samples this minute.")

        sense_hat = manifest["outputs"].get("sense_hat")
        if isinstance(sense_hat, dict):
            path = Path(str(sense_hat.get("path", "")))
            if path.exists():
                sense_hat["path_bytes"] = path.stat().st_size

        manifest["capture_started"] = capture_started
        manifest["capture_finished"] = iso_now()
        manifest["host"] = os.uname().nodename
        manifest["status"] = "success" if not manifest["errors"] else "partial" if manifest["warnings"] else "error"
        manifest_file = output_dir / "manifest.json"
        merge_home_assistant_status()
        write_json_atomic(manifest_file, manifest)

        chown_to_invoking_user(output_dir)

    print(f"Capture finished: {manifest['capture_finished']}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    if manifest["warnings"]:
        print("Warnings:")
        for warning in manifest["warnings"]:
            print(f"  - {warning}")
    if manifest["errors"]:
        print("Errors:")
        for error in manifest["errors"]:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
