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
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.config import Config  # type: ignore
    from backend.radar_analysis import analyze_radar_chunk, compile_minute_xy_payload, load_room_config  # type: ignore
else:
    from .config import Config
    from .radar_analysis import analyze_radar_chunk, compile_minute_xy_payload, load_room_config

THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
RADAR_CFG = MMW_RELEASE / "radar_config" / "config_3rx_3m"
DATA_ROOT = Path(Config.CAPTURE_DATA_DIR).expanduser()
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
    radar_threads: list[threading.Thread] = []
    radar_chunk_results: list[dict[str, Any]] = []
    room_config = load_room_config()

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
            chunk_index = 0
            while True:
                remaining = stop_at - time.monotonic()
                if remaining <= 0:
                    break
                current_chunk_seconds = min(chunk_seconds, remaining)
                radar_prefix = output_dir / f"mmw_radar_raw_{chunk_index:02d}"
                radar_started = iso_now()
                radar = None
                try:
                    radar = start_radar_capture(radar_prefix)
                    chunk_stop = time.monotonic() + current_chunk_seconds
                    while True:
                        remaining_chunk = chunk_stop - time.monotonic()
                        if remaining_chunk <= 0:
                            break
                        time.sleep(min(remaining_chunk, 0.25))
                except Exception as exc:
                    manifest["warnings"].append(f"Radar chunk {chunk_index} failed to start: {exc}")
                finally:
                    stop_radar_capture(radar)

                radar_files = sorted(output_dir.glob(f"mmw_radar_raw_{chunk_index:02d}_*.bin"))
                if not radar_files:
                    chunk_index += 1
                    continue
                radar_path = radar_files[-1]
                csv_path = output_dir / f"mmw_radar_xy_{chunk_index:02d}.csv"
                chunk_entry: dict[str, Any] = {
                    "chunk_index": chunk_index,
                    "bin_path": str(radar_path),
                    "csv_path": str(csv_path),
                    "started": radar_started,
                    "chunk_seconds": current_chunk_seconds,
                    "status": "analyzing",
                }

                def _analyze(
                    entry: dict[str, Any] = chunk_entry,
                    idx: int = chunk_index,
                    duration: float = current_chunk_seconds,
                    bin_file: Path = radar_path,
                    csv_file: Path = csv_path,
                    room_snapshot: dict[str, Any] = room_config,
                ) -> None:
                    try:
                        entry["result"] = analyze_radar_chunk(
                            bin_file,
                            csv_file,
                            chunk_index=idx,
                            chunk_seconds=duration,
                            room=room_snapshot,
                        )
                    except Exception as exc:
                        entry["error"] = str(exc)

                thread = threading.Thread(target=_analyze, name=f"RadarChunk-{chunk_index}", daemon=True)
                thread.start()
                radar_threads.append(thread)
                radar_chunk_results.append(chunk_entry)
                manifest["outputs"]["radar"]["chunks"].append(chunk_entry)
                chunk_index += 1

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
        for thread in radar_threads:
            thread.join(timeout=10)
        completed_chunks: list[dict[str, Any]] = []
        for chunk in radar_chunk_results:
            if not isinstance(chunk, dict):
                continue
            result = chunk.pop("result", None)
            error = chunk.pop("error", None)
            if isinstance(result, dict):
                completed_chunks.append(result)
                chunk.update({
                    "status": "complete",
                    "detected_frames": result.get("occupancy", {}).get("detected_frames", 0),
                    "evaluated_frames": result.get("occupancy", {}).get("evaluated_frames", 0),
                    "occupied": result.get("occupancy", {}).get("label") == "occupied",
                    "location": result.get("location"),
                    "score": result.get("score"),
                    "targets": result.get("targets", []),
                    "finished": iso_now(),
                })
            elif error:
                chunk.update({
                    "status": "error",
                    "error": error,
                    "finished": iso_now(),
                })

        if completed_chunks:
            xy_payload = compile_minute_xy_payload(completed_chunks)
            xy_payload_path = output_dir / "xy-tracking.json"
            with open(xy_payload_path, "w", encoding="utf-8") as fd:
                json.dump(xy_payload, fd, indent=2)
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
            with open(predictions_path, "w", encoding="utf-8") as fd:
                json.dump(prediction_payload, fd, indent=2)

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

        manifest["status"] = "success" if not manifest["errors"] else "partial" if manifest["warnings"] else "error"
        manifest_file = output_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as fd:
            json.dump(manifest, fd, indent=2)

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
