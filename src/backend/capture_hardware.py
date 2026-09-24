"""Hardware-facing capture helpers for the minute collector.

Serial/CSI receivers, Sense HAT, USB camera snapshots, and the BGT60TR13C
radar lifecycle (start/stop, GPIO settle, cross-process hardware lock) live
here so ``minute_collector.py`` can stay focused on orchestration. These
functions are intentionally free of collector state — they take explicit
arguments and report errors via return values or ``.error.json`` sidecars.
"""

from __future__ import annotations

import csv
import datetime as dt
try:
    import fcntl  # Unix-only; Windows uses msvcrt.locking below
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
import glob
import json
import logging
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.capture_container import _split_radar_packets  # type: ignore
    from backend.sensor_detection import likely_csi_serial_candidates, usable_usb_camera_devices  # type: ignore
else:
    from .capture_container import _split_radar_packets
    from .sensor_detection import likely_csi_serial_candidates, usable_usb_camera_devices

THOTH_ROOT = Path(__file__).resolve().parents[2]
MMW_RELEASE = THOTH_ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
RADAR_CFG = MMW_RELEASE / "radar_config" / "config_3rx_3m"
RADAR_LOCK_PATH = Path(tempfile.gettempdir()) / "thoth-radar-hardware.lock"
RADAR_GPIO_RETRY_SECONDS = 12.0
RADAR_GPIO_SETTLE_SECONDS = 0.5
# The BGT60TR13C driver delivers radar frames in 10-frame batches. This is a
# hardware/driver detail, not a user-facing "chunk" — the minute is the atomic
# unit and per-frame timestamps carry the real timing.
RADAR_FRAMES_PER_SECOND = 10


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def find_camera(requested: str | None) -> str | None:
    devices = usable_usb_camera_devices()
    if requested and requested in devices:
        return requested
    return devices[0] if devices else None


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
                if "CSI_DATA" in text:
                    return True
    except Exception:
        return False
    return False


def find_csi_ports(
    requested: list[str] | str | None,
    baud: int,
    detect_seconds: float,
) -> tuple[list[str], list[str]]:
    requested_ports = [requested] if isinstance(requested, str) else list(requested or [])
    requested_ports = list(dict.fromkeys(port for port in requested_ports if port and port != "auto"))
    candidates = serial_candidates()
    if requested_ports:
        return requested_ports, candidates

    esp32_ports = likely_csi_serial_candidates()
    if esp32_ports:
        return sorted(esp32_ports), candidates

    if detect_seconds <= 0:
        return candidates, candidates

    per_port_timeout = max(0.2, detect_seconds / max(1, len(candidates)))
    detected: list[str] = []
    for port in candidates:
        if probe_csi_port(port, baud, per_port_timeout):
            detected.append(port)
    return (detected or candidates), candidates


def csi_capture_stats(path: Path) -> dict[str, float | int | None]:
    """Calculate the average observed packet rate from one receiver CSV."""
    sample_count = 0
    first_ns: int | None = None
    last_ns: int | None = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    timestamp_ns = int(row.get("monotonic_ns") or 0)
                except (TypeError, ValueError):
                    timestamp_ns = 0
                sample_count += 1
                if timestamp_ns > 0:
                    first_ns = timestamp_ns if first_ns is None else min(first_ns, timestamp_ns)
                    last_ns = timestamp_ns if last_ns is None else max(last_ns, timestamp_ns)
    except OSError:
        pass
    span_seconds = ((last_ns - first_ns) / 1_000_000_000) if first_ns is not None and last_ns is not None else 0.0
    rate = ((sample_count - 1) / span_seconds) if sample_count > 1 and span_seconds > 0 else 0.0
    return {
        "sample_count": sample_count,
        "average_sampling_rate_hz": round(rate, 3),
        "observed_span_seconds": round(span_seconds, 3),
    }


def _iso_seconds(value: object) -> float:
    """ISO timestamp -> unix seconds, 0.0 when unparseable."""
    if not value:
        return 0.0
    try:
        parsed = dt.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _minute_radar_frames(output_dir: Path, manifest: dict[str, Any]) -> tuple[list[bytes], list[float]]:
    """All radar wire packets of the minute plus per-frame timestamps (s).

    Prefers the per-frame CLOCK_MONOTONIC stamps recorded in each chunk
    (same clock as the CSI CSVs); falls back to uniform interpolation
    between the chunk's started/finished_capture ISO timestamps.
    """
    frames: list[bytes] = []
    times: list[float] = []
    radar = manifest.get("outputs", {}).get("radar", {})
    chunks = radar.get("chunks") if isinstance(radar.get("chunks"), list) else []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        path = Path(str(chunk.get("bin_path") or ""))
        if not path.is_absolute():
            path = output_dir / path.name
        try:
            packets = list(_split_radar_packets(path.read_bytes()))
        except OSError:
            continue
        mono = chunk.get("frame_monotonic_ns")
        mono = [int(v) for v in mono] if isinstance(mono, list) else []
        start = _iso_seconds(chunk.get("started"))
        finish = _iso_seconds(chunk.get("finished_capture") or chunk.get("finished"))
        for index, packet in enumerate(packets):
            frames.append(packet)
            if index < len(mono) and mono[index] > 0:
                times.append(mono[index] / 1_000_000_000)
            elif len(packets) > 1 and finish > start:
                times.append(start + (finish - start) * index / (len(packets) - 1))
            else:
                times.append(start or (times[-1] + 0.1 if times else 0.0))
    return frames, times


def _minute_csi_samples(manifest: dict[str, Any]) -> list[tuple[int, float, str]]:
    """(receiver_index, monotonic_seconds, raw CSI_DATA line) for the minute."""
    samples: list[tuple[int, float, str]] = []
    wifi = manifest.get("outputs", {}).get("wifi_csi", {})
    receivers = wifi.get("receivers") if isinstance(wifi, dict) else None
    if not isinstance(receivers, list):
        receivers = [wifi] if isinstance(wifi, dict) else []
    for receiver_index, receiver in enumerate(receivers):
        if not isinstance(receiver, dict) or not receiver.get("path"):
            continue
        try:
            with open(str(receiver["path"]), "r", encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    line = str(row.get("raw_csi_line") or row.get("data") or "")
                    if "CSI_DATA" not in line:
                        continue
                    try:
                        t = int(row.get("monotonic_ns") or 0) / 1_000_000_000
                    except (TypeError, ValueError):
                        t = 0.0
                    if t <= 0:
                        t = _iso_seconds(row.get("host_timestamp"))
                    samples.append((receiver_index, t, line.strip()))
        except OSError:
            continue
    return samples


def collect_csi(
    port: str,
    baud: int,
    output_file: Path,
    stop_event: threading.Event,
) -> None:
    try:
        import serial
    except Exception as exc:
        with open(output_file.with_suffix(".error.json"), "w", encoding="utf-8") as fd:
            json.dump({"timestamp": iso_now(), "error": f"pyserial import failed: {exc}"}, fd, indent=2)
        return

    error_path = output_file.with_suffix(".error.json")
    last_error: Exception | None = None
    sample_count = 0
    try:
        with open(output_file, "w", encoding="utf-8", newline="", buffering=1) as output_fd:
            writer = csv.writer(output_fd)
            writer.writerow(["host_timestamp", "monotonic_ns", "serial_port", "raw_csi_line"])
            while not stop_event.is_set():
                try:
                    with open_serial_without_reset(port, baud, 0.05) as ser:
                        last_error = None
                        error_path.unlink(missing_ok=True)
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
                            marker = text.find("CSI_DATA")
                            if marker >= 0:
                                writer.writerow([host_timestamp, monotonic_ns, port, text[marker:]])
                                sample_count += 1
                except (OSError, serial.SerialException) as exc:
                    last_error = exc
                    if not stop_event.wait(0.2):
                        continue
            if last_error is not None and sample_count == 0:
                with open(error_path, "w", encoding="utf-8") as fd:
                    json.dump(
                        {"timestamp": iso_now(), "port": port, "baud": baud, "error": str(last_error)},
                        fd,
                        indent=2,
                    )
    except Exception as exc:
        with open(error_path, "w", encoding="utf-8") as fd:
            json.dump({"timestamp": iso_now(), "port": port, "baud": baud, "error": str(exc)}, fd, indent=2)


def collect_sensehat(output_file: Path, stop_event: threading.Event, errors: list[str], interval: float = 0.2) -> None:
    try:
        from sense_hat import SenseHat
    except Exception as exc:
        message = f"Sense HAT unavailable: {exc}"
        errors.append(message)
        logging.getLogger(__name__).error(message)
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
        message = f"Sense HAT capture failed: {exc}"
        errors.append(message)
        logging.getLogger(__name__).error(message)


def capture_camera_image(camera: str, output_file: Path) -> None:
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
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_file),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )
    if result.returncode != 0 or not output_file.exists() or output_file.stat().st_size == 0:
        output_file.unlink(missing_ok=True)
        detail = result.stderr.decode("utf-8", errors="replace")[-300:]
        raise RuntimeError(f"camera snapshot failed ({result.returncode}): {detail}")


def _open_radar_chip(output_prefix: Path | None = None) -> Any:
    """Initialise the BGT60TR13C and start streaming (runs in the driver
    process — see RadarProcess)."""
    if not Path("/dev/spidev0.0").exists():
        raise RuntimeError("/dev/spidev0.0 is missing; enable SPI and reboot the Raspberry Pi.")

    from utility.BGT60TR13C import BGT60TR13C, RET_VAL_OK
    from utility.helper import calculate_frame_size, find_register_config_in_directory, find_setting_in_directory
    bgt60tr13c = None
    try:
        deadline = time.monotonic() + RADAR_GPIO_RETRY_SECONDS
        while True:
            try:
                bgt60tr13c = BGT60TR13C(
                    spi_speed=50_000_000,
                    save_to_file=str(output_prefix) if output_prefix is not None else None,
                    strict_gpio=True,
                )
                break
            except Exception as exc:
                if time.monotonic() >= deadline or "GPIO busy" not in str(exc):
                    raise
                logging.info("Waiting for radar GPIO handoff: %s", exc)
                time.sleep(0.2)
        if bgt60tr13c.check_chip_id() != RET_VAL_OK:
            raise RuntimeError("BGT60TR13C chip ID check failed.")

        reg_file = find_register_config_in_directory(str(RADAR_CFG))
        setting_file = find_setting_in_directory(str(RADAR_CFG))
        bgt60tr13c.load_register_config_file(reg_file)

        with open(setting_file, "r", encoding="utf-8") as fd:
            setting_data = json.load(fd)

        frame_size = calculate_frame_size(setting_data)
        # Use the full 8192-sample hardware FIFO (was 4096): under CPU/IO
        # contention the drain thread gets twice the slack before an
        # overflow wedges the stream. Burst stays 2048 — spidev's xfer2
        # buffer is capped at 4096 bytes.
        bgt60tr13c.set_fifo_parameters(frame_size, 8192, 2048)
        if bgt60tr13c.start() != RET_VAL_OK:
            raise RuntimeError("BGT60TR13C failed to start.")

        return bgt60tr13c
    except Exception:
        stop_radar_capture(bgt60tr13c)
        raise


def _radar_driver_main(frame_queue: Any, control_queue: Any, output_prefix: Any) -> None:
    """Dedicated-process entry: own the chip, forward frames to the parent.

    The hardware FIFO must be drained within ~20-30 ms while chirps stream.
    Inside the collector process, 60 ms+ GIL-holding analysis bursts starve
    the drain thread and the FIFO overflows every frame. A dedicated process
    has its own GIL, so SPI draining is never preempted by analysis work.
    """
    # Die with the parent: daemon-children only exit on a *clean* parent
    # exit — a SIGKILLed collector would orphan us holding the radar GPIO.
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 15)  # PR_SET_PDEATHSIG = SIGTERM
    except Exception:
        pass
    radar = None
    try:
        radar = _open_radar_chip(Path(output_prefix) if output_prefix else None)
        control_queue.put(("ready", None))
        while True:
            try:
                frame = radar.frame_buffer.get(timeout=1.0)
            except queue.Empty:
                frame = None
            if os.getppid() == 1:
                return  # parent died — release the radar for the next owner
            if frame is None:
                continue
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # parent behind — drop oldest-style, keep newest flowing
    except Exception as exc:
        try:
            control_queue.put(("error", str(exc)))
        except Exception:
            pass
    finally:
        stop_radar_capture(radar)


class _FrameQueueProxy:
    """Duck-types BGT60TR13C.frame_buffer for the reader loop."""

    def __init__(self, source: Any) -> None:
        self._source = source

    def get(self, timeout: float | None = None) -> bytes:
        return self._source.get(timeout=timeout)


class RadarProcess:
    """Radar driver running in its own process (own GIL → no SPI starvation).

    Exposes the same surface the collector uses: ``frame_buffer`` queue and
    ``stop()``. Init errors in the child are re-raised in the parent.
    """

    def __init__(self, output_prefix: Path | None = None) -> None:
        self._frames: multiprocessing.Queue = multiprocessing.Queue(maxsize=256)
        self._control: multiprocessing.Queue = multiprocessing.Queue()
        self._proc = multiprocessing.Process(
            target=_radar_driver_main,
            args=(
                self._frames,
                self._control,
                str(output_prefix) if output_prefix is not None else None,
            ),
            daemon=True,
            name="RadarDriver",
        )
        self._proc.start()
        try:
            kind, payload = self._control.get(timeout=RADAR_GPIO_RETRY_SECONDS + 30.0)
        except queue.Empty as exc:
            self.stop()
            raise RuntimeError("Radar driver process did not initialise in time") from exc
        if kind != "ready":
            self.stop()
            raise RuntimeError(f"Radar driver failed to start: {payload}")
        self.frame_buffer = _FrameQueueProxy(self._frames)

    def stop(self) -> None:
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5.0)


def start_radar_capture(output_prefix: Path | None = None) -> Any:
    return RadarProcess(output_prefix)


def stop_radar_capture(radar: Any | None) -> None:
    if radar is not None:
        radar.stop()
        # gpiozero's per-process pin factory can retain lgpio line claims after
        # individual devices close. This worker will not touch GPIO again, so
        # close the factory explicitly before the next minute takes ownership.
        try:
            from gpiozero import Device

            if Device.pin_factory is not None:
                Device.pin_factory.close()
        except Exception as exc:
            logging.warning("Unable to close radar GPIO factory cleanly: %s", exc)


def settle_radar_gpio() -> None:
    """Give the GPIO daemon time to publish released lines before handoff."""
    time.sleep(RADAR_GPIO_SETTLE_SECONDS)


def acquire_radar_lock() -> Any:
    """Serialize physical radar ownership across overlapping minute workers."""
    handle = open(RADAR_LOCK_PATH, "a+", encoding="utf-8")
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    elif os.name == "nt":  # Windows: byte-range lock via msvcrt
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    return handle


def release_radar_lock(handle: Any | None) -> None:
    if handle is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        elif os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


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
