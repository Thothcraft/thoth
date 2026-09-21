#!/usr/bin/env python3
"""Capture one synchronized minute of Thoth sensor data.

This is the thin CLI entry point. The collection runtime (shared
``CollectorContext`` + per-sensor worker threads + finalization) lives in
``collector_runtime.py``; hardware drivers in ``capture_hardware.py``; manifest
and settings helpers in ``collector_manifest.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.collector_runtime import CollectorContext, run  # type: ignore
else:
    from .collector_runtime import CollectorContext, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one synchronized Thoth sensor minute into the configured data folder."
    )
    parser.add_argument("--duration", type=float, default=59.5, help="Capture duration in seconds.")
    parser.add_argument("--camera", default=None, help="USB camera device, for example /dev/video0.")
    parser.add_argument(
        "--csi-port",
        action="append",
        default=None,
        help="ESP32 CSI receiver serial port. Repeat for multiple receivers; omit for auto detection.",
    )
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
    parser.add_argument(
        "--scheduled-start",
        default=None,
        help="Exact ISO-8601 wall-clock minute assigned by the continuous supervisor.",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Stream sensors into the live dir without minute chunking, model inference, or finalization.",
    )
    return parser.parse_args()


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def main() -> int:
    return run(CollectorContext(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
