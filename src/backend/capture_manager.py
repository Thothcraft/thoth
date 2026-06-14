"""Minute capture management for Thoth."""

from __future__ import annotations

import binascii
import io
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config

MINUTE_DIR_RE = re.compile(r"^\d{8}_\d{4}$")
TIMESTAMP_FMT = "%Y%m%d_%H%M"


def _capture_root() -> Path:
    root = Path(Config.CAPTURE_DATA_DIR).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_minute(name: str) -> Optional[datetime]:
    try:
        return datetime.strptime(name, TIMESTAMP_FMT)
    except ValueError:
        return None


def is_minute_folder(path: Path) -> bool:
    return path.is_dir() and MINUTE_DIR_RE.match(path.name) is not None and _parse_minute(path.name) is not None


def list_minute_folders() -> List[Path]:
    root = _capture_root()
    folders = [item for item in root.iterdir() if is_minute_folder(item)]
    folders.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return folders


def _file_map(minute_dir: Path) -> Dict[str, Path]:
    files = {item.name: item for item in minute_dir.iterdir() if item.is_file()}
    result = {
        "manifest": files.get("manifest.json"),
        "video": files.get("usb_camera.mp4"),
        "video_log": files.get("usb_camera.ffmpeg.log"),
        "radar": None,
        "csi_csv": files.get("wifi_csi_raw.csv"),
        "csi_timestamped": files.get("wifi_csi_timestamped.csv"),
        "csi_serial": files.get("wifi_csi_serial_all.jsonl"),
    }
    radar_candidates = sorted(
        [item for item in minute_dir.iterdir() if item.is_file() and item.name.startswith("mmw_radar_raw_") and item.suffix == ".bin"],
        key=lambda p: p.name,
    )
    result["radar"] = radar_candidates[0] if radar_candidates else None
    return result


def minute_summary(minute_dir: Path) -> Dict[str, object]:
    files = _file_map(minute_dir)
    stat = minute_dir.stat()
    manifest = None
    if files["manifest"] and files["manifest"].exists():
        try:
            import json

            with open(files["manifest"], "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = None

    return {
        "minute": minute_dir.name,
        "path": str(minute_dir),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "files": {
            "video": bool(files["video"] and files["video"].exists()),
            "radar": bool(files["radar"] and files["radar"].exists()),
            "csi": bool((files["csi_csv"] and files["csi_csv"].exists()) or (files["csi_timestamped"] and files["csi_timestamped"].exists()) or (files["csi_serial"] and files["csi_serial"].exists())),
            "manifest": bool(files["manifest"] and files["manifest"].exists()),
        },
        "sizes": {
            "video": files["video"].stat().st_size if files["video"] and files["video"].exists() else 0,
            "radar": files["radar"].stat().st_size if files["radar"] and files["radar"].exists() else 0,
            "csi_csv": files["csi_csv"].stat().st_size if files["csi_csv"] and files["csi_csv"].exists() else 0,
            "csi_timestamped": files["csi_timestamped"].stat().st_size if files["csi_timestamped"] and files["csi_timestamped"].exists() else 0,
            "csi_serial": files["csi_serial"].stat().st_size if files["csi_serial"] and files["csi_serial"].exists() else 0,
        },
        "manifest": manifest,
    }


def list_minutes() -> List[Dict[str, object]]:
    return [minute_summary(path) for path in list_minute_folders()]


def get_minute(minute: str) -> Optional[Path]:
    if not MINUTE_DIR_RE.match(minute) or _parse_minute(minute) is None:
        return None
    minute_dir = _capture_root() / minute
    if minute_dir.exists() and minute_dir.is_dir():
        return minute_dir
    return None


def current_minute() -> Optional[Path]:
    folders = list_minute_folders()
    return folders[0] if folders else None


def capture_files(minute_dir: Path) -> Dict[str, Optional[Path]]:
    return _file_map(minute_dir)


def preview_text(path: Optional[Path], limit: int = 20000) -> str:
    if not path or not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except Exception:
        return ""


def read_binary_tail(path: Path, offset: int = 0, limit: int = 8192) -> Tuple[int, str]:
    if not path.exists():
        return offset, ""
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read(limit)
        new_offset = handle.tell()
    return new_offset, binascii.hexlify(data).decode("ascii") if data else ""


def zip_minute_folder(minute_dir: Path) -> Path:
    temp_dir = Path(tempfile.gettempdir())
    zip_path = temp_dir / f"{minute_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(minute_dir.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(minute_dir))
    return zip_path


def cleanup_old_minutes(keep: Optional[int] = None) -> Dict[str, object]:
    keep = int(keep or Config.CAPTURE_KEEP_MINUTES)
    folders = list_minute_folders()
    removed: List[str] = []
    if len(folders) <= keep:
        return {"kept": len(folders), "removed": removed}

    for minute_dir in folders[keep:]:
        shutil.rmtree(minute_dir, ignore_errors=True)
        removed.append(minute_dir.name)

    return {"kept": min(keep, len(folders)), "removed": removed}
