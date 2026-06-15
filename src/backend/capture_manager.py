"""Minute capture management for Thoth."""

from __future__ import annotations

import binascii
import io
import json
import os
import re
import shutil
import math
import subprocess
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


def _parse_iso_datetime(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _load_manifest(minute_dir: Path) -> Optional[Dict[str, object]]:
    manifest_path = minute_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _manifest_seconds(manifest: Optional[Dict[str, object]]) -> Optional[float]:
    if not isinstance(manifest, dict):
        return None

    duration = manifest.get("duration_seconds")
    try:
        if duration is not None:
            value = float(duration)
            if value > 0:
                return value
    except Exception:
        pass

    started = _parse_iso_datetime(manifest.get("capture_started") or manifest.get("scheduled_start"))
    finished = _parse_iso_datetime(manifest.get("capture_finished"))
    if started and finished and finished >= started:
        return (finished - started).total_seconds()
    return None


def _split_csv_line(line: str) -> List[str]:
    cells: List[str] = []
    current: List[str] = []
    in_quotes = False
    idx = 0
    while idx < len(line):
        char = line[idx]
        if char == '"':
            if in_quotes and idx + 1 < len(line) and line[idx + 1] == '"':
                current.append('"')
                idx += 2
                continue
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            cells.append(''.join(current))
            current = []
        else:
            current.append(char)
        idx += 1
    cells.append(''.join(current))
    return cells


def _extract_csi_payload(raw: str) -> str:
    payloads = re.findall(r'\[([^\]]*)\]', raw or '')
    return payloads[-1] if payloads else (raw or '')


def _parse_csi_payload(raw: str) -> List[float]:
    payload = _extract_csi_payload(raw)
    values: List[float] = []
    for token in re.findall(r'[-+]?\d+(?:\.\d+)?', payload):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _parse_csi_average_series(path: Path, limit: int = 5000) -> List[float]:
    if not path or not path.exists():
        return []

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except Exception:
        return []

    if not lines:
        return []

    def _mean_from_payload(payload: str) -> Optional[float]:
        values = _parse_csi_payload(payload)
        if len(values) < 2:
            return None
        mags: List[float] = []
        for idx in range(0, len(values) - 1, 2):
            imag = values[idx]
            real = values[idx + 1]
            mags.append(math.sqrt((real * real) + (imag * imag)))
        if not mags:
            return None
        return sum(mags) / len(mags)

    series: List[float] = []
    first = lines[0]
    if first.startswith('{'):
        for line in lines:
            try:
                row = json.loads(line)
            except Exception:
                continue
            raw = str(row.get('line') or row.get('raw') or row.get('raw_csi_line') or '').strip()
            if not raw.startswith('CSI_DATA,'):
                continue
            cells = _split_csv_line(raw)
            mean = _mean_from_payload(cells[-1] if cells else '')
            if mean is not None:
                series.append(mean)
    else:
        header = _split_csv_line(first)
        data_index = -1
        for candidate in ('data', 'raw_csi_line'):
            if candidate in header:
                data_index = header.index(candidate)
                break
        if data_index < 0:
            data_index = len(header) - 1
        if data_index < 0:
            return []
        for line in lines[1:]:
            cells = _split_csv_line(line)
            if len(cells) <= data_index:
                continue
            mean = _mean_from_payload(cells[data_index])
            if mean is not None:
                series.append(mean)

    return series[-limit:]


def _iter_radar_frames(path: Path):
    with open(path, 'rb') as handle:
        while True:
            version_bytes = handle.read(4)
            if not version_bytes or len(version_bytes) < 4:
                break
            version = int.from_bytes(version_bytes, byteorder='little', signed=False)
            if version != 0:
                break
            seq = int.from_bytes(handle.read(4), byteorder='little', signed=False)
            data_len_bytes = handle.read(4)
            if len(data_len_bytes) < 4:
                break
            data_len = int.from_bytes(data_len_bytes, byteorder='little', signed=False)
            raw_data = handle.read(data_len)
            if len(raw_data) != data_len:
                break
            yield seq, raw_data


def _count_radar_frames(path: Path) -> int:
    count = 0
    for _seq, _raw_data in _iter_radar_frames(path):
        count += 1
    return count


def _probe_video_metadata(video_path: Optional[Path]) -> Dict[str, object]:
    if not video_path or not video_path.exists():
        return {}

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if result.returncode != 0 or not result.stdout:
            return {}
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [{}])[0]
        fps = None
        avg_frame_rate = stream.get("avg_frame_rate")
        if isinstance(avg_frame_rate, str) and avg_frame_rate and avg_frame_rate != "0/0":
            try:
                num, den = avg_frame_rate.split("/", 1)
                fps = float(num) / float(den)
            except Exception:
                fps = None
        duration = None
        try:
            duration = float(stream.get("duration")) if stream.get("duration") is not None else None
        except Exception:
            duration = None
        frame_count = None
        try:
            frame_count = int(stream.get("nb_frames")) if stream.get("nb_frames") not in (None, "N/A") else None
        except Exception:
            frame_count = None
        return {
            "codec": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "fps": fps,
            "duration_seconds": duration,
            "frame_count": frame_count,
            "data_shape": f"{stream.get('height') or '?'} x {stream.get('width') or '?'}",
        }
    except Exception:
        return {}


def _csi_subcarrier_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except Exception:
        return None

    if not lines:
        return None

    def _count_from_raw(raw: str) -> Optional[int]:
        values = _parse_csi_payload(raw)
        if len(values) < 2:
            return None
        return max(1, len(values) // 2)

    first = lines[0]
    if first.startswith("{"):
        for line in lines:
            try:
                row = json.loads(line)
            except Exception:
                continue
            raw = str(row.get("line") or row.get("raw") or row.get("raw_csi_line") or "").strip()
            if raw.startswith("CSI_DATA,"):
                cells = _split_csv_line(raw)
                if cells:
                    count = _count_from_raw(cells[-1])
                    if count:
                        return count
    else:
        header = _split_csv_line(first)
        data_index = -1
        for candidate in ("data", "raw_csi_line"):
            if candidate in header:
                data_index = header.index(candidate)
                break
        if data_index < 0:
            data_index = len(header) - 1
        if data_index >= 0:
            for line in lines[1:]:
                cells = _split_csv_line(line)
                if len(cells) <= data_index:
                    continue
                count = _count_from_raw(cells[data_index])
                if count:
                    return count
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
    csi_csv = files.get("wifi_csi_raw.csv")
    csi_timestamped = files.get("wifi_csi_timestamped.csv")
    csi_serial = files.get("wifi_csi_serial_all.jsonl")
    result = {
        "manifest": files.get("manifest.json"),
        "predictions": files.get("predictions.json"),
        "video": files.get("usb_camera.mp4"),
        "video_log": files.get("usb_camera.ffmpeg.log"),
        "radar": None,
        "csi_csv": csi_csv,
        "csi_timestamped": csi_timestamped,
        "csi_serial": csi_serial,
        "csi": csi_timestamped or csi_csv or csi_serial,
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
    manifest = _load_manifest(minute_dir)
    seconds_recorded = _manifest_seconds(manifest)

    return {
        "minute": minute_dir.name,
        "path": str(minute_dir),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "capture_started": manifest.get("capture_started") if isinstance(manifest, dict) else None,
        "capture_finished": manifest.get("capture_finished") if isinstance(manifest, dict) else None,
        "seconds_recorded": seconds_recorded,
        "labels": list(manifest.get("labels") or []) if isinstance(manifest, dict) else [],
        "predictions": bool(files["predictions"] and files["predictions"].exists()),
        "files": {
            "video": bool(files["video"] and files["video"].exists()),
            "radar": bool(files["radar"] and files["radar"].exists()),
            "csi": bool((files["csi_csv"] and files["csi_csv"].exists()) or (files["csi_timestamped"] and files["csi_timestamped"].exists()) or (files["csi_serial"] and files["csi_serial"].exists())),
            "manifest": bool(files["manifest"] and files["manifest"].exists()),
            "predictions": bool(files["predictions"] and files["predictions"].exists()),
        },
        "sizes": {
            "video": files["video"].stat().st_size if files["video"] and files["video"].exists() else 0,
            "radar": files["radar"].stat().st_size if files["radar"] and files["radar"].exists() else 0,
            "csi_csv": files["csi_csv"].stat().st_size if files["csi_csv"] and files["csi_csv"].exists() else 0,
            "csi_timestamped": files["csi_timestamped"].stat().st_size if files["csi_timestamped"] and files["csi_timestamped"].exists() else 0,
            "csi_serial": files["csi_serial"].stat().st_size if files["csi_serial"] and files["csi_serial"].exists() else 0,
        },
        "manifest": manifest,
        "predictions": files["predictions"].exists() if files["predictions"] else False,
    }


def minute_metrics(minute_dir: Path) -> Dict[str, object]:
    files = _file_map(minute_dir)
    manifest = _load_manifest(minute_dir)
    seconds_recorded = _manifest_seconds(manifest)

    video_meta = _probe_video_metadata(files.get("video"))
    csi_path = files.get("csi_timestamped") or files.get("csi_csv") or files.get("csi_serial")
    csi_points = _parse_csi_average_series(csi_path) if csi_path and csi_path.exists() else []
    csi_width = _csi_subcarrier_count(csi_path) if csi_path and csi_path.exists() else None
    radar_path = files.get("radar")
    radar_frames = 0
    radar_sampled = 0
    radar_shape = None
    radar_fps = None
    if radar_path and radar_path.exists():
        try:
            radar_frames = _count_radar_frames(radar_path)
            try:
                with open(radar_path, 'rb') as handle:
                    handle.read(8)
                    payload_len_bytes = handle.read(4)
                    if len(payload_len_bytes) == 4:
                        payload_len = int.from_bytes(payload_len_bytes, byteorder='little', signed=False)
                        radar_shape = f"{payload_len} bytes/frame"
            except Exception:
                radar_shape = None
            radar_sampled = radar_frames
            if seconds_recorded and seconds_recorded > 0:
                radar_fps = radar_frames / seconds_recorded
        except Exception:
            pass

    csi_shape = None
    if csi_width:
        csi_shape = f"{len(csi_points)} x {csi_width}"

    csi_fps = (len(csi_points) / seconds_recorded) if seconds_recorded and csi_points else None

    return {
        "seconds_recorded": seconds_recorded,
        "video": {
            "duration_seconds": video_meta.get("duration_seconds") or seconds_recorded,
            "fps": video_meta.get("fps"),
            "codec": video_meta.get("codec"),
            "width": video_meta.get("width"),
            "height": video_meta.get("height"),
            "frame_count": video_meta.get("frame_count"),
            "data_shape": video_meta.get("data_shape"),
            "file_size": files["video"].stat().st_size if files["video"] and files["video"].exists() else 0,
        },
        "csi": {
            "sample_count": len(csi_points),
            "subcarrier_count": csi_width,
            "effective_rate_hz": csi_fps,
            "data_shape": csi_shape,
            "file_size": csi_path.stat().st_size if csi_path and csi_path.exists() else 0,
        },
        "radar": {
            "frame_count": radar_frames,
            "sampled_frames": radar_sampled,
            "effective_fps": radar_fps,
            "data_shape": radar_shape,
            "file_size": radar_path.stat().st_size if radar_path and radar_path.exists() else 0,
        },
        "manifest": manifest or {},
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


def _normalize_labels(labels: object) -> List[str]:
    if isinstance(labels, str):
        items = labels.split(",")
    elif isinstance(labels, list):
        items = labels
    else:
        items = []

    cleaned: List[str] = []
    for item in items:
        value = str(item or "").strip().replace("\n", " ")
        value = re.sub(r"\s+", " ", value)
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def update_minute_labels(minute_dir: Path, labels: object, replace: bool = True) -> List[str]:
    manifest_path = minute_dir / "manifest.json"
    manifest: Dict[str, object] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    incoming = _normalize_labels(labels)
    if replace:
        merged = incoming
    else:
        merged = _normalize_labels(manifest.get("labels", []))
        for label in incoming:
            if label not in merged:
                merged.append(label)

    manifest["labels"] = merged
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return merged


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
