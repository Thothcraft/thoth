#!/usr/bin/env python3
"""Convert legacy Thoth minute folders to synchronized NPZ containers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.capture_container import CONTAINER_FILENAME, build_capture_container  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Minute folders or a parent data folder")
    parser.add_argument("--keep-fragments", action="store_true", help="Retain old CSV/JPEG/BIN files")
    parser.add_argument("--force", action="store_true", help="Replace an existing capture.npz")
    args = parser.parse_args()
    minute_dirs: list[Path] = []
    for value in args.paths:
        if (value / "manifest.json").is_file():
            minute_dirs.append(value)
        elif value.is_dir():
            minute_dirs.extend(
                child for child in value.rglob("*")
                if child.is_dir() and (child / "manifest.json").is_file()
            )

    failures = 0
    for minute_dir in sorted(set(path.resolve() for path in minute_dirs)):
        destination = minute_dir / CONTAINER_FILENAME
        if destination.exists() and not args.force:
            print(f"skip {minute_dir}: {CONTAINER_FILENAME} already exists")
            continue
        try:
            manifest = json.loads((minute_dir / "manifest.json").read_text(encoding="utf-8"))
            info = build_capture_container(
                minute_dir,
                manifest,
                remove_fragments=not args.keep_fragments,
            )
            manifest["schema"] = "thoth-minute-manifest/v6"
            manifest["container"] = info
            (minute_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"migrated {minute_dir}: {info['second_count']} seconds, {info['bytes']} bytes")
        except Exception as exc:
            failures += 1
            print(f"failed {minute_dir}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
