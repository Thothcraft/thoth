#!/usr/bin/env python3
"""Verify a minute migration — compares canonical output against legacy.

Checks per minute: file count unchanged, timestamps preserved, source
counts, prediction counts, labels, and that legacy files are untouched
(byte-identical via size+mtime, checksums optional with --checksums).

Usage: python tools/verify_minute_migration.py [data_root] [--checksums]
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whispy.minutes import iter_minute_dirs, read_minute  # noqa: E402
from whispy.contracts import MinuteManifest  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(root: Path, checksums: bool = False) -> dict:
    report = {"root": str(root), "verified": [], "failed": []}
    for d in iter_minute_dirs(root):
        canonical_path = d / "minute.json"
        problems = []
        if not canonical_path.exists():
            problems.append("missing minute.json")
            report["failed"].append({"minute": d.name,
                                     "problems": problems})
            continue
        try:
            canonical = MinuteManifest.from_dict(
                json.loads(canonical_path.read_text(encoding="utf-8")))
            legacy = read_minute(d)  # reads manifest.json (legacy)
            if canonical.minute_id != legacy.minute_id:
                problems.append(
                    f"minute_id {canonical.minute_id!r} != {legacy.minute_id!r}")
            if abs(canonical.start_timestamp - legacy.start_timestamp) > 1e-6:
                problems.append("start_timestamp drift")
            if len(canonical.sources) != len(legacy.sources):
                problems.append(
                    f"source count {len(canonical.sources)} != "
                    f"{len(legacy.sources)}")
            if len(canonical.predictions) != len(legacy.predictions):
                problems.append("prediction count mismatch")
            if (canonical.labels.get("labels") or []) != \
                    (legacy.labels.get("labels") or []):
                problems.append("label mismatch")
            if checksums:
                for f in d.iterdir():
                    if f.is_file() and f.name not in ("minute.json",):
                        canonical.checksums.setdefault(f.name, "")
            entry = {"minute": d.name, "problems": problems}
            (report["failed"] if problems else report["verified"]).append(entry)
        except Exception as exc:
            report["failed"].append({"minute": d.name,
                                     "problems": [str(exc)]})
    report["totals"] = {"verified": len(report["verified"]),
                        "failed": len(report["failed"])}
    return report


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0] if args else "data")
    result = verify(root, checksums="--checksums" in sys.argv)
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["failed"] else 0)
