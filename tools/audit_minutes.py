#!/usr/bin/env python3
"""Audit minute captures under a data root — read-only.

Prints a JSON report: file counts, schema versions, timestamp ranges,
source counts, and which manifests are legacy (chunk-era) vs canonical
``thoth-minute/v1``.

Usage: python tools/audit_minutes.py [data_root]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whispy.minutes import iter_minute_dirs, read_minute  # noqa: E402


def audit(root: Path) -> dict:
    minutes = []
    totals = {"minutes": 0, "legacy": 0, "canonical": 0, "unreadable": 0,
              "files": 0, "bytes": 0}
    for d in iter_minute_dirs(root):
        files = [f for f in d.rglob("*") if f.is_file()]
        totals["files"] += len(files)
        totals["bytes"] += sum(f.stat().st_size for f in files)
        entry = {"minute": d.name, "path": str(d), "file_count": len(files)}
        try:
            m = read_minute(d)
            legacy = (d / "minute.json").exists() is False
            entry.update({
                "schema": m.metadata.get("legacy_schema") or m.format,
                "canonical": not legacy,
                "start_timestamp": m.start_timestamp,
                "end_timestamp": m.end_timestamp,
                "sources": len(m.sources),
                "predictions": len(m.predictions),
                "labels": m.labels.get("labels") or [],
            })
            totals["legacy" if legacy else "canonical"] += 1
        except Exception as exc:
            entry["error"] = str(exc)
            totals["unreadable"] += 1
        totals["minutes"] += 1
        minutes.append(entry)
    return {"root": str(root), "totals": totals, "minutes": minutes}


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    print(json.dumps(audit(root), indent=2))
