#!/usr/bin/env python3
"""Explicit minute migration — writes canonical ``minute.json`` manifests.

Non-destructive: legacy ``manifest.json`` and all data files are left
untouched; a canonical ``minute.json`` (thoth-minute/v1) is written next
to them. Re-run safe (idempotent). Never runs on daemon startup.

Usage:
    python tools/migrate_minutes.py [data_root] [--dry-run]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whispy.minutes import (  # noqa: E402
    iter_minute_dirs, read_minute, write_minute_manifest,
)


def migrate(root: Path, dry_run: bool = False) -> dict:
    report = {"root": str(root), "dry_run": dry_run,
              "migrated": [], "skipped": [], "failed": []}
    for d in iter_minute_dirs(root):
        target = d / "minute.json"
        if target.exists():
            report["skipped"].append({"minute": d.name,
                                      "reason": "already canonical"})
            continue
        try:
            manifest = read_minute(d)
            if not dry_run:
                write_minute_manifest(d, manifest)
            report["migrated"].append({
                "minute": d.name,
                "sources": len(manifest.sources),
                "predictions": len(manifest.predictions),
                "start_timestamp": manifest.start_timestamp,
            })
        except Exception as exc:
            report["failed"].append({"minute": d.name, "error": str(exc)})
    report["totals"] = {k: len(v) for k, v in report.items()
                        if isinstance(v, list)}
    return report


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0] if args else "data")
    print(json.dumps(migrate(root, dry_run="--dry-run" in sys.argv),
                     indent=2))
