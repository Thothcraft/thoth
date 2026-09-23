"""``thoth doctor`` — node diagnostics."""

from __future__ import annotations

import platform
import shutil
import sys
from typing import Any, Dict


def run_doctor(config, client) -> Dict[str, Any]:
    report: Dict[str, Any] = {"checks": []}

    def check(name: str, ok: bool, detail: str = ""):
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        return ok

    check("python", sys.version_info >= (3, 10), platform.python_version())
    check("platform", True, f"{platform.system()} {platform.machine()}")

    try:
        import whispy  # noqa: F401
        check("whispy", True, getattr(whispy, "__version__", "installed"))
    except ImportError:
        check("whispy", False, "whispy not installed")

    check("config_dir", True, str(config.path.parent))
    check("device_id", bool(config.device_id), config.device_id)
    check("local_token", bool(config.local_token), "present")

    running = client.is_running()
    check("daemon", running,
          "running" if running else "not running (start with `thoth daemon`)")
    if running:
        try:
            st = client.status()
            check("sensors", bool(st.get("sensors")),
                  f"{len(st.get('sensors', []))} sensor(s)")
            check("streams", True,
                  f"{len(st.get('streams', {}))} stream(s)")
        except Exception as exc:
            check("daemon_status", False, str(exc))

    report["ok"] = all(c["ok"] for c in report["checks"]
                       if c["name"] not in ("daemon",))
    return report
