#!/bin/bash
set -euo pipefail

PYTHON="/home/pi/Desktop/thoth/WS/MMW-HAT/venv/bin/python"
SCRIPT="/home/pi/Desktop/thoth/capture_dreamhat_minute.py"

if [[ "${EUID}" -eq 0 ]]; then
  exec "${PYTHON}" "${SCRIPT}" "$@"
fi

exec sudo -E "${PYTHON}" "${SCRIPT}" "$@"
