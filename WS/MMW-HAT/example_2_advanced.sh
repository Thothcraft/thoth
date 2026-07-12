#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$SCRIPT_DIR/MMW-HAT-Release/example_2_advanced"

cd "$EXAMPLE_DIR" || exit 1
python3 run_example_advanced.py
