#!/bin/bash
for f in $(ls -t /home/gad/Desktop/thoth/data/*/manifest.json | head -6); do
  python3 /tmp/diag_one.py "$f" 2>/dev/null
done