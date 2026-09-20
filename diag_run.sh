#!/bin/bash
for f in $(ls -t /home/gad/Desktop/thoth/data/*/manifest.json | head -3); do
  echo "===== $f"
  python3 /tmp/diag_one.py "$f"
done
