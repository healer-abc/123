#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhsu/jyluo
exec "$ROOT/projects/openpi/.venv/bin/python" -u \
  "$ROOT/experiments/pi05_libero_adaptation/development_watchdog.py" "$@"
