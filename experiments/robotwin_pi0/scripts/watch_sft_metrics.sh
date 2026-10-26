#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL="${1:-30}"
if [[ ! "$INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash watch_sft_metrics.sh [refresh_seconds; 0 prints once]" >&2
  exit 2
fi
while true; do
  # Metrics use only CPU-side readers, without the training linker overrides.
  /usr/bin/env -u LD_PRELOAD -u LD_LIBRARY_PATH \
    ROBOTWIN_EXPERIMENT_ROOT="$PROJECT_DIR" \
    "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/summarize_sft_metrics.py"
  if (( INTERVAL == 0 )); then break; fi
  /usr/bin/env -u LD_PRELOAD -u LD_LIBRARY_PATH /usr/bin/sleep "$INTERVAL"
done
