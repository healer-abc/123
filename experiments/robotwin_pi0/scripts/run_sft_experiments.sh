#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
exec 8> "$PROJECT_DIR/.sft-experiments.lock"
flock -n 8 || exit 75
if [[ -f "$PROJECT_DIR/report/sft-experiments-state.json" ]]; then
  echo "Existing experiment state; inspect it and use explicit resume rather than starting duplicate experiments." >&2
  exit 1
fi
printf '%s\n' "$$" > "$PROJECT_DIR/report/sft-experiments.pid"
python -u "$PROJECT_DIR/scripts/run_sft_experiments.py"
