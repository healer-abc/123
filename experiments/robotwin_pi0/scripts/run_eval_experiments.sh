#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
exec 8> "$PROJECT_DIR/.eval-experiments.lock"
flock -n 8 || exit 75
if [[ -f "$PROJECT_DIR/report/eval-experiments-state.json" && "${1:-}" != "--resume" ]]; then
  echo "Existing evaluation state; inspect it before starting another run." >&2
  exit 1
fi
printf '%s\n' "$$" > "$PROJECT_DIR/report/eval-experiments.pid"
python -u "$PROJECT_DIR/scripts/run_eval_experiments.py" "$@"
