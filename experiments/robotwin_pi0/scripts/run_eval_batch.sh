#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
RUN_DIR="${1:?absolute batch directory required}"
export RLINF_EXT_MODULE=robotwin_pi0_eval_ext
export ROBOTWIN_EVAL_AUDIT_DIR="$RUN_DIR/audit"
export RAY_ADDRESS=local
export RAY_TMPDIR="/dev/shm/rw-eval-$(id -u)-$$"
export TMPDIR="$RAY_TMPDIR/tmp"
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=268435456
export HYDRA_FULL_ERROR=1
mkdir -p "$TMPDIR"
printf '%s\n' "$RAY_TMPDIR" > "$RUN_DIR/ray-tmp-dir.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 2 > "$RUN_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
cd "$RLINF_ROOT"
set +e
python evaluations/eval_embodied_agent.py --config-path "$RUN_DIR" --config-name config > "$RUN_DIR/run.log" 2>&1
CODE=$?
set -e
printf '%s\n' "$CODE" > "$RUN_DIR/exit-code.txt"
exit "$CODE"
