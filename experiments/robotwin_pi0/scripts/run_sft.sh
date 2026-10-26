#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
CONFIG="${1:?config name required}"
RUN_DIR="${2:?absolute run directory required}"
shift 2
mkdir -p "$RUN_DIR"
exec 9> "$RUN_DIR/.run.lock"
flock -n 9 || exit 75
if [[ -f "$RUN_DIR/exit-code.txt" ]]; then
  echo "Existing completed or failed run; use a new run directory or an explicit resume config: $RUN_DIR" >&2
  exit 1
fi
printf '%s\n' "$$" > "$RUN_DIR/launcher.pid"
# Each experiment owns a separate Ray cluster, rendezvous and log directory.
export RAY_ADDRESS=local
export RAY_TMPDIR="/tmp/rw-sft-$(id -u)-$$"
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=1073741824
export HYDRA_FULL_ERROR=1
cp "$PROJECT_DIR/configs/$CONFIG.yaml" "$RUN_DIR/input-config.yaml"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$RUN_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
cd "$RLINF_ROOT"
set +e
python examples/sft/train_vla_sft.py \
  --config-path "$PROJECT_DIR/configs" --config-name "$CONFIG" \
  "runner.logger.log_path=$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/run.log"
STATUS=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${STATUS[0]}" > "$RUN_DIR/exit-code.txt"
if (( STATUS[0] != 0 )); then exit "${STATUS[0]}"; fi
exit "${STATUS[1]}"
