#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
cd "$RLINF_ROOT"
RUN_DIR="$PROJECT_DIR/runs/sft_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$PROJECT_DIR/report/latest-sft-smoke.txt"
export HYDRA_FULL_ERROR=1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$RUN_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
set +e
python examples/sft/train_vla_sft.py \
  --config-path "$PROJECT_DIR/configs" --config-name sft_smoke_2gpu \
  "runner.logger.log_path=$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/run.log"
STATUS=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${STATUS[0]}" > "$RUN_DIR/exit-code.txt"
if (( STATUS[0] != 0 )); then exit "${STATUS[0]}"; fi
exit "${STATUS[1]}"
