#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
cd "$RLINF_ROOT"
RUN_DIR="$PROJECT_DIR/runs/eval_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$PROJECT_DIR/report/latest-eval-smoke.txt"
export HYDRA_FULL_ERROR=1
set +e
python evaluations/eval_embodied_agent.py \
  --config-path "$PROJECT_DIR/configs" \
  --config-name eval_smoke_2gpu \
  "runner.logger.log_path=$RUN_DIR" \
  "$@" 2>&1 | tee "$RUN_DIR/run.log"
STATUS=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${STATUS[0]}" > "$RUN_DIR/exit-code.txt"
if (( STATUS[0] != 0 )); then exit "${STATUS[0]}"; fi
exit "${STATUS[1]}"
