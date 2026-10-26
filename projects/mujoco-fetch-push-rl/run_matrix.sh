#!/usr/bin/env bash
set -euo pipefail
# Run only after the single-seed pilot succeeds and resources are allocated.
MODE="${1:-core}"
SAC_DEVICE="${2:-cpu}"
STEPS="${3:-1000000}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p logs
case "$MODE" in
  core) groups=(D C B A) ;;
  extension) groups=(E) ;;
  *) echo 'Usage: bash run_matrix.sh core|extension cpu|cuda:0 [total_steps]' >&2; exit 2 ;;
esac
for group in "${groups[@]}"; do
  for seed in 0 1 2; do
    device="$SAC_DEVICE"
    if [[ "$group" == A ]]; then device=cpu; fi
    run="runs/${group}_seed${seed}"
    if [[ -e "$run" ]]; then
      echo "Output already exists: $run. Inspect it; no run will be overwritten." >&2
      exit 2
    fi
    python -u run.py train --group "$group" --seed "$seed" --steps "$STEPS" \
      --device "$device" --out "$run" 2>&1 | tee "logs/${group}_seed${seed}.log"
  done
done
