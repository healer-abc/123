#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
python -u tools/check_cpu_environment.py 2>&1 | tee logs/cpu_environment.log
for task in Reach Push; do
    python -u run.py check --task "$task" 2>&1 | tee "logs/check_${task}.log"
done
python -u run.py train --task Reach --group D --seed 99 --steps 3000 \
    --buffer-size 20000 --learning-starts 100 --eval-every 1000 \
    --eval-episodes 5 --checkpoint-every 1000 --device cpu \
    --out runs/reach_smoke 2>&1 | tee logs/reach_smoke.log
python -u run.py eval --checkpoint runs/reach_smoke/final --episodes 10 \
    --seed-base 10000 --device cpu --out evaluations/reach_smoke \
    2>&1 | tee logs/reach_smoke_eval.log
python -u run.py train --resume runs/reach_smoke/final --steps 5000 \
    --eval-every 1000 --eval-episodes 5 --checkpoint-every 1000 --device cpu \
    --out runs/reach_resume 2>&1 | tee logs/reach_resume.log
for group in A B C D E; do
    steps=300
    if [[ "$group" == A ]]; then steps=2048; fi
    python -u run.py train --task Push --group "$group" --seed 99 --steps "$steps" \
        --buffer-size 20000 --learning-starts 100 --eval-every "$steps" \
        --eval-episodes 2 --device cpu --out "runs/smoke_${group}_push" \
        2>&1 | tee "logs/smoke_${group}_push.log"
done
for group in D E; do
    python -u run.py eval --checkpoint "runs/smoke_${group}_push/final" \
        --episodes 2 --gains 0.6 0.8 1.0 1.2 1.4 --device cpu \
        --out "evaluations/smoke_${group}_gains" \
        2>&1 | tee "logs/smoke_${group}_gains.log"
done
python -u tools/verify_acceptance.py 2>&1 | tee logs/verify_acceptance.log
