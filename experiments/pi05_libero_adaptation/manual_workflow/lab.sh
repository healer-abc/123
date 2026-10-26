# Source this file in bash. Sourcing defines commands; it starts no experiment.
# Override LAB_TAG before sourcing to start a distinct set of experiments.
export LAB_WORKSPACE=/home/zhsu/jyluo
export LAB_OPENPI="$LAB_WORKSPACE/projects/openpi"
export LAB_PROJECT="$LAB_WORKSPACE/experiments/pi05_libero_adaptation"
export LAB_TAG="${LAB_TAG:-manual_20261023_01}"
export LAB_ROOT="$LAB_PROJECT/manual_runs/$LAB_TAG"
export LAB_TRAIN_PY="${LAB_TRAIN_PY:-$LAB_OPENPI/.venv/bin/python}"
export LAB_CLIENT_PY="${LAB_CLIENT_PY:-$LAB_OPENPI/examples/libero/.venv/bin/python}"
export OPENPI_DATA_HOME="$LAB_WORKSPACE/myopenpi/dataset/ckpt"
export HF_LEROBOT_HOME="$LAB_WORKSPACE/dataset_root"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$LAB_OPENPI/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"

lab_use() {
    case "$1" in
        E0) export LAB_CONFIG=pi05_libero ;;
        E1) export LAB_CONFIG=pi05_libero_lora ;;
        E2) export LAB_CONFIG=pi05_libero_lora_frozen_vision ;;
        E3) export LAB_CONFIG=pi05_libero_action_only ;;
        *) echo 'usage: lab_use E0|E1|E2|E3 [physical_GPU=0] [port=8123]' >&2; return 2 ;;
    esac
    export LAB_ARM="$1" LAB_GPU="${2:-0}" LAB_PORT="${3:-8123}"
    export LAB_RUN="${LAB_TAG}_${LAB_ARM}_s42"
    export LAB_LOG="$LAB_ROOT/$LAB_ARM"
    export LAB_CKPT="$LAB_OPENPI/checkpoints/$LAB_CONFIG/$LAB_RUN/29999"
    mkdir -p "$LAB_LOG"
    printf 'run=%s\nconfig=%s\nGPU=%s port=%s\ncheckpoint=%s\n' "$LAB_RUN" "$LAB_CONFIG" "$LAB_GPU" "$LAB_PORT" "$LAB_CKPT"
}

lab_smoke() (
    set -euo pipefail
    cd "$LAB_OPENPI"
    test ! -e "checkpoints/$LAB_CONFIG/${LAB_RUN}_smoke" || { echo 'Smoke already exists; choose a new LAB_TAG.' >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="$LAB_GPU" "$LAB_TRAIN_PY" -u scripts/train.py "$LAB_CONFIG" \
        --exp-name="${LAB_RUN}_smoke" --batch-size=32 --seed=42 \
        --num-train-steps=10 --log-interval=1 2>&1 | tee "$LAB_LOG/smoke.log"
    test -f "checkpoints/$LAB_CONFIG/${LAB_RUN}_smoke/9/_CHECKPOINT_METADATA"
)

lab_train() (
    set -euo pipefail
    cd "$LAB_OPENPI"
    local resume=()
    if [ "${1:-}" = resume ]; then
        test -d "checkpoints/$LAB_CONFIG/$LAB_RUN"
        resume=(--resume)
    elif [ -n "${1:-}" ]; then
        echo 'usage: lab_train [resume]' >&2; exit 2
    else
        test ! -e "checkpoints/$LAB_CONFIG/$LAB_RUN" || { echo 'Run exists: use resume or a new LAB_TAG.' >&2; exit 2; }
    fi
    CUDA_VISIBLE_DEVICES="$LAB_GPU" "$LAB_TRAIN_PY" -u scripts/train.py "$LAB_CONFIG" \
        --exp-name="$LAB_RUN" --batch-size=32 --seed=42 --num-train-steps=30000 \
        "${resume[@]}" 2>&1 | tee "$LAB_LOG/train_$(date +%Y%m%d_%H%M%S)_$$.log"
    test -f "$LAB_CKPT/_CHECKPOINT_METADATA"
)

lab_serve() (
    set -euo pipefail
    cd "$LAB_OPENPI"
    test -f "$LAB_CKPT/_CHECKPOINT_METADATA"
    CUDA_VISIBLE_DEVICES="$LAB_GPU" "$LAB_TRAIN_PY" -u "$LAB_PROJECT/seeded_policy_server.py" \
        --config="$LAB_CONFIG" --checkpoint="$LAB_CKPT" --port="$LAB_PORT" \
        2>&1 | tee "$LAB_LOG/server_$(date +%Y%m%d_%H%M%S)_$$.log"
)

lab_eval() (
    set -euo pipefail
    local phase="${1:-development}"
    local suites=(libero_spatial libero_10)
    case "$phase" in
        development) ;;
        confirmation) test -f "$LAB_ROOT/freeze.json" || { echo 'Freeze candidate first.' >&2; exit 2; }
                      suites=(libero_spatial libero_object libero_goal libero_10) ;;
        *) echo 'phase must be development or confirmation' >&2; exit 2 ;;
    esac
    cd "$LAB_OPENPI"
    if [ "$phase" = confirmation ]; then
        "$LAB_TRAIN_PY" "$LAB_PROJECT/manual_workflow/results.py" check-freeze --root "$LAB_ROOT" --require-arm "$LAB_ARM"
    fi
    CUDA_VISIBLE_DEVICES="$LAB_GPU" MUJOCO_EGL_DEVICE_ID="$LAB_GPU" \
        "$LAB_CLIENT_PY" -u "$LAB_PROJECT/eval_manifest.py" \
        --run-id="$LAB_RUN" --config="$LAB_CONFIG" --checkpoint="$LAB_CKPT" \
        --phase="$phase" --suites "${suites[@]}" --host=127.0.0.1 --port="$LAB_PORT" \
        2>&1 | tee "$LAB_LOG/eval_${phase}_$(date +%Y%m%d_%H%M%S)_$$.log"
)

lab_profile() (
    set -euo pipefail
    cd "$LAB_OPENPI"
    test ! -e "checkpoints/$LAB_CONFIG/${LAB_RUN}_profile" || { echo 'Profile exists; choose a new LAB_TAG.' >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="$LAB_GPU" "$LAB_TRAIN_PY" -u "$LAB_PROJECT/profile_training.py" \
        --config="$LAB_CONFIG" --exp-name="${LAB_RUN}_profile" --output="$LAB_LOG/profile_steps.jsonl" \
        2>&1 | tee "$LAB_LOG/profile.log"
)

lab_status() {
    local phase="${1:-development}"
    "$LAB_TRAIN_PY" -m json.tool "$LAB_PROJECT/evaluations/$LAB_RUN/$phase/summary.json"
}
