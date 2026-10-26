# Source this file from bash before working on this project.
FETCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONDA_ENVS_PATH="$FETCH_ROOT/envs"
export CONDA_PKGS_DIRS="$FETCH_ROOT/.cache/conda"
source /home/zhsu/jyluo/miniforge3/etc/profile.d/conda.sh
conda activate "$FETCH_ROOT/envs/fetch-rl"
export UV_CACHE_DIR="$FETCH_ROOT/.cache/uv"
export PIP_CACHE_DIR="$FETCH_ROOT/.cache/pip"
export XDG_CACHE_HOME="$FETCH_ROOT/.cache"
export MPLCONFIGDIR="$FETCH_ROOT/.cache/matplotlib"
export PATH="$FETCH_ROOT/.local/tmux/bin:$PATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
cd "$FETCH_ROOT"
unset FETCH_ROOT
