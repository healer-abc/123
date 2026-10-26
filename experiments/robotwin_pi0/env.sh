# Run: source /home/zhsu/jyluo/experiments/robotwin_pi0/env.sh
ROBOTWIN_EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROBOTWIN_EXPERIMENT_ROOT
export ROBOTWIN_WORKSPACE="$(cd "$ROBOTWIN_EXPERIMENT_ROOT/../.." && pwd)"
export RLINF_ROOT="$ROBOTWIN_WORKSPACE/projects/RLinf"
export ROBOTWIN_PATH="$ROBOTWIN_WORKSPACE/projects/RoboTwin"
export REPO_PATH="$RLINF_ROOT"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export ROBOT_PLATFORM=ALOHA
export PATH="$ROBOTWIN_EXPERIMENT_ROOT/toolchain/bin:$PATH"
source "$ROBOTWIN_EXPERIMENT_ROOT/.venv/bin/activate"
export LD_LIBRARY_PATH="$ROBOTWIN_EXPERIMENT_ROOT/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROBOTWIN_EXPERIMENT_ROOT/compat:$RLINF_ROOT:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export RLINF_EXT_MODULE=robotwin_pi0_ext
export XDG_CACHE_HOME="$ROBOTWIN_EXPERIMENT_ROOT/.cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_LEROBOT_HOME="$ROBOTWIN_WORKSPACE/data/robotwin_pi0/lerobot"
export OPENPI_DATA_HOME="$ROBOTWIN_WORKSPACE/models/robotwin_pi0/openpi"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export RAY_TMPDIR="/tmp/robotwin-pi0-ray-$(id -u)"
export DOWNLOAD_DIR="$ROBOTWIN_EXPERIMENT_ROOT"
export UV_HTTP_TIMEOUT=300
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CC="$ROBOTWIN_WORKSPACE/miniforge3/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ROBOTWIN_WORKSPACE/miniforge3/bin/x86_64-conda-linux-gnu-g++"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
export PYOPENGL_PLATFORM=egl
# Use the same explicit rendering protocol in collection, SFT evaluation and RL.
export ROBOTWIN_DENOISER=none
export CUDA_HOME="$ROBOTWIN_EXPERIMENT_ROOT/toolchain"
if [[ -x "$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++" ]]; then
    export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
fi
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
# NCCL 2.26.2 bundled with Torch crashes on large broadcasts on this Blackwell
# pair. Keep Torch's pinned dependencies, preload the verified project runtime.
ROBOTWIN_NCCL_LIBRARY="$ROBOTWIN_EXPERIMENT_ROOT/runtime/nccl-2.31.2/nvidia/nccl/lib/libnccl.so.2"
if [[ -f "$ROBOTWIN_NCCL_LIBRARY" && ":${LD_PRELOAD:-}:" != *":$ROBOTWIN_NCCL_LIBRARY:"* ]]; then
    export LD_PRELOAD="$ROBOTWIN_NCCL_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
fi
unset ROBOTWIN_NCCL_LIBRARY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}pypi.tuna.tsinghua.edu.cn,mirrors.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org,download.pytorch.org,hf-mirror.com,.xethub.hf.co"
export no_proxy="$NO_PROXY"
