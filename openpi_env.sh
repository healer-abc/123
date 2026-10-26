# openpi 环境激活脚本 —— 需要时执行: source ~/jyluo/openpi_env.sh
# 内容仅作用于当前 shell，不污染 ~/.bashrc

# C 编译器 (conda gcc)
export CC=/home/zhsu/jyluo/miniforge3/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/zhsu/jyluo/miniforge3/bin/x86_64-conda-linux-gnu-g++

# openpi 数据/缓存目录
export OPENPI_DATA_HOME=/home/zhsu/jyluo/myopenpi/dataset/ckpt
export HF_LEROBOT_HOME=/home/zhsu/jyluo/dataset_root
export UV_CACHE_DIR=/tmp/uv-cache
export UV_HTTP_TIMEOUT=300
export GIT_LFS_SKIP_SMUDGE=1

# wandb 离线模式(需要时取消注释)
# export WANDB_MODE=offline

echo "[openpi] 环境已激活: OPENPI_DATA_HOME=$OPENPI_DATA_HOME"

# LIBERO 评测用(在 openpi 目录下跑评测脚本时需要)
export PYTHONPATH=$PYTHONPATH:/home/zhsu/jyluo/projects/openpi/third_party/libero

# HuggingFace 镜像(服务器连不上官方 hf,用国内镜像)
export HF_ENDPOINT=https://hf-mirror.com
