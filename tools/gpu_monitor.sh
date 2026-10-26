#!/usr/bin/env bash
# GPU 状态 1Hz 采样 -> CSV（给将来的实验记录显存/利用率/功耗曲线用）
#
# 用法:
#   ./gpu_monitor.sh [输出文件] [采样间隔秒] [GPU 编号]
#   ./gpu_monitor.sh gpu.csv 1            # 每秒一次，所有卡
#   ./gpu_monitor.sh gpu1.csv 1 1         # 每秒一次，只看 GPU1
#
# 后台跑（不占终端）:
#   nohup ./gpu_monitor.sh ~/jyluo/gpu.csv 1 >/dev/null 2>&1 &
#   停止: pkill -f gpu_monitor.sh
#
# 重要说明（关于"显存峰值"）:
#   openpi 用 JAX，且默认开启预分配（XLA_PYTHON_CLIENT_PREALLOCATE=true），
#   因此显存占用在训练开始后很快就贴着 XLA_PYTHON_CLIENT_MEM_FRACTION 设定的
#   上限（我们的设定是 0.9）并基本走平。所以"峰值"≈预分配上限，
#   逐秒采样得到的基本是一条平线 —— 这是正常现象，不是采样失效。
#   （进程中途可能因异步 checkpoint 缓冲而小幅上浮，如 78G -> 85G。）

OUT="${1:-gpu_monitor.csv}"
INTERVAL="${2:-1}"
GPU_FILTER="${3:-}"

QUERY="index,memory.used,memory.total,utilization.gpu,power.draw"
if [ -n "$GPU_FILTER" ]; then
  IDS="-i ${GPU_FILTER}"
else
  IDS=""
fi

echo "timestamp,index,mem_used_mib,mem_total_mib,util_pct,power_w" > "$OUT"
echo "[gpu_monitor] 开始采样 -> $OUT  (间隔 ${INTERVAL}s, GPU='${GPU_FILTER:-all}')"
echo "[gpu_monitor] 停止: pkill -f gpu_monitor.sh"

while true; do
  TS="$(date '+%Y-%m-%d %H:%M:%S')"
  nvidia-smi $IDS --query-gpu="$QUERY" --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' \
    | while IFS=, read -r idx mu mt ut pw; do
        [ -z "$idx" ] && continue
        printf '%s,%s,%s,%s,%s,%s\n' "$TS" "$idx" "$mu" "$mt" "$ut" "$pw" >> "$OUT"
      done
  sleep "$INTERVAL"
done
