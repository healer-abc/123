#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ─────────────────────────────────────────────────────────────────────────────
# auto_eval_after_train.sh
#
#   等 GPU1 上的**全参微调**自己结束 → 等最后一个 checkpoint 落盘 →
#   在 GPU1 上起 policy server → 跑 LIBERO 四套件评测 → 收尾并汇总。
#
# 为什么需要它：
#   全参训练预计 2026-10-14 14:39 结束；而 LoRA 备用实验还要在 **GPU0** 上
#   跑到 2026-10-15 20:22。中间若无人值守，GPU1 会空转几十小时。
#   本脚本把这段空档利用起来，且**只碰 GPU1**（红线：绝不能占 GPU0）。
#
# 用法（在 VSCode 远程终端里执行；先用 `source ~/jyluo/openpi_env.sh` 带好环境）：
#
#   # ① 只看就绪状态，不做任何启动（随时可跑，安全）
#   bash ~/jyluo/tools/auto_eval_after_train.sh --check
#
#   # ② 正式挂后台（断开 SSH 也不受影响）
#   nohup bash ~/jyluo/tools/auto_eval_after_train.sh \
#       > ~/jyluo/auto_eval.log 2>&1 & echo "PID=$!"
#
# 查看进度：tail -f ~/jyluo/auto_eval.log
#
# 停止：
#   pgrep -f auto_eval_after_train | xargs -r kill        # 停本脚本的等待循环
#   pgrep -f serve_policy         | xargs -r kill        # 停 policy server
#
# 结果：
#   ~/jyluo/eval_auto/eval.log          原始评测输出
#   ~/jyluo/eval_auto/server.log        policy server 日志
#   ~/jyluo/eval_auto/videos/           失败回放视频（默认只留失败）
#   ~/jyluo/eval_auto/parsed/           parse_eval_log.py 的结构化结果
# ─────────────────────────────────────────────────────────────────────────────
set -u

OPENPI_DIR="$HOME/jyluo/projects/openpi"
CKPT_ROOT="$OPENPI_DIR/checkpoints/pi05_libero/pi05_libero_full_gpu1"
OUT="$HOME/jyluo/eval_auto"
PARSER="$HOME/jyluo/tools/parse_eval_log.py"
SUITES=(libero_spatial libero_object libero_goal libero_10)

# 尾随空格很关键：把 pi05_libero 与 pi05_libero_lora 区分开（绝不误等/误杀 LoRA）
FULL_TRAIN_PAT='scripts/train\.py pi05_libero '

GPU=1                      # 红线：评测只准用 GPU1，GPU0 被 LoRA 占着
GPU_FREE_MIB=2000          # GPU 显存低于此值才算空闲
MEM_FRACTION=0.5

log() { printf '[auto-eval %s] %s\n' "$(date '+%F %T')" "$*"; }

# ── 0. 环境 ─────────────────────────────────────────────────────────────────
# 本脚本用 `nohup bash ... &` 从交互 shell 启动，环境变量会继承过来；
# 下面这行只是兜底（在脚本里直接 source 可能因缺少 conda init 而报错，故加 || true）。
[ -f "$HOME/jyluo/openpi_env.sh" ] && . "$HOME/jyluo/openpi_env.sh" >/dev/null 2>&1 || true

cd "$OPENPI_DIR" || { log "!! 找不到 $OPENPI_DIR"; exit 1; }
mkdir -p "$OUT" "$OUT/videos"

if ! command -v uv >/dev/null 2>&1; then
  log "!! PATH 里没有 uv —— 请先在交互 shell 里 source ~/jyluo/openpi_env.sh 再启动本脚本"
fi

# ── 工具函数 ────────────────────────────────────────────────────────────────

gpu_used() {  # $1 = GPU index -> 已用显存 MiB（失败则空）
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

latest_ckpt() {  # 最新的「已 finalize」checkpoint 目录名
  local d b
  for d in "$CKPT_ROOT"/*/; do
    b="$(basename "$d")"
    case "$b" in *tmp*|*.*) continue;; esac          # 跳过 N.orbax-checkpoint-tmp-M
    [ -f "$d/_CHECKPOINT_METADATA" ] && echo "$b"    # 只看写完了的
  done | sort -n | tail -1
}

readiness() {
  local last ck gu
  log "──────── 就绪检查 ────────"
  if pgrep -f "$FULL_TRAIN_PAT" >/dev/null; then
    last="$(pgrep -f "$FULL_TRAIN_PAT" | tail -1)"
    log "全参训练 : 仍在运行（PID $last，已跑 $(ps -o etime= -p "$last" 2>/dev/null | tr -d ' ')）"
  else
    log "全参训练 : 已结束 ✓"
  fi
  ck="$(latest_ckpt)"
  log "最新 ckpt: ${ck:-（暂无已 finalize 的 checkpoint）}"
  gu="$(gpu_used "$GPU")"
  log "GPU$GPU 显存: ${gu:-?} MiB  （< $GPU_FREE_MIB 才能起评测）"
  ls -d "$OPENPI_DIR"/checkpoints/pi05_libero_lora/*/ 2>/dev/null | sed 's/^/  LoRA 侧: /' || true
  log "解析器   : $([ -f "$PARSER" ] && echo '在 ✓' || echo '缺失 !!')"
  log "──────── 检查结束 ────────"
}

# ── --check：只读，不动任何东西 ─────────────────────────────────────────────
if [ "${1:-}" = "--check" ]; then
  readiness
  echo
  log "以上仅为检查，未启动任何进程。去掉 --check 即为正式运行。"
  exit 0
fi

# ── 1. 等全参训练结束 ───────────────────────────────────────────────────────
log "开始等待全参训练结束（pattern: $FULL_TRAIN_PAT）"
while pgrep -f "$FULL_TRAIN_PAT" >/dev/null; do sleep 60; done
log "全参训练已退出。等最后一个 checkpoint finalize…"

# ── 2. 等最后一个 checkpoint ────────────────────────────────────────────────
STEP=""
for _ in $(seq 1 120); do        # 最多等 60 分钟
  STEP="$(latest_ckpt)"
  [ -n "$STEP" ] && break
  sleep 30
done
if [ -z "$STEP" ]; then
  log "!! 等不到已 finalize 的 checkpoint，放弃（检查 $CKPT_ROOT）"
  exit 1
fi
log "使用 checkpoint：$CKPT_ROOT/$STEP"

# ── 3. GPU1 必须空闲（绝不抢卡）─────────────────────────────────────────────
U=""
for _ in $(seq 1 30); do         # 最多等 30 分钟
  U="$(gpu_used "$GPU")"
  [ -n "$U" ] && [ "$U" -lt "$GPU_FREE_MIB" ] && break
  log "GPU$GPU 仍被占用 ${U:-?} MiB，等 60s…"
  sleep 60
done
U="$(gpu_used "$GPU")"
if [ -z "$U" ] || [ "$U" -ge "$GPU_FREE_MIB" ]; then
  log "!! GPU$GPU 仍被占用（${U:-?} MiB），主动放弃，以免影响他人"
  exit 1
fi
log "GPU$GPU 空闲（${U} MiB）"

# ── 4. policy server（GPU1）──────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"
export WANDB_MODE=offline

nohup uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir="$CKPT_ROOT/$STEP" > "$OUT/server.log" 2>&1 &
SRV=$!
log "policy server PID=$SRV（GPU$GPU），等它起来…"

READY=0
for _ in $(seq 1 60); do         # 最多等 10 分钟
  sleep 10
  if ! kill -0 "$SRV" 2>/dev/null; then
    log "!! server 进程已退出，日志末尾："
    tail -20 "$OUT/server.log"
    exit 1
  fi
  if grep -q "Creating server" "$OUT/server.log" 2>/dev/null; then READY=1; break; fi
done
if [ "$READY" != 1 ]; then
  log "!! server 10 分钟内未就绪，日志末尾："
  tail -20 "$OUT/server.log"
  kill "$SRV" 2>/dev/null
  exit 1
fi
log "server 就绪：$(grep -m1 'Creating server' "$OUT/server.log")"

# ── 5. 客户端（LIBERO 3.8 venv）跑四套件 ────────────────────────────────────
# shellcheck disable=SC1091
. "$OPENPI_DIR/examples/libero/.venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_DIR/third_party/libero"
cd "$OPENPI_DIR"

# 探测"只留失败视频"的开关写法（tyro 对 bool 可能生成 --no-xxx，也可能接受 --xxx False）
HELP="$(python examples/libero/main.py --help 2>&1 || true)"
if printf '%s' "$HELP" | grep -q -- '--args.no-save-success-videos'; then
  SAVE_FLAG="--args.no-save-success-videos"
elif printf '%s' "$HELP" | grep -q -- '--args.save-success-videos'; then
  SAVE_FLAG="--args.save-success-videos False"
else
  SAVE_FLAG=""
  log "⚠️ 未识别到放弃保存成功视频的开关，将保留全部视频（占磁盘更多）"
fi
log "视频开关：${SAVE_FLAG:-（不使用）}"

for S in "${SUITES[@]}"; do
  log "════ $S 开始 ════"
  python examples/libero/main.py \
      --args.task-suite-name "$S" \
      --args.video-out-path "$OUT/videos" \
      $SAVE_FLAG 2>&1 | tee -a "$OUT/eval.log"
  log "════ $S 结束 ════"
done

# ── 6. 收尾 ─────────────────────────────────────────────────────────────────
kill "$SRV" 2>/dev/null
log "policy server 已停止"

if [ -f "$PARSER" ]; then
  log "解析评测日志…"
  python3 "$PARSER" "$OUT/eval.log" -o "$OUT/parsed" --video-dir "$OUT/videos" \
      >> "$OUT/auto_eval_tail.log" 2>&1 || log "（解析器返回非 0，结果见 $OUT/auto_eval_tail.log）"
fi

log "全部完成。各 suite 汇总："
grep -E "Total success rate|Task suite:" "$OUT/eval.log" 2>/dev/null | tail -20 || true
log "原始日志 : $OUT/eval.log"
log "结构化   : $OUT/parsed/eval_summary.json"
log "视频     : $OUT/videos"
log "★ 现在可以回来跑 LoRA 侧的解析了："
log "   python3 ~/jyluo/tools/parse_train_log.py -o ~/jyluo/metrics_lora --log train_lora.log"
