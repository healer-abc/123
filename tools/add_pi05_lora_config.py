#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 src/openpi/training/config.py 追加 `pi05_libero_lora` 配置（LoRA 微调，备选实验）

背景
----
openpi 官方仓库（本 commit）只提供了 **pi0** 的 LoRA 示例：
    pi0_libero_low_mem_finetune / pi0_fast_libero_low_mem_finetune
**没有 pi05 的 LoRA 配置**。本脚本按官方示例的同构写法，为 pi05 + LIBERO 生成一份。

关键点（源码级核实）
----
1. `freeze_filter` 必须与 `model` 的变体一致，`TrainConfig` 不会自动推导
   （`config.py:689-695` 官方注释亦如此强调）。
2. `TrainConfig.trainable_filter = nnx.All(nnx.Param, nnx.Not(self.freeze_filter))`
   （`config.py:549-552`）。
3. `Pi0Config.get_freeze_filter()`（`models/pi0_config.py:88-117`）对
   paligemma=lora + action_expert=lora 返回：
       nnx.All(PathRegex(".*llm.*"), nnx.Not(PathRegex(".*lora.*")))
   ⇒ 冻结 `PaliGemma/llm/**` 里所有非 lora 参数；
      可训练 = 所有 `*lora*` 参数 + **所有路径不含 "llm" 的参数**
      （`PaliGemma/img/*` SigLIP 视觉塔、action_in_proj、pi05 的 time_mlp_*、action_out_proj）。
4. `CheckpointWeightLoader.load` 内部 `missing_regex=".*lora.*"`
   （`training/weight_loaders.py:54`）—— 加载 pi05_base 时会自动把缺失的 LoRA 权重
   用随机初始化补齐，**可以直接加载**，无需额外处理。

用法
----
  python3 add_pi05_lora_config.py --dry-run     # 先看将插入什么
  python3 add_pi05_lora_config.py               # 应用（自动备份）

回退
----
  cp src/openpi/training/config.py.bak.<时间戳> src/openpi/training/config.py

之后即可直接跑（注意 assets / norm_stats 复用 pi05_libero 的，不必重算）：
  uv run scripts/train.py pi05_libero_lora --exp-name=pi05_libero_lora_gpu1 --batch-size=32
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

DEFAULT_FILE = "src/openpi/training/config.py"

# 插入锚点：pi05_libero 配置块的结尾（已核实该文本在文件中唯一）
ANCHOR = (
    '        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),\n'
    '        pytorch_weight_path="/path/to/your/pytorch_weight_path",\n'
    "        num_train_steps=30_000,\n"
    "    ),\n"
)

MARKER = "pi05_libero_lora"

NEW_BLOCK = '''
    #
    # pi05 + LIBERO 的 LoRA 微调（本仓库自行添加；官方示例只有 pi0 的 LoRA 配置）
    #
    TrainConfig(
        name="pi05_libero_lora",
        # 与 pi05_libero 的唯一区别：两个变体改为 *_lora，并配 freeze_filter + 关 EMA。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            # 注意：pi05 用 False；pi0 的 LoRA 示例用的是 True，别抄错。
            extra_delta_transform=False,
            # ⚠️ 必须显式指向 pi05_libero 已有的 assets：
            #   assets_dirs = assets_base_dir / config.name （config.py:537-540）
            #   新配置名会去找 assets/pi05_libero_lora/，那里没有 norm stats
            #   → data_loader.py:176-180 直接 raise ValueError("Normalization stats not found")
            # 用 AssetsConfig 显式指定父目录（asset_id 默认取 repo_id），即可复用
            # 已算好的归一化统计，省掉约 42 分钟重算。
            assets=AssetsConfig(assets_dir="./assets/pi05_libero"),
        ),
        # LoRA 可训练参数远少于全参微调，显存更省，batch 可在命令行调大。
        lr_schedule=_optimizer.CosineDecaySchedule(
            # 建议起点：LoRA 常用比全参微调更大的学习率（pi05_libero 全参是 5e-5）。
            warmup_steps=1_000,
            peak_lr=2e-4,
            decay_steps=30_000,
            decay_lr=2e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,
        # freeze_filter 必须与上面的 model 变体一致（官方注释：just make sure it matches）。
        # 实际上只有 paligemma_variant / action_expert_variant 两个字段参与计算，
        # 其余字段原样复制一份只为便于人工核对。
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # LoRA 微调关闭 EMA（与官方 pi0 LoRA 示例一致）。
        ema_decay=None,
    ),
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description="add pi05_libero_lora config")
    ap.add_argument("--file", default=DEFAULT_FILE, help="config.py 路径")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    path = os.path.abspath(os.path.expanduser(args.file))
    if not os.path.exists(path):
        print("[error] 找不到文件: %s" % path, file=sys.stderr)
        return 2

    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    if 'name="%s"' % MARKER in src:
        print("[skip] 已存在 name=%r 的配置，无需重复添加。" % MARKER)
        return 0

    if src.count(ANCHOR) != 1:
        print(
            "[error] 插入锚点匹配 %d 次（期望 1 次），请确认 config.py 版本。" % src.count(ANCHOR),
            file=sys.stderr,
        )
        return 2

    new_src = src.replace(ANCHOR, ANCHOR + NEW_BLOCK, 1)

    if args.dry_run:
        print("=" * 70)
        print("DRY RUN —— 将插入以下内容（不会被写入）")
        print("=" * 70)
        print(NEW_BLOCK)
        print("=" * 70)
        print("插入位置: %s" % path)
        print("预计新增 %d 字节" % (len(new_src) - len(src)))
        return 0

    backup = "%s.bak.%s" % (path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_src)

    print("=" * 70)
    print("已添加配置 pi05_libero_lora")
    print("=" * 70)
    print("文件   : %s" % path)
    print("备份   : %s" % backup)
    print("回退   : cp '%s' '%s'" % (backup, path))
    print()
    print("本配置已用 AssetsConfig 显式复用 assets/pi05_libero（否则会因缺 norm stats 报错）。")
    print("等价替代做法（二选一即可，本配置已内建，无需再做）：")
    print("  cd assets && ln -s pi05_libero pi05_libero_lora")
    print()
    print("自检建议（在服务器上执行）:")
    print("  python3 -c \"import ast; ast.parse(open('%s').read()); print('syntax OK')\"" % path)
    print("  uv run scripts/train.py pi05_libero_lora --help >/dev/null && echo 'config 可被 tyro 加载'")
    print()
    print("⚠️ 小规模验证（必须在空闲卡上；若用 GPU0 一定要锁卡，否则 JAX 会去抢 GPU1）:")
    print("  CUDA_VISIBLE_DEVICES=0 uv run python -c \"import jax; print(jax.devices())\"")
    print("      # 期望只列出 1 个设备；若列出 2 个 → 立刻中止，不要继续")
    print("  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \\")
    print("    uv run scripts/train.py pi05_libero_lora \\")
    print("        --exp-name=pi05_libero_lora_smoke --batch-size=2 --num-train-steps=10")
    print("      # 日志里要能看到 'Loaded norm stats from' 和 'Step 0: loss=...'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
