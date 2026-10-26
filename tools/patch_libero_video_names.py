#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 examples/libero/main.py 打补丁：视频文件名带 trial 序号 + 可选只存失败视频

为什么需要：
  官方 main.py:168-174 保存视频的文件名是
      rollout_{task_description}_{success|failure}.mp4
  **不含 trial 序号**。同一个 task 跑 50 次 rollout，同名文件互相覆盖，
  最终每个 task 只剩最后 1 个成功 + 1 个失败视频。
  想让"失败视频按 task 收集"这件事成立，必须把序号写进文件名。

本脚本做两件事（幂等，可重复执行）：
  1) Args 增加开关 `save_success_videos: bool = True`
     （跑全量四套件时建议用 --save-success-videos=False，只留失败视频，磁盘占用降一个数量级）
  2) 文件名加入 `t{task_id}_ep{episode_idx}`，并按开关跳过成功视频

用法
----
  # 先看要改什么（不改文件）
  python patch_libero_video_names.py --dry-run

  # 实际打补丁（自动备份为 main.py.bak.YYYYMMDD-HHMMSS）
  python patch_libero_video_names.py

  # 指定目标文件
  python patch_libero_video_names.py --file ~/jyluo/projects/openpi/examples/libero/main.py

回退
----
  cp examples/libero/main.py.bak.<时间戳> examples/libero/main.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

DEFAULT_FILE = "examples/libero/main.py"
MARKER = "[patched-by-tools/patch_libero_video_names.py]"

OLD_ARG = '    video_out_path: str = "data/libero/videos"  # Path to save videos\n'
NEW_ARG = (
    OLD_ARG
    + "    save_success_videos: bool = True  "
    + "# False = 只保存失败视频（跑全量四件套时省磁盘）  " + MARKER + "\n"
)

OLD_BLOCK = '''            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )'''

NEW_BLOCK = '''            suffix = "success" if done else "failure"
            # ''' + MARKER + '''
            # 原代码文件名不含 trial 序号，同 task 的多次 rollout 会互相覆盖。
            if (not done) or args.save_success_videos:
                task_segment = task_description.replace(" ", "_")[:60]
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_t{task_id}_ep{episode_idx}_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )'''


def main(argv=None):
    ap = argparse.ArgumentParser(description="patch LIBERO eval video filenames")
    ap.add_argument("--file", default=DEFAULT_FILE, help="main.py 路径")
    ap.add_argument("--dry-run", action="store_true", help="只显示将要做出的改动")
    args = ap.parse_args(argv)

    path = os.path.abspath(os.path.expanduser(args.file))
    if not os.path.exists(path):
        print("[error] 找不到文件: %s" % path, file=sys.stderr)
        return 2

    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print("[skip] 该文件已经打过补丁（检测到标记），无需重复执行。")
        return 0

    if OLD_ARG not in src:
        print("[error] 没找到 Args 里的 video_out_path 行，请确认文件版本是否符合预期。",
              file=sys.stderr)
        return 2
    if OLD_BLOCK not in src:
        print("[error] 没找到视频保存代码块，请确认文件版本是否符合预期。", file=sys.stderr)
        return 2

    new_src = src.replace(OLD_ARG, NEW_ARG, 1).replace(OLD_BLOCK, NEW_BLOCK, 1)

    if args.dry_run:
        print("=" * 66)
        print("DRY RUN —— 以下内容不会被写入")
        print("=" * 66)
        print("[改动 1/2] Args 新增开关：")
        print("  + " + NEW_ARG.rstrip().split(MARKER)[0].rstrip())
        print()
        print("[改动 2/2] 视频保存块：")
        print(NEW_BLOCK)
        print("=" * 66)
        print("预计新增 %d 字节" % (len(new_src) - len(src)))
        return 0

    backup = "%s.bak.%s" % (path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_src)

    print("=" * 66)
    print("补丁已应用")
    print("=" * 66)
    print("目标文件 : %s" % path)
    print("备份文件 : %s   （回退: cp '%s' '%s'）" % (backup, backup, path))
    print("改动     : 2 处（Args 新增 save_success_videos；视频名加 t{task_id}_ep{episode_idx}）")
    print()
    print("用法：")
    print("  # 小规模试跑：保留全部视频（默认）")
    print("  python examples/libero/main.py --task-suite-name libero_spatial --num-trials-per-task 5")
    print()
    print("  # 全量四套件：只留失败视频，磁盘占用降一个数量级")
    print("  python examples/libero/main.py --task-suite-name libero_10 --save-success-videos False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
