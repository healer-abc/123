#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LIBERO 评测日志 -> 结构化结果（JSON / CSV）

解析 examples/libero/main.py 的标准输出（stdout/stderr 重定向到文件即可），产出：

  eval_summary.json   每个 suite 的 task 成功率、总体成功率、轮次数、异常数
  episodes.csv        每次 rollout 一行：seq, suite, task, success
  task_rates.csv      每个 task 一行：suite, task, episodes, 官方日志里的成功率
  failure_videos.json 失败视频清单（按 task 分组，含空的 failure_type 字段供人工/模型标注）

用法
----
  # 只解析日志
  python parse_eval_log.py eval_spatial.log -o ./eval_out

  # 同时扫描视频目录（默认 data/libero/videos），统计成功/失败视频
  python parse_eval_log.py eval_spatial.log -o ./eval_out \
      --video-dir ~/jyluo/projects/openpi/data/libero/videos

仅依赖标准库（Python 3.8+ 可用）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

SUITE_RE = re.compile(r"Task suite:\s*(?P<suite>\S+)")
TASK_RE = re.compile(r"Task:\s*(?P<task>.+?)\s*$")
SUCCESS_RE = re.compile(r"Success:\s*(?P<val>True|False)")
CUR_TASK_RATE_RE = re.compile(r"Current task success rate:\s*(?P<v>[0-9.]+)")
CUR_TOTAL_RATE_RE = re.compile(r"Current total success rate:\s*(?P<v>[0-9.]+)")
TOTAL_RATE_RE = re.compile(r"^.*Total success rate:\s*(?P<v>[0-9.]+)")
TOTAL_EPISODES_RE = re.compile(r"Total episodes:\s*(?P<v>\d+)")
EXC_RE = re.compile(r"Caught exception:")
EPISODES_SO_FAR_RE = re.compile(r"# episodes completed so far:\s*(?P<v>\d+)")


def parse_logs(paths):
    suite = None
    task = None
    episodes = []
    task_rates = []
    final = {}
    exceptions = []
    for path in paths:
        if not os.path.exists(path):
            print("[warn] 日志不存在，跳过: %s" % path, file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                m = SUITE_RE.search(line)
                if m:
                    suite = m.group("suite")
                    continue
                m = TASK_RE.search(line)
                if m and "Task suite" not in line:
                    t = m.group("task")
                    if t and t != task:
                        task = t
                    continue
                if EXC_RE.search(line):
                    exceptions.append({"suite": suite, "task": task, "line": line.strip()})
                    continue
                m = SUCCESS_RE.search(line)
                if m:
                    episodes.append(
                        {
                            "seq": len(episodes) + 1,
                            "suite": suite,
                            "task": task,
                            "success": m.group("val") == "True",
                        }
                    )
                    continue
                m = CUR_TASK_RATE_RE.search(line)
                if m:
                    task_rates.append(
                        {"suite": suite, "task": task, "reported_rate": float(m.group("v"))}
                    )
                    continue
                m = CUR_TOTAL_RATE_RE.search(line)
                if m:
                    final.setdefault("per_suite_total_rate", []).append(
                        {"suite": suite, "rate": float(m.group("v"))}
                    )
                    continue
                m = TOTAL_RATE_RE.search(line)
                if m:
                    final["total_success_rate"] = float(m.group("v"))
                    continue
                m = TOTAL_EPISODES_RE.search(line)
                if m:
                    final["total_episodes_reported"] = int(m.group("v"))
                    continue
    return episodes, task_rates, final, exceptions


def scan_videos(video_dir):
    """返回 {'success': {...}, 'failure': {...}}，按 task 名分组计数并列出文件。"""
    out = {"success": {}, "failure": {}}
    if not video_dir or not os.path.isdir(video_dir):
        return out
    for name in sorted(os.listdir(video_dir)):
        if not name.startswith("rollout_") or not name.endswith(".mp4"):
            continue
        kind = "success" if name.endswith("_success.mp4") else "failure"
        task = name[len("rollout_"):-len("_" + kind + ".mp4")]
        entry = out[kind].setdefault(task, {"count": 0, "files": []})
        entry["count"] += 1
        entry["files"].append(os.path.join(video_dir, name))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="LIBERO 评测日志解析器")
    ap.add_argument("logfiles", nargs="*", help="评测日志文件（等价于 --log）")
    ap.add_argument("--log", action="append", default=[], help="可重复")
    ap.add_argument("-o", "--outdir", default="eval_out")
    ap.add_argument("--video-dir", default=None, help="视频目录，如 data/libero/videos")
    args = ap.parse_args(argv)

    paths = list(args.logfiles) + list(args.log)
    if not paths:
        ap.error("请至少给一个评测日志路径")

    episodes, task_rates, final, exceptions = parse_logs(paths)
    if not episodes and not final:
        print("[error] 没解析到评测结果（未找到 'Success: True/False' 等行）", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    ep_csv = os.path.join(args.outdir, "episodes.csv")
    with open(ep_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seq", "suite", "task", "success"])
        w.writeheader()
        for r in episodes:
            w.writerow(r)

    tr_csv = os.path.join(args.outdir, "task_rates.csv")
    with open(tr_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["suite", "task", "reported_rate"])
        w.writeheader()
        for r in task_rates:
            w.writerow(r)

    # 按 suite 汇总（用逐轮记录自己算，避免依赖日志里的汇总行）
    by_suite = {}
    for r in episodes:
        s = by_suite.setdefault(r["suite"], {"episodes": 0, "successes": 0})
        s["episodes"] += 1
        s["successes"] += 1 if r["success"] else 0
    for s, v in by_suite.items():
        v["success_rate"] = (v["successes"] / v["episodes"]) if v["episodes"] else None

    videos = scan_videos(args.video_dir)
    failure_list = []
    for task, entry in sorted(videos["failure"].items()):
        failure_list.append(
            {
                "task": task,
                "video_count": entry["count"],
                "files": entry["files"],
                "failure_type": "",  # 留空，供人工 / 多模态模型事后标注
            }
        )

    total_ep = len(episodes)
    total_succ = sum(1 for r in episodes if r["success"])
    summary = {
        "logs": paths,
        "video_dir": args.video_dir,
        "episodes_parsed": total_ep,
        "successes": total_succ,
        "success_rate_from_episodes": (total_succ / total_ep) if total_ep else None,
        "per_suite": by_suite,
        "reported": final,
        "exceptions": exceptions,
        "videos": {
            "success_videos": sum(v["count"] for v in videos["success"].values()),
            "failure_videos": sum(v["count"] for v in videos["failure"].values()),
            "note": (
                "openpi 原生 examples/libero/main.py 的视频文件名不含 trial 序号，"
                "同一 task 的多次 rollout 会互相覆盖（每 task 最多只剩 1 个 success + 1 个 failure）。"
                "若要保留全部失败视频，需按 tools 里的说明给 main.py 打补丁。"
            ),
        },
        "failures": failure_list,
    }

    sum_path = os.path.join(args.outdir, "eval_summary.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("=" * 62)
    print("评测结果解析完成")
    print("=" * 62)
    print("轮次: %d   成功: %d   成功率: %s" % (
        total_ep, total_succ,
        ("%.4f" % summary["success_rate_from_episodes"]) if total_ep else "n/a"))
    for s in sorted(by_suite):
        v = by_suite[s]
        print("  %-16s %4d/%4d = %.4f" % (s, v["successes"], v["episodes"],
                                          v["success_rate"] or 0.0))
    if "total_success_rate" in final:
        print("日志里的 Total success rate: %.4f" % final["total_success_rate"])
    if exceptions:
        print("!! 捕获到 %d 次异常，详见 summary.json" % len(exceptions))
    if args.video_dir:
        print("视频: success=%d failure=%d (目录 %s)" % (
            summary["videos"]["success_videos"], summary["videos"]["failure_videos"],
            args.video_dir))
        if summary["videos"]["failure_videos"]:
            print("  注意: 同名视频会互相覆盖，每 task 最多剩 1 个失败视频（见 summary.json 的 note）")
    print("-" * 62)
    print("产出:")
    for p in (ep_csv, tr_csv, sum_path):
        print("  %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
