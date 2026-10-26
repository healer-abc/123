#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""openpi 训练日志 -> 结构化指标（CSV / JSON）

把 openpi 训练过程中落进日志的两类行，解析成可直接进实验记录的表：

  1) 指标行（scripts/train.py 的 pbar.write 产出，每 log_interval 步一条）
     Step 1100: grad_norm=0.2270, loss=0.0297, param_norm=1802.3903
  2) 进度行（tqdm_loggable 产出，每 ~10 秒一条）
     19:11:20.989 [I] Progress on: 1.10kit/30.0kit rate:5.4s/it remaining:43:07:31 ...

纯事后解析：不改动、也不影响正在跑的训练进程。

用法
----
  python parse_train_log.py ~/jyluo/projects/openpi/train_resume.log -o ./metrics

  # 多个日志合并（如重启前的 train.log + 重启后的 train_resume.log）
  python parse_train_log.py -o ./metrics \
      --log ~/jyluo/projects/openpi/train.log \
      --log ~/jyluo/projects/openpi/train_resume.log

产出（-o 指定目录下）
----
  steps.csv        每次记录一行：step, loss, grad_norm, param_norm
  throughput.csv   每 ~10 秒一行：timestamp, step, s_per_it, remaining_s, elapsed_s
  summary.json     汇总：步数区间 / loss 首末最小 / NaN 检查 / 吞吐 / ETA

仅依赖标准库（Python 3.8+ 可用）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta

STEP_RE = re.compile(r"^Step (?P<step>\d+): (?P<body>.+)$")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+|nan|-?inf)")
PROG_RE = re.compile(r"(?P<ts>\d{2}:\d{2}:\d{2})\.\d+ \[I\] Progress on: (?P<body>.+)$")
PROG_BODY_RE = re.compile(
    r"(?P<cur>\S+)/(?P<tot>\S+) rate:(?P<rate>\S+) "
    r"remaining:(?P<rem>\S+) elapsed:(?P<ela>\S+)"
)
METRIC_KEYS = ("loss", "grad_norm", "param_norm")


def parse_num(tok):
    """'1.10kit' -> 1100.0 ; '878it' -> 878.0 ; '20.0it' -> 20.0 ; '-' -> None"""
    if tok in ("-", "?", ""):
        return None
    for suffix, mult in (("kit", 1000.0), ("Mit", 1e6), ("k it", 1000.0), ("it", 1.0)):
        if tok.endswith(suffix):
            try:
                return float(tok[: -len(suffix)]) * mult
            except ValueError:
                return None
    try:
        return float(tok)
    except ValueError:
        return None


def parse_hms(tok):
    """'09:30' -> 570 ; '1:43:11' -> 6191 ; '43:07:31' -> 155251 ; '?' -> None"""
    if tok in ("?", "-", ""):
        return None
    try:
        parts = [int(p) for p in tok.split(":")]
    except ValueError:
        return None
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


def parse_s_per_it(tok):
    """'5.4s/it' -> 5.4 ; '2.5it/s' -> 0.4 ; '-' -> None"""
    if tok in ("-", "?", ""):
        return None
    if tok.endswith("s/it"):
        try:
            return float(tok[:-4])
        except ValueError:
            return None
    if tok.endswith("it/s"):
        try:
            return 1.0 / float(tok[:-4])
        except ValueError:
            return None
    return None


def read_logs(paths):
    steps = []       # list of dict: step + metrics
    progress = []    # list of dict: timestamp/step/s_per_it/remaining_s/elapsed_s
    total_steps = None
    for path in paths:
        if not os.path.exists(path):
            print("[warn] 日志不存在，跳过: %s" % path, file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                m = STEP_RE.match(line.strip())
                if m:
                    rec = {"step": int(m.group("step"))}
                    for key, val in KV_RE.findall(m.group("body")):
                        try:
                            rec[key] = float(val)
                        except ValueError:
                            rec[key] = float("nan")
                    rec["_source"] = path
                    steps.append(rec)
                    continue
                m = PROG_RE.search(line)
                if m:
                    b = PROG_BODY_RE.search(m.group("body"))
                    if not b:
                        continue
                    cur = parse_num(b.group("cur"))
                    tot = parse_num(b.group("tot"))
                    if tot is not None:
                        total_steps = int(round(tot))
                    progress.append(
                        {
                            "timestamp": m.group("ts"),
                            "step": int(round(cur)) if cur is not None else None,
                            "s_per_it": parse_s_per_it(b.group("rate")),
                            "remaining_s": parse_hms(b.group("rem")),
                            "elapsed_s": parse_hms(b.group("ela")),
                            "_source": path,
                        }
                    )
    steps.sort(key=lambda r: r["step"])
    return steps, progress, total_steps


def metric_stats(steps, key):
    vals = [(r["step"], r[key]) for r in steps if key in r]
    if not vals:
        return None
    finite = [(s, v) for s, v in vals if v == v and abs(v) != float("inf")]
    bad = [s for s, v in vals if v != v or abs(v) == float("inf")]
    if not finite:
        return {"count": len(vals), "nan_or_inf_at": bad}
    first_s, first_v = finite[0]
    last_s, last_v = finite[-1]
    min_s, min_v = min(finite, key=lambda t: t[1])
    max_s, max_v = max(finite, key=lambda t: t[1])
    return {
        "count": len(vals),
        "first": {"step": first_s, "value": first_v},
        "last": {"step": last_s, "value": last_v},
        "min": {"step": min_s, "value": min_v},
        "max": {"step": max_s, "value": max_v},
        "nan_or_inf_at": bad,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="openpi 训练日志解析器")
    ap.add_argument("logfiles", nargs="*", help="日志文件路径（等价于 --log）")
    ap.add_argument("--log", action="append", default=[], help="日志文件路径，可重复")
    ap.add_argument("-o", "--outdir", default="metrics", help="输出目录（默认 ./metrics）")
    args = ap.parse_args(argv)

    paths = list(args.logfiles) + list(args.log)
    if not paths:
        ap.error("请至少给一个日志文件路径")

    steps, progress, total_steps = read_logs(paths)
    if not steps and not progress:
        print("[error] 没解析到任何指标行或进度行，请确认日志内容", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    steps_csv = os.path.join(args.outdir, "steps.csv")
    with open(steps_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["step"] + list(METRIC_KEYS), extrasaction="ignore")
        w.writeheader()
        for r in steps:
            w.writerow({k: r.get(k, "") for k in ["step"] + list(METRIC_KEYS)})

    tp_csv = os.path.join(args.outdir, "throughput.csv")
    with open(tp_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["timestamp", "step", "s_per_it", "remaining_s", "elapsed_s"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in progress:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in w.fieldnames})

    # 吞吐：用首末进度行做区间平均，更稳（避免被启动/checkpoint 阶段拖偏）
    valid = [r for r in progress if r["step"] and r["elapsed_s"] is not None]
    derived = None
    if len(valid) >= 2:
        a, b = valid[0], valid[-1]
        dstep = b["step"] - a["step"]
        dt = b["elapsed_s"] - a["elapsed_s"]
        if dstep > 0 and dt > 0:
            derived = dt / float(dstep)
    recent = [r["s_per_it"] for r in progress if r["s_per_it"]][-20:]
    mean_recent = sum(recent) / len(recent) if recent else None

    remaining_last = next(
        (r["remaining_s"] for r in reversed(progress) if r["remaining_s"] is not None), None
    )
    mtimes = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    eta = None
    if remaining_last is not None and mtimes:
        eta = datetime.fromtimestamp(max(mtimes)) + timedelta(seconds=remaining_last)
        eta = eta.replace(microsecond=0)

    loss_stat = metric_stats(steps, "loss")
    summary = {
        "logs": paths,
        "runs_parsed": len(paths),
        "total_steps_target": total_steps,
        "steps": {
            "records": len(steps),
            "first": steps[0]["step"] if steps else None,
            "last": steps[-1]["step"] if steps else None,
            "interval_guess": (
                steps[1]["step"] - steps[0]["step"] if len(steps) >= 2 else None
            ),
        },
        "metrics": {k: metric_stats(steps, k) for k in METRIC_KEYS},
        "throughput": {
            "progress_records": len(progress),
            "derived_s_per_it_first_to_last": derived,
            "mean_s_per_it_last_20": mean_recent,
            "elapsed_hours_last": (
                round(valid[-1]["elapsed_s"] / 3600.0, 3) if valid else None
            ),
            "remaining_human_last": (
                str(timedelta(seconds=remaining_last)) if remaining_last is not None else None
            ),
            "eta_guess_utc_plus8": eta.strftime("%Y-%m-%d %H:%M:%S") if eta else None,
        },
        "health": {
            "nan_detected": bool(loss_stat and loss_stat.get("nan_or_inf_at")),
            "loss_decreasing": bool(
                loss_stat
                and loss_stat.get("first")
                and loss_stat.get("last")
                and loss_stat["last"]["value"] < loss_stat["first"]["value"]
            ),
        },
    }

    sum_path = os.path.join(args.outdir, "summary.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # ---- 控制台报告 ----
    print("=" * 62)
    print("解析完成")
    print("=" * 62)
    print("指标行: %d 条  (step %s .. %s)" % (
        summary["steps"]["records"], summary["steps"]["first"], summary["steps"]["last"]))
    print("进度行: %d 条" % summary["throughput"]["progress_records"])
    if loss_stat and loss_stat.get("first"):
        print("loss  : %.4f (step %d)  ->  %.4f (step %d)   最小 %.4f @ %d" % (
            loss_stat["first"]["value"], loss_stat["first"]["step"],
            loss_stat["last"]["value"], loss_stat["last"]["step"],
            loss_stat["min"]["value"], loss_stat["min"]["step"]))
        if summary["health"]["nan_detected"]:
            print("  !! 检测到 NaN/Inf，位于 step: %s" % loss_stat["nan_or_inf_at"])
        else:
            print("  NaN/Inf 检查: 通过")
    if derived:
        print("吞吐  : 区间平均 %.3f s/it (%.2f it/s)" % (derived, 1.0 / derived))
    if mean_recent:
        print("       最近 20 条采样均值 %.3f s/it" % mean_recent)
    if summary["throughput"]["remaining_human_last"]:
        print("剩余  : %s   ETA(按日志 mtime) %s" % (
            summary["throughput"]["remaining_human_last"], eta))
    print("-" * 62)
    print("产出:")
    print("  %s" % steps_csv)
    print("  %s" % tp_csv)
    print("  %s" % sum_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
