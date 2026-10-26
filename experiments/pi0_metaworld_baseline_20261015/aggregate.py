"""Validate completed episodes and export auditable baseline tables."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def aggregate():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol_hash = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    rows, episodes = [], []
    for task in protocol["tasks"]:
        records = []
        for index in range(protocol["trials_per_task"]):
            file = ROOT / "results" / task / f"episode_{index:02d}.json"
            if not file.exists():
                continue
            record = json.loads(file.read_text())
            assert record["task"] == task and record["episode"] == index
            assert record["seed"] == protocol["episode_seeds"][index]
            assert record["protocol_sha256"] == protocol_hash
            assert record["checkpoint_sha256"] == protocol["checkpoint_sha256"]
            assert Path(record["trajectory"]).is_file() and Path(record["video"]).is_file()
            assert 0 < record["steps"] <= protocol["max_policy_steps"]
            assert type(record["success"]) is bool
            records.append(record)
        successes = sum(int(r["success"]) for r in records)
        final_successes = sum(int(r["success_at_last_step"]) for r in records)
        complete = len(records) == protocol["trials_per_task"]
        row = {"task": task, "planned_rollouts": protocol["trials_per_task"],
               "completed_rollouts": len(records), "successes": successes,
               "success_rate": successes / len(records) if records else None,
               "last_step_successes": final_successes, "status": "complete" if complete else "pending"}
        rows.append(row)
        episodes.extend(records)
    fields = list(rows[0])
    with (ROOT / "results/summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    episode_fields = ["task", "episode", "seed", "task_index", "success", "success_at_last_step",
                      "first_success_step", "steps", "return", "elapsed_seconds", "mean_inference_ms", "video"]
    with (ROOT / "results/episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=episode_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episodes)
    count = len(episodes)
    success_count = sum(int(x["success"]) for x in episodes)
    summary = {"complete": all(r["status"] == "complete" for r in rows), "tasks": rows,
               "completed_rollouts": count, "planned_rollouts": 100, "successes": success_count,
               "overall_success_rate": success_count / count if count else None,
               "protocol_sha256": protocol_hash, "checkpoint_sha256": protocol["checkpoint_sha256"]}
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# π0 MetaWorld v2 baseline", "", "| 场景 | 已完成 / 计划 | 成功次数 | 成功率 | 状态 |",
             "|---|---:|---:|---:|---|"]
    for row in rows:
        rate = f"{row['success_rate']:.1%}" if row["success_rate"] is not None else "—"
        lines.append(f"| {row['task']} | {row['completed_rollouts']} / 20 | {row['successes']} | {rate} | {row['status']} |")
    rate = f"{summary['overall_success_rate']:.1%}" if count else "—"
    lines += [f"| 总计 | {count} / 100 | {success_count} | {rate} | {'complete' if summary['complete'] else 'pending'} |", "",
              "使用 RLinf 官方发布的 π0 MetaWorld SFT 检查点，冻结权重，无额外训练或模型改进。",
              "每次先执行 15 个零动作等待物体稳定，再最多执行 160 个策略控制步；动作块长度 5，去噪步数 5。",
              "主要指标为策略控制期间至少一次 `info['success']` 为真；另记录最后一步成功状态。运行错误不计为失败。",
              "每场景使用固定种子 42–61，独立选择 20 个不同的 MT1 初始化配置。", "",
              "环境固定为用户指定的 v2；参考脚本当前使用 v3，因此这些结果不能直接当作官方 MT50 v3 分数的复现。", "",
              f"检查点：ModelScope `RLinf/RLinf-Pi0-MetaWorld`，revision `{protocol['checkpoint_revision']}`。",
              f"检查点 SHA256：`{protocol['checkpoint_sha256']}`。",
              f"MetaWorld commit：`{protocol['metaworld_commit']}`。",
              f"协议 SHA256：`{protocol_hash}`。", "",
              "文件：`protocol.json`、`results/summary.csv`、`results/episodes.csv`、逐次 JSON/NPZ、`videos/results/` 和 `logs/`。", "",
              "参考：[RLinf MetaWorld 文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/metaworld.html)、"
              "[官方评测脚本](https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py)。"]
    (ROOT / "results/report.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(aggregate(), indent=2))
