"""Side-by-side environment-version summary; v2 and v3 instances are not paired."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
protocol = json.loads((ROOT / "protocol.json").read_text())
old_root = Path(protocol["previous_run"])
old = json.loads((old_root / "results/summary.json").read_text())
new = json.loads((ROOT / "results/summary.json").read_text())
assert old["complete"] and new["complete"]
old_by_task = {x["task"].replace("-v2", ""): x for x in old["tasks"]}
rows = []
for entry in new["tasks"]:
    name = entry["task"].replace("-v3", "")
    before = old_by_task[name]
    rows.append({"task": name, "v2_successes": before["successes"],
                 "v3_successes": entry["successes"],
                 "v3_last_step_successes": entry["last_step_successes"],
                 "trials_per_version": 20})
with (ROOT / "results/version_comparison.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = ["# 同一冻结 π0 检查点在 Farama MetaWorld v2 与 3.1.1 v3 的结果", "",
         "| 任务语义 | 旧 v2 曾成功 | 新 v3 曾成功 | 新 v3 最后一步成功 |",
         "|---|---:|---:|---:|"]
for row in rows:
    lines.append(f"| {row['task']} | {row['v2_successes']}/20 | {row['v3_successes']}/20 | {row['v3_last_step_successes']}/20 |")
lines.append(f"| 总计 | {old['successes']}/100 | {new['successes']}/100 | {new['last_step_successes']}/100 |")
lines += ["", "两轮均冻结同一权重，使用相同的 42–61 种子序列、步数上限和 success_once 口径。",
          "Farama v3 环境与旧 v2 的任务实例、物理版本不同，不能将每行差值解释为严格配对的模型性能提升。",
          "MTBench 2026 年 4 月的约 90% 是另一套 Isaac Gym/Franka 环境训练约 200M 帧后的强化学习成绩，不是本 π0 的对照目标。"]
(ROOT / "results/version_comparison.md").write_text("\n".join(lines) + "\n")
print(json.dumps(rows, indent=2))
