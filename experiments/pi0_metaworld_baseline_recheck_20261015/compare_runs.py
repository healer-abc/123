"""Paired comparison; every new episode must retain its original scene and seed."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
protocol = json.loads((ROOT / "protocol.json").read_text())
OLD = Path(protocol["previous_run"])
rows, paired = [], []
for task in protocol["tasks"]:
    old_records, new_records = [], []
    for episode in range(protocol["trials_per_task"]):
        name = f"episode_{episode:02d}.json"
        old = json.loads((OLD / "results" / task / name).read_text())
        new = json.loads((ROOT / "results" / task / name).read_text())
        assert old["seed"] == new["seed"]
        assert old["task_data_sha256"] == new["task_data_sha256"]
        assert old["checkpoint_sha256"] == new["checkpoint_sha256"]
        with np.load(old["trajectory"]) as before, np.load(new["trajectory"]) as after:
            initial_delta = float(np.abs(before["states"][0] - after["states"][0]).max())
            assert initial_delta < 1e-6
        paired.append({"task": task, "episode": episode, "seed": new["seed"],
                       "scene_sha256": new["task_data_sha256"], "old_success": old["success"],
                       "new_success": new["success"], "same_initial_state_max_abs_error": initial_delta})
        old_records.append(old)
        new_records.append(new)
    row = {"task": task, "old_successes": sum(x["success"] for x in old_records),
           "new_successes": sum(x["success"] for x in new_records),
           "new_last_step_successes": sum(x["success_at_last_step"] for x in new_records),
           "trials": 20}
    rows.append(row)
(ROOT / "diagnostics/paired_comparison.json").write_text(json.dumps(paired, indent=2))
with (ROOT / "results/comparison.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = ["# 原始结果与修复后重跑对照", "", "| 场景 | 原始成功 | 重跑成功 | 重跑成功率 | 重跑末步成功 |",
         "|---|---:|---:|---:|---:|"]
for row in rows:
    lines.append(f"| {row['task']} | {row['old_successes']}/20 | {row['new_successes']}/20 | {row['new_successes']/20:.0%} | {row['new_last_step_successes']}/20 |")
old_total=sum(row['old_successes'] for row in rows)
new_total=sum(row['new_successes'] for row in rows)
last_total=sum(row['new_last_step_successes'] for row in rows)
lines += [f"| 总计 | {old_total}/100 | {new_total}/100 | {new_total/100:.0%} | {last_total}/100 |", "",
          "每一对回合的权重SHA、seed、MT1场景SHA一致；初始四维状态一致。主指标均为160步内success_once。",
          "旧100回合逐步动作重放无差异；新100回合另由独立模拟器重放验证，详见diagnostics/new_trajectory_replay.json。",
          "修复了v2种子API调用并加强异常/数据校验，不代表旧31/100算错，也不能把配对成绩变化直接解释为模型改进。",
          "推理为BF16、固定随机种子；未强制所有GPU算子逐位确定性。相同种子不等于跨进程轨迹逐位相同。",
          "排查细节见 ../DIAGNOSIS.md。"]
(ROOT / "results/comparison.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"old_successes": old_total,"new_successes":new_total,"rows":rows},indent=2))
