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
        assert old["checkpoint_sha256"] == protocol["previous_checkpoint_sha256"]
        assert new["checkpoint_sha256"] == protocol["checkpoint_sha256"]
        assert old["checkpoint_sha256"] != new["checkpoint_sha256"]
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
lines = ["# Farama MetaWorld v3: π0 SFT 与 π0 RL FlowNoise 配对对照", "",
         "| 场景 | 旧 SFT 成功 | 新 RL FlowNoise 成功 | 新成功率 | 新末步成功 |",
         "|---|---:|---:|---:|---:|"]
for row in rows:
    lines.append(f"| {row['task']} | {row['old_successes']}/20 | {row['new_successes']}/20 | {row['new_successes']/20:.0%} | {row['new_last_step_successes']}/20 |")
old_total=sum(row['old_successes'] for row in rows)
new_total=sum(row['new_successes'] for row in rows)
last_total=sum(row['new_last_step_successes'] for row in rows)
lines += [f"| 总计 | {old_total}/100 | {new_total}/100 | {new_total/100:.0%} | {last_total}/100 |", "",
          "每一对回合的 seed、MT1 场景 SHA 一致，初始四维状态一致；模型权重 SHA 不同。主指标均为160步内 success_once。",
          "两组均在 Farama MetaWorld 3.1.1 v3 上执行，分别为 RLinf 官方 SFT π0 与 PPO FlowNoise 微调 π0。评测中冻结权重，无本地训练或模型调整。",
          "新100回合另由独立模拟器重放验证，详见 diagnostics/new_trajectory_replay.json。",
          "推理为 BF16、固定随机种子；未强制所有 GPU 算子逐位确定性。",
          f"补充口径：新模型末步成功 {last_total}/100，旧模型末步成功 {sum(x['last_step_successes'] for x in json.loads((OLD / 'results/summary.json').read_text())['tasks'])}/100。`success_once` 衡量回合中任一时刻达成，末步成功衡量 160 步结束时是否仍达成。"]
(ROOT / "results/comparison.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"old_successes": old_total,"new_successes":new_total,"rows":rows},indent=2))
