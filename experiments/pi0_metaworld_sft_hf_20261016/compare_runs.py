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
lines = ["# π0 MetaWorld SFT 5,000 步与 10,000 步检查点对照", "", "| 场景 | 5,000 步成功 | 10,000 步成功 | 10,000 步成功率 | 10,000 步末步成功 |",
         "|---|---:|---:|---:|---:|"]
for row in rows:
    lines.append(f"| {row['task']} | {row['old_successes']}/20 | {row['new_successes']}/20 | {row['new_successes']/20:.0%} | {row['new_last_step_successes']}/20 |")
old_total=sum(row['old_successes'] for row in rows)
new_total=sum(row['new_successes'] for row in rows)
last_total=sum(row['new_last_step_successes'] for row in rows)
lines += [f"| 总计 | {old_total}/100 | {new_total}/100 | {new_total/100:.0%} | {last_total}/100 |", "",
          "每一对回合的seed、MT1场景SHA一致，初始四维状态一致；模型检查点SHA不同。主指标均为160步内success_once。",
          "新100回合另由独立模拟器重放验证，详见diagnostics/new_trajectory_replay.json。",
          "两组均为RLinf发布的MetaWorld SFT π0：旧组来自ModelScope镜像，训练元数据为5,000步；新组来自Hugging Face，训练元数据为10,000步。无本地训练或模型调整。",
          "推理为BF16、固定随机种子；未强制所有GPU算子逐位确定性。相同种子不等于跨进程轨迹逐位相同。",
          "上轮排查细节见 ../pi0_metaworld_baseline_recheck_20261015/DIAGNOSIS.md。"]
(ROOT / "results/comparison.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"old_successes": old_total,"new_successes":new_total,"rows":rows},indent=2))
