"""Validate each saved v3 rollout and export the frozen Pi05 SFT result table."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "pi0_metaworld_native_v3_20261016" / "aggregate.py"
spec = importlib.util.spec_from_file_location("pi0_v3_integrity", SOURCE)
integrity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integrity)
integrity.ROOT = SOURCE.parent

def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol_hash = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    rows, episodes = [], []
    for task in protocol["tasks"]:
        records = []
        for index in range(protocol["trials_per_task"]):
            path = ROOT / "results" / task / f"episode_{index:02d}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            assert record["task"] == task and record["episode"] == index
            assert record["seed"] == protocol["episode_seeds"][index]
            assert record["protocol_sha256"] == protocol_hash
            assert record["checkpoint_sha256"] == protocol["checkpoint_sha256"]
            assert 0 < record["steps"] <= protocol["max_policy_steps"]
            integrity.validate_trajectory(record, protocol)
            records.append(record)
        assert len({r["task_index"] for r in records}) == len(records)
        assert len({r["task_data_sha256"] for r in records}) == len(records)
        successes = sum(r["success"] for r in records)
        last = sum(r["success_at_last_step"] for r in records)
        rows.append({"task": task, "completed": len(records), "planned": 20,
                     "successes": successes,
                     "success_rate": successes / len(records) if records else None,
                     "last_step_successes": last})
        episodes.extend(records)
    count = len(episodes)
    summary = {"complete": count == 100, "completed": count, "planned": 100,
               "successes": sum(x["success"] for x in episodes),
               "success_rate": sum(x["success"] for x in episodes) / count if count else None,
               "last_step_successes": sum(x["success_at_last_step"] for x in episodes),
               "tasks": rows, "checkpoint_sha256": protocol["checkpoint_sha256"],
               "protocol_sha256": protocol_hash, "validated_episodes": count}
    result = ROOT / "results"
    result.mkdir(exist_ok=True)
    (result / "summary.json").write_text(json.dumps(summary, indent=2))
    with (result / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (result / "episodes.csv").open("w", newline="") as file:
        fields = ["task", "episode", "seed", "task_index", "success", "success_at_last_step",
                  "first_success_step", "steps", "return", "elapsed_seconds", "mean_inference_ms", "video"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episodes)
    old_root = Path(protocol["previous_v2_run"])
    old_summary = json.loads((old_root / "results/summary.json").read_text())
    assert old_summary["complete"] and old_summary["checkpoint_sha256"] == protocol["checkpoint_sha256"]
    old_counts = {}
    for row in rows:
        old_task = row["task"].replace("-v3", "-v2")
        old_records = [json.loads((old_root / "results" / old_task / f"episode_{i:02d}.json").read_text())
                       for i in range(protocol["trials_per_task"])]
        old_counts[row["task"]] = {"within_160": sum(r["first_success_step"] is not None
                                                      and r["first_success_step"] <= 160 for r in old_records),
                                   "within_500": sum(r["success"] for r in old_records)}
    lines = ["# π0.5 MetaWorld SFT 在 Farama MetaWorld 3.1.1 v3 的评测", "",
             "| 任务 | 成功/完成 | 成功率 | 最后一步成功 |", "|---|---:|---:|---:|"]
    for row in rows:
        rate = f"{row['success_rate']:.1%}" if row["success_rate"] is not None else "—"
        lines.append(f"| {row['task']} | {row['successes']}/{row['completed']} | {rate} | {row['last_step_successes']} |")
    rate = f"{summary['success_rate']:.1%}" if count else "—"
    lines += [f"| 总计 | {summary['successes']}/{count} | {rate} | {summary['last_step_successes']} |", "",
              "模型：RLinf/RLinf-Pi05-MetaWorld-SFT，冻结权重，严格加载，使用发布者的分位数归一化统计与原生 4D MetaWorld 动作。",
              "环境：Farama MetaWorld 3.1.1 / v3，MuJoCo 3.3.0。五任务每项 20 个 MT1 配置，种子 42–61。",
              "按 RLinf 官方评测脚本设定先做 15 个零动作稳定，再至多 160 个策略步；动作块长度 5、去噪步数 5。",
              "成功口径：策略控制期间至少一次 info.success 为真，不要求最后一帧仍为真；另列最后一帧成功数。",
              "完整观测和逐步 success 仅用于审计；策略只看到相机图像和前四维手部状态。每条轨迹和视频均校验 SHA256。",
              "前轮 v2 使用同一权重，但环境版本、初始化实例和每回合最大步数（500）不同，不能把差异归因于单一因素。", "",
              "| 任务 | 旧 v2 ≤160 步 | 旧 v2 ≤500 步 | 新 v3 ≤160 步 |",
              "|---|---:|---:|---:|"]
    for row in rows:
        before = old_counts[row["task"]]
        lines.append(f"| {row['task'].replace('-v3', '')} | {before['within_160']}/20 | {before['within_500']}/20 | {row['successes']}/20 |")
    lines += [f"| 总计 | {sum(v['within_160'] for v in old_counts.values())}/100 | "
              f"{sum(v['within_500'] for v in old_counts.values())}/100 | {summary['successes']}/100 |", "",
              "旧 v2 的 ≤160 步列仅是原 500 步轨迹的离线截断；v3 有 15 步稳定阶段，而且场景和动力学不同，不能视为严格配对对照。", "",
              "2026-04-19 的约 90% 属于 MTBench 的 Isaac Gym/Franka PPO 奖励修复；不适用于此 Farama MuJoCo/Sawyer 环境或本冻结 π0.5 权重。",
              f"模型 commit：{protocol['checkpoint_revision']}；权重 SHA256：{protocol['checkpoint_sha256']}。",
              f"MetaWorld wheel SHA256：{protocol['metaworld_wheel_sha256']}；协议 SHA256：{protocol_hash}。", "",
              "参考：[MTBench 更新](https://github.com/Viraj-Joshi/MTBench#updates)、[RLinf MetaWorld 权重](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT)、[RLinf 官方评测脚本](https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py)。"]
    replay_file = result / "replay_validation.json"
    if replay_file.is_file():
        replay = json.loads(replay_file.read_text())
        if replay.get("complete") and replay.get("passed") and len(replay.get("episodes", [])) == 100:
            lines += ["", "100/100 回合通过独立 MetaWorld v3 仿真器逐步重放；每步完整观测、奖励与成功标志一致。"]
    (result / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
