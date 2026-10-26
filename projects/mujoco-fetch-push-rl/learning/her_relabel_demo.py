"""Use one real trajectory to demonstrate future-goal relabeling, without training."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import run

out = ROOT / "learning/her_relabel_samples.csv"
env = run.make_env({"group": "D", "task": "Push"})
rows, transitions = [], []
try:
    obs, _ = env.reset(seed=99)
    env.action_space.seed(99)
    for step in range(50):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append((step, action.copy(), next_obs["achieved_goal"].copy(), next_obs["desired_goal"].copy(), reward, info))
        if terminated or truncated:
            break
    assert len(transitions) == 50
    # The final achieved goal is a valid future goal for every transition here.
    future_goal = transitions[-1][2]
    for step, action, achieved, desired, reward, info in transitions:
        changed = float(env.compute_reward(achieved, future_goal, info))
        row = {"step": step, "source_seed": 99, "future_step": 49,
               "original_reward": float(reward), "relabeled_reward": changed}
        for label, vector in (("action", action), ("achieved", achieved), ("original_goal", desired), ("relabeled_goal", future_goal)):
            row.update({f"{label}_{i}": float(v) for i, v in enumerate(vector)})
        rows.append(row)
    assert rows[-1]["relabeled_reward"] == 0.0
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"env": "FetchPush-v4", "seed": 99, "transitions": len(rows),
                "changed_rewards": sum(r["original_reward"] != r["relabeled_reward"] for r in rows),
                "trajectory_and_actions_unchanged": True,
                "note": "Learning example only; formal training uses SB3 HerReplayBuffer."}
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
finally:
    env.close()
