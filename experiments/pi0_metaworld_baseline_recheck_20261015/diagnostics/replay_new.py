"""Replay every new trajectory in a separate simulator after its JSON is committed."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rollout
import numpy as np

deadline = time.monotonic() + 1800
report = {"protocol_sha256": rollout.PROTOCOL_HASH, "episodes": [], "complete": False}
for task in rollout.PROTOCOL["tasks"]:
    env, tasks, indices = rollout.make_env(task)
    try:
        for episode, seed in enumerate(rollout.PROTOCOL["episode_seeds"]):
            record_file = ROOT / "results" / task / f"episode_{episode:02d}.json"
            while not record_file.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Waiting for formal result {record_file}")
                time.sleep(1)
            record = json.loads(record_file.read_text())
            with np.load(record["trajectory"], allow_pickle=False) as data:
                np.random.seed(seed)
                obs, _ = rollout.reset_episode(env, tasks[indices[episode]], seed)
                for _ in range(15):
                    obs, _, _, _, _ = env.step(np.zeros(4))
                state_err = next_err = reward_err = 0.0
                matches = True
                for step, action in enumerate(data["actions"]):
                    state_err = max(state_err, float(np.abs(obs - data["full_observations"][step]).max()))
                    obs, reward, term, trunc, info = env.step(action)
                    rollout.check_simulation(env)
                    next_err = max(next_err, float(np.abs(obs - data["next_observations"][step]).max()))
                    reward_err = max(reward_err, abs(float(reward) - float(data["rewards"][step])))
                    matches &= bool(info["success"]) == bool(data["successes"][step])
                row = {"task": task, "episode": episode, "full_state_max_abs_error": state_err,
                       "next_state_max_abs_error": next_err, "reward_max_abs_error": reward_err,
                       "success_matches": matches,
                       "passed": state_err < 1e-8 and next_err < 1e-8 and reward_err < 1e-8 and matches}
                report["episodes"].append(row)
                (ROOT / "diagnostics/new_trajectory_replay.json").write_text(json.dumps(report, indent=2))
                if not row["passed"]:
                    raise AssertionError(row)
        print(task, "20/20 independent simulator replays passed", flush=True)
    finally:
        env.close()
report["complete"] = True
report["passed"] = all(e["passed"] for e in report["episodes"])
(ROOT / "diagnostics/new_trajectory_replay.json").write_text(json.dumps(report, indent=2))
