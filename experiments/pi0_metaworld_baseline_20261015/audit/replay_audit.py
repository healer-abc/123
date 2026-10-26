"""Independently replay recorded actions; no policy or model changes."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rollout
import numpy as np


def main():
    report = {"started": time.time(), "tasks": [], "episodes": []}
    for task in rollout.PROTOCOL["tasks"]:
        env, tasks, indices = rollout.make_env(task)
        failures = []
        for episode, seed in enumerate(rollout.PROTOCOL["episode_seeds"]):
            record = json.loads((ROOT / "results" / task / f"episode_{episode:02d}.json").read_text())
            saved = np.load(record["trajectory"])
            np.random.seed(seed)
            env.set_task(tasks[indices[episode]])
            observation, info = env.reset(seed=seed)
            for _ in range(rollout.PROTOCOL["settling_steps"]):
                observation, _, _, _, _ = env.step(np.zeros(4))
            states, rewards, successes = [], [], []
            for action in saved["actions"]:
                states.append(observation[:4].astype(np.float32))
                observation, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                successes.append(bool(info["success"]))
            state_error = float(np.max(np.abs(np.asarray(states) - saved["states"])))
            reward_error = float(np.max(np.abs(np.asarray(rewards) - saved["rewards"])))
            success_matches = bool(np.array_equal(successes, saved["successes"]))
            row = {"task": task, "episode": episode, "state_max_abs_error": state_error,
                   "reward_max_abs_error": reward_error, "success_matches": success_matches,
                   "simulation_exception": bool(env._did_see_sim_exception),
                   "success": bool(any(successes)), "target": env._target_pos.tolist()}
            report["episodes"].append(row)
            if state_error > 1e-5 or reward_error > 1e-4 or not success_matches or env._did_see_sim_exception:
                failures.append(row)
        env.close()
        report["tasks"].append({"task": task, "episodes": 20, "mismatches": len(failures)})
        print(json.dumps(report["tasks"][-1]), flush=True)
        (ROOT / "audit/replay_audit.json").write_text(json.dumps(report, indent=2))
    report["completed"] = time.time()
    report["all_match"] = all(t["mismatches"] == 0 for t in report["tasks"])
    (ROOT / "audit/replay_audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
