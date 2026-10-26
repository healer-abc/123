"""Independently replay all recorded actions in a new MetaWorld v3 simulator."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import rollout
import numpy as np

def main():
    reference = rollout.reference
    result = {"protocol_sha256": rollout.protocol_hash, "episodes": [], "complete": False}
    for task in rollout.protocol["tasks"]:
        env, tasks, indices = reference.make_env(task)
        try:
            for episode, seed in enumerate(rollout.protocol["episode_seeds"]):
                record_file = ROOT / "results" / task / f"episode_{episode:02d}.json"
                record = json.loads(record_file.read_text())
                with np.load(record["trajectory"], allow_pickle=False) as data:
                    np.random.seed(seed)
                    observation, _ = reference.reset_episode(env, tasks[indices[episode]], seed)
                    for _ in range(rollout.protocol["settling_steps"]):
                        observation, _, terminated, truncated, _ = env.step(np.zeros(4))
                        assert not terminated and not truncated
                    state_error = next_error = reward_error = 0.0
                    success_matches = True
                    for step, action in enumerate(data["actions"]):
                        state_error = max(state_error, float(np.abs(observation - data["full_observations"][step]).max()))
                        observation, reward, _, _, info = env.step(action)
                        reference.check_simulation(env)
                        next_error = max(next_error, float(np.abs(observation - data["next_observations"][step]).max()))
                        reward_error = max(reward_error, abs(float(reward) - float(data["rewards"][step])))
                        success_matches &= bool(info["success"]) == bool(data["successes"][step])
                    row = {"task": task, "episode": episode,
                           "full_state_max_abs_error": state_error,
                           "next_state_max_abs_error": next_error,
                           "reward_max_abs_error": reward_error,
                           "success_matches": success_matches,
                           "passed": state_error < 1e-8 and next_error < 1e-8 and reward_error < 1e-8 and success_matches}
                    result["episodes"].append(row)
                    if not row["passed"]:
                        raise AssertionError(row)
            print(f"{task}: 20/20 independent replays passed", flush=True)
        finally:
            env.close()
    result["complete"] = True
    result["passed"] = True
    path = ROOT / "results" / "replay_validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {len(result['episodes'])} episodes", flush=True)

if __name__ == "__main__":
    main()
