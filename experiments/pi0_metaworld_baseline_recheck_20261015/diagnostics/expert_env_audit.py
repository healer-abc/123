"""Environment diagnostic only: published scripted policies; never baseline scores."""
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rollout
import numpy as np

names = {
    "assembly-v2": "Assembly", "shelf-place-v2": "ShelfPlace", "disassemble-v2": "Disassemble",
    "peg-unplug-side-v2": "PegUnplugSide", "stick-pull-v2": "StickPull",
}
report = {"role": "diagnostic_only_not_pi0_results", "protocol_sha256": rollout.PROTOCOL_HASH, "tasks": []}
for task, classname in names.items():
    module = importlib.import_module("metaworld.policies.sawyer_" + task.replace("-", "_") + "_policy")
    expert = getattr(module, "Sawyer" + classname + "V2Policy")()
    env, tasks, indices = rollout.make_env(task)
    episodes = []
    try:
        for episode, seed in enumerate(rollout.PROTOCOL["episode_seeds"]):
            np.random.seed(seed)
            obs, _ = rollout.reset_episode(env, tasks[indices[episode]], seed)
            for _ in range(15):
                obs, _, _, _, _ = env.step(np.zeros(4))
            success = False
            first = None
            for step in range(160):
                obs, reward, term, trunc, info = env.step(expert.get_action(obs))
                rollout.check_simulation(env)
                if info["success"] and first is None:
                    first = step + 1
                success |= bool(info["success"])
                if term or trunc:
                    break
            episodes.append({"episode": episode, "success": success, "first_success_step": first})
    finally:
        env.close()
    row = {"task": task, "scripted_diagnostic_successes": sum(e["success"] for e in episodes), "episodes": episodes}
    report["tasks"].append(row)
    (ROOT / "diagnostics/expert_environment_check.json").write_text(json.dumps(report, indent=2))
    print(task, row["scripted_diagnostic_successes"], "/20 (diagnostic only)", flush=True)
