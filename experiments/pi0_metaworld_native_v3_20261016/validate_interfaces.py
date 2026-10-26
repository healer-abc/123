"""Validate the released Farama v3 simulator before any policy result is counted."""
import hashlib
import json
from pathlib import Path

import numpy as np

import rollout

ROOT = Path(__file__).resolve().parent
rows = []
for task in rollout.PROTOCOL["tasks"]:
    env, tasks, indices = rollout.make_env(task)
    try:
        assert len(tasks) >= 20 and len(set(indices)) == 20
        selected = [hashlib.sha256(tasks[i].data).hexdigest() for i in indices]
        assert len(set(selected)) == 20
        seed = rollout.PROTOCOL["episode_seeds"][0]
        first, _ = rollout.reset_episode(env, tasks[indices[0]], seed)
        first_image = rollout.render_image(env)
        second, _ = rollout.reset_episode(env, tasks[indices[0]], seed)
        assert np.array_equal(first, second)
        assert first.shape == (39,) and env.action_space.shape == (4,)
        assert first_image.shape == (480, 480, 3)
        observation, reward, terminated, truncated, info = env.step(np.zeros(4))
        rollout.check_simulation(env)
        assert np.isfinite(observation).all() and np.isfinite(reward)
        assert "success" in info and not terminated and not truncated
        rows.append({"task": task, "candidate_tasks": len(tasks),
                     "selected_task_hashes_unique": len(set(selected)),
                     "repeat_seed_same_observation": True,
                     "observation_shape": list(first.shape),
                     "action_shape": list(env.action_space.shape),
                     "image_shape": list(first_image.shape)})
    finally:
        env.close()
result = {"passed": True, "protocol_sha256": rollout.PROTOCOL_HASH,
          "metaworld_version": rollout.PROTOCOL["metaworld_environment_version"],
          "simulator": "MuJoCo 3.3.0", "tasks": rows}
(ROOT / "diagnostics/interface_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
