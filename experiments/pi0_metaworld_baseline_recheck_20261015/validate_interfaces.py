"""Check the actual v2 seed bug, fixed reset, and unchanged initial policy inputs."""
import json
from pathlib import Path
import numpy as np
import rollout

ROOT = Path(__file__).resolve().parent
OLD = Path(rollout.PROTOCOL["previous_run"])

rows = []
for task in rollout.PROTOCOL["tasks"]:
    env, tasks, indices = rollout.make_env(task)
    try:
        env.set_task(tasks[indices[0]])
        env.seed(1234)
        env.reset(seed=42)
        ignored = env.np_random.bit_generator.state == np.random.default_rng(1234).bit_generator.state
        assert ignored, "Expected the pinned native v2 reset API to ignore its seed argument"
        obs, _ = rollout.reset_episode(env, tasks[indices[0]], 42)
        seed_applied = env.np_random.bit_generator.state == np.random.default_rng(42).bit_generator.state
        assert seed_applied
        for _ in range(15):
            obs, _, _, _, _ = env.step(np.zeros(4))
            rollout.check_simulation(env)
        previous = np.load(OLD / "results" / task / "episode_00.npz")["states"][0]
        error = float(np.abs(obs[:4].astype(np.float32) - previous).max())
        assert error < 1e-5
        image = rollout.render_image(env)
        assert image.shape == (480,480,3) and image.dtype == np.uint8
        # Simulate MetaWorld's internal failure flag to verify that it cannot become a 0% task score.
        env._did_see_sim_exception = True
        try:
            rollout.check_simulation(env)
        except RuntimeError:
            error_caught = True
        else:
            raise AssertionError("Internal simulation failures must abort the episode")
        env._did_see_sim_exception = False
        rows.append({"task": task, "legacy_seed_ignored": ignored, "fixed_seed_applied": seed_applied,
                     "same_initial_state_max_abs_error": error, "image_shape": list(image.shape),
                     "simulation_error_guard_passed": error_caught})
    finally:
        env.close()
result = {"passed": True, "protocol_sha256": rollout.PROTOCOL_HASH, "tasks": rows}
(ROOT / "diagnostics/interface_validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result,indent=2))
