"""Independent CPU preflight, not a replacement for the original run.py."""
import json
import os
from pathlib import Path
import platform
import shutil
import time
import traceback
import sys
from importlib.metadata import version

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import make_env
gym.register_envs(gymnasium_robotics)


def inspect_environment(env_id):
    task = "Reach" if "Reach" in env_id else "Push"
    env = make_env({"task": task, "group": "B" if "Dense" in env_id else "D"})
    try:
        check_env(env, warn=True)
        obs, _ = env.reset(seed=99)
        assert env.observation_space.contains(obs)
        actions = np.random.default_rng(99).uniform(-1, 1, (100, 4)).astype(np.float32)
        trajectory = []
        episode_ends = []
        started = time.perf_counter()
        for step, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert all(np.isfinite(value).all() for value in obs.values())
            assert np.isfinite(reward)
            computed = env.unwrapped.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
            np.testing.assert_allclose(reward, computed)
            trajectory.append({k: v.copy() for k, v in obs.items()})
            if terminated or truncated:
                episode_ends.append({"step": step + 1, "terminated": bool(terminated), "truncated": bool(truncated)})
                env.reset(seed=99 + step + 1)
        elapsed = time.perf_counter() - started
        env.reset(seed=99)
        for step, action in enumerate(actions):
            obs, _, terminated, truncated, _ = env.step(action)
            for key in obs:
                np.testing.assert_allclose(obs[key], trajectory[step][key], atol=1e-10, rtol=0)
            if terminated or truncated:
                env.reset(seed=99 + step + 1)
        assert len(episode_ends) == 2
        assert all(item["truncated"] and not item["terminated"] for item in episode_ends)
        goals = np.stack([obs["achieved_goal"], obs["achieved_goal"]])
        targets = np.stack([obs["desired_goal"], obs["desired_goal"]])
        assert env.unwrapped.compute_reward(goals, targets, {}).shape == (2,)
        return {
            "status": "passed", "env_id": env_id,
            "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
            "action_shape": list(env.action_space.shape),
            "episode_ends": episode_ends,
            "deterministic_seed_replay": True,
            "reward_recomputation_and_batching": True,
            "steps": 100,
            "simulation_steps_per_second": 100 / elapsed,
            "render_mode": env.render_mode,
        }
    finally:
        env.close()


def main():
    output = ROOT / "checks" / "cpu_environment.json"
    output.parent.mkdir(exist_ok=True)
    report = {
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "packages": {name: version(name) for name in (
            "torch", "gymnasium", "gymnasium-robotics", "mujoco", "stable-baselines3")},
        "gpu_hidden": os.environ["CUDA_VISIBLE_DEVICES"] == "",
        "cuda_build": torch.version.cuda,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "rendering_tested": False,
        "original_project_training_tested": False,
        "environments": [],
    }
    try:
        torch.set_num_threads(2)
        assert torch.version.cuda is None, "Preflight expects the CPU PyTorch wheel"
        x = torch.randn(256, 256, device="cpu")
        assert torch.isfinite(x @ x).all()
        for task in ("Reach", "Push"):
            for reward in ("", "Dense"):
                result = inspect_environment(f"Fetch{task}{reward}-v4")
                report["environments"].append(result)
                print(json.dumps(result), flush=True)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
