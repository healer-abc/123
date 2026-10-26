"""Create adjust_bottle, read its three cameras and 14-D state, and execute a hold action."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from robotwin.envs.vector_env import VectorEnv

parser = argparse.ArgumentParser()
parser.add_argument("--randomized", action="store_true")
args = parser.parse_args()
root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
robotwin = Path(os.environ["ROBOTWIN_PATH"])
variant = "randomized" if args.randomized else "clean"
config = yaml.safe_load((robotwin / "task_config" / f"demo_{variant}.yml").read_text())
config.update(task_name="adjust_bottle", planner_backend="mplib", step_lim=200, collect_data=False, save_path=str(Path(os.environ["ROBOTWIN_WORKSPACE"]) / "data/robotwin_pi0/smoke"))
config["camera"]["collect_wrist_camera"] = True
os.environ["ASSETS_PATH"] = str(robotwin)
seeds = json.loads((Path(os.environ["RLINF_ROOT"]) / "rlinf/envs/sim/robotwin/seeds/eval_seeds.json").read_text())
seed = seeds["adjust_bottle"]["success_seeds"][0]
print(f"Creating {variant} adjust_bottle, seed={seed}, planner=mplib", flush=True)
env = VectorEnv(config, n_envs=1, env_seeds=[seed])
try:
    env.reset()
    obs = env.get_obs()[0]
    state = np.asarray(obs["state"])
    assert state.shape == (14,) and np.isfinite(state).all(), state
    cameras = {}
    for key in ["full_image", "left_wrist_image", "right_wrist_image"]:
        frame = np.asarray(obs[key])
        assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.std() > 0, (key, frame.shape)
        cameras[key] = {"shape": list(frame.shape), "std": float(frame.std())}
    Image.fromarray(np.asarray(obs["full_image"], dtype=np.uint8)).save(root / "report" / f"adjust_bottle_{variant}.png")
    stepped = env.step(np.repeat(state[None, None, :], 1, axis=1))
    assert np.isfinite(np.asarray(stepped[1])).all(), stepped[1]
    result = {"status": "passed", "variant": variant, "seed": seed, "planner": "mplib", "state_shape": list(state.shape), "cameras": cameras, "instruction": obs.get("instruction"), "hold_action_reward": np.asarray(stepped[1]).tolist()}
    (root / "report" / f"task-check-{variant}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
finally:
    env.close()
