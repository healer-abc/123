"""Project episode statistics using the official SB3 load/predict API."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from custom_envs import ActionGain


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--group", choices=list("ABCDE"), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--gains", type=float, nargs="+", default=[1.0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or len(set(args.gains)) != len(args.gains):
        parser.error("episodes must be positive; gains must be unique")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    torch.set_num_threads(2)
    env_id = "FetchPushDense-v4" if args.group in "AB" else "FetchPush-v4"
    env = ActionGain(gym.make(env_id))
    model_class = PPO if args.group == "A" else SAC
    options = {} if args.group == "A" else {"buffer_size": 1}
    model = model_class.load(args.model, env=env, device="cpu", **options)
    rows, metrics = [], []
    try:
        for gain in args.gains:
            if gain <= 0:
                parser.error("gains must be positive")
            env.gain_low = env.gain_high = gain
            for scene in range(args.seed_base, args.seed_base + args.episodes):
                obs, _ = env.reset(seed=scene)
                initial_distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
                reached, steps = False, 0
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    reached |= bool(info["is_success"])
                    if terminated or truncated:
                        break
                assert steps == 50
                rows.append({"group": args.group, "training_seed": args.training_seed,
                             "episode_seed": scene, "gain": gain, "steps": steps,
                             "success": int(info["is_success"]), "ever_success": int(reached),
                             "initial_distance_m": initial_distance,
                             "distance_m": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))})
            subset = [r for r in rows if r["gain"] == gain]
            metrics.append({"gain": gain, "episodes": len(subset),
                            "success_rate": float(np.mean([r["success"] for r in subset])),
                            "distance_m": float(np.mean([r["distance_m"] for r in subset]))})
    finally:
        env.close()
    args.out.mkdir(parents=True)
    with (args.out / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    result = {"model": str(args.model.resolve()),
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "group": args.group, "training_seed": args.training_seed,
              "model_timesteps": model.num_timesteps, "seed_base": args.seed_base,
              "metrics": metrics}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
