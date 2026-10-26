"""Small, single-process Fetch experiment runner. See SERVER_GUIDE.md."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

gym.register_envs(gymnasium_robotics)
GROUPS = {
    "A": {"algo": "ppo", "dense": True, "her": False, "gain": False},
    "B": {"algo": "sac", "dense": True, "her": False, "gain": False},
    "C": {"algo": "sac", "dense": False, "her": False, "gain": False},
    "D": {"algo": "sac", "dense": False, "her": True, "gain": False},
    "E": {"algo": "sac", "dense": False, "her": True, "gain": True},
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


class ActionGain(gym.Wrapper):
    """Change only commanded translation; leave goals and rewards intact."""
    def __init__(self, env, low=1.0, high=1.0):
        super().__init__(env)
        self.low, self.high, self.gain = low, high, 1.0
        self.gain_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.gain_rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
        # Independent RNG preserves paired environment initial states across gains.
        self.gain = float(self.gain_rng.uniform(self.low, self.high))
        return obs, info

    def step(self, action):
        executed = np.asarray(action, dtype=np.float32).copy()
        executed[:3] = np.clip(executed[:3] * self.gain, -1, 1)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        info["action_gain"] = self.gain
        return obs, float(reward), terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)


def make_env(config, *, training=False, gain=1.0, render=False):
    spec = GROUPS[config["group"]]
    suffix = "Dense" if spec["dense"] else ""
    env_id = f"Fetch{config['task']}{suffix}-v4"
    env = gym.make(env_id, render_mode="rgb_array" if render else None)
    low, high = (0.8, 1.2) if training and spec["gain"] else (gain, gain)
    return ActionGain(env, low, high)


def assess(model, config, *, episodes, seed_base, gain=1.0, video=None):
    env = make_env(config, gain=gain, render=video is not None)
    writer = None
    rows = []
    try:
        if video is not None:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(video), fps=25)
        for episode in range(episodes):
            obs, info = env.reset(seed=seed_base + episode)
            if writer:
                writer.append_data(env.render())
            total_reward, steps, reached = 0.0, 0, False
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                reached = reached or bool(info.get("is_success", False))
                if writer:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            rows.append({"episode_seed": seed_base + episode, "gain": gain,
                         "success": int(bool(info.get("is_success", False))),
                         "ever_success": int(reached),
                         "distance_m": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
                         "return": total_reward, "steps": steps})
    finally:
        env.close()
        if writer:
            writer.close()
    return rows


def save_bundle(model, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / "model.zip")
    if isinstance(model, SAC):
        model.save_replay_buffer(directory / "replay.pkl")
    write_json(directory / "config.json", config)
    # Written last: an interrupted write must not be mistaken for a usable checkpoint.
    write_json(directory / "complete.json", {"steps": model.num_timesteps})


def load_bundle(directory, device, training=False):
    directory = Path(directory)
    if not (directory / "complete.json").is_file():
        raise ValueError(f"Incomplete checkpoint: {directory}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    env = Monitor(make_env(config, training=training))
    cls = PPO if GROUPS[config["group"]]["algo"] == "ppo" else SAC
    model = cls.load(directory / "model.zip", env=env, device=device)
    if training and isinstance(model, SAC):
        model.load_replay_buffer(directory / "replay.pkl", truncate_last_traj=True)
    return model, config, env


class ExperimentCallback(BaseCallback):
    def __init__(self, config, out, eval_every, episodes, checkpoint_every):
        super().__init__()
        self.config, self.out = config, out
        self.eval_every, self.episodes = eval_every, episodes
        self.checkpoint_every = checkpoint_every
        self.stop_requested = False
        self.elapsed = 0.0

    def _on_training_start(self):
        self.start = time.perf_counter()
        self.first_step = self.model.num_timesteps
        self.next_eval = (self.first_step // self.eval_every + 1) * self.eval_every
        self.next_checkpoint = (self.first_step // self.checkpoint_every + 1) * self.checkpoint_every

    def _on_step(self):
        if self.stop_requested:
            return False
        if self.num_timesteps >= self.next_eval:
            # Evaluation must not change the training RNG stream.
            py_rng, np_rng = random.getstate(), np.random.get_state()
            torch_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            try:
                rows = assess(self.model, self.config, episodes=self.episodes, seed_base=10000)
            finally:
                random.setstate(py_rng)
                np.random.set_state(np_rng)
                torch.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
            self.elapsed = time.perf_counter() - self.start
            metrics = {"timesteps": self.num_timesteps, "session_seconds": self.elapsed,
                       "success_rate": float(np.mean([r["success"] for r in rows])),
                       "distance_m": float(np.mean([r["distance_m"] for r in rows])),
                       "session_steps_per_second": (self.num_timesteps - self.first_step) / self.elapsed}
            append_csv(self.out / "validation.csv", [metrics])
            append_csv(self.out / "validation_episodes.csv",
                       [{"timesteps": self.num_timesteps, **r} for r in rows])
            self.logger.record("validation/success_rate", metrics["success_rate"])
            self.logger.record("validation/distance_m", metrics["distance_m"])
            print(json.dumps(metrics), flush=True)
            self.next_eval += self.eval_every
        if self.num_timesteps >= self.next_checkpoint:
            save_bundle(self.model, self.config, self.out / "checkpoints" / f"step_{self.num_timesteps}")
            self.next_checkpoint += self.checkpoint_every
        return not self.stop_requested


def train(args):
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the CUDA check before GPU training.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.resume:
        model, config, env = load_bundle(args.resume, args.device, training=True)
        config = {**config, "resumed_from": str(Path(args.resume).resolve())}
    else:
        config = {"group": args.group, "task": args.task, "seed": args.seed,
                  "learning_rate": args.lr, "buffer_size": args.buffer_size,
                  "learning_starts": args.learning_starts, "gamma": 0.98,
                  "batch_size": 256, "net_arch": [256, 256], "n_envs": 1}
        env = Monitor(make_env(config, training=True), filename=str(out / "monitor.csv"))
        spec = GROUPS[config["group"]]
        common = dict(policy="MultiInputPolicy", env=env, seed=args.seed,
                      learning_rate=args.lr, gamma=0.98, batch_size=256,
                      policy_kwargs=dict(net_arch=[256, 256]), device=args.device, verbose=1)
        if spec["algo"] == "ppo":
            model = PPO(**common, n_steps=2048, n_epochs=10, clip_range=0.2, gae_lambda=0.95)
        else:
            her = {"replay_buffer_class": HerReplayBuffer,
                   "replay_buffer_kwargs": dict(n_sampled_goal=4, goal_selection_strategy="future")} if spec["her"] else {}
            model = SAC(**common, buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                        train_freq=1, gradient_steps=1, tau=0.005, ent_coef="auto", **her)
    remaining = args.steps - model.num_timesteps
    if remaining <= 0:
        env.close()
        raise ValueError("--steps is the TOTAL target; it must exceed checkpoint timesteps.")
    config.update(target_steps=args.steps, device=args.device, threads=args.threads)
    write_json(out / "config.json", config)
    packages = {name: importlib.metadata.version(name) for name in
                ["torch", "stable-baselines3", "gymnasium", "gymnasium-robotics", "mujoco", "numpy"]}
    write_json(out / "system.json", {"python": sys.version, "platform": platform.platform(),
               "packages": packages, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (out / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")
    model.set_logger(configure(str(out), ["stdout", "csv", "tensorboard"]))
    callback = ExperimentCallback(config, out, args.eval_every, args.eval_episodes, args.checkpoint_every)
    previous_handlers = {}
    def stop(signum, frame):
        callback.stop_requested = True
        print("Stop requested; saving at the next callback. Please wait.", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, stop)
    start, initial = time.perf_counter(), model.num_timesteps
    try:
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not bool(args.resume))
        status = "interrupted" if callback.stop_requested else "final"
        save_bundle(model, config, out / status)
        seconds = time.perf_counter() - start
        write_json(out / "summary.json", {"status": status, "timesteps": model.num_timesteps,
                   "session_seconds": seconds, "session_steps": model.num_timesteps - initial,
                   "session_steps_per_second": (model.num_timesteps - initial) / seconds})
    finally:
        env.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if callback.stop_requested:
        raise SystemExit(130)


def evaluate(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    model, config, bound_env = load_bundle(args.checkpoint, args.device)
    summary = []
    try:
        for gain in args.gains:
            rows = assess(model, config, episodes=args.episodes, seed_base=args.seed_base, gain=gain)
            append_csv(out / "episodes.csv", rows)
            summary.append({"gain": gain, "success_rate": float(np.mean([r["success"] for r in rows])),
                            "distance_m": float(np.mean([r["distance_m"] for r in rows])),
                            "episodes": len(rows)})
    finally:
        bound_env.close()
    write_json(out / "summary.json", {"checkpoint": str(Path(args.checkpoint).resolve()),
               "group": config["group"], "training_seed": config["seed"],
               "seed_base": args.seed_base, "metrics": summary})
    print(json.dumps(summary, indent=2))


def check(args):
    from stable_baselines3.common.env_checker import check_env
    config = {"group": "D", "task": args.task}
    env = make_env(config)
    try:
        check_env(env, warn=True)
        obs, _ = env.reset(seed=0)
        print("observation:", {k: v.shape for k, v in obs.items()})
        print("action:", env.action_space)
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            if terminated or truncated:
                env.reset()
        print("Environment check passed; training quality has not been tested.")
    finally:
        env.close()


def video(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model, config, env = load_bundle(args.checkpoint, args.device)
    try:
        rows = assess(model, config, episodes=args.episodes, seed_base=args.seed_base,
                      gain=args.gain, video=out)
        print(json.dumps(rows, indent=2))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--task", choices=["Reach", "Push"], default="Push")
    p.set_defaults(func=check)
    p = sub.add_parser("train")
    p.add_argument("--group", choices=list(GROUPS), default="D")
    p.add_argument("--task", choices=["Reach", "Push"], default="Push")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--out", required=True)
    p.add_argument("--resume")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=1000000)
    p.add_argument("--learning-starts", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=20000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=250000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=train)
    p = sub.add_parser("eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=20000)
    p.add_argument("--gains", type=float, nargs="+", default=[1.0])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=evaluate)
    p = sub.add_parser("video")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=30000)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=video)
    args = parser.parse_args()
    for key in ("steps", "episodes", "eval_every", "eval_episodes", "checkpoint_every", "threads", "buffer_size"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
