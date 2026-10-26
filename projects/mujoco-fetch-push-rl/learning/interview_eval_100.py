"""Minimal explicit-scene evaluation exercise; outputs are separate from formal results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from run import make_env


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--group', choices=list('ABCDE'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-base', type=int, default=20000)
    parser.add_argument('--gain', type=float, default=1.)
    args = parser.parse_args()
    if args.out.exists() or args.episodes < 1 or not np.isfinite(args.gain) or args.gain <= 0:
        parser.error('Use a new output directory, positive episode count and finite positive gain')
    torch.set_num_threads(2)
    env = make_env({'group': args.group, 'task': 'Push'}, gain=args.gain)
    try:
        cls = PPO if args.group == 'A' else SAC
        kwargs = {} if cls is PPO else {'buffer_size': 1}
        model = cls.load(args.model, env=env, device='cpu', **kwargs)
        records = []
        for scene_seed in range(args.seed_base, args.seed_base + args.episodes):
            obs, _ = env.reset(seed=scene_seed)
            reached = False
            for step in range(1, 51):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                reached |= bool(info['is_success'])
                if terminated or truncated:
                    break
            records.append({'episode_seed': scene_seed, 'gain': args.gain,
                            'success': int(bool(info['is_success'])),
                            'ever_success': int(reached), 'steps': step,
                            'distance_m': float(np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']))})
    finally:
        env.close()
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    summary = {'purpose': 'Interview exercise; not additional formal results',
               'model': str(args.model.resolve()),
               'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
               'group': args.group, 'gain': args.gain, 'episodes': len(records),
               'seed_base': args.seed_base, 'model_timesteps': model.num_timesteps,
               'success_rate': float(np.mean([r['success'] for r in records])),
               'distance_m': float(np.mean([r['distance_m'] for r in records]))}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
