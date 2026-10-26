"""CPU diagnostic on fixed demo observations. No optimizer steps or policy modification."""
import json
import os
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'metaworld_official_baselines_50demo_20261019'
sys.path.insert(0, str(BASE))
from runtime import environment, bootstrap, atomic_json, sha256
os.environ.update(environment(''))
bootstrap()
import h5py
import numpy as np
import torch
import worker
torch.set_num_threads(2)
torch.manual_seed(0)
task = 'pick-place-wall'
run = BASE / 'runs' / ('ACT_' + task) / 'attempt_01'
sys.argv = worker.act_args(run, task)
config = json.loads((run / 'config.json').read_text())
checkpoint = run / 'policy_best.ckpt'
original_hash = sha256(checkpoint)
with patch.object(torch.nn.Module, 'cuda', lambda self, *args, **kwargs: self):
    policy = worker.act_functions()['make_policy']('ACT', config['policy_config'])
policy.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False), strict=True)
policy.eval()
with (run / 'dataset_stats.pkl').open('rb') as f:
    stats = pickle.load(f)
split = json.loads((BASE / 'manifests/dataset_split.json').read_text())
ids = split['train_episode_ids'][:5] + split['validation_episode_ids']
records = []
with torch.inference_mode():
    for episode in ids:
        with h5py.File(BASE / 'datasets' / ('act_' + task) / f'episode_{episode}.hdf5', 'r') as f:
            q = f['observations/qpos'][0]
            image = f['observations/images/corner2'][0]
            expert = f['action'][:100]
        q = torch.as_tensor((q-stats['qpos_mean'])/stats['qpos_std'], dtype=torch.float32)[None]
        image = torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32)[None, None] / 255.
        target = torch.as_tensor((expert-stats['action_mean'])/stats['action_std'], dtype=torch.float32)[None]
        prior = policy(q, image)
        posterior = policy(q, image, target, torch.zeros((1,100),dtype=torch.bool))
        raw = prior[0].numpy()*stats['action_std']+stats['action_mean']
        records.append({'episode': int(episode), 'split': 'train' if episode in split['train_episode_ids'] else 'validation',
            'normalized_prior_l1': float((prior-target).abs().mean()),
            'normalized_posterior_l1': float(posterior['l1']), 'posterior_kl': float(posterior['kl']),
            'first_predicted_action': raw[0].tolist(), 'first_expert_action': expert[0].tolist(),
            'raw_first_action_l1': float(np.abs(raw[0]-expert[0]).mean()),
            'finite_predictions': bool(torch.isfinite(prior).all())})
        print(json.dumps(records[-1]), flush=True)
assert sha256(checkpoint) == original_hash
result = {'task': task, 'checkpoint_sha256': original_hash, 'device': 'cpu',
    'observations': 'first timestep from five fixed training episodes and all ten validation episodes',
    'note': 'Offline diagnostic only. Posterior uses expert future actions as an encoder input, while rollout uses the official zero latent prior. Neither is substituted into rollout.',
    'checkpoint_unchanged': True, 'records': records,
    'means': {s: {k: float(np.mean([r[k] for r in records if r['split']==s]))
        for k in ['normalized_prior_l1','normalized_posterior_l1','raw_first_action_l1']}
        for s in ['train','validation']}}
atomic_json(HERE / 'ACT_pick-place-wall' / 'offline_model_diagnostic.json', result)
