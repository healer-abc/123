"""Inspect frozen models, observation dimensions and reward boundaries, without learning."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from stable_baselines3 import PPO, SAC
from run import make_env


def main():
    env = make_env({'group': 'D', 'task': 'Push'})
    obs, _ = env.reset(seed=20000)
    u = env.unwrapped
    result = {
        'observation_shapes': {k: list(v.shape) for k, v in obs.items()},
        'dict_space_order': list(env.observation_space.spaces),
        'nq': u.model.nq, 'nv': u.model.nv, 'njnt': u.model.njnt,
        'joint_names': [u.model.joint(i).name for i in range(u.model.njnt)],
        'dt': u.dt, 'n_substeps': u.n_substeps,
        'physics_timestep': u.model.opt.timestep,
    }
    achieved = np.array([[0., 0., 0.], [.05, 0., 0.], [.051, 0., 0.]])
    desired = np.zeros_like(achieved)
    rewards = u.compute_reward(achieved, desired, [{}] * 3)
    successes = u._is_success(achieved, desired)
    assert rewards.shape == (3,) and rewards.tolist() == [0, 0, -1]
    assert successes.tolist() == [1, 0, 0]
    result['sparse_boundary_probe'] = {'distances': [0, .05, .051], 'rewards': rewards.tolist(), 'success': successes.tolist()}
    for group in 'AD':
        cls = PPO if group == 'A' else SAC
        kwargs = {} if group == 'A' else {'buffer_size': 1, 'env': env}
        model = cls.load(ROOT / f'runs/formal_20261022/{group}_seed0/final/model.zip', device='cpu', **kwargs)
        extractor = model.policy.features_extractor if group == 'A' else model.actor.features_extractor
        result[group] = {
            'policy_parameters_including_targets': sum(v.numel() for v in model.policy.parameters()),
            'feature_dimension': extractor.features_dim,
            'target_entropy': getattr(model, 'target_entropy', None),
            'policy_activation': str(getattr(model.policy, 'activation_fn', None)),
        }
        if group == 'D':
            result[group].update({name + '_parameters': sum(v.numel() for v in network.parameters()) for name, network in [('actor', model.actor), ('critic', model.critic), ('target_critic', model.critic_target)]})
        del model
    result['ppo_last_logged_metrics'] = {}
    for seed in range(3):
        with (ROOT / f'runs/formal_20261022/A_seed{seed}/progress.csv').open() as stream:
            records = list(csv.DictReader(stream))
        keys = ['train/approx_kl', 'train/clip_fraction', 'train/entropy_loss', 'train/explained_variance', 'train/value_loss', 'train/policy_gradient_loss']
        result['ppo_last_logged_metrics'][str(seed)] = {key: next(float(r[key]) for r in reversed(records) if r.get(key)) for key in keys}
    destination = ROOT / 'checks/interview_runtime_20261023.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    env.close()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
