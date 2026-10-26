"""Common scored protocol; preserves official 200-step MetaWorld horizon."""
import collections
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from runtime import atomic_json, install_render_fix, ROOT, sha256


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def observation(obs, method):
    if method == 'SimpleDP3':
        return {'point_cloud': obs['point_cloud'].astype(np.float32),
                'agent_pos': obs['agent_pos'].astype(np.float32)}
    return {'image': obs['image'].astype(np.float32) / 255.,
            'agent_pos': obs['agent_pos'].astype(np.float32)}


def evaluate(policy, task, method, output_dir, *, final=False, checkpoint=None,
             stats=None, round_id=0):
    install_render_fix()
    from diffusion_policy_3d.env import MetaWorldEnv
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if final:
        assert checkpoint and Path(checkpoint).is_file()
        ckpt_hash = sha256(checkpoint)
        identity = {'task': task, 'method': method, 'checkpoint_sha256': ckpt_hash,
                    'protocol': sha256(ROOT / 'protocol.json')}
    else:
        identity = {'task': task, 'method': method, 'round': round_id}
    records = []
    env = None
    policy.eval()
    n_obs = 2
    try:
        for episode in range(20):
            seed = (2026101900 if final else 1026101900) + episode
            dest = output_dir / f'episode_{episode:02d}.json'
            if final and dest.exists():
                old = json.loads(dest.read_text())
                assert old['identity'] == identity and old['seed'] == seed
                records.append(old)
                continue
            seed_all(seed)
            if env is None:
                env = MetaWorldEnv(task, device='cuda:0', num_points=512, use_point_crop=True)
            # Official wrapper.seed is a no-op: explicitly seed the underlying v2 env.
            env.env.seed(seed)
            obs = env.reset()
            initial = np.concatenate([env.env.sim.get_state().flatten(), obs['full_state'],
                                      np.asarray(env.env._target_pos).ravel()])
            initial_hash = __import__('hashlib').sha256(
                np.asarray(initial, dtype=np.float64).tobytes()).hexdigest()
            history = collections.deque([observation(obs, method)] * n_obs, maxlen=n_obs)
            if method != 'ACT':
                policy.reset()
            success, first_success, reward_sum, steps = False, None, 0., 0
            frames = []
            while steps < 200:
                if method == 'ACT':
                    q = (obs['agent_pos'] - stats['qpos_mean']) / stats['qpos_std']
                    with torch.no_grad():
                        actions = policy(torch.as_tensor(q, dtype=torch.float32, device='cuda')[None],
                            torch.as_tensor(obs['image'], dtype=torch.float32, device='cuda')[None, None] / 255.)
                    actions = actions[0].cpu().numpy() * stats['action_std'] + stats['action_mean']
                else:
                    batch = {k: torch.as_tensor(np.stack([x[k] for x in history]),
                        dtype=torch.float32, device='cuda')[None] for k in history[0]}
                    with torch.no_grad():
                        actions = policy.predict_action(batch)['action'][0].cpu().numpy()
                assert actions.ndim == 2 and actions.shape[1] == 4
                if not np.isfinite(actions).all():
                    raise FloatingPointError('Nonfinite model action')
                for action in actions:
                    # No additional controller, residual, expert fallback or postprocessing.
                    obs, reward, done, info = env.step(action)
                    reward_sum += float(reward)
                    if info['success'] and not success:
                        first_success = steps + 1
                    success = success or bool(info['success'])
                    history.append(observation(obs, method))
                    if final and steps % 2 == 0:
                        frames.append(obs['image'].transpose(1, 2, 0))
                    steps += 1
                    if done or steps >= 200:
                        break
                atomic_json(output_dir / 'heartbeat.json', {'time': time.time(), 'episode': episode,
                    'steps': steps, 'final': final})
                if done or steps >= 200:
                    break
            record = {'identity': identity, 'episode': episode, 'seed': seed,
                      'initial_state_sha256': initial_hash, 'steps': steps,
                      'success': success, 'first_success_step': first_success,
                      'return': reward_sum, 'time': time.time()}
            atomic_json(dest, record)
            records.append(record)
            if final:
                import imageio.v2 as imageio
                imageio.mimsave(output_dir / f'episode_{episode:02d}.mp4', frames, fps=10)
            print(json.dumps({'rollout': record}, ensure_ascii=False), flush=True)
    finally:
        if env is not None:
            env.env.close()
            del env
    assert len(records) == 20
    result = {'identity': identity, 'episodes': 20,
              'successes': sum(x['success'] for x in records),
              'success_rate': sum(x['success'] for x in records) / 20,
              'mean_return': float(np.mean([x['return'] for x in records]))}
    atomic_json(output_dir / 'summary.json', result)
    return result


class RunnerMixin:
    def configure(self, output_dir, task_name, method, **kw):
        self.output_dir, self.task_name, self.method = output_dir, task_name, method
        self.round_id = 0
        self.scores = []

    def run(self, policy, **kw):
        r = evaluate(policy, self.task_name, self.method,
            Path(self.output_dir) / 'development_rollouts' / f'round_{self.round_id:04d}',
            round_id=self.round_id)
        self.round_id += 1
        self.scores.append(r['success_rate'])
        return {'test_mean_score': r['success_rate'], 'test/mean_score': r['success_rate'],
                'mean_success_rates': r['success_rate'], 'mean_traj_rewards': r['mean_return'],
                'SR_test_L3': float(np.mean(sorted(self.scores)[-3:])),
                'SR_test_L5': float(np.mean(sorted(self.scores)[-5:]))}
