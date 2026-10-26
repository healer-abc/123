"""Five-step DDIM with eight predicted positions and the original official action slicing."""
import collections
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from runtime import ROOT, atomic_json, install_render_fix, sha256


def evaluate(policy, task, method, output_dir, *, final=False, checkpoint=None):
    assert final and method == 'SimpleDP3' and checkpoint
    assert (policy.horizon, policy.n_obs_steps, policy.n_action_steps, policy.num_inference_steps) == (8, 2, 8, 5)
    install_render_fix()
    from diffusion_policy_3d.env import MetaWorldEnv
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    seeds = protocol['final_episode_seeds']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {'task': task, 'method': method, 'config': {'horizon': 8, 'trained_horizon': 16,
        'n_obs_steps': 2, 'configured_n_action_steps': 8, 'action_horizon': 7,
        'num_inference_steps': 5, 'scheduler': 'DDIM', 'prediction_type': 'sample'},
        'protocol': sha256(ROOT / 'protocol.json'), 'dataset_episodes': 50,
        'dataset_manifest_sha256': sha256(ROOT / 'datasets' / (task + '.json')),
        'split_sha256': sha256(ROOT / 'manifests/dataset_split.json'), 'training_seed': 0,
        'stage': 'exploratory_epoch300_ddim5_horizon8_20', 'round': 0,
        'ablation_plan_sha256': sha256(Path(__file__).resolve().parent / 'plan.json'),
        'checkpoint_completed_epochs': 300, 'checkpoint_sha256': sha256(checkpoint)}
    policy.eval()
    step_calls, forward_calls = [0], [0]
    original_step = policy.noise_scheduler.step
    def counted_step(*args, **kwargs):
        step_calls[0] += 1
        return original_step(*args, **kwargs)
    policy.noise_scheduler.step = counted_step
    def check_forward(module, args, kwargs):
        assert kwargs['sample'].shape == (1, 8, 4), 'Wrong actual denoising trajectory length'
        forward_calls[0] += 1
    hook = policy.model.register_forward_pre_hook(check_forward, with_kwargs=True)
    env, records = None, []
    try:
        for i, seed in enumerate(seeds):
            ep = output_dir / f'episode_{i:02d}'
            if ep.with_suffix('.json').exists():
                record = json.loads(ep.with_suffix('.json').read_text())
                assert record['identity'] == identity and record['seed'] == seed
                assert ep.with_suffix('.npz').is_file() and ep.with_suffix('.mp4').is_file()
                records.append(record)
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if env is None:
                env = MetaWorldEnv(task, device='cuda:0', num_points=512, use_point_crop=True)
            env.env.seed(seed)
            obs = env.reset()
            initial = np.concatenate([env.env.sim.get_state().flatten(), obs['full_state'], np.asarray(env.env._target_pos).ravel()])
            initial_hash = hashlib.sha256(np.asarray(initial, dtype=np.float64).tobytes()).hexdigest()
            def observation(observation):
                return {'point_cloud': observation['point_cloud'].astype(np.float32),
                        'agent_pos': observation['agent_pos'].astype(np.float32)}
            history = collections.deque([observation(obs)] * 2, maxlen=2)
            policy.reset()
            frames, actions, agent_pos, infos, rewards, chunks, predictions, query_steps = [], [], [], [], [], [], [], []
            chunk, index = None, 0
            sampler_before, forward_before = step_calls[0], forward_calls[0]
            with torch.inference_mode():
                for step in range(200):
                    agent_pos.append(np.asarray(obs['agent_pos']).copy())
                    if chunk is None or index == len(chunk):
                        batch = {k: torch.as_tensor(np.stack([x[k] for x in history]), dtype=torch.float32,
                                                    device='cuda')[None] for k in history[0]}
                        prediction = policy.predict_action(batch)
                        full = prediction['action_pred'][0].cpu().numpy()
                        chunk = prediction['action'][0].cpu().numpy()
                        assert full.shape == (8, 4) and chunk.shape == (7, 4)
                        assert np.array_equal(chunk, full[1:8]), 'Official observation-offset action slice changed'
                        assert np.isfinite(full).all()
                        predictions.append(full.copy())
                        chunks.append(chunk.copy())
                        query_steps.append(step)
                        index = 0
                    action = chunk[index]
                    index += 1
                    assert action.shape == (4,) and np.isfinite(action).all(), 'Nonfinite model action'
                    actions.append(action.copy())
                    obs, reward, done, info = env.step(action)
                    infos.append({k: np.asarray(v).tolist() for k, v in info.items()})
                    rewards.append(float(reward))
                    history.append(observation(obs))
                    if step % 2 == 0:
                        frames.append(obs['image'].transpose(1, 2, 0))
                    if (step + 1) % 10 == 0:
                        atomic_json(output_dir / 'heartbeat.json', {'time': time.time(), 'episode': i,
                                                                  'steps': step + 1, 'final': True})
                    if done:
                        break
            assert len(actions) == 200 and query_steps == list(range(0, 200, 7))
            assert step_calls[0]-sampler_before == forward_calls[0]-forward_before == 145
            successful_steps = [j+1 for j, info in enumerate(infos) if bool(info['success'])]
            arr = np.asarray(actions)
            record = {'identity': identity, 'episode': i, 'seed': seed, 'steps': 200,
                'initial_state_sha256': initial_hash, 'success': bool(successful_steps),
                'first_success_step': successful_steps[0] if successful_steps else None,
                'return': float(sum(rewards)), 'infos': infos, 'nonfinite_actions': int((~np.isfinite(arr)).sum()),
                'fraction_components_outside_env_bounds': float(((arr < env.action_space.low) | (arr > env.action_space.high)).mean()),
                'num_inference_steps': 5, 'inference_horizon': 8, 'configured_n_action_steps': 8,
                'effective_action_steps': 7, 'inference_queries': 29, 'denoising_step_calls': 145,
                'model_forward_calls': 145, 'actual_denoising_sample_shape': [1, 8, 4],
                'trajectory': str(ep.with_suffix('.npz')), 'video': str(ep.with_suffix('.mp4')), 'time': time.time()}
            np.savez_compressed(ep.with_suffix('.npz'), actions=arr, agent_pos=np.asarray(agent_pos),
                rewards=np.asarray(rewards), query_steps=np.asarray(query_steps), predicted_chunks=np.asarray(chunks),
                full_predicted_trajectories=np.asarray(predictions), initial_state=initial)
            import imageio.v2 as imageio
            imageio.mimsave(ep.with_suffix('.mp4'), frames, fps=10)
            atomic_json(ep.with_suffix('.json'), record)
            records.append(record)
            print(json.dumps({'task': task, 'episode': i, 'seed': seed, 'success': record['success'], 'time': record['time']}), flush=True)
    finally:
        hook.remove()
        policy.noise_scheduler.step = original_step
        if env is not None:
            env.env.close()
    assert len(records) == 20
    result = {'identity': identity, 'episodes': 20, 'seeds': seeds,
        'successes': sum(bool(r['success']) for r in records),
        'success_rate': sum(bool(r['success']) for r in records)/20,
        'mean_return': float(np.mean([r['return'] for r in records])),
        'successful_episodes': [r['video'] for r in records if r['success']]}
    atomic_json(output_dir / 'summary.json', result)
    return result
