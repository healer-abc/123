"""Run the fixed 5 x 20 unfinetuned base evaluation; retain every attempt."""
import argparse
from collections import deque
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OPENPI_DATA_HOME', '/home/zhsu/jyluo/myopenpi/dataset/ckpt')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
PROTOCOL_HASH = hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def status(phase, **kwargs):
    record = dict(time=time.time(), pid=os.getpid(), phase=phase, **kwargs)
    save(ROOT / 'logs/status.json', record)
    print(json.dumps(record), flush=True)


def summarize():
    rows, records = [], []
    for task in PROTOCOL['tasks']:
        task_records = []
        for file in sorted((ROOT / 'results' / task).glob('episode_*.json')):
            record = json.loads(file.read_text())
            assert record['protocol_sha256'] == PROTOCOL_HASH
            task_records.append(record)
        valid = [r for r in task_records if r['status'] == 'valid']
        successes = sum(r['success'] for r in valid)
        rows.append(dict(task=task, completed=len(task_records), valid=len(valid),
                         infrastructure_missing=len(task_records) - len(valid), successes=successes,
                         success_rate=successes / len(valid) if valid else None))
        records.extend(task_records)
    summary = dict(tasks=rows, planned=100, completed=len(records), valid=sum(r['valid'] for r in rows),
                   successes=sum(r['successes'] for r in rows), protocol_sha256=PROTOCOL_HASH)
    summary['complete'] = summary['completed'] == summary['planned']
    summary['overall_success_rate'] = summary['successes'] / summary['valid'] if summary['valid'] else None
    save(ROOT / 'results/summary.json', summary)
    with (ROOT / 'results/summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fields = ['task','episode','seed','task_index','status','success','first_success_step','steps',
              'elapsed_seconds','clipped_action_components','video','protocol_sha256']
    with (ROOT / 'results/episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(records)
    lines = ['# 未微调 π0.5 Base：MetaWorld v2', '', '| 任务 | 完成 | 有效 | 成功 | 成功率 |', '|---|---:|---:|---:|---:|']
    for row in rows:
        rate = f"{row['success_rate']:.1%}" if row['success_rate'] is not None else '待评测'
        lines.append(f"| {row['task']} | {row['completed']}/20 | {row['valid']} | {row['successes']} | {rate} |")
    lines += ['', '模型为原始 pi05_base；每轮首次 info.success=true 立即结束并计成功。',
              '使用固定直接动作接口，没有经过验证的 Sawyer 专用解码器/归一化；低成功率不能单独归因于模型理解能力。',
              f"已完成 {summary['completed']}/100，有效 {summary['valid']}，成功 {summary['successes']}。"]
    (ROOT / 'results/report.md').write_text('\n'.join(lines) + '\n')
    return summary


def execute_episode(env, tasks, indices, task, episode, policy, adapter, attempt, smoke=False):
    import imageio.v2 as imageio
    import numpy as np
    seed = PROTOCOL['episode_seeds'][episode]
    chosen = tasks[indices[episode]]
    env.set_task(chosen)
    obs, _ = env.reset(seed=seed)
    policy.reset(seed)
    actions, raw_plans, rewards, successes, latencies, states, images = [], [], [], [], [], [], []
    plan = deque()
    success = False
    clipped_total = 0
    start = time.monotonic()
    prefix = 'smoke' if smoke else 'results'
    video = ROOT / 'videos' / prefix / task / f'ep{episode:02d}_attempt{attempt}.mp4'
    video.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(video, fps=10, codec='libx264', quality=6) as writer:
            for step in range(PROTOCOL['max_policy_steps']):
                image = adapter.render(env)
                if step % 5 == 0:
                    writer.append_data(image)
                state = adapter.encode_state(env, obs)
                states.append(state)
                if not plan:
                    status('rollout', task=task, episode=episode, attempt=attempt, step=step)
                    t0 = time.monotonic()
                    raw = policy.infer(image, state, PROTOCOL['prompts'][task])
                    latencies.append(time.monotonic() - t0)
                    raw_plans.append(raw)
                    decoded, _ = adapter.decode_actions(env, raw)
                    for action, raw_action in zip(decoded[:PROTOCOL['replan_steps']], raw[:PROTOCOL['replan_steps'], :4]):
                        plan.append((action, int(np.count_nonzero(action != raw_action))))
                action, clipped_count = plan.popleft()
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise FloatingPointError('nonfinite_environment')
                clipped_total += clipped_count
                actions.append(action); rewards.append(float(reward)); successes.append(bool(info['success']))
                if bool(info['success']):
                    success = True
                    writer.append_data(adapter.render(env))
                    break
                if terminated or truncated:
                    break
        trajectory = ROOT / prefix / task / f'episode_{episode:02d}_attempt{attempt}.npz'
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trajectory, states=states, actions=actions, raw_plans=raw_plans,
                            rewards=rewards, successes=successes, latency_s=latencies)
        result = dict(task=task, episode=episode, seed=seed, task_index=indices[episode], status='valid',
                      success=success, first_success_step=len(actions) if success else None, steps=len(actions),
                      elapsed_seconds=time.monotonic()-start, clipped_action_components=clipped_total,
                      action_components=4*len(actions), video=str(video), trajectory=str(trajectory),
                      task_data_sha256=hashlib.sha256(chosen.data).hexdigest(),
                      reset_rand_vec=np.asarray(pickle.loads(chosen.data)['rand_vec']).tolist(),
                      mean_inference_seconds=float(np.mean(latencies)),
                      attempt=attempt, protocol_sha256=PROTOCOL_HASH)
        save(ROOT / prefix / task / f'episode_{episode:02d}.json', result)
        status('episode_complete', **result)
        return result
    except Exception as error:
        record = dict(task=task, episode=episode, seed=seed, attempt=attempt, error=traceback.format_exc(),
                      exception_type=type(error).__name__, completed_steps=len(actions), video=str(video))
        save(ROOT / 'logs' / task / f'episode_{episode:02d}_attempt{attempt}_error.json', record)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-smoke', action='store_true')
    parser.add_argument('--policy-smoke', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from adapter import FrozenBasePolicy, MetaWorldAdapter
    adapter = MetaWorldAdapter(PROTOCOL)
    if args.env_smoke:
        import imageio.v2 as imageio
        import numpy as np
        manifest, checks = [], []
        for task in PROTOCOL['tasks']:
            env, tasks, indices = adapter.make_env(task)
            try:
                for episode, index in enumerate(indices):
                    env.set_task(tasks[index]); obs, _ = env.reset(seed=PROTOCOL['episode_seeds'][episode])
                    manifest.append(dict(task=task, episode=episode, seed=PROTOCOL['episode_seeds'][episode],
                                         task_index=index, task_data_sha256=hashlib.sha256(tasks[index].data).hexdigest()))
                    if episode == 0:
                        image = adapter.render(env)
                        imageio.imwrite(ROOT / 'videos' / f'{task}_interface.png', image)
                        encoded = adapter.encode_state(env, obs)
                        decoded, _ = adapter.decode_actions(env, np.zeros((PROTOCOL['action_horizon'], 32)))
                        _, _, _, _, info = env.step(decoded[0])
                        checks.append(dict(task=task, image_shape=list(image.shape), state_shape=list(encoded.shape),
                                           action_shape=list(decoded.shape), has_success='success' in info))
            finally:
                env.close()
        save(ROOT / 'manifests/episodes.json', manifest)
        save(ROOT / 'logs/interface_checks.json', checks)
        print(json.dumps(checks), flush=True)
        return
    status('loading_base_model')
    policy = FrozenBasePolicy(PROTOCOL)
    status('model_loaded')
    selected = PROTOCOL['tasks'][:1] if args.policy_smoke else PROTOCOL['tasks']
    for task in selected:
        env, tasks, indices = adapter.make_env(task)
        try:
            count = 1 if args.policy_smoke else PROTOCOL['trials_per_task']
            for episode in range(count):
                if not args.policy_smoke and (ROOT / 'results' / task / f'episode_{episode:02d}.json').exists():
                    continue
                for attempt in (1, 2):
                    try:
                        execute_episode(env, tasks, indices, task, episode, policy, adapter, attempt, args.policy_smoke)
                        break
                    except (ValueError, FloatingPointError):
                        status('policy_or_interface_error', task=task, episode=episode, error=traceback.format_exc())
                        raise
                    except Exception:
                        if attempt == 2:
                            save(ROOT / 'results' / task / f'episode_{episode:02d}.json',
                                 dict(task=task, episode=episode, seed=PROTOCOL['episode_seeds'][episode],
                                      task_index=indices[episode], status='infrastructure_missing', success=None,
                                      protocol_sha256=PROTOCOL_HASH))
                if not args.policy_smoke:
                    summarize()
        finally:
            env.close()
    if args.policy_smoke:
        status('policy_smoke_complete')
    else:
        status('complete', summary=summarize())


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('error', error=traceback.format_exc())
        raise
