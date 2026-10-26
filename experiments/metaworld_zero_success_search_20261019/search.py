"""Finite, auditable inference-only search. Never writes into the baseline campaign."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'metaworld_official_baselines_50demo_20261019'
sys.path.insert(0, str(BASE))
from runtime import ROOT, PYTHON, atomic_json, bootstrap, environment, sha256


def configs(method):
    if method == 'ACT':
        return [{'temporal_agg': True, 'action_horizon': 1, 'aggregation_k': 0.01}] + [
            {'temporal_agg': False, 'action_horizon': h} for h in [1, 2, 4, 8, 16, 32, 100]]
    steps = [100, 50, 20, 10] if method == 'DP' else [10, 20, 50]
    return [{'num_inference_steps': n, 'action_horizon': h}
            for h in [1, 2, 8] for n in steps]


def get_plan():
    dest = HERE / 'plan.json'
    if dest.exists():
        return json.loads(dest.read_text())
    rows = list(csv.DictReader((BASE / 'report/results.csv').open()))
    targets = [dict(r) for r in rows if int(r['episodes']) == 20 and int(r['successes']) == 0]
    plan = {'created_at': time.time(), 'baseline_root': str(BASE),
        'baseline_results_sha256': sha256(BASE / 'report/results.csv'),
        'script_sha256': sha256(__file__), 'targets': targets,
        'scope_note': 'stick-pull-v2 is not in the existing five-task campaign; no matching baseline checkpoint is available in this campaign.',
        'blocks': 3, 'episodes_per_trial': 3, 'search_seed_start': 3026101900,
        'holdout_seeds': list(range(4026101900, 4026101920)),
        'max_episode_steps': 200, 'success_definition': 'any official info.success within 200 steps',
        'stop_rule': 'Complete each three-episode trial. Lock the first trial with >0 successes; evaluate exactly one independent 20-episode holdout. Do not select on holdout results.',
        'grids': {m: configs(m) for m in ['ACT', 'DP', 'SimpleDP3']},
        'training_changes': None, 'classification': 'inference-parameter/seed-screened evaluation; original baseline retained'}
    atomic_json(dest, plan)
    (HERE / 'baseline_results_snapshot.csv').write_text((BASE / 'report/results.csv').read_text())
    return plan


def load_model(row):
    import torch
    run = Path(row['output'])
    method = row['method']
    if method == 'ACT':
        import worker
        sys.argv = worker.act_args(run, row['task'])
        cfg = json.loads((run / 'config.json').read_text())
        policy = worker.act_functions()['make_policy']('ACT', cfg['policy_config'])
        checkpoint = run / 'policy_best.ckpt'
        policy.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False), strict=True)
        with (run / 'dataset_stats.pkl').open('rb') as f:
            stats = pickle.load(f)
    else:
        import dill
        import hydra
        checkpoint = Path(json.loads((run / 'trained.json').read_text())['checkpoint'])
        payload = torch.load(checkpoint, pickle_module=dill, map_location='cpu', weights_only=False)
        cfg = payload['cfg']
        policy = hydra.utils.instantiate(cfg.policy)
        key = 'ema_model' if cfg.training.use_ema else 'model'
        policy.load_state_dict(payload['state_dicts'][key], strict=True)
        stats = None
    assert sha256(checkpoint) == row['checkpoint_sha256'], 'Checkpoint identity changed'
    return policy.cuda().eval(), stats, checkpoint


def rollout(policy, stats, row, config, seeds, dest, stage, expert=False):
    import collections
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from runtime import install_render_fix
    from rollout import seed_all, observation
    install_render_fix()
    from diffusion_policy_3d.env import MetaWorldEnv
    dest.mkdir(parents=True, exist_ok=True)
    identity = {'method': row['method'], 'task': row['task'],
        'checkpoint_sha256': row['checkpoint_sha256'], 'config': config,
        'stage': stage, 'expert_positive_control': expert, 'plan_sha256': sha256(HERE / 'plan.json')}
    env, records = None, []
    method = row['method']
    if method != 'ACT' and not expert:
        policy.n_action_steps = config['action_horizon']
        policy.num_inference_steps = config['num_inference_steps']
    try:
        for episode, seed in enumerate(seeds):
            ep = dest / f'episode_{episode:02d}'
            if ep.with_suffix('.json').exists():
                previous = json.loads(ep.with_suffix('.json').read_text())
                assert previous['identity'] == identity and previous['seed'] == seed
                assert ep.with_suffix('.npz').is_file() and ep.with_suffix('.mp4').is_file()
                records.append(previous)
                continue
            seed_all(seed)
            if env is None:
                env = MetaWorldEnv(row['task'], device='cuda:0', num_points=512, use_point_crop=True)
            env.env.seed(seed)
            obs = env.reset()
            initial = np.concatenate([env.env.sim.get_state().flatten(), obs['full_state'], np.asarray(env.env._target_pos).ravel()])
            initial_hash = hashlib.sha256(np.asarray(initial, dtype=np.float64).tobytes()).hexdigest()
            history = collections.deque([observation(obs, method)] * 2, maxlen=2)
            if method != 'ACT' and not expert:
                policy.reset()
            if expert:
                from metaworld import policies
                expert_policy = getattr(policies, 'Sawyer' + ''.join(x.title() for x in row['task'].split('-')) + 'V2Policy')()
            all_time = torch.zeros((200, 300, 4), device='cuda') if method == 'ACT' and config.get('temporal_agg') and not expert else None
            frames, actions_log, states_log, infos, rewards = [], [], [], [], []
            predicted_chunks, query_steps = [], []
            success, first_success, steps = False, None, 0
            chunk, chunk_index = None, 0
            while steps < 200:
                states_log.append(np.asarray(obs['agent_pos']).copy())
                if expert:
                    action = expert_policy.get_action(obs['full_state'])
                elif method == 'ACT':
                    interval = config['action_horizon']
                    if steps % interval == 0:
                        q = (obs['agent_pos'] - stats['qpos_mean']) / stats['qpos_std']
                        with torch.inference_mode():
                            normalized = policy(torch.as_tensor(q, dtype=torch.float32, device='cuda')[None],
                                torch.as_tensor(obs['image'], dtype=torch.float32, device='cuda')[None, None] / 255.)
                        assert normalized.shape == (1, 100, 4)
                        assert bool(torch.isfinite(normalized).all()), 'Nonfinite normalized ACT output'
                        predicted_chunks.append(normalized[0].cpu().numpy())
                        query_steps.append(steps)
                    if config['temporal_agg']:
                        assert interval == 1, 'Official temporal aggregation queries every step'
                        all_time[[steps], steps:steps+100] = normalized
                        current = all_time[:, steps]
                        current = current[torch.all(current != 0, axis=1)]
                        assert len(current) > 0
                        weights = np.exp(-config['aggregation_k'] * np.arange(len(current)))
                        weights /= weights.sum()
                        raw = (current * torch.from_numpy(weights).cuda().unsqueeze(1)).sum(0)
                    else:
                        raw = normalized[0, steps % interval]
                    action = raw.cpu().numpy() * stats['action_std'] + stats['action_mean']
                else:
                    if chunk is None or chunk_index == len(chunk):
                        batch = {k: torch.as_tensor(np.stack([x[k] for x in history]), dtype=torch.float32, device='cuda')[None] for k in history[0]}
                        with torch.inference_mode():
                            chunk = policy.predict_action(batch)['action'][0].cpu().numpy()
                        assert chunk.shape == (config['action_horizon'], 4)
                        predicted_chunks.append(chunk.copy())
                        query_steps.append(steps)
                        chunk_index = 0
                    action = chunk[chunk_index]
                    chunk_index += 1
                assert np.asarray(action).shape == (4,) and np.isfinite(action).all(), 'Invalid model action'
                actions_log.append(np.asarray(action).copy())
                obs, reward, done, info = env.step(action)
                history.append(observation(obs, method))
                info = {k: np.asarray(v).tolist() for k, v in info.items()}
                infos.append(info)
                rewards.append(float(reward))
                if bool(info['success']) and not success:
                    first_success = steps + 1
                success = success or bool(info['success'])
                if steps % 2 == 0:
                    frames.append(obs['image'].transpose(1, 2, 0))
                steps += 1
                if steps % 10 == 0:
                    atomic_json(HERE / 'heartbeat.json', {'time': time.time(), 'pid': os.getpid(),
                        'target': row['method'] + '_' + row['task'], 'stage': stage,
                        'directory': str(dest), 'episode': episode, 'steps': steps})
                if done:
                    break
            actions_arr = np.asarray(actions_log)
            record = {'identity': identity, 'episode': episode, 'seed': seed, 'steps': steps,
                'initial_state_sha256': initial_hash, 'success': success, 'first_success_step': first_success,
                'return': float(sum(rewards)), 'infos': infos,
                'action_min': actions_arr.min(0).tolist(), 'action_max': actions_arr.max(0).tolist(),
                'action_std': actions_arr.std(0).tolist(), 'nonfinite_actions': int((~np.isfinite(actions_arr)).sum()),
                'fraction_components_outside_env_bounds': float(((actions_arr < env.action_space.low) | (actions_arr > env.action_space.high)).mean()),
                'video': str(ep.with_suffix('.mp4')), 'trajectory': str(ep.with_suffix('.npz')), 'time': time.time()}
            np.savez_compressed(ep.with_suffix('.npz'), actions=actions_arr, agent_pos=np.asarray(states_log),
                rewards=np.asarray(rewards), query_steps=np.asarray(query_steps),
                predicted_chunks=np.asarray(predicted_chunks), initial_state=initial)
            imageio.mimsave(ep.with_suffix('.mp4'), frames, fps=10)
            # JSON is the commit marker: all raw artifacts must exist before it is written.
            atomic_json(ep.with_suffix('.json'), record)
            records.append(record)
            print(json.dumps({'stage': stage, 'episode': episode, 'seed': seed, 'success': success,
                              'return': record['return'], 'path': str(ep)}, ensure_ascii=False), flush=True)
    finally:
        if env is not None:
            env.env.close()
    result = {'identity': identity, 'episodes': len(records), 'successes': sum(x['success'] for x in records),
        'success_rate': sum(x['success'] for x in records) / len(records),
        'seeds': list(seeds), 'successful_episodes': [x['video'] for x in records if x['success']]}
    atomic_json(dest / 'summary.json', result)
    return result


def run_target(row, plan):
    bootstrap()
    import torch
    torch.set_num_threads(2)
    policy, stats, checkpoint = load_model(row)
    target = HERE / (row['method'] + '_' + row['task'])
    target.mkdir(exist_ok=True)
    lock_path = target / 'locked.json'
    locked = json.loads(lock_path.read_text()) if lock_path.exists() else None
    grid = plan['grids'][row['method']]
    if locked is None:
        for block in range(plan['blocks']):
            for index, cfg in enumerate(grid):
                trial = block * len(grid) + index
                seeds = list(range(plan['search_seed_start'] + 3 * trial, plan['search_seed_start'] + 3 * trial + 3))
                result = rollout(policy, stats, row, cfg, seeds, target / 'search' / f'trial_{trial:03d}', 'search')
                assert result['episodes'] == 3
                if result['successes'] > 0:
                    locked = {'config': cfg, 'screening_seeds': seeds, 'screening_result': result,
                        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'locked_at': time.time()}
                    atomic_json(lock_path, locked)
                    break
            if locked is not None:
                break
    if locked is not None:
        result = rollout(policy, stats, row, locked['config'], plan['holdout_seeds'], target / 'holdout20', 'holdout20')
        before = {'temporal_agg': False, 'action_horizon': 100} if row['method'] == 'ACT' else {
            'num_inference_steps': 100 if row['method'] == 'DP' else 10, 'action_horizon': 8}
        atomic_json(target / 'parameter_diff.json', {'baseline': before, 'locked': locked['config'],
            'training_or_checkpoint_changes': False, 'trained_prediction_horizon_unchanged': 100 if row['method'] == 'ACT' else 16})
        atomic_json(target / 'complete.json', {'state': 'holdout_complete', 'screening': locked['screening_result'], 'holdout': result})
    else:
        controls = rollout(policy, stats, row, grid[0], plan['holdout_seeds'][:3], target / 'expert_positive_control', 'expert_control', expert=True)
        atomic_json(target / 'complete.json', {'state': 'bounded_search_exhausted',
            'trials': len(grid) * plan['blocks'], 'search_episodes': len(grid) * plan['blocks'] * 3,
            'expert_positive_control': controls, 'nonzero_result': False})
    assert sha256(checkpoint) == row['checkpoint_sha256'], 'Checkpoint modified during evaluation'


def report(plan):
    baseline = list(csv.DictReader((BASE / 'report/results.csv').open()))
    lines = ['# MetaWorld 推理参数搜索与独立重评测', '',
        '原始基线保留。搜索命中后的 20 局使用预先固定的独立 seed，不将筛选用 3 局混入成功率。',
        '这些结果属于参数与 seed 筛选后的评测，不能当作未调参的官方基线。训练和 checkpoint 未改动。', '',
        '| 任务 | 方法 | 原始基线 | 筛选 | 独立20局 | 状态 |', '|---|---|---:|---:|---:|---|']
    output_rows = []
    for task in ['shelf-place', 'pick-place-wall', 'pick-place', 'stick-push', 'door-open']:
        for method in ['DP', 'ACT', 'SimpleDP3']:
            old = next((r for r in baseline if r['task'] == task and r['method'] == method), None)
            target = HERE / (method + '_' + task)
            complete = json.loads((target / 'complete.json').read_text()) if (target / 'complete.json').exists() else None
            orig = f"{old['successes']}/{old['episodes']} ({100*float(old['success_rate']):.0f}%)" if old else '未完成'
            screen, holdout, status = '—', '—', '未调参' if old else '基线未完成'
            if complete:
                if complete['state'] == 'holdout_complete':
                    s, h = complete['screening'], complete['holdout']
                    screen = f"{s['successes']}/3"
                    holdout = f"{h['successes']}/20 ({h['success_rate']*100:.0f}%)"
                    status = '搜索后评测完成'
                else:
                    screen = f"0/{complete['search_episodes']}"
                    status = '有限搜索空间穷尽；详见排错报告'
            lines.append(f'| {task} | {method} | {orig} | {screen} | {holdout} | {status} |')
            output_rows.append({'task': task, 'method': method, 'baseline': orig, 'screening': screen, 'holdout20': holdout, 'status': status})
    lines += ['', '## 范围与记录', '', plan['scope_note'], '',
        f"每个方法预先固定 {plan['blocks']} 个 seed 轮次，每个参数组合 3 局；首个命中即锁定。",
        'ACT 使用原始 100 步预测块，action_horizon 仅控制执行多少步后重新查询。时间聚合严格采用官方 k=0.01 和原有权重方向。',
        'DP 的 action_horizon 对应 policy.n_action_steps；训练预测 horizon=16 保持不变。',
        '所有尝试的 episode JSON、MP4、动作/模型输出 NPZ 均保留在各任务 search/ 中。',
        '独立20局 seeds: 4026101900–4026101919。成功定义与基线一致：200步内任意 official info.success。', '']
    for row in plan['targets']:
        target = HERE / (row['method'] + '_' + row['task'])
        cp = target / 'complete.json'
        if not cp.exists():
            continue
        complete = json.loads(cp.read_text())
        if complete['state'] == 'bounded_search_exhausted':
            episodes = [json.loads(p.read_text()) for p in sorted((target / 'search').glob('trial_*/episode_*.json'))]
            diag = ['# 排错分析', '', f"目标：{row['method']} / {row['task']}",
                f"预先登记的有限空间已穷尽：{complete['trials']} 组 × 3 局，全部0成功。没有宣称穷尽所有可能整数参数或无限seed。",
                f"官方专家正对照：{complete['expert_positive_control']['successes']}/3。专家仅用于环境诊断，未用于模型评测动作。",
                f"模型动作非有限数总量：{sum(e['nonfinite_actions'] for e in episodes)}。",
                f"动作分量超过原生环境动作边界的平均比例：{sum(e['fraction_components_outside_env_bounds'] for e in episodes)/len(episodes):.6f}。原生环境自行裁剪，未额外干预。",
                f"回报范围：{min(e['return'] for e in episodes):.6f} 至 {max(e['return'] for e in episodes):.6f}。",
                '每步完整官方 info、动作范围及变化幅度见 episode JSON；原始动作、归一化预测块、查询时刻和初始状态见同名 NPZ。',
                '严格加载已验证的原始 checkpoint，使用原有相机、图像归一化、9维状态、4维动作及数据统计。',
                '若专家成功且模型全失败，说明该环境和成功判定可达；不能据此证明所有模型实现均无误，也不能保证单纯推理参数能解决训练策略的泛化失败。',
                '搜索已按停止规则结束，不删除失败样本，不继续筛选holdout，不把专家成功计入模型成功率。', '']
            (target / 'diagnostic_report.md').write_text('\n'.join(diag))
        elif complete['state'] == 'holdout_complete':
            lines += [f"### {row['method']} / {row['task']} 成功 episode", '']
            hits = complete['holdout']['successful_episodes']
            lines += hits if hits else ['独立20局未出现成功；筛选命中不保证独立评测非0。']
            lines += ['']
    (HERE / 'summary.md').write_text('\n'.join(lines))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
    writer.writeheader()
    writer.writerows(output_rows)
    (HERE / 'summary.csv').write_text(stream.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    plan = get_plan()
    if args.worker is not None:
        run_target(plan['targets'][args.worker], plan)
        return
    atomic_json(HERE / 'supervisor_pid.json', {'pid': os.getpid(), 'time': time.time()})
    for index, row in enumerate(plan['targets']):
        target = HERE / (row['method'] + '_' + row['task'])
        if (target / 'complete.json').exists():
            continue
        for attempt in range(1, 4):
            # Inference admission leaves >=4 GiB of physical free memory after a 2 GiB allocation.
            while True:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
                candidates = [(int(free), int(gpu)) for gpu, free in (line.split(',') for line in out.splitlines())]
                free, gpu = max(candidates)
                if free >= 6144:
                    break
                atomic_json(HERE / 'status.json', {'time': time.time(), 'state': 'waiting_for_inference_memory', 'free_mib': free})
                time.sleep(15)
            log_path = HERE / f'worker_{index:02d}_attempt_{attempt:02d}.log'
            with log_path.open('a') as log:
                child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker', str(index)], env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.time()
                while child.poll() is None:
                    state = {'time': time.time(), 'state': 'running', 'target': row['method'] + '_' + row['task'],
                        'pid': os.getpid(), 'worker_pid': child.pid, 'gpu': gpu, 'attempt': attempt, 'log': str(log_path)}
                    atomic_json(HERE / 'status.json', state)
                    hb = HERE / 'heartbeat.json'
                    latest = max(started, hb.stat().st_mtime if hb.exists() else started, log_path.stat().st_mtime)
                    if time.time() - latest > 600:
                        child.terminate()
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            child.kill()
                        atomic_json(HERE / f'alert_{index}_{attempt}.json', {**state, 'error': '600s without progress'})
                    time.sleep(5)
                if child.returncode == 0:
                    break
                atomic_json(HERE / f'error_{index}_{attempt}.json', {'time': time.time(), 'exit_code': child.returncode, 'log': str(log_path), 'attempt': attempt})
                if attempt == 3:
                    atomic_json(HERE / 'status.json', {'time': time.time(), 'state': 'infrastructure_error', 'log': str(log_path)})
                    raise RuntimeError(f'Worker repeatedly failed: {log_path}')
        report(plan)
    report(plan)
    atomic_json(HERE / 'status.json', {'time': time.time(), 'state': 'complete', 'pid': os.getpid()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
