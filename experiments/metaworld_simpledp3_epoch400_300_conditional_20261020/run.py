"""Transparent outcome-dependent checkpoint exploration, with a frozen two-branch rule."""
import csv
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'metaworld_community_baselines_50demo_20261020'
REPLAY = HERE.parent / 'metaworld_simpledp3_epoch500_ddim10_20261020'
sys.path.insert(0, str(BASE))
from runtime import TASKS, PYTHON, atomic_json, environment, bootstrap, sha256


def plan():
    path = HERE / 'plan.json'
    if path.exists():
        return json.loads(path.read_text())
    protocol = json.loads((BASE / 'protocol.json').read_text())
    p = {'created': time.time(), 'method': 'SimpleDP3', 'tasks': TASKS,
        'screen_task': 'pick-place', 'screen_epoch': 400, 'screen_episodes': 20,
        'selection_rule': 'screen successes <18/20 => epoch400 for all5; >=18/20 => epoch300 for all5',
        'threshold_successes': 18, 'below_threshold_epoch': 400, 'at_or_above_threshold_epoch': 300,
        'reuse_screen_as_pick_place_result_if_epoch400_selected': True,
        'episodes_per_task': 20, 'seeds': protocol['final_episode_seeds'], 'max_episode_steps': 200,
        'training_seed': 0, 'num_inference_steps': 10, 'horizon': 16, 'n_obs_steps': 2, 'action_horizon': 8,
        'source_replay_plan_sha256': sha256(REPLAY / 'plan.json'),
        'baseline_protocol_sha256': sha256(BASE / 'protocol.json'),
        'execution_files': {n: sha256(HERE / n) for n in ['run.py', 'rollout.py']},
        'classification': 'Post-hoc outcome-dependent checkpoint exploration on the already observed common20 seeds. Not an independently selected baseline or an unbiased comparison. Retain screen and all original500/1000 results.',
        'max_parallel': 2, 'inference_budget_mib': 2048, 'reserve_mib': 4096,
        'stop_rule': 'Complete20 screening episodes before selecting one branch; then finish all5 tasks20 episodes regardless of scores. No other branch changes, seeds or hyperparameter search.'}
    atomic_json(path, p)
    (HERE / 'parameters.diff').write_text('--- fixed_epoch500\n+++ outcome_selected_epoch400_or300\n'
        '-checkpoint_completed_epochs: 500\n+screen: epoch400 pick-place20episodes\n'
        '+if successes <18: epoch400 for allfive\n+else: epoch300 for allfive\n'
        ' num_inference_steps: 10\n training_seed: 0\n n_obs_steps: 2\n action_horizon: 8\n')
    return p


def verify(p):
    for name, digest in p['execution_files'].items():
        assert sha256(HERE / name) == digest, 'Execution source changed: ' + name
    assert sha256(BASE / 'protocol.json') == p['baseline_protocol_sha256']
    assert sha256(REPLAY / 'plan.json') == p['source_replay_plan_sha256']


def target(epoch, task):
    path = HERE / 'targets' / f'epoch{epoch}_{task}.json'
    if path.exists():
        return json.loads(path.read_text())
    for out in sorted((REPLAY / 'runs' / task).glob('attempt_*')):
        receipt_path = out / 'snapshot_receipts' / f'epoch_{epoch:04d}.json'
        if not receipt_path.exists():
            continue
        r = json.loads(receipt_path.read_text())
        assert r['completed_epochs'] == epoch and r['optimizer_updates'] == 61 * epoch
        assert sha256(r['checkpoint']) == r['sha256']
        t = {'task': task, 'epoch': epoch, 'checkpoint': r['checkpoint'], 'checkpoint_sha256': r['sha256'],
            'training_output': str(out), 'config_sha256': sha256(out / 'config.yaml'),
            'snapshot_receipt': str(receipt_path), 'snapshot_receipt_sha256': sha256(receipt_path)}
        atomic_json(path, t)
        return t
    return None


def folder(epoch, task):
    return HERE / 'runs' / f'epoch{epoch}' / task / 'rollouts20'


def result(p, epoch, task):
    path = folder(epoch, task) / 'summary.json'
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    t = target(epoch, task)
    assert s['episodes'] == 20 and s['seeds'] == p['seeds']
    assert s['identity']['checkpoint_completed_epochs'] == epoch
    assert s['identity']['checkpoint_sha256'] == t['checkpoint_sha256']
    assert s['identity']['ablation_plan_sha256'] == sha256(HERE / 'plan.json')
    return s


def worker(p, epoch, task):
    verify(p)
    t = target(epoch, task)
    assert t and sha256(t['checkpoint']) == t['checkpoint_sha256']
    assert sha256(Path(t['training_output']) / 'config.yaml') == t['config_sha256']
    bootstrap()
    import torch
    import dill
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    torch.set_num_threads(2)
    payload = torch.load(t['checkpoint'], pickle_module=dill, map_location='cpu', weights_only=False)
    assert dill.loads(payload['pickles']['epoch']) == epoch
    assert dill.loads(payload['pickles']['global_step']) == 61 * epoch
    cfg = payload['cfg']
    assert cfg.training.seed == 0 and cfg.training.use_ema and cfg.policy.num_inference_steps == 10
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
    assert policy.horizon == 16 and policy.n_obs_steps == 2 and policy.n_action_steps == 8
    assert type(policy.noise_scheduler).__name__ == 'DDIMScheduler'
    assert policy.noise_scheduler.config.prediction_type == 'sample'
    del payload
    policy.cuda().eval()
    spec = importlib.util.spec_from_file_location('conditional_rollout', HERE / 'rollout.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.evaluate(policy, task, 'SimpleDP3', folder(epoch, task), final=True,
                       checkpoint=t['checkpoint'], checkpoint_epoch=epoch)
    assert sha256(t['checkpoint']) == t['checkpoint_sha256']


def audit_group(p, epoch, task):
    import numpy as np
    s, t = result(p, epoch, task), target(epoch, task)
    assert s and t
    assert sha256(t['checkpoint']) == t['checkpoint_sha256']
    # Pair initial states with the preserved original1000 reference; never change seeds.
    ref = json.loads((REPLAY / 'targets' / (task + '.json')).read_text())
    original = Path(ref['baseline_summary']).parent
    hits, records = 0, []
    for i, seed in enumerate(p['seeds']):
        path = folder(epoch, task) / f'episode_{i:02d}.json'
        e = json.loads(path.read_text())
        before = json.loads((original / path.name).read_text())
        assert e['seed'] == before['seed'] == seed and e['identity'] == s['identity']
        assert e['initial_state_sha256'] == before['initial_state_sha256']
        assert e['steps'] == len(e['infos']) == 200 and e['nonfinite_actions'] == 0
        native = [j+1 for j, info in enumerate(e['infos']) if bool(info['success'])]
        assert e['success'] == bool(native) and e['first_success_step'] == (native[0] if native else None)
        assert e['num_inference_steps'] == 10 and e['denoising_step_calls'] == 250
        assert Path(e['video']).stat().st_size > 0
        with np.load(e['trajectory']) as data:
            assert np.array_equal(data['query_steps'], np.arange(0, 200, 8))
            assert data['predicted_chunks'].shape == (25, 8, 4)
            assert np.allclose(data['predicted_chunks'].reshape(200, 4), data['actions'], rtol=1e-6, atol=1e-6)
            assert np.isfinite(data['actions']).all() and np.isclose(data['rewards'].sum(), e['return'])
        hits += bool(e['success'])
        records.append({'epoch': epoch, 'task': task, 'seed': seed, 'success': bool(e['success']),
                        'record': str(path), 'sha256': sha256(path), 'video': e['video'], 'trajectory': e['trajectory']})
    assert hits == s['successes']
    atomic_json(HERE / 'audits' / f'epoch{epoch}_{task}.json',
        {'passed': True, 'epoch': epoch, 'task': task, 'episodes': 20, 'successes': hits,
         'same_initial_states_as_original1000': True, 'native_success_recomputed': True,
         'actions_reconstructed': True, 'actual_denoising_calls_per_episode': 250})
    return records


def selection(p):
    path = HERE / 'selection.json'
    s = result(p, 400, 'pick-place')
    if not s:
        return None
    records = audit_group(p, 400, 'pick-place')
    chosen = 400 if s['successes'] < p['threshold_successes'] else 300
    value = {'screen_epoch': 400, 'screen_task': 'pick-place', 'screen_successes': s['successes'],
        'screen_episodes': 20, 'screen_rate': s['success_rate'], 'threshold': .9, 'selected_epoch': chosen,
        'screen_summary': str(folder(400, 'pick-place') / 'summary.json'),
        'screen_summary_sha256': sha256(folder(400, 'pick-place') / 'summary.json'),
        'plan_sha256': sha256(HERE / 'plan.json'), 'classification': p['classification']}
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Selection cannot change after it is locked'
    else:
        atomic_json(path, value)
        atomic_json(HERE / 'screen_episode_manifest.json', records)
    return value


def report(p, jobs, chosen, state='running'):
    rows = []
    for job in jobs:
        s = result(p, job['epoch'], job['task'])
        episodes = [json.loads(path.read_text()) for path in sorted(folder(job['epoch'], job['task']).glob('episode_*.json'))]
        rows.append({'key': job['key'], 'task': job['task'], 'epoch': job['epoch'],
            'successes': s['successes'] if s else sum(bool(e['success']) for e in episodes),
            'episodes': 20 if s else len(episodes), 'state': job['state']})
    atomic_json(HERE / 'status.json', {'time': time.time(), 'pid': os.getpid(), 'state': state,
        'selection': chosen, 'jobs': jobs, 'rows': rows})
    lines = ['# SimpleDP3：400轮筛选与条件评测', '',
        '先测400轮pick-place20局；成功少于18局→五任务均用400轮，否则五任务均用300轮。',
        'DDIM10、观测2、动作间隔8、训练seed0、统一评测seed2026102000–2026102019，每局200步。',
        '按观测结果选轮次的探索实验；保留全部筛选与原500/1000轮结果，不替换固定基线。', '',
        f'状态：{state}；选择：{chosen["selected_epoch"] if chosen else "等待400轮pick-place完整20局"}。', '',
        '| 阶段 | 任务 | 训练轮数 | 当前成功/已测 | 状态 |', '|---|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['key']} | {row['task']} | {row['epoch']} | {row['successes']}/{row['episodes']} | {row['state']} |")
    (HERE / 'progress.md').write_text('\n'.join(lines) + '\n')


def finalize(p, chosen):
    epoch = chosen['selected_epoch']
    rows, records = [], []
    for task in TASKS:
        s = result(p, epoch, task)
        records += audit_group(p, epoch, task)
        rows.append({'task': task, 'checkpoint_epoch': epoch, 'successes': s['successes'], 'episodes': 20, 'success_rate': s['success_rate']})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (HERE / 'summary.csv').write_text(stream.getvalue())
    atomic_json(HERE / 'summary.json', {'classification': p['classification'], 'selection': chosen, 'results': rows})
    atomic_json(HERE / 'episode_manifest.json', records)
    atomic_json(HERE / 'successful_episodes.json', [r for r in records if r['success']])
    lines = [f'# SimpleDP3：条件选择{epoch}轮，DDIM10', '',
        f"400轮pick-place筛选：{chosen['screen_successes']}/20（{chosen['screen_rate']:.0%}）；按预先记录的90%阈值选择{epoch}轮。", '',
        '| 任务 | 权重轮数 | 成功次数 | 成功率 |', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['task']} | {epoch} | {r['successes']}/20 | {r['success_rate']:.0%} |")
    lines += ['', f"五项平均：{sum(r['successes'] for r in rows)}%。", '',
        '这是按结果选轮次的探索评测，不能替代预先固定的基线；400轮筛选、固定500轮和1000轮结果均保留。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n')
    atomic_json(HERE / 'complete.json', {'state': 'complete', 'selected_epoch': epoch, 'selected_episodes': 100,
        'screen_episodes': 20, 'unique_episodes': 100 if epoch == 400 else 120, 'audited': True})


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    p = plan()
    verify(p)
    if len(sys.argv) > 1:
        assert sys.argv[1] == '--worker'
        worker(p, int(sys.argv[2]), sys.argv[3])
        return
    lock = (HERE / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def handler(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    atomic_json(HERE / 'supervisor_pid.json', {'pid': os.getpid(), 'started': time.time()})
    jobs = [{'key': 'screen400_pick-place', 'epoch': 400, 'task': 'pick-place',
             'state': 'complete' if result(p, 400, 'pick-place') else 'pending', 'failures': 0, 'attempt': 0, 'not_before': 0}]
    active, chosen = {}, None
    try:
        while True:
            free = {int(g): int(f) for g, f in (line.split(',') for line in subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines())}
            for key, item in list(active.items()):
                proc, job = item['proc'], item['job']
                heartbeat = folder(job['epoch'], job['task']) / 'heartbeat.json'
                last = max(job['started'], Path(job['log']).stat().st_mtime, heartbeat.stat().st_mtime if heartbeat.exists() else 0)
                reason = None
                if proc.poll() is None and (free[job['gpu']] < 3500 or time.time()-last > 600):
                    reason = 'low_free_memory' if free[job['gpu']] < 3500 else 'no_progress_600s'
                    stop(proc)
                if proc.poll() is not None:
                    item['log'].close()
                    if proc.returncode == 0 and result(p, job['epoch'], job['task']):
                        audit_group(p, job['epoch'], job['task'])
                        job['state'] = 'complete'
                    else:
                        job['failures'] += reason != 'low_free_memory'
                        job['state'] = 'pending' if job['failures'] < 3 else 'error'
                        job['not_before'] = time.time()+30
                        atomic_json(HERE / f'alert_{key}_attempt{job["attempt"]}.json',
                            {'time': time.time(), 'job': job, 'reason': reason or 'worker_error', 'exit_code': proc.returncode})
                    del active[key]
            # A failed screen must never silently select another branch.
            if jobs[0]['state'] == 'complete' and chosen is None:
                chosen = selection(p)
                for task in TASKS:
                    epoch = chosen['selected_epoch']
                    if epoch == 400 and task == 'pick-place':
                        continue  # The audited20 screening episodes are also this group's20 episodes.
                    jobs.append({'key': f'epoch{epoch}_{task}', 'epoch': epoch, 'task': task,
                        'state': 'complete' if result(p, epoch, task) else 'pending', 'failures': 0, 'attempt': 0, 'not_before': 0})
            occupied = {item['job']['gpu'] for item in active.values()}
            for job in jobs:
                if job['state'] not in ['pending', 'waiting_checkpoint'] or len(active) >= p['max_parallel'] or time.time() < job['not_before']:
                    continue
                if target(job['epoch'], job['task']) is None:
                    job['state'] = 'waiting_checkpoint'
                    continue
                candidates = [(f, g) for g, f in free.items() if g not in occupied and f >= p['inference_budget_mib']+p['reserve_mib']]
                if not candidates:
                    continue
                _, gpu = max(candidates)
                job.update(state='running', attempt=job['attempt']+1, gpu=gpu, started=time.time())
                path = HERE / 'logs' / f'{job["key"]}_attempt{job["attempt"]}.log'
                path.parent.mkdir(exist_ok=True)
                log = path.open('a')
                job['log'] = str(path)
                proc = subprocess.Popen([str(PYTHON), '-B', str(HERE / 'run.py'), '--worker', str(job['epoch']), job['task']],
                    cwd=BASE, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job['pid'] = proc.pid
                active[job['key']] = {'proc': proc, 'job': job, 'log': log}
                occupied.add(gpu)
            report(p, jobs, chosen)
            if chosen and all(job['state'] == 'complete' for job in jobs):
                finalize(p, chosen)
                report(p, jobs, chosen, 'complete')
                return
            if all(job['state'] in ['complete', 'error'] for job in jobs) and any(job['state'] == 'error' for job in jobs):
                report(p, jobs, chosen, 'partial_failure')
                return
            time.sleep(5)
    finally:
        for item in active.values():
            stop(item['proc'])
            item['log'].close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        atomic_json(HERE / 'supervisor_error.json', {'time': time.time(), 'traceback': traceback.format_exc()})
        raise
