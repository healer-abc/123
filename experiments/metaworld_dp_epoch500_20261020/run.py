"""Independent fresh DP500 training/evaluation supervisor; preserve the paused DP1000 run."""
import csv
import fcntl
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
sys.path.insert(0, str(BASE))
from runtime import TASKS, PYTHON, atomic_json, environment, sha256


def plan():
    path = HERE / 'plan.json'
    if path.exists():
        return json.loads(path.read_text())
    protocol = json.loads((BASE / 'protocol.json').read_text())
    source = 'sources/dp/diffusion_policy-main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py'
    value = {'created': time.time(), 'method': 'DP', 'tasks': TASKS, 'training_seed': 0,
        'completed_epoch_target': 500, 'saved_completed_epochs': [300, 400, 500],
        'lr_schedule_epochs': 1000, 'batch_size': 128, 'horizon': 16, 'n_obs_steps': 2,
        'action_horizon': 8, 'num_inference_steps': 10, 'episodes_per_task': 20,
        'seeds': protocol['final_episode_seeds'], 'max_episode_steps': 200,
        'demonstrations': 50, 'training_demonstrations': 40, 'validation_demonstrations': 10,
        'initialization': 'from scratch, seed0, resume=False',
        'classification': 'Separate post-hoc training-duration sensitivity experiment. Original DP1000 processes ended by user; existing disk checkpoints and results retained; no score selection.',
        'resume_context': 'Save LR scheduler, EMA counters, RNG, train sampling batch and runner state alongside each300/400/500 checkpoint.',
        'continue_training_only_on_future_user_instruction': True,
        'baseline_worker_sha256': sha256(BASE / 'worker.py'),
        'baseline_protocol_sha256': sha256(BASE / 'protocol.json'),
        'official_training_relative_path': source, 'official_training_sha256': sha256(BASE / source),
        'execution_files': {n: sha256(HERE / n) for n in ['run.py', 'replay500.py', 'rollout500.py']},
        'max_parallel': 2, 'training_budget_mib': 9000, 'reserve_mib': 3500,
        'stop_rule': 'Train all five tasks500 epochs, save300/400/500, evaluate all20 episodes per task regardless of outcome'}
    atomic_json(path, value)
    (HERE / 'parameters.diff').write_text(
        '--- original_DP1000\n+++ independent_DP500\n@@ Training budget @@\n'
        '-completed_epochs: 1000\n+completed_epochs: 500\n'
        '+independent_completed_epoch_snapshots: [300, 400, 500]\n'
        ' lr_schedule_epochs: 1000\n training_seed: 0\n num_inference_steps: 10\n'
        ' horizon: 16\n n_obs_steps: 2\n action_horizon: 8\n batch_size: 128\n')
    return value


def output(task):
    attempts = sorted((HERE / 'runs' / task).glob('attempt_*'))
    return attempts[-1] if attempts else HERE / 'runs' / task / 'attempt_01'


def result(task):
    path = output(task) / 'complete.json'
    return json.loads(path.read_text()) if path.exists() else None


def report(jobs, state='running'):
    snapshots, rows = [], []
    for job in jobs:
        out = output(job['task'])
        hb = out / 'heartbeat.json'
        heartbeat = json.loads(hb.read_text()) if hb.exists() else {}
        summary = result(job['task'])
        for path in sorted((out / 'snapshot_receipts').glob('epoch_*.json')):
            snapshots.append(dict(task=job['task'], **json.loads(path.read_text())))
        rows.append({'task': job['task'], 'state': job['state'], 'epoch': heartbeat.get('epoch'),
            'successes': summary['successes'] if summary else None,
            'episodes': summary['episodes'] if summary else 0})
    atomic_json(HERE / 'checkpoint_index.json', snapshots)
    atomic_json(HERE / 'status.json', {'time': time.time(), 'pid': os.getpid(), 'state': state, 'jobs': jobs, 'rows': rows})
    lines = ['# DP独立500轮实验', '', '从头训练seed0；50示教（40训练/10验证）；保留原1000轮学习率计划。',
        '完成300、400、500轮分别保存模型、EMA与优化器快照；500轮EMA以10步去噪评测每任务20局。',
        '原DP1000进程及看门狗已按用户指令结束，已有磁盘权重另存于preserved_baseline；SimpleDP3实验继续运行。',
        '各快照另存学习率调度、EMA计数、随机数等续训状态；500轮后停止训练，等待用户指令才续训。', '',
        f'状态：{state}', '', '| 任务 | 阶段 | 当前epoch（约） | 正式评测 |', '|---|---|---:|---|']
    for row in rows:
        evaluation = f"{row['successes']}/20" if row['episodes'] else '待评测'
        lines.append(f"| {row['task']} | {row['state']} | {row['epoch'] if row['epoch'] is not None else '—'} | {evaluation} |")
    lines += ['', '## 已保存的独立快照', '']
    lines += [f"- {s['task']}，完成{s['completed_epochs']}轮：[权重]({s['checkpoint']})" for s in snapshots]
    (HERE / 'progress.md').write_text('\n'.join(lines) + '\n')


def audit(p):
    import numpy as np
    rows, successes = [], []
    for task in TASKS:
        out = output(task)
        s = result(task)
        assert s['episodes'] == 20 and s['seeds'] == p['seeds']
        assert s['identity']['method'] == 'DP'
        assert s['identity']['ablation_plan_sha256'] == sha256(HERE / 'plan.json')
        assert s['identity']['checkpoint_completed_epochs'] == 500
        trained = json.loads((out / 'trained.json').read_text())
        assert trained['completed_epochs'] == 500 and trained['global_step'] == 30500
        assert sha256(trained['checkpoint']) == trained['sha256'] == s['identity']['checkpoint_sha256']
        assert json.loads((out / 'training_audit.json').read_text())['passed']
        hits = 0
        for i, seed in enumerate(p['seeds']):
            path = out / 'final_rollouts' / f'episode_{i:02d}.json'
            e = json.loads(path.read_text())
            assert e['identity'] == s['identity'] and e['seed'] == seed
            assert len(e['infos']) == e['steps'] == 200 and e['nonfinite_actions'] == 0
            assert bool(e['success']) == any(bool(x['success']) for x in e['infos'])
            assert e['num_inference_steps'] == 10 and e['denoising_step_calls'] == 250
            assert Path(e['video']).stat().st_size > 0
            with np.load(e['trajectory']) as data:
                assert np.array_equal(data['query_steps'], np.arange(0, 200, 8))
                assert data['predicted_chunks'].shape == (25, 8, 4)
                assert np.allclose(data['predicted_chunks'].reshape(200, 4), data['actions'], atol=1e-6, rtol=1e-6)
                assert np.isfinite(data['actions']).all()
            hits += bool(e['success'])
            if e['success']:
                successes.append({'task': task, 'seed': seed, 'record': str(path), 'video': e['video'], 'trajectory': e['trajectory']})
        assert hits == s['successes']
        rows.append({'task': task, 'successes': hits, 'episodes': 20, 'success_rate': hits/20})
    atomic_json(HERE / 'summary.json', rows)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (HERE / 'summary.csv').write_text(stream.getvalue())
    lines = ['# DP500轮，DDIM10，20局/任务', '', '| 任务 | 成功次数 | 成功率 |', '|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['task']} | {row['successes']}/20 | {row['success_rate']:.0%} |")
    lines += ['', f"五项平均：{sum(r['successes'] for r in rows)}%。", '', '原DP1000结果保留，本表为独立训练时长敏感性实验。']
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n')
    atomic_json(HERE / 'successful_episodes.json', successes)
    atomic_json(HERE / 'audit.json', {'passed': True, 'episodes': 100, 'snapshot_epochs_verified': [300, 400, 500],
        'native_success_recomputed': True, 'actions_reconstructed': True, 'denoising_calls_per_episode': 250})


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    lock = (HERE / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = plan()
    for rel, digest in p['execution_files'].items():
        assert sha256(HERE / rel) == digest
    def handler(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    atomic_json(HERE / 'supervisor_pid.json', {'pid': os.getpid(), 'started': time.time()})
    jobs = [{'task': t, 'state': 'complete' if result(t) else 'pending', 'failures': 0, 'not_before': 0,
        'attempt': max([int(q.name.split('_')[-1]) for q in (HERE / 'runs' / t).glob('attempt_*')] + [0])} for t in TASKS]
    active = {}
    try:
        while True:
            free = {int(g): int(f) for g, f in (line.split(',') for line in subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines())}
            for task, item in list(active.items()):
                proc, job = item['proc'], item['job']
                activity = [job['started'], Path(job['log']).stat().st_mtime]
                activity += [x.stat().st_mtime for x in output(task).rglob('heartbeat.json')]
                reason = None
                if proc.poll() is None and (free[job['gpu']] < 3500 or time.time()-max(activity) > 1200):
                    reason = 'low_free_memory' if free[job['gpu']] < 3500 else 'no_progress_1200s'
                    stop(proc)
                if proc.poll() is not None:
                    item['log'].close()
                    if proc.returncode == 0 and result(task):
                        job['state'] = 'complete'
                    else:
                        job['failures'] += reason != 'low_free_memory'
                        job['state'] = 'pending' if job['failures'] < 3 else 'error'
                        job['not_before'] = time.time()+60
                        atomic_json(HERE / f'alert_{task}_attempt_{job["attempt"]}.json',
                            {'time': time.time(), 'job': job, 'reason': reason or 'worker_error', 'exit_code': proc.returncode})
                    del active[task]
            occupied = {item['job']['gpu'] for item in active.values()}
            for job in jobs:
                if job['state'] != 'pending' or len(active) >= p['max_parallel'] or time.time() < job['not_before']:
                    continue
                candidates = [(f, g) for g, f in free.items() if g not in occupied and f >= p['training_budget_mib']+p['reserve_mib']]
                if not candidates:
                    continue
                _, gpu = max(candidates)
                job.update(state='running', attempt=job['attempt']+1, gpu=gpu, started=time.time())
                path = HERE / 'logs' / f'{job["task"]}_attempt_{job["attempt"]:02d}.log'
                path.parent.mkdir(exist_ok=True)
                log = path.open('a')
                job['log'] = str(path)
                proc = subprocess.Popen([str(PYTHON), '-B', str(HERE / 'replay500.py'), job['task'], str(job['attempt'])],
                    cwd=BASE, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job['pid'] = proc.pid
                active[job['task']] = {'proc': proc, 'job': job, 'log': log}
                occupied.add(gpu)
            report(jobs)
            if all(job['state'] == 'complete' for job in jobs):
                audit(p)
                report(jobs, 'complete')
                atomic_json(HERE / 'complete.json', {'state': 'complete', 'episodes': 100, 'audited': True})
                return
            if all(job['state'] in ['complete', 'error'] for job in jobs):
                report(jobs, 'partial_failure')
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
