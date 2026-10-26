"""Persistent GPU admission, process watchdog, bounded retries and final reporting."""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from runtime import ROOT, PYTHON, TASKS, METHODS, environment, atomic_json, sha256

RESERVE = 4096
BUDGET = {'collect': 2500, 'SimpleDP3': 4096, 'ACT': 4096, 'DP': 10000}
MAX_RUNNING = 4
MAX_PER_GPU = 2


class ExistingProcess:
    """Adopt this campaign's verified preflight worker when starting its watchdog."""
    def __init__(self, pid, out):
        self.pid, self.out, self.returncode = pid, out, None

    def poll(self):
        try:
            cmd = Path(f'/proc/{self.pid}/cmdline').read_bytes().split(b'\0')
            alive = str(ROOT / 'worker.py').encode() in cmd and str(self.out).encode() in cmd
            stat = Path(f'/proc/{self.pid}/stat').read_text().split(') ')[1].split()[0]
            alive = alive and stat != 'Z'
        except FileNotFoundError:
            alive = False
        if not alive:
            self.returncode = 0 if (self.out / 'complete.json').exists() else 1
        return self.returncode

    def wait(self, timeout=10):
        deadline = time.time() + timeout
        while self.poll() is None:
            if time.time() >= deadline:
                raise subprocess.TimeoutExpired('adopted worker', timeout)
            time.sleep(.2)
        return self.returncode


def gpu_memory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    return {int(i): int(m) for i, m in (r.split(',') for r in rows.splitlines())}


def process_memory():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    result = {}
    for row in rows.splitlines():
        pid, mem = row.split(',')
        if pid.strip().isdigit() and mem.strip().isdigit():
            result[int(pid)] = result.get(int(pid), 0)+int(mem)
    return result


def ready_jobs(jobs, active):
    by_key = {j['key']: j for j in jobs}
    ready = [j for j in jobs if j['status'] == 'pending' and
             all(by_key[n]['status'] == 'complete' for n in j['needs'])]
    priority = {'DP': 0, 'ACT': 1, 'SimpleDP3': 2}
    def key(j):
        if j['stage'] != 'train':
            return (0 if j['stage'] == 'collect' else 1, 0, 0, TASKS.index(j['task']))
        return (2, sum(a['method'] == j['method'] for a in active if a['stage'] == 'train'),
                priority[j['method']], TASKS.index(j['task']))
    return sorted(ready, key=key)


def eligible_gpus(free, reserved, active, budget):
    return [g for g in free if free[g]-reserved[g] >= budget+RESERVE
            and sum(a['gpu'] == g for a in active) < MAX_PER_GPU]


def append(path, data):
    with open(path, 'a') as f:
        f.write(json.dumps({'time': time.time(), **data}, ensure_ascii=False) + '\n')


def alert(job, reason, retry):
    msg = {'job': job['key'], 'reason': reason, 'retry': retry,
           'log': str(job.get('log')), 'time': time.time()}
    append(ROOT / 'logs/alerts.jsonl', msg)
    atomic_json(ROOT / 'logs/latest_alert.json', msg)
    print('ALERT ' + json.dumps(msg, ensure_ascii=False), flush=True)


def stop_job(job):
    proc = job['process']
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def aggregate(jobs):
    rows = []
    initial_states = {}
    for j in jobs:
        if j['stage'] != 'train' or j['status'] != 'complete':
            continue
        out = Path(j['out'])
        r = json.loads((out / 'complete.json').read_text())
        assert r['episodes'] == 20 and r['identity']['task'] == j['task']
        assert r['identity']['method'] == j['method']
        trained = json.loads((out / 'trained.json').read_text())
        assert r['identity']['dataset_episodes'] == trained['dataset_episodes'] == 50
        assert trained['training_episodes'] == 40
        assert r['identity']['split_sha256'] == trained['split_sha256']
        assert r['identity']['dataset_manifest_sha256'] == trained['dataset_manifest_sha256']
        assert sha256(trained['checkpoint']) == r['identity']['checkpoint_sha256']
        records = [json.loads((out / 'final_rollouts' / f'episode_{i:02d}.json').read_text())
                   for i in range(20)]
        assert sum(x['success'] for x in records) == r['successes']
        for i, record in enumerate(records):
            assert record['identity'] == r['identity'] and record['episode'] == i
            assert record['seed'] == 2026102000 + i and record['steps'] == 200
            assert len(record['infos']) == 200 and record['success'] == any(bool(x['success']) for x in record['infos'])
            assert Path(record['trajectory']).is_file() and Path(record['video']).stat().st_size > 0
            assert r['seeds'] == list(range(2026102000,2026102020))
            assert record['identity']['protocol'] == sha256(ROOT / 'protocol.json')
            key = (j['task'], i)
            prior = initial_states.setdefault(key, record['initial_state_sha256'])
            assert prior == record['initial_state_sha256'], f'Initial state mismatch: {key}'
        rows.append({'task': j['task'], 'method': j['method'], 'successes': r['successes'],
            'episodes': 20, 'success_rate': r['success_rate'],
            'checkpoint_sha256': r['identity']['checkpoint_sha256'], 'output': str(out)})
    report = ROOT / 'report'
    report.mkdir(exist_ok=True)
    with open(report / 'results.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'method', 'successes', 'episodes',
            'success_rate', 'checkpoint_sha256', 'output'])
        writer.writeheader()
        writer.writerows(rows)
    table = ['# MetaWorld 文献配置新基线（50示教，40训练/10验证，训练seed0）', '',
        '每格为成功次数 / 20（成功率）。未完成实验记为“待完成”；训练中的评测不计入本表。', '',
        '| 场景 | DP | ACT | SimpleDP3 |', '|---|---:|---:|---:|']
    for task in TASKS:
        values = []
        for method in METHODS:
            found = [r for r in rows if r['task'] == task and r['method'] == method]
            values.append(f"{found[0]['successes']}/20（{found[0]['success_rate']:.0%}）" if found else '待完成')
        table.append('| ' + ' | '.join([task] + values) + ' |')
    if len(rows) == 15:
        averages = [f"{sum(r['successes'] for r in rows if r['method'] == m)}/100"
                    f"（{sum(r['successes'] for r in rows if r['method'] == m):.0f}%）" for m in METHODS]
        table.append('| 五场景平均 | ' + ' | '.join(averages) + ' |')
    table += ['', '配置出处、适配与单训练seed限制见 ../PROTOCOL.md。所有组使用同一套新评测seed2026102000–2026102019。',
        '最终评测用独立固定 seeds；成功定义为官方 info.success 在 200 步内曾为真。',
        '本表不会用开发评测的最高成功率或截图中的历史结果补值。', '']
    (report / 'results.md').write_text('\n'.join(table))
    return len(rows)


def make_jobs():
    jobs=[]
    for method in METHODS:
        jobs.append({'key':'smoke_'+method,'stage':'smoke','task':TASKS[0],'method':method,'needs':[]})
    for task in TASKS:
        for method in ['DP','ACT','SimpleDP3']:
            jobs.append({'key':method+'_'+task,'stage':'train','task':task,'method':method,'needs':['smoke_'+m for m in METHODS]})
    for job in jobs:job.update(status='pending',attempt=0,failures=0)
    return jobs


def public(j):
    return {k: v for k, v in j.items() if k not in ['process', 'handle']}


def update_progress(jobs,count,free):
    lines=['# 新一轮 MetaWorld 规范基线进度','',
        '训练seed统一0；50条示教预算=40训练+10验证；每组20局，测试seed统一2026102000–2026102019。',
        'ACT chunk10/聚合开启/KL10，2000epochs；DP/SimpleDP3 DDIM10/sample，1000epochs、batch128。',
        '所有模型从头训练；配置和测试seed冻结；0%如实报告，不触发调参搜索。','',
        f'最终评测已完成：{count}/15组。GPU空闲MiB：{free}。','',
        '| 任务 | 方法 | 阶段 | epoch/总数 | 已用小时 | 训练预计剩余小时 |','|---|---|---|---:|---:|---:|']
    for j in jobs:
        if j['stage']!='train':continue
        total=2000 if j['method']=='ACT' else 1000
        epoch=None;remaining=None;elapsed=None
        if j.get('started'):elapsed=(time.time()-j['started'])/3600
        if j.get('out'):
            f=Path(j['out'])/'heartbeat.json'
            if f.exists():
                hb=json.loads(f.read_text())
                if 'epoch' in hb:epoch=int(hb['epoch'])
                elif j['method']=='ACT' and 'batches' in hb:epoch=int(hb['batches'])//7
                if epoch and elapsed and epoch>=10:remaining=elapsed*(total-epoch)/epoch
        lines.append('| '+' | '.join([j['task'],j['method'],j['status'],
            f'{epoch}/{total}' if epoch is not None else f'—/{total}',
            f'{elapsed:.2f}' if elapsed is not None else '—',
            f'{remaining:.1f}' if remaining is not None else '测量中'])+' |')
    lines+=['','训练剩余时间按当前平均吞吐估计，排队与最终评测另计。共享GPU负载变化会影响估计。','',
        '结果表见report/results.md；协议见PROTOCOL.md；异常见logs/latest_alert.json。','']
    (ROOT/'progress.md').write_text('\n'.join(lines))


def main():
    global MAX_RUNNING
    lock = open(ROOT / 'watchdog.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(ROOT / 'logs/watchdog_pid.json', {'pid': os.getpid(), 'started': time.time()})
    freeze = json.loads((ROOT / 'source_manifest.json').read_text())
    for rel, digest in freeze['execution_files'].items():
        assert sha256(ROOT / rel) == digest, 'Source changed: ' + rel
    jobs = make_jobs()
    by_key = {j['key']: j for j in jobs}
    stopping = [False]
    def handler(*_):
        stopping[0] = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    # Restart supervisor without rerunning successfully verified jobs.
    for job in jobs:
        attempts = sorted((ROOT / 'runs' / job['key']).glob('attempt_*'))
        for path in attempts:
            job['attempt'] = max(job['attempt'], int(path.name.split('_')[-1]))
            if (path / 'complete.json').exists():
                job.update(status='complete', out=str(path))
        if job['status'] == 'pending' and attempts:
            path = attempts[-1]
            heartbeat = path / 'heartbeat.json'
            if heartbeat.exists():
                receipt = json.loads(heartbeat.read_text())
                proc = ExistingProcess(receipt['pid'], path)
                if proc.poll() is None:
                    assert os.getpgid(proc.pid) == proc.pid
                    environ = Path(f'/proc/{proc.pid}/environ').read_bytes().split(b'\0')
                    gpu = int(next(e.split(b'=', 1)[1] for e in environ if e.startswith(b'CUDA_VISIBLE_DEVICES=')))
                    log = path / 'worker.log'
                    job.update(status='running', out=str(path), log=log, handle=open(log, 'a'),
                        process=proc, pid=proc.pid, gpu=gpu, started=receipt['time'],
                        budget=BUDGET['collect' if job['stage'] == 'collect' else job['method']])
                    append(ROOT / 'logs/events.jsonl', {'event': 'adopt_preflight_worker',
                        'job': job['key'], 'pid': proc.pid, 'gpu': gpu})
    fatal = False
    try:
        while not stopping[0]:
            free = gpu_memory()
            active = [j for j in jobs if j['status'] == 'running']
            yielded_gpus = set()
            for j in sorted(active, key=lambda j: -j['budget']):
                proc = j['process']
                out = Path(j['out'])
                last_activity = max([j['started'], j['log'].stat().st_mtime] +
                    [p.stat().st_mtime for p in out.rglob('heartbeat.json')])
                reason = None
                if j['gpu'] in yielded_gpus and proc.poll() is None:
                    continue
                if proc.poll() is None and free[j['gpu']] < 3500:
                    reason = 'GPU low memory; yield only this campaign process'
                    stop_job(j)
                    yielded_gpus.add(j['gpu'])
                elif proc.poll() is None and time.time() - last_activity > 1200:
                    reason = 'No heartbeat or log progress for 1200 seconds'
                    stop_job(j)
                elif proc.poll() is not None:
                    reason = f'worker exited {proc.returncode} without valid completion'
                else:
                    continue
                j['handle'].close()
                if proc.returncode == 0 and (out / 'complete.json').exists():
                    j['status'] = 'complete'
                    append(ROOT / 'logs/events.jsonl', {'event': 'complete', **public(j)})
                else:
                    tail = j['log'].read_text(errors='replace')[-12000:]
                    scientific = any(s in tail for s in ['FloatingPointError', 'Nonfinite', 'Initial state mismatch', 'Source changed'])
                    j['failures'] += 1
                    retry = j['failures'] < 3 and not scientific
                    alert(j, reason + '\n' + tail[-3000:], retry)
                    if retry:
                        j['status'] = 'pending'
                        # Quarantine incomplete collection; never overwrite a good dataset.
                        if j['stage'] == 'collect':
                            for p in [ROOT / 'datasets' / f"metaworld_{j['task']}_expert.zarr",
                                      ROOT / 'datasets/additional40' / f"metaworld_{j['task']}_expert.zarr",
                                      ROOT / 'datasets' / ('act_' + j['task']),
                                      ROOT / 'datasets' / (j['task'] + '.json')]:
                                if p.exists():
                                    label = 'additional40_' if p.parent.name == 'additional40' else ''
                                    p.rename(out / ('incomplete_' + label + p.name))
                        # OOM retries reduce parallelism rather than batch size or model.
                        if 'out of memory' in tail.lower():
                            MAX_RUNNING = max(1, MAX_RUNNING - 1)
                    else:
                        j['status'] = 'failed'
                        # Isolate exhausted infrastructure failures; unrelated healthy jobs continue.
                        fatal = fatal or scientific
            if fatal:
                break
            active = [j for j in jobs if j['status'] == 'running']
            used = process_memory()
            reserved = {g: sum(max(0, j['budget']-used.get(j['pid'], 0)) for j in active if j['gpu'] == g)
                        for g in free}
            while len(active) < MAX_RUNNING:
                selected = None
                for candidate in ready_jobs(jobs, active):
                    budget = BUDGET['collect' if candidate['stage'] == 'collect' else candidate['method']]
                    candidates = eligible_gpus(free, reserved, active, budget)
                    if candidates:
                        selected = candidate
                        break
                if selected is None:
                    break
                j = selected
                if j['failures'] >= 3:
                    j['status'] = 'failed'
                    alert(j, 'Three infrastructure attempts exhausted; inspect saved traceback and alert', False)
                    fatal = True
                    break
                gpu = max(candidates, key=lambda g: free[g] - reserved[g])
                j['attempt'] += 1
                out = ROOT / 'runs' / j['key'] / f"attempt_{j['attempt']:02d}"
                out.mkdir(parents=True, exist_ok=False)
                log = out / 'worker.log'
                handle = open(log, 'w')
                argv = [str(PYTHON), '-B', str(ROOT / 'worker.py'), '--stage', j['stage'],
                        '--method', j['method'], '--task', j['task'], '--out', str(out)]
                proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                    cwd=ROOT, env=environment(gpu), start_new_session=True)
                j.update(status='running', out=str(out), log=log, handle=handle, process=proc,
                         gpu=gpu, budget=budget, pid=proc.pid, started=time.time())
                # Path is serialized explicitly below; never serialize live process handles.
                j['log'] = log
                append(ROOT / 'logs/events.jsonl', {'event': 'launch', 'job': j['key'],
                    'pid': proc.pid, 'gpu': gpu, 'argv': argv, 'attempt': j['attempt']})
                active.append(j)
                reserved[gpu] += budget
            if fatal:
                break
            count = aggregate(jobs)
            update_progress(jobs,count,free)
            state = {'time': time.time(), 'pid': os.getpid(), 'status': 'running',
                'final_groups_complete': count, 'final_groups_total': 15,
                'gpu_free_mib': free,
                'jobs': [dict(public(j), log=str(j.get('log', ''))) for j in jobs]}
            if all(j['status'] in ['complete','failed'] for j in jobs):
                state['status'] = 'complete' if all(j['status']=='complete' for j in jobs) else 'partial_failure'
                atomic_json(ROOT / 'logs/status.json', state)
                break
            atomic_json(ROOT / 'logs/status.json', state)
            time.sleep(10)
    except BaseException as e:
        fatal = True
        atomic_json(ROOT / 'logs/supervisor_error.json', {'type': type(e).__name__, 'error': str(e), 'time': time.time()})
        raise
    finally:
        for j in jobs:
            if j['status'] == 'running':
                stop_job(j)
                j['handle'].close()
                j['status'] = 'stopped'
        aggregate(jobs)
        if fatal or stopping[0]:
            atomic_json(ROOT / 'logs/status.json', {'time': time.time(), 'pid': os.getpid(),
                'status': 'failed_awaiting_fix' if fatal else 'stopped_by_signal',
                'jobs': [dict(public(j), log=str(j.get('log', ''))) for j in jobs]})


if __name__ == '__main__':
    # Convert Path values only for append/public JSON serialization.
    _original_public = public
    def public(j):
        return {k: str(v) if isinstance(v, Path) else v for k, v in _original_public(j).items()}
    main()
