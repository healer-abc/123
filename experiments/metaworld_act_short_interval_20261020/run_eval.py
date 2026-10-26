"""Fixed ACT inference protocol on five tasks; four new evaluations, one explicit reuse."""
import argparse
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'metaworld_official_baselines_50demo_20261019'
ENGINE = HERE.parent / 'metaworld_zero_success_search_20261019/search.py'
REUSED = HERE.parent / 'metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20'
sys.path.insert(0, str(BASE))
from runtime import PYTHON, TASKS, atomic_json, bootstrap, environment, sha256


def make_plan():
    path = HERE / 'plan.json'
    if path.exists():
        return json.loads(path.read_text())
    rows = list(csv.DictReader((BASE / 'report/results.csv').open()))
    targets = []
    for task in TASKS:
        row = next(r for r in rows if r['method'] == 'ACT' and r['task'] == task)
        run = Path(row['output'])
        assert sha256(run / 'policy_best.ckpt') == row['checkpoint_sha256']
        row.update(stats_sha256=sha256(run / 'dataset_stats.pkl'), config_sha256=sha256(run / 'config.json'))
        row['reuse_directory'] = str(REUSED) if task == 'pick-place-wall' else None
        if row['reuse_directory']:
            summary = json.loads((REUSED / 'summary.json').read_text())
            assert summary['episodes'] == 20 and summary['identity']['checkpoint_sha256'] == row['checkpoint_sha256']
            assert summary['identity']['config'] == {'temporal_agg': False, 'action_horizon': 1}
            assert summary['seeds'] == list(range(4026101900, 4026101920))
            row['reused_summary_sha256'] = sha256(REUSED / 'summary.json')
        targets.append(row)
    plan = {'created_at': time.time(), 'targets': targets, 'method': 'ACT', 'config': {
        'temporal_agg': False, 'action_horizon': 1}, 'prediction_chunk_size': 100,
        'seeds': list(range(4026101900, 4026101920)), 'episodes_per_task': 20, 'max_episode_steps': 200,
        'success_definition': 'any original environment info.success within 200 steps',
        'engine_path': str(ENGINE), 'engine_sha256': sha256(ENGINE),
        'runner_sha256': sha256(__file__), 'training_changes': None,
        'classification': 'Fixed short-execution-interval evaluation. Setting selected previously on pick-place-wall; no selection on the four new tasks.',
        'historical_baseline_seeds': list(range(2026101900, 2026101920)),
        'comparison_note': 'Historical baselines use different seeds. Pick-place-wall has an additional same-seed original-setting control in the previous campaign.'}
    atomic_json(path, plan)
    (HERE / 'baseline_results_snapshot.csv').write_text((BASE / 'report/results.csv').read_text())
    (HERE / 'parameters.diff').write_text('--- original_ACT_inference.json\n+++ fixed_short_interval.json\n@@ -1,4 +1,4 @@\n {\n-  "action_horizon": 100,\n+  "action_horizon": 1,\n   "temporal_agg": false\n }\n')
    return plan


def worker(task, plan):
    row = next(r for r in plan['targets'] if r['task'] == task)
    assert row['reuse_directory'] is None
    assert sha256(ENGINE) == plan['engine_sha256']
    assert sha256(__file__) == plan['runner_sha256']
    run = Path(row['output'])
    assert sha256(run/'dataset_stats.pkl') == row['stats_sha256']
    assert sha256(run/'config.json') == row['config_sha256']
    progress = HERE / 'progress' / task
    progress.mkdir(parents=True, exist_ok=True)
    (progress / 'plan.json').write_bytes((HERE/'plan.json').read_bytes())
    spec = importlib.util.spec_from_file_location('verified_zero_search_engine', ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.HERE = progress
    bootstrap()
    import torch
    torch.set_num_threads(2)
    policy, stats, checkpoint = engine.load_model(row)
    out = HERE / 'runs' / task
    out.mkdir(parents=True, exist_ok=True)
    result = engine.rollout(policy, stats, row, plan['config'], plan['seeds'], out / 'rollouts20', 'fixed_ACT_short_interval20')
    assert sha256(checkpoint) == row['checkpoint_sha256']
    atomic_json(out/'complete.json', result)


def summaries(plan, jobs):
    lines = ['# ACT 五任务统一短执行间隔评测', '',
        '50条示教预算（40训练/10验证），原始checkpoint。chunk_size=100，action_horizon=1，temporal_agg=false；每任务固定20局，最长200步。',
        '配置沿用此前pick-place-wall筛选结果；其他四任务不再筛选参数。历史原始基线使用不同seed，保留供参考，不能把差值全部归因于执行间隔。', '',
        '| 任务 | 原始100步基线 | 本轮1步执行 | 状态/来源 |', '|---|---:|---:|---|']
    rows = []
    for row in plan['targets']:
        task = row['task']
        folder = Path(row['reuse_directory']) if row['reuse_directory'] else HERE/'runs'/task/'rollouts20'
        p = folder/'summary.json'
        value = json.loads(p.read_text()) if p.exists() else None
        job = next((j for j in jobs if j['task']==task), None)
        partial = [json.loads(p.read_text()) for p in sorted(folder.glob('episode_*.json'))] if not value else []
        finished = value['episodes'] if value else len(partial)
        hits = value['successes'] if value else sum(r['success'] for r in partial)
        status = '复用此前已完成20局' if row['reuse_directory'] else '完成' if value else (job['state'] if job else 'pending')
        score = f'{hits}/20 ({100*hits/20:.0f}%)' if finished == 20 else f'已完成{finished}/20局，成功{hits}局（未定稿）'
        old = f"{row['successes']}/20 ({100*float(row['success_rate']):.0f}%)"
        lines.append(f'| {task} | {old} | {score} | {status} |')
        rows.append({'task':task, 'historical_successes':int(row['successes']), 'historical_episodes':20,
            'short_interval_successes':hits,'completed_episodes':finished,
            'short_interval_success_rate': hits/20 if finished==20 else '', 'status':status,
            'rollout_directory':str(folder), 'checkpoint_sha256':row['checkpoint_sha256']})
    if all(r['completed_episodes']==20 for r in rows):
        total = sum(r['short_interval_successes'] for r in rows)
        lines += [f'| **五任务平均** | **41%（历史seed）** | **{total}%（{total}/100）** | 每任务20局 |']
    lines += ['', '固定评测seeds：4026101900–4026101919。pick-place-wall复用原路径，没有复制为新采样，也没有再次计入新增rollout数量。',
        'pick-place-wall此前同seed原始100步对照为5/20，1步执行为8/20。',
        '所有新增局保存MP4、NPZ动作/模型输出及JSON官方逐步info。零成功或退化结果同样保留。', '']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    (HERE/'summary.csv').write_text(stream.getvalue())
    (HERE/'summary.md').write_text('\n'.join(lines))
    return rows


def audit(plan):
    import numpy as np
    audited, success_paths = [], []
    for row in plan['targets']:
        run = Path(row['output'])
        assert sha256(run/'policy_best.ckpt') == row['checkpoint_sha256']
        assert sha256(run/'dataset_stats.pkl') == row['stats_sha256']
        with (run/'dataset_stats.pkl').open('rb') as f:
            stats = pickle.load(f)
        folder = Path(row['reuse_directory']) if row['reuse_directory'] else HERE/'runs'/row['task']/'rollouts20'
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['episodes']==20 and summary['seeds']==plan['seeds']
        assert summary['identity']['checkpoint_sha256']==row['checkpoint_sha256']
        assert summary['identity']['config']==plan['config'] and not summary['identity']['expert_positive_control']
        if row['reuse_directory']:
            assert sha256(folder/'summary.json')==row['reused_summary_sha256']
        records=[]
        for i,seed in enumerate(plan['seeds']):
            p=folder/f'episode_{i:02d}.json'
            record=json.loads(p.read_text())
            assert record['seed']==seed and record['identity']==summary['identity']
            assert record['steps']==len(record['infos'])==200
            assert record['success']==any(bool(x['success']) for x in record['infos'])
            assert record['first_success_step']==next((j+1 for j,x in enumerate(record['infos']) if x['success']),None)
            with np.load(p.with_suffix('.npz')) as data:
                assert data['actions'].shape==(200,4)
                assert data['predicted_chunks'].shape==(200,100,4)
                assert np.array_equal(data['query_steps'],np.arange(200))
                assert np.isfinite(data['actions']).all() and np.isfinite(data['predicted_chunks']).all()
                expected=data['predicted_chunks'][:,0]*stats['action_std']+stats['action_mean']
                assert np.allclose(expected,data['actions'],rtol=1e-6,atol=1e-6)
                assert abs(float(data['rewards'].sum())-record['return'])<1e-8
            assert p.with_suffix('.mp4').stat().st_size>1000
            records.append(record)
            if record['success']:
                success_paths.append({'task':row['task'],'episode':i,'seed':seed,
                    'first_success_step':record['first_success_step'],'video':str(p.with_suffix('.mp4')),
                    'trajectory':str(p.with_suffix('.npz')),'record':str(p)})
        assert len({r['initial_state_sha256'] for r in records})==20
        assert sum(r['success'] for r in records)==summary['successes']
        audited.append({'task':row['task'],'episodes':20,'successes':summary['successes'],
            'reused':bool(row['reuse_directory']),'checkpoint_unchanged':True})
    atomic_json(HERE/'successful_episodes.json', success_paths)
    atomic_json(HERE/'artifact_audit.json',{'passed':True,'episodes_audited':100,'new_episodes':80,
        'reused_episodes':20,'scores_recomputed_from_official_info':True,
        'actions_reconstructed_from_original_model_predictions':True,'tasks':audited})
    lines=['# 成功episode路径','','| 任务 | Episode | Seed | 首次成功步 | 视频 | 原始轨迹 |','|---|---:|---:|---:|---|---|']
    for r in success_paths:
        lines.append(f"| {r['task']} | {r['episode']:02d} | {r['seed']} | {r['first_success_step']} | [视频]({r['video']}) | [NPZ]({r['trajectory']}) |")
    (HERE/'successful_episodes.md').write_text('\n'.join(lines)+'\n')


def supervise(plan):
    atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time()})
    jobs=[{'task':r['task'],'state':'complete' if (HERE/'runs'/r['task']/'complete.json').exists() else 'pending','attempt':0}
          for r in plan['targets'] if not r['reuse_directory']]
    active={}
    while any(j['state']!='complete' for j in jobs):
        for task,item in list(active.items()):
            proc,job,log=item['process'],item['job'],item['log']
            hb=HERE/'progress'/task/'heartbeat.json'
            latest=max(job['started'],hb.stat().st_mtime if hb.exists() else job['started'],Path(job['log']).stat().st_mtime)
            if proc.poll() is None and time.time()-latest>600:
                atomic_json(HERE/f'alert_{task}_{job["attempt"]}.json',{'time':time.time(),'reason':'600s without progress','job':job})
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill()
            if proc.poll() is not None:
                log.close()
                if proc.returncode==0 and (HERE/'runs'/task/'complete.json').exists():
                    job['state']='complete'
                else:
                    atomic_json(HERE/f'error_{task}_{job["attempt"]}.json',{'time':time.time(),'job':job,'exit_code':proc.returncode})
                    job['state']='pending' if job['attempt']<3 else 'error'
                    if job['state']=='error':
                        atomic_json(HERE/'status.json',{'time':time.time(),'state':'error','jobs':jobs})
                        raise RuntimeError(f'Repeated failure: {task}; inspect {job["log"]}')
                del active[task]
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(g):int(f) for g,f in (line.split(',') for line in out.splitlines())}
        occupied={item['job']['gpu'] for item in active.values()}
        for job in jobs:
            if job['state']!='pending' or len(active)>=2:continue
            choices=[(f,g) for g,f in free.items() if g not in occupied and f>=6144]
            if not choices:continue
            _,gpu=max(choices)
            job.update(attempt=job['attempt']+1,state='running',gpu=gpu,started=time.time())
            # Preserve uncommitted partial artifacts from an interrupted attempt.
            folder=HERE/'runs'/job['task']/'rollouts20'
            for p in folder.glob('episode_*.*') if folder.exists() else []:
                if p.suffix in ['.npz','.mp4'] and not p.with_suffix('.json').exists():
                    archive=HERE/'runs'/job['task']/f'orphaned_before_attempt_{job["attempt"]}'
                    archive.mkdir(parents=True,exist_ok=True)
                    shutil.move(str(p),str(archive/p.name))
            log_path=HERE/f'{job["task"]}_attempt_{job["attempt"]:02d}.log'
            log=log_path.open('a')
            proc=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--worker',job['task']],env=environment(gpu),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job.update(pid=proc.pid,log=str(log_path))
            active[job['task']]={'process':proc,'job':job,'log':log}
            occupied.add(gpu)
        rows=summaries(plan,jobs)
        atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':'running',
            'completed_tasks':sum(r['completed_episodes']==20 for r in rows),'total_tasks':5,
            'new_episodes_completed':sum(r['completed_episodes'] for r in rows if r['task']!='pick-place-wall'),
            'gpu_free_mib':free,'jobs':jobs})
        if all(j['state']=='complete' for j in jobs):break
        time.sleep(5)
    atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':'auditing','jobs':jobs})
    audit(plan)
    rows=summaries(plan,jobs)
    atomic_json(HERE/'complete.json',{'time':time.time(),'new_episodes':80,'reused_episodes':20,
        'total_episodes':100,'results':rows,'audit_passed':True})
    atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':'complete',
        'completed_tasks':5,'total_tasks':5,'new_episodes_completed':80,'jobs':jobs})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=TASKS)
    args=parser.parse_args()
    plan=make_plan()
    try:
        worker(args.worker,plan) if args.worker else supervise(plan)
    except Exception:
        traceback.print_exc()
        raise
