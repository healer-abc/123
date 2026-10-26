"""ACT joint horizon search on one frozen 20-seed suite for all five tasks."""
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'metaworld_official_baselines_50demo_20261019'
ENGINE=HERE.parent/'metaworld_zero_success_search_20261019/search_followup.py'
SHORT=HERE.parent/'metaworld_act_short_interval_20261020'
WALL=HERE.parent/'metaworld_zero_success_search_20261019/ACT_pick-place-wall/paired_original20'
sys.path.insert(0,str(BASE))
from runtime import PYTHON,TASKS,atomic_json,bootstrap,environment,sha256


def plan():
    p=HERE/'plan.json'
    if p.exists():return json.loads(p.read_text())
    baseline=list(csv.DictReader((BASE/'report/results.csv').open()))
    rows=[next(r for r in baseline if r['method']=='ACT' and r['task']==t) for t in TASKS]
    for row in rows:
        run=Path(row['output'])
        assert sha256(run/'policy_best.ckpt')==row['checkpoint_sha256']
        row.update(stats_sha256=sha256(run/'dataset_stats.pkl'),config_sha256=sha256(run/'config.json'))
    short=json.loads((SHORT/'plan.json').read_text())
    refs=[]
    for row in rows:
        src=next(r for r in short['targets'] if r['task']==row['task'])
        folder=Path(src['reuse_directory']) if src['reuse_directory'] else SHORT/'runs'/row['task']/'rollouts20'
        refs.append({'task':row['task'],'horizon':1,'directory':str(folder),'summary_sha256':sha256(folder/'summary.json')})
    refs.append({'task':'pick-place-wall','horizon':100,'directory':str(WALL),'summary_sha256':sha256(WALL/'summary.json')})
    value={'created_at':time.time(),'method':'ACT','targets':rows,
        'seeds':list(range(4026101900,4026101920)),'training_seed':0,
        'candidates':[100,32,16,8,4,2,1],'temporal_agg':False,'chunk_size':100,
        'hard_tasks':['shelf-place','pick-place','pick-place-wall'],
        'easy_tasks':['door-open','stick-push'],'reference_results':refs,
        'engine_sha256':sha256(ENGINE),'runner_sha256':sha256(__file__),
        'stop_rule':'First ONE horizon with 20 completed episodes and >0 successes on EVERY task. Same ordered seed list on all tasks and candidates; no task-specific horizons or seed changes.',
        'classification':'Common-horizon parameter-selection validation on one fixed common seed suite; not an independent unseen test estimate.',
        'training_changes':None,'inference_budget_mib':2048,'reserve_mib':4096}
    atomic_json(p,value)
    return value


def reference(p,h,task):
    return next((r for r in p['reference_results'] if r['horizon']==h and r['task']==task),None)


def folder_for(p,h,task):
    ref=reference(p,h,task)
    return Path(ref['directory']) if ref else HERE/'runs'/f'horizon_{h}'/task/'rollouts20'


def read_result(p,h,task):
    folder=folder_for(p,h,task);f=folder/'summary.json'
    if not f.exists():return None
    s=json.loads(f.read_text());row=next(r for r in p['targets'] if r['task']==task)
    assert s['episodes']==20 and s['seeds']==p['seeds']
    assert s['identity']['config']=={'temporal_agg':False,'action_horizon':h}
    assert s['identity']['checkpoint_sha256']==row['checkpoint_sha256'] and not s['identity']['expert_positive_control']
    ref=reference(p,h,task)
    if ref:assert sha256(f)==ref['summary_sha256']
    return s


def worker(p,h,task):
    assert sha256(ENGINE)==p['engine_sha256'] and sha256(__file__)==p['runner_sha256']
    row=next(r for r in p['targets'] if r['task']==task);run=Path(row['output'])
    assert sha256(run/'dataset_stats.pkl')==row['stats_sha256']
    assert sha256(run/'config.json')==row['config_sha256']
    progress=HERE/'progress'/f'horizon_{h}'/task;progress.mkdir(parents=True,exist_ok=True)
    (progress/'plan.json').write_bytes((HERE/'plan.json').read_bytes())
    spec=importlib.util.spec_from_file_location('joint_ACT_engine',ENGINE)
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);engine.HERE=progress
    bootstrap()
    import torch
    torch.set_num_threads(2)
    policy,stats,checkpoint=engine.load_model(row)
    engine.rollout(policy,stats,row,{'temporal_agg':False,'action_horizon':h},p['seeds'],
        folder_for(p,h,task),'joint_common_seed_horizon_validation20')
    assert sha256(checkpoint)==row['checkpoint_sha256']


def progress_report(p,h,phase,history,jobs):
    lines=['# ACT：同一seed集合、同一执行间隔的五任务验证','',
        '**训练seed=0；所有任务、所有候选统一rollout seeds 4026101900–4026101919；每任务20局、最长200步。**',
        '同一个候选间隔必须让五项均非零才接受。任何一项0/20就淘汰该共同间隔；不拼接不同间隔或不同seed批次。',
        '模型和训练配置保持原样。该固定seed集合参与参数选择，结果标注为搜索后验证，原始基线和所有失败记录保留。','',
        f'当前阶段：{phase}；候选执行间隔：{h}。','',
        '| 任务 | 共同间隔 | 已完成局数 | 成功数 | 来源 |','|---|---:|---:|---:|---|']
    for task in TASKS:
        s=read_result(p,h,task);folder=folder_for(p,h,task)
        records=[] if s else [json.loads(q.read_text()) for q in sorted(folder.glob('episode_*.json'))]
        n=s['episodes'] if s else len(records);hits=s['successes'] if s else sum(r['success'] for r in records)
        lines.append(f"| {task} | {h} | {n}/20 | {hits} | {'完全匹配的既有记录' if reference(p,h,task) else '本次采样'} |")
    lines+=['','## 候选历史','']
    for item in history:lines.append(f"- 间隔{item['horizon']}：{item['state']}，{item['successes']}。")
    (HERE/'progress.md').write_text('\n'.join(lines)+'\n')
    atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':phase,
        'current_horizon':h,'common_seeds':p['seeds'],'history':history,'jobs':jobs})


def run_group(p,h,tasks,history):
    jobs=[{'task':t,'state':'complete' if read_result(p,h,t) else 'pending','attempt':0} for t in tasks]
    active={};failed=False
    while True:
        for task,item in list(active.items()):
            proc,job=item['process'],item['job'];hb=HERE/'progress'/f'horizon_{h}'/task/'heartbeat.json'
            last=max(job['started'],hb.stat().st_mtime if hb.exists() else job['started'],Path(job['log']).stat().st_mtime)
            if proc.poll() is None and time.time()-last>600:
                atomic_json(HERE/f'alert_h{h}_{task}_{job["attempt"]}.json',{'time':time.time(),'job':job,'reason':'600s without progress'})
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill()
            if proc.poll() is not None:
                item['log'].close()
                if proc.returncode==0 and read_result(p,h,task):job['state']='complete'
                else:
                    atomic_json(HERE/f'error_h{h}_{task}_{job["attempt"]}.json',{'time':time.time(),'job':job,'exit_code':proc.returncode})
                    job['state']='pending' if job['attempt']<3 else 'error'
                    if job['state']=='error':raise RuntimeError(f'Repeated infrastructure failure: h={h}, task={task}')
                del active[task]
        for job in jobs:
            s=read_result(p,h,job['task'])
            if s and s['successes']==0:failed=True
        if not failed:
            out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(g):int(f) for g,f in (line.split(',') for line in out.splitlines())}
            occupied={i['job']['gpu'] for i in active.values()}
            for job in jobs:
                if job['state']!='pending' or len(active)>=2:continue
                options=[(f,g) for g,f in free.items() if g not in occupied and f>=6144]
                if not options:continue
                _,gpu=max(options);job.update(state='running',attempt=job['attempt']+1,gpu=gpu,started=time.time())
                log_path=HERE/f'h{h}_{job["task"]}_attempt_{job["attempt"]:02d}.log';log=log_path.open('a')
                proc=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--worker',str(h),job['task']],
                    env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(pid=proc.pid,log=str(log_path));active[job['task']]={'process':proc,'job':job,'log':log};occupied.add(gpu)
        progress_report(p,h,'running',history,jobs)
        if not active and (failed or all(j['state']=='complete' for j in jobs)):return not failed
        time.sleep(5)


def audit_candidate(p,h):
    import numpy as np
    manifest=[];successful=[];rows=[]
    for task in TASKS:
        row=next(r for r in p['targets'] if r['task']==task)
        s=read_result(p,h,task);assert s and s['successes']>0
        run=Path(row['output'])
        assert sha256(run/'policy_best.ckpt')==row['checkpoint_sha256']
        assert sha256(run/'dataset_stats.pkl')==row['stats_sha256']
        assert sha256(run/'config.json')==row['config_sha256']
        with (run/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
        folder=folder_for(p,h,task);hits=0
        for i,seed in enumerate(p['seeds']):
            path=folder/f'episode_{i:02d}.json';r=json.loads(path.read_text())
            assert r['identity']==s['identity'] and r['seed']==seed and r['episode']==i
            assert r['steps']==200 and len(r['infos'])==200
            truth=[j+1 for j,x in enumerate(r['infos']) if bool(x['success'])]
            assert bool(truth)==r['success'] and r['first_success_step']==(truth[0] if truth else None)
            assert Path(r['video']).stat().st_size>0
            with np.load(r['trajectory']) as data:
                q=data['query_steps'];chunks=data['predicted_chunks'];actions=data['actions']
                assert np.array_equal(q,np.arange(0,200,h))
                assert chunks.shape==(len(q),100,4) and actions.shape==(200,4)
                assert np.isfinite(chunks).all() and np.isfinite(actions).all()
                qi=np.searchsorted(q,np.arange(200),side='right')-1
                expected=chunks[qi,np.arange(200)-q[qi]]*stats['action_std']+stats['action_mean']
                assert np.allclose(actions,expected,atol=1e-6,rtol=1e-6)
                assert np.isclose(data['rewards'].sum(),r['return'])
            reference_record=json.loads((folder_for(p,1,task)/path.name).read_text())
            assert r['initial_state_sha256']==reference_record['initial_state_sha256'], (task,seed,'initial state differs')
            hits+=r['success']
            item={'task':task,'seed':seed,'record':str(path),'record_sha256':sha256(path),
                'video':r['video'],'trajectory':r['trajectory'],'success':r['success']}
            manifest.append(item)
            if r['success']:successful.append(item)
        assert hits==s['successes']
        rows.append({'method':'ACT','task':task,'action_horizon':h,'chunk_size':100,'temporal_agg':False,
            'successes':hits,'episodes':20,'success_rate':hits/20,'seed_start':p['seeds'][0],
            'seed_end':p['seeds'][-1],'source_directory':str(folder),'checkpoint_sha256':row['checkpoint_sha256']})
    return rows,manifest,successful


def finalize(p,h,history):
    rows,manifest,successful=audit_candidate(p,h)
    atomic_json(HERE/'audit.json',{'passed':True,'episodes':100,'same_seed_list_all_tasks':True,
        'same_horizon_all_tasks':True,'initial_states_match_common_seed_reference':True,
        'native_success_recomputed':True,'model_actions_reconstructed':True,'checkpoint_unchanged':True})
    atomic_json(HERE/'locked.json',{'config':{'action_horizon':h,'temporal_agg':False},
        'seeds':p['seeds'],'training_seed':0,'chunk_size':100})
    atomic_json(HERE/'source_manifest.json',manifest)
    atomic_json(HERE/'successful_episodes.json',successful)
    atomic_json(HERE/'summary.json',{'common_seed_batch':True,'common_horizon':h,
        'all_five_tasks_nonzero':True,'seeds':p['seeds'],'classification':p['classification'],'results':rows})
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'summary.csv').write_text(stream.getvalue())
    lines=['# ACT 同seed、同执行间隔五任务结果','',
        f'所有任务训练seed=0；rollout seeds={p["seeds"][0]}–{p["seeds"][-1]}，顺序完全相同。',
        f'共同action_horizon={h}，chunk_size=100，temporal_agg=false。每任务20局，每局200步。',
        '这是固定seed集合上的共同参数搜索后验证；原始基线和失败记录保留。','',
        '| 任务 | 成功数 | 成功率 |','|---|---:|---:|']
    lines += [f'| {r["task"]} | {r["successes"]}/20 | {r["success_rate"]:.0%} |' for r in rows]
    lines += ['',f'五项平均：{sum(r["success_rate"] for r in rows)/5:.0%}。100局审计通过。','',
        '完整成功episode路径见successful_episodes.json；原始记录及哈希见source_manifest.json。','']
    (HERE/'summary.md').write_text('\n'.join(lines))
    (HERE/'parameters.diff').write_text(f'--- original_ACT_protocol\n+++ joint_seed_interval_validation\n'
        f'- action_horizon: 100\n+ action_horizon: {h}\n'
        '- rollout_seeds: 2026101900..2026101919\n+ rollout_seeds: 4026101900..4026101919 (identical on all five tasks)\n'
        '  temporal_agg: false\n  chunk_size: 100\n  training_seed: 0\n  checkpoints: unchanged\n')
    atomic_json(HERE/'complete.json',{'state':'complete','accepted_horizon':h,'all_five_tasks_nonzero':True,'history':history})
    progress_report(p,h,'complete',history,[])


def exhausted_report(p,history):
    lines=['# 有限共同间隔搜索未达成五项非零','',
        '固定同一组20个seed；所有失败均保留。不能由有限搜索保证存在五项同时非零的间隔。','',
        '| 共同间隔 | 已完成任务成功数 |','|---:|---|']
    for item in history:lines.append(f'| {item["horizon"]} | {item["successes"]} |')
    lines+=['','## 环境与模型输出检查','']
    for h in p['candidates']:
        for task in TASKS:
            s=read_result(p,h,task)
            if not s or s['successes']!=0:continue
            records=[json.loads(q.read_text()) for q in sorted(folder_for(p,h,task).glob('episode_*.json'))]
            fields={k:sum(any(bool(x.get(k,False)) for x in r['infos']) for r in records)
                for k in ['success','near_object','grasp_success']}
            nonfinite=sum(r['nonfinite_actions'] for r in records)
            clipping=sum(r['fraction_components_outside_env_bounds'] for r in records)/len(records)
            lines.append(f'- h={h}, {task}: 原生info判定{fields}；非有限动作数{nonfinite}；动作分量超环境边界比例{clipping:.3%}。完整状态、奖励、模型chunk和动作保存在NPZ与JSON。')
    lines+=['','未执行的任务标记为跳过，不计为0；一项0/20即已足够淘汰该共同间隔。',
        '上述检查不能证明模型对所有初始状态均无能力；结果只覆盖本次固定seed和有限间隔集合。','']
    (HERE/'exhaustion_report.md').write_text('\n'.join(lines))
    atomic_json(HERE/'complete.json',{'state':'finite_joint_grid_exhausted','all_five_tasks_nonzero':False,'history':history})
    progress_report(p,p['candidates'][-1],'finite_joint_grid_exhausted',history,[])


def main():
    p=plan()
    assert sha256(__file__)==p['runner_sha256'] and sha256(ENGINE)==p['engine_sha256']
    if len(sys.argv)>1:
        assert sys.argv[1]=='--worker'
        worker(p,int(sys.argv[2]),sys.argv[3]);return
    atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time()})
    history=[]
    for h in p['candidates']:
        ok=run_group(p,h,p['hard_tasks'],history)
        if ok:ok=run_group(p,h,p['easy_tasks'],history)
        scores={t:read_result(p,h,t)['successes'] for t in TASKS if read_result(p,h,t)}
        history.append({'horizon':h,'state':'accepted' if ok else 'rejected','successes':scores})
        atomic_json(HERE/'candidate_history.json',history)
        if ok:finalize(p,h,history);return
    exhausted_report(p,history)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(HERE/'supervisor_error.json',{'time':time.time(),'pid':os.getpid(),'traceback':traceback.format_exc()})
        raise
