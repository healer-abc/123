"""Replayed500-epoch training-prefix ablation; DDIM10 and original1000-epoch LR schedule."""
import csv
import difflib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'metaworld_community_baselines_50demo_20261020'
sys.path.insert(0,str(BASE))
from runtime import TASKS,PYTHON,atomic_json,environment,bootstrap,sha256


def plan():
    f=HERE/'plan.json'
    if f.exists():return json.loads(f.read_text())
    protocol=json.loads((BASE/'protocol.json').read_text())
    before={'method':'SimpleDP3','horizon':16,'n_obs_steps':2,'n_action_steps':8,
        'scheduler':'DDIM','prediction_type':'sample','num_train_timesteps':100,'num_inference_steps':10,
        'completed_training_epochs':1000,'lr_schedule_epochs':1000}
    after=dict(before,completed_training_epochs=500)
    p={'created':time.time(),'baseline_root':str(BASE),'baseline_protocol_sha256':sha256(BASE/'protocol.json'),
        'baseline_evaluator_sha256':sha256(BASE/'rollout.py'),'runner_sha256':sha256(__file__),
        'evaluator_sha256':sha256(HERE/'rollout500.py'),
        'replay_sha256':sha256(HERE/'replay500.py'),'baseline_worker_sha256':sha256(BASE/'worker.py'),
        'official_training_sha256':sha256(BASE/'sources/dp3/3D-Diffusion-Policy-master/3D-Diffusion-Policy/train.py'),'tasks':TASKS,'seeds':protocol['final_episode_seeds'],
        'training_seed':0,'episodes_per_task':20,'max_episode_steps':200,'before':before,'after':after,
        'checkpoint_rule':'Replay baseline initialization/data/config through500 completed epochs; keep1000-epoch LR schedule; save independent300/400 checkpoints; evaluate500 EMA. No outcome selection.',
        'classification':'Post-hoc training-duration sensitivity on the observed common seed suite. Fresh500-epoch replay, not an originally retained500 checkpoint; original1000 baseline retained.',
        'stop_rule':'Complete all20 episodes for allfive tasks, regardless of scores; no seed/config search.',
        'saved_completed_epochs':[300,400,500],'max_parallel':2,'training_budget_mib':4096,'reserve_mib':4096}
    atomic_json(f,p)
    (HERE/'parameters.diff').write_text(''.join(difflib.unified_diff((json.dumps(before,indent=2)+'\n').splitlines(True),
        (json.dumps(after,indent=2)+'\n').splitlines(True),fromfile='epoch1000_baseline',tofile='epoch500_sensitivity')))
    (HERE/'evaluator.diff').write_text(''.join(difflib.unified_diff((BASE/'rollout.py').read_text().splitlines(True),
        (HERE/'rollout500.py').read_text().splitlines(True),fromfile='baseline/rollout.py',tofile='ablation/rollout500.py')))
    return p


def verify(p):
    assert sha256(__file__)==p['runner_sha256']
    assert sha256(HERE/'rollout500.py')==p['evaluator_sha256']
    assert sha256(BASE/'protocol.json')==p['baseline_protocol_sha256']
    assert sha256(BASE/'rollout.py')==p['baseline_evaluator_sha256']
    assert sha256(HERE/'replay500.py')==p['replay_sha256']
    assert sha256(BASE/'worker.py')==p['baseline_worker_sha256']


def bind_target(p,task):
    receipt=HERE/'targets'/(task+'.json')
    if receipt.exists():return json.loads(receipt.read_text())
    status=json.loads((BASE/'logs/status.json').read_text())
    j=next(j for j in status['jobs'] if j['key']=='SimpleDP3_'+task)
    if j['status']!='complete':return None
    out=Path(j['out']);trained=json.loads((out/'trained.json').read_text())
    summary_path=out/'final_rollouts/summary.json';s=json.loads(summary_path.read_text())
    assert trained['completed_epochs']==1000 and trained['global_step']==61000
    assert s['episodes']==20 and s['seeds']==p['seeds']
    assert s['identity']['protocol']==p['baseline_protocol_sha256']
    assert s['identity']['config']=={'horizon':16,'n_obs_steps':2,'action_horizon':8,'num_inference_steps':10,'scheduler':'DDIM','prediction_type':'sample'}
    assert sha256(trained['checkpoint'])==trained['sha256']==s['identity']['checkpoint_sha256']
    value={'task':task,'method':'SimpleDP3','output':str(out),'checkpoint':trained['checkpoint'],
        'checkpoint_sha256':trained['sha256'],'config_sha256':sha256(out/'config.yaml'),
        'trained_receipt_sha256':sha256(out/'trained.json'),'baseline_summary':str(summary_path),
        'baseline_summary_sha256':sha256(summary_path),'baseline_successes':s['successes']}
    atomic_json(receipt,value)
    return value


def folder(task):
    paths=sorted((HERE/'runs'/task).glob('attempt_*'))
    return (paths[-1] if paths else HERE/'runs'/task/'attempt_01')/'final_rollouts'


def result(p,task):
    out=folder(task).parent;f=out/'complete.json'
    if not f.exists():return None
    s=json.loads(f.read_text());t=bind_target(p,task);assert t
    trained=json.loads((out/'trained.json').read_text())
    assert trained['completed_epochs']==500 and trained['global_step']==30500
    assert s['episodes']==20 and s['seeds']==p['seeds']
    assert s['identity']['checkpoint_sha256']==trained['sha256']
    assert s['identity']['ablation_plan_sha256']==sha256(HERE/'plan.json')
    assert s['identity']['config']['num_inference_steps']==10
    assert s['identity']['checkpoint_completed_epochs']==500
    return s


def report(p,jobs,state='running'):
    rows=[]
    snapshots=[]
    for task in TASKS:
        t=bind_target(p,task);s=result(p,task)
        eps=[json.loads(f.read_text()) for f in sorted(folder(task).glob('episode_*.json'))]
        out=folder(task).parent
        heartbeat=out/'heartbeat.json'
        hb=json.loads(heartbeat.read_text()) if heartbeat.exists() else {}
        for epoch in [300,400]:
            f=out/'snapshot_receipts'/f'epoch_{epoch:04d}.json'
            if f.exists():snapshots.append(dict(task=task,**json.loads(f.read_text())))
        f=out/'trained.json'
        if f.exists():
            tr=json.loads(f.read_text())
            snapshots.append({'task':task,'completed_epochs':tr['completed_epochs'],'checkpoint':tr['checkpoint'],'sha256':tr['sha256']})
        rows.append({'task':task,'epoch1000_successes':t['baseline_successes'] if t else '',
            'epoch1000_episodes':20 if t else '', 'epoch500_successes':s['successes'] if s else sum(e['success'] for e in eps),
            'epoch500_episodes':s['episodes'] if s else len(eps),
            'training_epoch':hb.get('epoch'),
            'state':'complete' if s else next(j['state'] for j in jobs if j['task']==task)})
    lines=['# SimpleDP3 epoch500：独立敏感性实验','',
        '训练前缀1000→500轮，保留1000轮学习率计划；去噪10步、horizon16、obs2、action8、训练seed0。',
        '500轮模型为同配置重放训练所得；完成300/400轮另存独立权重，不用原800/1000轮权重冒充。',
        '五任务各20局；统一评测seeds2026102000–2026102019，每局200步。',
        '保留原1000轮基线；这是观察主表后的参数敏感性实验，不替换主表，也不声称独立未见测试。','',
        f'状态：{state}。','', '| 任务 | epoch1000基线 | epoch500当前 | 阶段 |','|---|---:|---:|---|']
    for r in rows:
        a=f"{r['epoch1000_successes']}/20" if r['epoch1000_episodes'] else '等待原训练/评测结束'
        phase=r['state']+(f"，训练epoch≈{int(r['training_epoch'])}/500" if r['state']=='running' and r['training_epoch'] is not None else '')
        lines.append(f"| {r['task']} | {a} | {r['epoch500_successes']}/{r['epoch500_episodes']} | {phase} |")
    lines+=['','## 已保存权重','']
    lines += [f"- {x['task']} 完成{x['completed_epochs']}轮：[checkpoint]({x['checkpoint']})" for x in snapshots]
    atomic_json(HERE/'checkpoint_index.json',snapshots)
    (HERE/'progress.md').write_text('\n'.join(lines)+'\n')
    atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':state,'jobs':jobs,'rows':rows})
    return rows


def audit(p):
    import numpy as np
    rows=[];successful=[];sources=[]
    for task in TASKS:
        t=bind_target(p,task);s=result(p,task);assert s
        baseline=Path(t['baseline_summary']).parent
        assert sha256(t['baseline_summary'])==t['baseline_summary_sha256']
        assert sha256(t['checkpoint'])==t['checkpoint_sha256']
        out=folder(task).parent;trained=json.loads((out/'trained.json').read_text())
        assert trained['completed_epochs']==500 and trained['global_step']==30500
        assert sha256(trained['checkpoint'])==trained['sha256']
        assert json.loads((out/'replay_audit.json').read_text())['passed']
        for epoch in [300,400]:
            receipt=json.loads((out/'snapshot_receipts'/f'epoch_{epoch:04d}.json').read_text())
            assert receipt['completed_epochs']==epoch and receipt['optimizer_updates']==61*epoch
            assert sha256(receipt['checkpoint'])==receipt['sha256']
        hits=0;paired=[]
        for i,seed in enumerate(p['seeds']):
            path=folder(task)/f'episode_{i:02d}.json';e=json.loads(path.read_text())
            b=json.loads((baseline/path.name).read_text())
            assert e['identity']==s['identity'] and e['seed']==b['seed']==seed
            assert e['initial_state_sha256']==b['initial_state_sha256']
            assert e['steps']==len(e['infos'])==200 and e['nonfinite_actions']==0
            success_steps=[j+1 for j,x in enumerate(e['infos']) if bool(x['success'])]
            assert e['success']==bool(success_steps) and e['first_success_step']==(success_steps[0] if success_steps else None)
            assert e['num_inference_steps']==10 and e['denoising_step_calls']==250
            assert Path(e['video']).stat().st_size>0
            with np.load(e['trajectory']) as d:
                assert np.array_equal(d['query_steps'],np.arange(0,200,8))
                assert d['predicted_chunks'].shape==(25,8,4)
                assert np.allclose(d['predicted_chunks'].reshape(200,4),d['actions'],rtol=1e-6,atol=1e-6)
                assert np.isfinite(d['actions']).all() and np.isclose(d['rewards'].sum(),e['return'])
            hits+=e['success'];paired.append({'seed':seed,'epoch1000_success':b['success'],'epoch500_success':e['success']})
            record={'task':task,'seed':seed,'record':str(path),'sha256':sha256(path),'video':e['video'],'trajectory':e['trajectory']}
            sources.append(record)
            if e['success']:successful.append(record)
        assert hits==s['successes']
        rows.append({'task':task,'epoch1000_successes':t['baseline_successes'],'epoch500_successes':hits,'episodes':20,
            'epoch1000_rate':t['baseline_successes']/20,'epoch500_rate':hits/20,'change_percentage_points':(hits-t['baseline_successes'])*5,
            'checkpoint_sha256':trained['sha256'],'epoch500_directory':str(folder(task))})
        atomic_json(HERE/'paired_episodes'/(task+'.json'),paired)
    atomic_json(HERE/'audit.json',{'passed':True,'episodes':100,'fresh500_epoch_prefix_replays':True,'snapshots300_and400_verified':True,'lr_schedule_epochs':1000,'same_initial_state_per_seed':True,
        'actual_denoising_calls_per_episode':250,'inference_queries_per_episode':25,'native_success_recomputed':True,
        'saved_actions_reconstructed':True,'baseline_unchanged':True})
    atomic_json(HERE/'source_manifest.json',sources);atomic_json(HERE/'successful_episodes.json',successful)
    atomic_json(HERE/'summary.json',{'classification':p['classification'],'seeds':p['seeds'],'results':rows})
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'summary.csv').write_text(stream.getvalue())
    lines=['# SimpleDP3：epoch1000与epoch500配对结果','',
        '相同20个seed、执行间隔8、观测2、去噪10步。500轮模型来自同配置重放；学习率计划仍为1000轮。原始1000轮基线保留。','',
        '| 任务 | epoch1000 | epoch500 | 差值（百分点） |','|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['epoch1000_successes']}/20（{r['epoch1000_rate']:.0%}） | {r['epoch500_successes']}/20（{r['epoch500_rate']:.0%}） | {r['change_percentage_points']:+d} |")
    a=sum(r['epoch1000_successes'] for r in rows);b=sum(r['epoch500_successes'] for r in rows)
    lines+=['',f'五项平均：epoch1000={a}%，epoch500={b}%。100局审计通过。','',
        '参数差异见parameters.diff；成功视频/轨迹路径见successful_episodes.json；逐seed配对结果见paired_episodes。','']
    (HERE/'summary.md').write_text('\n'.join(lines))


def main():
    p=plan();verify(p)
    import fcntl
    import signal
    def stop_handler(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    lock=(HERE/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time()})
    jobs=[{'task':t,'state':'complete' if result(p,t) else 'pending',
        'attempt':max([int(q.name.split('_')[-1]) for q in (HERE/'runs'/t).glob('attempt_*')]+[0]),
        'failures':0,'not_before':0} for t in TASKS]
    active={}
    try:
        while True:
            free={int(g):int(f) for g,f in (x.split(',') for x in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
            for task,item in list(active.items()):
                proc=item['proc'];j=item['job'];log=Path(j['log'])
                heartbeat=folder(task)/'heartbeat.json'
                training_hb=folder(task).parent/'heartbeat.json'
                last=max(j['started'],log.stat().st_mtime,heartbeat.stat().st_mtime if heartbeat.exists() else 0,training_hb.stat().st_mtime if training_hb.exists() else 0)
                reason=None
                if proc.poll() is None and (free[j['gpu']]<3500 or time.time()-last>600):
                    reason='low_free_memory' if free[j['gpu']]<3500 else 'no_progress_600s'
                    os.killpg(proc.pid,15)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,9);proc.wait()
                if proc.poll() is not None:
                    item['log'].close()
                    if proc.returncode==0 and result(p,task):j['state']='complete'
                    else:
                        if reason!='low_free_memory':j['failures']+=1
                        j['state']='pending' if j['failures']<3 else 'error';j['not_before']=time.time()+30
                        atomic_json(HERE/f'alert_{task}_attempt_{j["attempt"]}.json',{'time':time.time(),'job':j,'reason':reason or 'worker_error','exit_code':proc.returncode})
                    del active[task]
            occupied={x['job']['gpu'] for x in active.values()}
            for j in jobs:
                if j['state'] not in ['pending','waiting_baseline'] or len(active)>=2 or time.time()<j['not_before']:continue
                if bind_target(p,j['task']) is None:j['state']='waiting_baseline';continue
                candidates=[(f,g) for g,f in free.items() if g not in occupied and f>=8192]
                if not candidates:continue
                _,gpu=max(candidates);j.update(state='running',attempt=j['attempt']+1,gpu=gpu,started=time.time())
                path=HERE/'logs'/f'{j["task"]}_attempt_{j["attempt"]:02d}.log';path.parent.mkdir(exist_ok=True)
                log=path.open('a');j['log']=str(path)
                proc=subprocess.Popen([str(PYTHON),str(HERE/'replay500.py'),j['task'],str(j['attempt'])],
                    cwd=BASE,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                j['pid']=proc.pid;active[j['task']]={'proc':proc,'job':j,'log':log};occupied.add(gpu)
            report(p,jobs)
            if all(j['state']=='complete' for j in jobs):
                audit(p);report(p,jobs,'complete');atomic_json(HERE/'complete.json',{'state':'complete','episodes':100,'audited':True});return
            if all(j['state'] in ['complete','error'] for j in jobs):
                report(p,jobs,'partial_failure');return
            time.sleep(5)
    finally:
        for item in active.values():
            proc=item['proc']
            if proc.poll() is None:
                os.killpg(proc.pid,15)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,9);proc.wait()
            item['log'].close()


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(HERE/'supervisor_error.json',{'time':time.time(),'traceback':traceback.format_exc()})
        raise
