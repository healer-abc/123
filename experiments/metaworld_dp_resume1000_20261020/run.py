"""Supervise five full-state DP continuations and paired1000-epoch evaluations."""
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

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'metaworld_community_baselines_50demo_20261020'
SOURCE=HERE.parent/'metaworld_dp_epoch500_20261020'
sys.path.insert(0,str(BASE))
from runtime import TASKS,PYTHON,atomic_json,environment,sha256


def bind_source(task):
    f=HERE/'targets'/(task+'.json')
    if f.exists():return json.loads(f.read_text())
    out=sorted((SOURCE/'runs'/task).glob('attempt_*'))[-1]
    receipt=out/'snapshot_receipts/epoch_0500.json';r=json.loads(receipt.read_text())
    trained=json.loads((out/'trained.json').read_text());s=json.loads((out/'complete.json').read_text())
    assert r['completed_epochs']==trained['completed_epochs']==500 and r['optimizer_updates']==30500
    assert sha256(r['checkpoint'])==r['sha256']
    assert sha256(r['resume_context'])==r['resume_context_sha256']
    assert s['episodes']==20 and s['identity']['checkpoint_sha256']==trained['sha256']
    target={'task':task,'checkpoint':r['checkpoint'],'checkpoint_sha256':r['sha256'],
        'resume_context':r['resume_context'],'resume_context_sha256':r['resume_context_sha256'],
        'receipt':str(receipt),'receipt_sha256':sha256(receipt),'source_output':str(out),
        'source_summary':str(out/'complete.json'),'source_summary_sha256':sha256(out/'complete.json'),
        'epoch500_successes':s['successes']}
    atomic_json(f,target);return target


def plan():
    f=HERE/'plan.json'
    if f.exists():return json.loads(f.read_text())
    for task in TASKS:bind_source(task)
    protocol=json.loads((BASE/'protocol.json').read_text())
    p={'created':time.time(),'method':'DP','tasks':TASKS,'start_epoch':500,'target_epoch':1000,
       'additional_epochs':500,'snapshot_epochs':[600,700,800,900,1000],'training_seed':0,
       'horizon':16,'n_obs_steps':2,'action_horizon':8,'num_inference_steps':10,'batch_size':128,
       'lr_schedule_epochs':1000,'start_updates':30500,'target_updates':61000,
       'restore':['model','ema_model','optimizer','lr_scheduler','EMA_update_counter','Python_NumPy_Torch_CPU_CUDA_RNG','train_sampling_batch','runner_state'],
       'seeds':protocol['final_episode_seeds'],'episodes_per_task':20,'max_episode_steps':200,
       'classification':'Requested continuation of saved500-epoch runs to fixed terminal1000; paired evaluation on the already observed common20-seed suite. Preserve500 results and all checkpoints; no outcome selection.',
       'max_parallel':3,'gpu_slots':{'0':2,'1':1},'training_budget_mib':10000,'reserve_mib':4096,
       'execution_files':{n:sha256(HERE/n) for n in ['run.py','resume.py','rollout.py']},
       'targets_sha256':{task:sha256(HERE/'targets'/(task+'.json')) for task in TASKS},
       'source_plan_sha256':sha256(SOURCE/'plan.json'),'baseline_protocol_sha256':sha256(BASE/'protocol.json'),
       'baseline_worker_sha256':sha256(BASE/'worker.py'),
       'official_training_sha256':sha256(BASE/'sources/dp/diffusion_policy-main/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py'),
       'stop_rule':'Save every100 cumulative epochs, stop at1000 and evaluate all20 episodes per task; no further training without instruction.'}
    assert json.loads((HERE/'resume_preflight.json').read_text())['passed']
    atomic_json(f,p)
    (HERE/'parameters.diff').write_text('--- completed_DP500\n+++ requested_DP1000_continuation\n'
        '-terminal_epoch: 500\n+terminal_epoch: 1000\n+additional_epochs: 500\n+full_saved_state_resume: true\n'
        '+snapshot_epochs: [600,700,800,900,1000]\n training_seed: 0\n batch_size: 128\n'
        ' lr_schedule_epochs: 1000\n num_inference_steps: 10\n horizon: 16\n n_obs_steps: 2\n action_horizon: 8\n')
    return p


def output(task):
    attempts=sorted((HERE/'runs'/task).glob('attempt_*'))
    return attempts[-1] if attempts else HERE/'runs'/task/'attempt_01'


def result(p,task):
    f=output(task)/'complete.json'
    if not f.exists():return None
    s=json.loads(f.read_text())
    assert s['episodes']==20 and s['seeds']==p['seeds']
    assert s['identity']['checkpoint_completed_epochs']==1000
    assert s['identity']['ablation_plan_sha256']==sha256(HERE/'plan.json')
    return s


def report(p,jobs,state='running'):
    rows=[];snapshots=[]
    for j in jobs:
        out=output(j['task']);f=out/'heartbeat.json';hb=json.loads(f.read_text()) if f.exists() else {}
        s=result(p,j['task']);target=bind_source(j['task'])
        for f in sorted((out/'snapshot_receipts').glob('epoch_*.json')):snapshots.append(dict(task=j['task'],**json.loads(f.read_text())))
        rows.append({'task':j['task'],'state':j['state'],'epoch':1000 if s else hb.get('epoch'),
            'epoch500_successes':target['epoch500_successes'],'epoch1000_successes':s['successes'] if s else None,
            'episodes':20 if s else 0})
    atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':state,'jobs':jobs,'rows':rows})
    atomic_json(HERE/'checkpoint_index.json',snapshots)
    lines=['# DP500→1000轮续训与评测','',
        '恢复500轮的模型、EMA、优化器、学习率、EMA计数、随机数和训练采样批次；原1000轮学习率计划。',
        '累计600/700/800/900/1000轮保存独立快照；去噪10步、horizon16、obs2、action8、训练seed0。',
        '1000轮固定终点EMA，每项20局，同一评测seed；500轮结果保留。','',f'状态：{state}。','',
        '| 任务 | 当前累计epoch（约） | 500轮结果 | 1000轮结果 | 状态 |','|---|---:|---:|---:|---|']
    for row in rows:
        val=f"{row['epoch1000_successes']}/20" if row['episodes'] else '待评测'
        lines.append(f"| {row['task']} | {row['epoch']} | {row['epoch500_successes']}/20 | {val} | {row['state']} |")
    lines+=['','## 已保存快照','']
    lines += [f"- {s['task']}，{s['completed_epochs']}轮：[权重]({s['checkpoint']}) / [续训状态]({s['resume_context']})" for s in snapshots]
    (HERE/'progress.md').write_text('\n'.join(lines)+'\n')


def audit(p):
    import numpy as np
    rows=[];records=[]
    for task in TASKS:
        out=output(task);s=result(p,task);assert s
        target=bind_source(task);assert sha256(target['source_summary'])==target['source_summary_sha256']
        tr=json.loads((out/'trained.json').read_text());ta=json.loads((out/'training_audit.json').read_text())
        assert tr['completed_epochs']==1000 and tr['optimizer_updates']==61000
        assert tr['source_checkpoint_sha256']==target['checkpoint_sha256']
        assert sha256(tr['checkpoint'])==tr['sha256']==s['identity']['checkpoint_sha256']
        assert ta['passed'] and ta['additional_updates']==30500 and ta['snapshot_epochs_verified']==p['snapshot_epochs']
        hits=0
        for i,seed in enumerate(p['seeds']):
            path=out/'final_rollouts'/f'episode_{i:02d}.json';e=json.loads(path.read_text())
            b=json.loads((Path(target['source_output'])/'final_rollouts'/path.name).read_text())
            assert e['identity']==s['identity'] and e['seed']==b['seed']==seed
            assert e['initial_state_sha256']==b['initial_state_sha256']
            assert len(e['infos'])==e['steps']==200 and e['nonfinite_actions']==0
            native=[j+1 for j,x in enumerate(e['infos']) if bool(x['success'])]
            assert e['success']==bool(native) and e['first_success_step']==(native[0] if native else None)
            assert e['num_inference_steps']==10 and e['denoising_step_calls']==250
            assert Path(e['video']).stat().st_size>0
            with np.load(e['trajectory']) as data:
                assert np.array_equal(data['query_steps'],np.arange(0,200,8)) and data['predicted_chunks'].shape==(25,8,4)
                assert np.allclose(data['predicted_chunks'].reshape(200,4),data['actions'],rtol=1e-6,atol=1e-6)
                assert np.isfinite(data['actions']).all() and np.isclose(data['rewards'].sum(),e['return'])
            hits+=bool(e['success']);records.append({'task':task,'seed':seed,'success':e['success'],
                'record':str(path),'video':e['video'],'trajectory':e['trajectory']})
        assert hits==s['successes']
        rows.append({'task':task,'epoch500_successes':target['epoch500_successes'],'epoch1000_successes':hits,
            'episodes':20,'epoch500_rate':target['epoch500_successes']/20,'epoch1000_rate':hits/20})
    atomic_json(HERE/'summary.json',rows);atomic_json(HERE/'episode_manifest.json',records)
    successful=[r for r in records if r['success']];atomic_json(HERE/'successful_episodes.json',successful)
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'summary.csv').write_text(stream.getvalue())
    lines=['# DP500轮与续训1000轮结果','', '| 任务 | 500轮 | 1000轮 |','|---|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['epoch500_successes']}/20（{r['epoch500_rate']:.0%}） | {r['epoch1000_successes']}/20（{r['epoch1000_rate']:.0%}） |")
    lines+=['',f"五项平均：500轮36%；1000轮{sum(r['epoch1000_successes'] for r in rows)}%。",'',
        '完整恢复500轮状态后继续训练；固定1000轮EMA，统一20个已观测seed配对评测。所有历史结果保留。']
    (HERE/'summary.md').write_text('\n'.join(lines)+'\n')
    lines=['# 1000轮成功episode索引','', '| 任务 | seed | 视频 | 轨迹 | 原始判定 |','|---|---:|---|---|---|']
    for task in TASKS:
        candidates=[r for r in successful if r['task']==task]
        if candidates:
            r=candidates[0];lines.append(f"| {task} | {r['seed']} | [MP4]({r['video']}) | [NPZ]({r['trajectory']}) | [JSON]({r['record']}) |")
        else:lines.append(f'| {task} | — | 无成功episode | — | — |')
    (HERE/'successful_episodes.md').write_text('\n'.join(lines)+'\n')
    atomic_json(HERE/'audit.json',{'passed':True,'episodes':100,'source_epoch':500,'terminal_epoch':1000,
        'full_resume_state_restored':True,'snapshot_epochs':p['snapshot_epochs'],'native_success_recomputed':True,
        'actions_reconstructed':True,'same_initial_states_as500':True,'actual_denoising_calls_per_episode':250})


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def main():
    lock=(HERE/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=plan()
    for n,h in p['execution_files'].items():assert sha256(HERE/n)==h
    def handler(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time()})
    jobs=[{'task':t,'state':'complete' if result(p,t) else 'pending','failures':0,'not_before':0,
        'attempt':max([int(d.name.split('_')[-1]) for d in (HERE/'runs'/t).glob('attempt_*')]+[0])} for t in TASKS]
    active={}
    try:
        while True:
            free={int(g):int(f) for g,f in (x.split(',') for x in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
            yielded=set()
            for task,item in list(active.items()):
                proc,j=item['proc'],item['job'];activity=[j['started'],Path(j['log']).stat().st_mtime]
                activity += [f.stat().st_mtime for f in output(task).rglob('heartbeat.json')]
                reason=None
                if proc.poll() is None and j['gpu'] not in yielded and (free[j['gpu']]<3500 or time.time()-max(activity)>1200):
                    reason='low_free_memory' if free[j['gpu']]<3500 else 'no_progress_1200s';stop(proc);yielded.add(j['gpu'])
                if proc.poll() is not None:
                    item['log'].close()
                    if proc.returncode==0 and result(p,task):j['state']='complete'
                    else:
                        j['failures']+=reason!='low_free_memory';j['state']='pending' if j['failures']<3 else 'error';j['not_before']=time.time()+60
                        atomic_json(HERE/f'alert_{task}_attempt{j["attempt"]}.json',{'time':time.time(),'job':j,'reason':reason or 'worker_error','exit_code':proc.returncode})
                    del active[task]
            reserved={g:0 for g in free}
            for j in jobs:
                if j['state']!='pending' or len(active)>=p['max_parallel'] or time.time()<j['not_before']:continue
                counts={g:sum(item['job']['gpu']==g for item in active.values()) for g in free}
                candidates=[(f-reserved[g],g) for g,f in free.items() if counts[g]<p['gpu_slots'].get(str(g),1)
                    and f-reserved[g]>=p['training_budget_mib']+p['reserve_mib']]
                if not candidates:continue
                _,gpu=max(candidates);j.update(state='running',attempt=j['attempt']+1,gpu=gpu,started=time.time())
                path=HERE/'logs'/f'{j["task"]}_attempt{j["attempt"]}.log';path.parent.mkdir(exist_ok=True)
                log=path.open('a');j['log']=str(path)
                proc=subprocess.Popen([str(PYTHON),'-B',str(HERE/'resume.py'),j['task'],str(j['attempt'])],cwd=BASE,
                    env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                j['pid']=proc.pid;active[j['task']]={'proc':proc,'job':j,'log':log};reserved[gpu]+=p['training_budget_mib']
            report(p,jobs)
            if all(j['state']=='complete' for j in jobs):
                audit(p);report(p,jobs,'complete');atomic_json(HERE/'complete.json',{'state':'complete','episodes':100,'audited':True});return
            if all(j['state'] in ['complete','error'] for j in jobs):report(p,jobs,'partial_failure');return
            time.sleep(5)
    finally:
        for item in active.values():stop(item['proc']);item['log'].close()


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(HERE/'supervisor_error.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
