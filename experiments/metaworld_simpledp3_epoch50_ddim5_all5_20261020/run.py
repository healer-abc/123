"""User-fixed epoch50 assessment, retaining the previous unsuccessful threshold search."""
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
SOURCE = HERE.parent / 'metaworld_simpledp3_epoch200_ddim5_select_20261020'
sys.path.insert(0, str(BASE))
from runtime import TASKS, PYTHON, atomic_json, environment, sha256

# Reuse audited weight-loading and artifact-verification functions in this isolated process.
# No original source or original experiment result is modified.
spec = importlib.util.spec_from_file_location('reusable_eval_helpers', SOURCE/'run.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
helpers.HERE = HERE


def plan():
    f = HERE/'plan.json'
    if f.exists():return json.loads(f.read_text())
    protocol=json.loads((BASE/'protocol.json').read_text())
    reused=SOURCE/'evaluation/epoch50/pick-place/rollouts20/summary.json'
    p={'created':time.time(),'method':'SimpleDP3','checkpoint_completed_epochs':50,'tasks':TASKS,
       'num_inference_steps':5,'horizon':16,'n_obs_steps':2,'action_horizon':8,'training_seed':0,
       'seeds':protocol['final_episode_seeds'],'episodes_per_task':20,'max_episode_steps':200,
       'source_plan_sha256':sha256(SOURCE/'plan.json'),'source_runner_sha256':sha256(SOURCE/'run.py'),
       'source_exhausted_sha256':sha256(SOURCE/'selection_exhausted.json'),
       'reused_pick_place_summary':str(reused),'reused_pick_place_summary_sha256':sha256(reused),
       'baseline_protocol_sha256':sha256(BASE/'protocol.json'),'baseline_worker_sha256':sha256(BASE/'worker.py'),
       'execution_files':{n:sha256(HERE/n) for n in ['run.py','rollout.py']},
       'classification':'User explicitly fixed epoch50 after200/100/50 screening produced95/90/90%. This is post-hoc exploratory evaluation, not a successful <90% threshold selection and not a replacement for fixed original baselines.',
       'reuse_pick_place20':True,'new_episodes':80,'total_table_episodes':100,
       'stop_rule':'Finish all5 groups20episodes regardless of outcome, without further checkpoint/seed changes',
       'max_parallel':2,'inference_budget_mib':2048,'reserve_mib':4096}
    atomic_json(f,p)
    (HERE/'parameters.diff').write_text('--- completed200_100_50_threshold_search\n+++ user_fixed_epoch50_all5\n'
        '-selection: first pick-place <90%; exhausted with [95%,90%,90%]\n'
        '+checkpoint_epoch: 50, explicitly fixed by user after search\n'
        '+reuse_pick_place: 18/20 from the completed50-epoch screen\n'
        '+evaluate_other4_tasks: 20episodes each, no outcome-based changes\n'
        ' num_inference_steps: 5\n horizon: 16\n n_obs_steps: 2\n action_horizon: 8\n training_seed: 0\n')
    return p


def verify(p):
    for n,h in p['execution_files'].items():assert sha256(HERE/n)==h
    assert sha256(SOURCE/'run.py')==p['source_runner_sha256']
    assert sha256(SOURCE/'plan.json')==p['source_plan_sha256']
    assert sha256(SOURCE/'selection_exhausted.json')==p['source_exhausted_sha256']
    assert sha256(BASE/'protocol.json')==p['baseline_protocol_sha256']
    assert sha256(BASE/'worker.py')==p['baseline_worker_sha256']
    assert sha256(p['reused_pick_place_summary'])==p['reused_pick_place_summary_sha256']


def folder(epoch,task):
    assert epoch==50
    return SOURCE/'evaluation/epoch50/pick-place/rollouts20' if task=='pick-place' else HERE/'runs'/task/'rollouts20'


def target(epoch,task):
    assert epoch==50
    f=HERE/'targets'/(task+'.json')
    if f.exists():return json.loads(f.read_text())
    out=sorted((SOURCE/'training'/task).glob('attempt_*'))[-1]
    receipt=out/'snapshot_receipts/epoch_0050.json';r=json.loads(receipt.read_text())
    assert r['completed_epochs']==50 and r['optimizer_updates']==3050
    assert sha256(r['checkpoint'])==r['sha256']
    t={'task':task,'epoch':50,'checkpoint':r['checkpoint'],'checkpoint_sha256':r['sha256'],
       'training_output':str(out),'config_sha256':sha256(out/'config.yaml'),
       'receipt':str(receipt),'receipt_sha256':sha256(receipt)}
    atomic_json(f,t);return t


def result(p,epoch,task):
    f=folder(epoch,task)/'summary.json'
    if not f.exists():return None
    s=json.loads(f.read_text());t=target(epoch,task)
    assert s['episodes']==20 and s['seeds']==p['seeds']
    assert s['identity']['checkpoint_completed_epochs']==50
    assert s['identity']['checkpoint_sha256']==t['checkpoint_sha256']
    expected=p['source_plan_sha256'] if task=='pick-place' else sha256(HERE/'plan.json')
    assert s['identity']['ablation_plan_sha256']==expected
    assert s['identity']['config']=={'horizon':16,'n_obs_steps':2,'action_horizon':8,
        'num_inference_steps':5,'scheduler':'DDIM','prediction_type':'sample'}
    return s


helpers.verify=verify
helpers.folder=folder
helpers.target=target
helpers.result=result


def report(p,jobs,state='running'):
    rows=[]
    for j in jobs:
        s=result(p,50,j['task'])
        episodes=[json.loads(f.read_text()) for f in sorted(folder(50,j['task']).glob('episode_*.json'))]
        rows.append({'task':j['task'],'checkpoint_epoch':50,'state':j['state'],
            'successes':s['successes'] if s else sum(bool(e['success']) for e in episodes),
            'episodes':20 if s else len(episodes),'reused':j['task']=='pick-place'})
    atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':state,'jobs':jobs,'rows':rows})
    lines=['# SimpleDP3：固定50轮，五任务评测','',
        '用户明确指定50轮；DDIM5、horizon16、obs2、action8；各20局，统一种子2026102000–2026102019。',
        'pick-place原筛选20局复用（18/20=90%），其他四项新测；旧筛选未满足<90%条件的结论保留。',
        '本轮是事后固定checkpoint的探索结果，不替换原基线。','',f'状态：{state}。','',
        '| 任务 | 权重轮次 | 当前成功/已测 | 状态 |','|---|---:|---:|---|']
    for r in rows:lines.append(f"| {r['task']} | 50 | {r['successes']}/{r['episodes']} | {r['state']} |")
    (HERE/'progress.md').write_text('\n'.join(lines)+'\n')


def finalize(p):
    records=[];rows=[];audits=[]
    for task in TASKS:
        s=result(p,50,task);assert s
        records+=helpers.audit_group(p,50,task)
        rows.append({'task':task,'checkpoint_epoch':50,'successes':s['successes'],'episodes':20,
                     'success_rate':s['success_rate'],'reused_screen':task=='pick-place'})
        audits.append(str(HERE/'audits'/f'epoch50_{task}.json'))
    atomic_json(HERE/'summary.json',{'classification':p['classification'],'results':rows})
    atomic_json(HERE/'episode_manifest.json',records)
    successful=[r for r in records if r['success']];atomic_json(HERE/'successful_episodes.json',successful)
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'summary.csv').write_text(stream.getvalue())
    lines=['# SimpleDP3：50轮，DDIM5，五任务结果','',
        'horizon16、obs2、action8、训练seed0；每任务20局，统一评测seed。pick-place复用已完成筛选的20局。','',
        '| 任务 | 权重轮次 | 成功次数 | 成功率 |','|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | 50 | {r['successes']}/20 | {r['success_rate']:.0%} |")
    lines+=['',f"五项平均：{sum(r['successes'] for r in rows)}%。",'',
        '事后由用户指定50轮的探索评测；此前三个快照筛选均≥90%的结果及原始基线完整保留。']
    (HERE/'summary.md').write_text('\n'.join(lines)+'\n')
    lines=['# 成功episode索引','','全部成功episode见successful_episodes.json；下列为各任务首个成功episode。','',
           '| 任务 | seed | 视频 | 轨迹 | 原始判定 |','|---|---:|---|---|---|']
    for task in TASKS:
        candidates=[x for x in successful if x['task']==task]
        if candidates:
            x=candidates[0];lines.append(f"| {task} | {x['seed']} | [MP4]({x['video']}) | [NPZ]({x['trajectory']}) | [JSON]({x['record']}) |")
        else:lines.append(f'| {task} | — | 无成功episode | — | — |')
    (HERE/'successful_episodes.md').write_text('\n'.join(lines)+'\n')
    atomic_json(HERE/'audit.json',{'passed':True,'episodes':100,'new_episodes':80,'reused_episodes':20,
        'checkpoint_epoch':50,'group_audits':audits,'native_success_recomputed':True,
        'actions_reconstructed':True,'same_initial_states':True,'source_results_unchanged':True})
    atomic_json(HERE/'complete.json',{'state':'complete','checkpoint_epoch':50,'episodes':100,
        'new_episodes':80,'reused_episodes':20,'audited':True})


def main():
    p=plan();verify(p)
    if len(sys.argv)>1:
        assert sys.argv[1]=='--worker';helpers.evaluate_worker(p,50,sys.argv[2]);return
    lock=(HERE/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def handler(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time()})
    jobs=[{'task':task,'state':'complete' if result(p,50,task) else 'pending','attempt':0,'failures':0,'not_before':0} for task in TASKS]
    # Verify the reused group before producing a combined table or starting new work.
    helpers.audit_group(p,50,'pick-place')
    active={}
    try:
        while True:
            free={int(g):int(f) for g,f in (x.split(',') for x in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
            for task,item in list(active.items()):
                proc,j=item['proc'],item['job'];hb=folder(50,task)/'heartbeat.json'
                last=max(j['started'],Path(j['log']).stat().st_mtime,hb.stat().st_mtime if hb.exists() else 0)
                reason=None
                if proc.poll() is None and (free[j['gpu']]<3500 or time.time()-last>600):
                    reason='low_free_memory' if free[j['gpu']]<3500 else 'no_progress_600s';helpers.stop(proc)
                if proc.poll() is not None:
                    item['log'].close()
                    if proc.returncode==0 and result(p,50,task):
                        helpers.audit_group(p,50,task);j['state']='complete'
                    else:
                        j['failures']+=reason!='low_free_memory';j['state']='pending' if j['failures']<3 else 'error';j['not_before']=time.time()+30
                        atomic_json(HERE/f'alert_{task}_attempt{j["attempt"]}.json',{'time':time.time(),'job':j,'reason':reason or 'worker_error','exit_code':proc.returncode})
                    del active[task]
            occupied={x['job']['gpu'] for x in active.values()}
            for j in jobs:
                if j['state']!='pending' or len(active)>=2 or time.time()<j['not_before']:continue
                target(50,j['task'])
                candidates=[(f,g) for g,f in free.items() if g not in occupied and f>=6144]
                if not candidates:continue
                _,gpu=max(candidates);j.update(state='running',attempt=j['attempt']+1,gpu=gpu,started=time.time())
                path=HERE/'logs'/f'{j["task"]}_attempt{j["attempt"]}.log';path.parent.mkdir(exist_ok=True)
                log=path.open('a');j['log']=str(path)
                proc=subprocess.Popen([str(PYTHON),'-B',str(HERE/'run.py'),'--worker',j['task']],
                    cwd=BASE,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                j['pid']=proc.pid;active[j['task']]={'proc':proc,'job':j,'log':log};occupied.add(gpu)
            report(p,jobs)
            if all(j['state']=='complete' for j in jobs):finalize(p);report(p,jobs,'complete');return
            if all(j['state'] in ['complete','error'] for j in jobs):report(p,jobs,'partial_failure');return
            time.sleep(5)
    finally:
        for item in active.values():helpers.stop(item['proc']);item['log'].close()


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(HERE/'supervisor_error.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
