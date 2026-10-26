"""One fixed DDIM5 ablation, same checkpoints/seeds as the DDIM10 baseline."""
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
        'scheduler':'DDIM','prediction_type':'sample','num_train_timesteps':100,'num_inference_steps':10}
    after=dict(before,num_inference_steps=5)
    p={'created':time.time(),'baseline_root':str(BASE),'baseline_protocol_sha256':sha256(BASE/'protocol.json'),
        'baseline_evaluator_sha256':sha256(BASE/'rollout.py'),'runner_sha256':sha256(__file__),
        'evaluator_sha256':sha256(HERE/'rollout5.py'),'tasks':TASKS,'seeds':protocol['final_episode_seeds'],
        'training_seed':0,'episodes_per_task':20,'max_episode_steps':200,'before':before,'after':after,
        'checkpoint_rule':'Reuse completed baseline terminal epoch1000 EMA; no new training or checkpoint selection',
        'classification':'Post-hoc DDIM-step sensitivity analysis on the already observed common seed suite; original baseline retained.',
        'stop_rule':'Complete all20 episodes for allfive tasks, regardless of scores; no seed/config search.',
        'max_parallel':2,'inference_budget_mib':2048,'reserve_mib':4096}
    atomic_json(f,p)
    (HERE/'parameters.diff').write_text(''.join(difflib.unified_diff((json.dumps(before,indent=2)+'\n').splitlines(True),
        (json.dumps(after,indent=2)+'\n').splitlines(True),fromfile='DDIM10_baseline',tofile='DDIM5_sensitivity')))
    (HERE/'evaluator.diff').write_text(''.join(difflib.unified_diff((BASE/'rollout.py').read_text().splitlines(True),
        (HERE/'rollout5.py').read_text().splitlines(True),fromfile='baseline/rollout.py',tofile='ablation/rollout5.py')))
    return p


def verify(p):
    assert sha256(__file__)==p['runner_sha256']
    assert sha256(HERE/'rollout5.py')==p['evaluator_sha256']
    assert sha256(BASE/'protocol.json')==p['baseline_protocol_sha256']
    assert sha256(BASE/'rollout.py')==p['baseline_evaluator_sha256']


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


def folder(task):return HERE/'runs'/task/'rollouts20'


def result(p,task):
    f=folder(task)/'summary.json'
    if not f.exists():return None
    s=json.loads(f.read_text());t=bind_target(p,task);assert t
    assert s['episodes']==20 and s['seeds']==p['seeds']
    assert s['identity']['checkpoint_sha256']==t['checkpoint_sha256']
    assert s['identity']['ablation_plan_sha256']==sha256(HERE/'plan.json')
    assert s['identity']['config']['num_inference_steps']==5
    return s


def worker(p,task):
    verify(p);target=bind_target(p,task);assert target
    assert sha256(target['checkpoint'])==target['checkpoint_sha256']
    assert sha256(Path(target['output'])/'config.yaml')==target['config_sha256']
    bootstrap()
    import torch
    import dill
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    torch.set_num_threads(2)
    data=torch.load(target['checkpoint'],pickle_module=dill,map_location='cpu',weights_only=False)
    cfg=data['cfg'];assert cfg.training.use_ema and cfg.training.seed==0
    policy=hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(data['state_dicts']['ema_model'],strict=True)
    assert policy.horizon==16 and policy.n_obs_steps==2 and policy.n_action_steps==8
    assert policy.num_inference_steps==10 and type(policy.noise_scheduler).__name__=='DDIMScheduler'
    assert policy.noise_scheduler.config.prediction_type=='sample'
    policy.num_inference_steps=5
    policy._ablation_step_calls=0
    original_step=policy.noise_scheduler.step
    def counted_step(*args,**kw):
        policy._ablation_step_calls+=1
        return original_step(*args,**kw)
    policy.noise_scheduler.step=counted_step
    policy.cuda().eval()
    spec=importlib.util.spec_from_file_location('paired_evaluator5',HERE/'rollout5.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    evaluator.evaluate(policy,task,'SimpleDP3',folder(task),final=True,checkpoint=target['checkpoint'])
    assert sha256(target['checkpoint'])==target['checkpoint_sha256']


def report(p,jobs,state='running'):
    rows=[]
    for task in TASKS:
        t=bind_target(p,task);s=result(p,task)
        eps=[json.loads(f.read_text()) for f in sorted(folder(task).glob('episode_*.json'))]
        rows.append({'task':task,'DDIM10_successes':t['baseline_successes'] if t else '',
            'DDIM10_episodes':20 if t else '', 'DDIM5_successes':s['successes'] if s else sum(e['success'] for e in eps),
            'DDIM5_episodes':s['episodes'] if s else len(eps),
            'state':'complete' if s else next(j['state'] for j in jobs if j['task']==task)})
    lines=['# SimpleDP3 DDIM5：独立敏感性实验','',
        '仅去噪步数10→5。相同terminal EMA模型、horizon16、obs2、action8、训练seed0。',
        '五任务各20局；统一评测seeds2026102000–2026102019，每局200步。',
        '保留DDIM10原始基线；这是观察主表后的参数敏感性实验，不替换主表，也不声称独立未见测试。','',
        f'状态：{state}。','', '| 任务 | DDIM10基线 | DDIM5当前 | 阶段 |','|---|---:|---:|---|']
    for r in rows:
        a=f"{r['DDIM10_successes']}/20" if r['DDIM10_episodes'] else '等待原训练/评测结束'
        lines.append(f"| {r['task']} | {a} | {r['DDIM5_successes']}/{r['DDIM5_episodes']} | {r['state']} |")
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
        hits=0;paired=[]
        for i,seed in enumerate(p['seeds']):
            path=folder(task)/f'episode_{i:02d}.json';e=json.loads(path.read_text())
            b=json.loads((baseline/path.name).read_text())
            assert e['identity']==s['identity'] and e['seed']==b['seed']==seed
            assert e['initial_state_sha256']==b['initial_state_sha256']
            assert e['steps']==len(e['infos'])==200 and e['nonfinite_actions']==0
            success_steps=[j+1 for j,x in enumerate(e['infos']) if bool(x['success'])]
            assert e['success']==bool(success_steps) and e['first_success_step']==(success_steps[0] if success_steps else None)
            assert e['num_inference_steps']==5 and e['denoising_step_calls']==125
            assert Path(e['video']).stat().st_size>0
            with np.load(e['trajectory']) as d:
                assert np.array_equal(d['query_steps'],np.arange(0,200,8))
                assert d['predicted_chunks'].shape==(25,8,4)
                assert np.allclose(d['predicted_chunks'].reshape(200,4),d['actions'],rtol=1e-6,atol=1e-6)
                assert np.isfinite(d['actions']).all() and np.isclose(d['rewards'].sum(),e['return'])
            hits+=e['success'];paired.append({'seed':seed,'DDIM10_success':b['success'],'DDIM5_success':e['success']})
            record={'task':task,'seed':seed,'record':str(path),'sha256':sha256(path),'video':e['video'],'trajectory':e['trajectory']}
            sources.append(record)
            if e['success']:successful.append(record)
        assert hits==s['successes']
        rows.append({'task':task,'DDIM10_successes':t['baseline_successes'],'DDIM5_successes':hits,'episodes':20,
            'DDIM10_rate':t['baseline_successes']/20,'DDIM5_rate':hits/20,'change_percentage_points':(hits-t['baseline_successes'])*5,
            'checkpoint_sha256':t['checkpoint_sha256'],'DDIM5_directory':str(folder(task))})
        atomic_json(HERE/'paired_episodes'/(task+'.json'),paired)
    atomic_json(HERE/'audit.json',{'passed':True,'episodes':100,'same_checkpoint_per_task':True,'same_initial_state_per_seed':True,
        'actual_denoising_calls_per_episode':125,'inference_queries_per_episode':25,'native_success_recomputed':True,
        'saved_actions_reconstructed':True,'baseline_unchanged':True})
    atomic_json(HERE/'source_manifest.json',sources);atomic_json(HERE/'successful_episodes.json',successful)
    atomic_json(HERE/'summary.json',{'classification':p['classification'],'seeds':p['seeds'],'results':rows})
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'summary.csv').write_text(stream.getvalue())
    lines=['# SimpleDP3：DDIM10与DDIM5配对结果','',
        '同模型、同20个seed、同执行间隔8、同观测2；仅去噪步数10→5。原始基线保留，此表为事后敏感性实验。','',
        '| 任务 | DDIM10 | DDIM5 | 差值（百分点） |','|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['DDIM10_successes']}/20（{r['DDIM10_rate']:.0%}） | {r['DDIM5_successes']}/20（{r['DDIM5_rate']:.0%}） | {r['change_percentage_points']:+d} |")
    a=sum(r['DDIM10_successes'] for r in rows);b=sum(r['DDIM5_successes'] for r in rows)
    lines+=['',f'五项平均：DDIM10={a}%，DDIM5={b}%。100局审计通过。','',
        '参数差异见parameters.diff；成功视频/轨迹路径见successful_episodes.json；逐seed配对结果见paired_episodes。','']
    (HERE/'summary.md').write_text('\n'.join(lines))


def main():
    p=plan();verify(p)
    if len(sys.argv)>1:
        assert sys.argv[1]=='--worker';worker(p,sys.argv[2]);return
    import fcntl
    import signal
    def stop_handler(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    lock=(HERE/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time()})
    jobs=[{'task':t,'state':'complete' if result(p,t) else 'pending','attempt':0,'failures':0,'not_before':0} for t in TASKS]
    active={}
    try:
        while True:
            free={int(g):int(f) for g,f in (x.split(',') for x in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
            for task,item in list(active.items()):
                proc=item['proc'];j=item['job'];log=Path(j['log'])
                heartbeat=folder(task)/'heartbeat.json'
                last=max(j['started'],log.stat().st_mtime,heartbeat.stat().st_mtime if heartbeat.exists() else 0)
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
                candidates=[(f,g) for g,f in free.items() if g not in occupied and f>=6144]
                if not candidates:continue
                _,gpu=max(candidates);j.update(state='running',attempt=j['attempt']+1,gpu=gpu,started=time.time())
                path=HERE/'logs'/f'{j["task"]}_attempt_{j["attempt"]:02d}.log';path.parent.mkdir(exist_ok=True)
                log=path.open('a');j['log']=str(path)
                proc=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--worker',j['task']],
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
