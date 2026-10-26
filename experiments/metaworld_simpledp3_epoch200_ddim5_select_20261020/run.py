"""Fresh200 training plus a fixed200->100->50 checkpoint-screening pipeline."""
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
REFERENCE = HERE.parent / 'metaworld_simpledp3_epoch500_ddim10_20261020'
sys.path.insert(0, str(BASE))
from runtime import TASKS, PYTHON, atomic_json, environment, bootstrap, sha256


def plan():
    f = HERE / 'plan.json'
    if f.exists():
        return json.loads(f.read_text())
    protocol = json.loads((BASE / 'protocol.json').read_text())
    p = {'created': time.time(), 'method': 'SimpleDP3', 'tasks': TASKS,
        'initialization': 'all5 tasks from scratch, training seed0, no prior checkpoint loaded',
        'stop_completed_epochs': 200, 'snapshot_epochs': [50,100,200], 'lr_schedule_epochs': 1000,
        'training_seed': 0, 'batch_size': 128, 'demonstrations': 50, 'training_demonstrations': 40,
        'validation_demonstrations': 10, 'horizon': 16, 'n_obs_steps': 2, 'action_horizon': 8,
        'num_inference_steps': 5, 'screen_task': 'pick-place', 'screen_epochs_in_order': [200,100,50],
        'threshold_successes': 18, 'selection_rule': 'first checkpoint with successes <18/20 on pick-place; >=18/20 advances to next earlier snapshot',
        'exhaustion_rule': 'If all3 checkpoints succeed >=18/20, preserve all results and report no qualifying checkpoint; no added checkpoint/seed/hyperparameter search.',
        'episodes_per_task': 20, 'seeds': protocol['final_episode_seeds'], 'max_episode_steps': 200,
        'classification': 'Post-hoc outcome-dependent checkpoint exploration. Cannot replace a pre-specified baseline or establish an unbiased improvement. Preserve all original and screening results.',
        'reuse_selected_pick_place_screen': True, 'train_all5_to200_even_if_screening_exhausted': True,
        'max_parallel_train': 2, 'max_parallel_eval': 2, 'train_budget_mib': 4096,
        'eval_budget_mib': 2048, 'reserve_mib': 4096,
        'baseline_protocol_sha256': sha256(BASE / 'protocol.json'),
        'baseline_worker_sha256': sha256(BASE / 'worker.py'),
        'official_training_sha256': sha256(BASE / 'sources/dp3/3D-Diffusion-Policy-master/3D-Diffusion-Policy/train.py'),
        'execution_files': {n: sha256(HERE / n) for n in ['run.py','train200.py','rollout.py']}}
    atomic_json(f, p)
    (HERE / 'parameters.diff').write_text('--- prior_fixed500_DDIM10\n+++ fresh200_DDIM5_with_checkpoint_selection\n'
        '-stop_completed_epochs: 500\n+stop_completed_epochs: 200\n-snapshot_epochs: [300,400,500]\n+snapshot_epochs: [50,100,200]\n'
        '-num_inference_steps: 10\n+num_inference_steps: 5\n horizon: 16\n n_obs_steps: 2\n action_horizon: 8\n'
        ' training_seed: 0\n lr_schedule_epochs: 1000\n+screen_order: [200,100,50]\n+select: first pick-place successes <18/20\n')
    return p


def verify(p):
    for name, digest in p['execution_files'].items():
        assert sha256(HERE / name) == digest, 'Changed frozen execution source: '+name
    assert sha256(BASE / 'protocol.json') == p['baseline_protocol_sha256']
    assert sha256(BASE / 'worker.py') == p['baseline_worker_sha256']


def training_output(task):
    attempts = sorted((HERE / 'training' / task).glob('attempt_*'))
    return attempts[-1] if attempts else HERE / 'training' / task / 'attempt_01'


def training_done(task):
    path = training_output(task) / 'training_complete.json'
    if not path.exists():
        return False
    r = json.loads(path.read_text())
    assert r['completed_epochs'] == 200 and r['optimizer_updates'] == 12200
    assert r['snapshot_epochs_verified'] == [50,100,200]
    return True


def target(epoch, task):
    path = HERE / 'targets' / f'epoch{epoch}_{task}.json'
    if path.exists():
        return json.loads(path.read_text())
    for out in sorted((HERE / 'training' / task).glob('attempt_*')):
        receipt_path = out / 'snapshot_receipts' / f'epoch_{epoch:04d}.json'
        if not receipt_path.exists():
            continue
        r = json.loads(receipt_path.read_text())
        assert r['completed_epochs'] == epoch and r['optimizer_updates'] == 61*epoch
        assert sha256(r['checkpoint']) == r['sha256']
        t = {'epoch': epoch, 'task': task, 'checkpoint': r['checkpoint'], 'checkpoint_sha256': r['sha256'],
            'training_output': str(out), 'config_sha256': sha256(out/'config.yaml'),
            'receipt': str(receipt_path), 'receipt_sha256': sha256(receipt_path)}
        atomic_json(path, t)
        return t
    return None


def folder(epoch, task):
    return HERE / 'evaluation' / f'epoch{epoch}' / task / 'rollouts20'


def result(p, epoch, task):
    f = folder(epoch, task) / 'summary.json'
    if not f.exists():
        return None
    s, t = json.loads(f.read_text()), target(epoch,task)
    assert t and s['episodes'] == 20 and s['seeds'] == p['seeds']
    assert s['identity']['checkpoint_completed_epochs'] == epoch
    assert s['identity']['checkpoint_sha256'] == t['checkpoint_sha256']
    assert s['identity']['ablation_plan_sha256'] == sha256(HERE/'plan.json')
    assert s['identity']['config'] == {'horizon':16,'n_obs_steps':2,'action_horizon':8,
        'num_inference_steps':5,'scheduler':'DDIM','prediction_type':'sample'}
    return s


def evaluate_worker(p, epoch, task):
    verify(p)
    t = target(epoch, task)
    assert t and sha256(t['checkpoint']) == t['checkpoint_sha256']
    assert sha256(Path(t['training_output'])/'config.yaml') == t['config_sha256']
    bootstrap()
    import torch
    import dill
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    torch.set_num_threads(2)
    data = torch.load(t['checkpoint'],pickle_module=dill,map_location='cpu',weights_only=False)
    assert dill.loads(data['pickles']['epoch']) == epoch and dill.loads(data['pickles']['global_step']) == 61*epoch
    cfg = data['cfg']
    assert cfg.training.seed == 0 and cfg.training.use_ema and cfg.policy.num_inference_steps == 5
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(data['state_dicts']['ema_model'],strict=True)
    del data
    assert policy.horizon == 16 and policy.n_obs_steps == 2 and policy.n_action_steps == 8
    policy._evaluation_checkpoint_epoch = epoch
    policy.cuda().eval()
    spec = importlib.util.spec_from_file_location('checkpoint_eval',HERE/'rollout.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.evaluate(policy,task,'SimpleDP3',folder(epoch,task),final=True,checkpoint=t['checkpoint'])
    assert sha256(t['checkpoint']) == t['checkpoint_sha256']


def audit_group(p, epoch, task):
    import numpy as np
    s, t = result(p,epoch,task), target(epoch,task)
    assert s and t and sha256(t['checkpoint']) == t['checkpoint_sha256']
    ref = json.loads((REFERENCE/'targets'/(task+'.json')).read_text())
    baseline = Path(ref['baseline_summary']).parent
    hits, records = 0, []
    for i,seed in enumerate(p['seeds']):
        path = folder(epoch,task)/f'episode_{i:02d}.json'
        e,b = json.loads(path.read_text()),json.loads((baseline/path.name).read_text())
        assert e['identity'] == s['identity'] and e['seed'] == b['seed'] == seed
        assert e['initial_state_sha256'] == b['initial_state_sha256']
        assert e['steps'] == len(e['infos']) == 200 and e['nonfinite_actions'] == 0
        native = [j+1 for j,x in enumerate(e['infos']) if bool(x['success'])]
        assert e['success'] == bool(native) and e['first_success_step'] == (native[0] if native else None)
        assert e['num_inference_steps'] == 5 and e['denoising_step_calls'] == 125
        assert Path(e['video']).stat().st_size > 0
        with np.load(e['trajectory']) as data:
            assert np.array_equal(data['query_steps'],np.arange(0,200,8))
            assert data['predicted_chunks'].shape == (25,8,4)
            assert np.allclose(data['predicted_chunks'].reshape(200,4),data['actions'],rtol=1e-6,atol=1e-6)
            assert np.isfinite(data['actions']).all() and np.isclose(data['rewards'].sum(),e['return'])
        hits += bool(e['success'])
        records.append({'epoch':epoch,'task':task,'seed':seed,'success':bool(e['success']),
            'record':str(path),'sha256':sha256(path),'video':e['video'],'trajectory':e['trajectory']})
    assert hits == s['successes']
    atomic_json(HERE/'audits'/f'epoch{epoch}_{task}.json',{'passed':True,'epoch':epoch,'task':task,
        'episodes':20,'successes':hits,'summary_sha256':sha256(folder(epoch,task)/'summary.json'),
        'original_initial_states_match':True,'native_success_recomputed':True,'actions_reconstructed':True,
        'actual_denoising_calls_per_episode':125})
    atomic_json(HERE/'episode_manifests'/f'epoch{epoch}_{task}.json',records)
    return records


def selection_state(p):
    history = []
    for epoch in p['screen_epochs_in_order']:
        s = result(p,epoch,'pick-place')
        if not s:
            return {'state':'screening','next_epoch':epoch,'history':history,'selected_epoch':None}
        audit = HERE/'audits'/f'epoch{epoch}_pick-place.json'
        if not audit.exists():
            audit_group(p,epoch,'pick-place')
        assert json.loads(audit.read_text())['summary_sha256'] == sha256(folder(epoch,'pick-place')/'summary.json')
        history.append({'epoch':epoch,'successes':s['successes'],'episodes':20,'success_rate':s['success_rate'],
            'summary':str(folder(epoch,'pick-place')/'summary.json'),'summary_sha256':sha256(folder(epoch,'pick-place')/'summary.json')})
        if s['successes'] < p['threshold_successes']:
            value = {'state':'selected','selected_epoch':epoch,'history':history,'plan_sha256':sha256(HERE/'plan.json')}
            f = HERE/'selection.json'
            if f.exists():
                assert json.loads(f.read_text()) == value
            else:
                atomic_json(f,value)
            return value
    value = {'state':'exhausted','selected_epoch':None,'history':history,'plan_sha256':sha256(HERE/'plan.json')}
    atomic_json(HERE/'selection_exhausted.json',value)
    return value


def report(p, jobs, selection, state='running'):
    snapshots, rows = [], []
    for task in TASKS:
        out = training_output(task)
        f = out/'heartbeat.json'
        hb = json.loads(f.read_text()) if f.exists() else {}
        for f in sorted((out/'snapshot_receipts').glob('epoch_*.json')):
            snapshots.append(dict(task=task,**json.loads(f.read_text())))
        job = next(j for j in jobs if j['kind']=='train' and j['task']==task)
        rows.append({'kind':'train','task':task,'epoch':hb.get('epoch'),'state':job['state']})
    for job in jobs:
        if job['kind'] != 'eval':
            continue
        s = result(p,job['epoch'],job['task'])
        episodes = [json.loads(f.read_text()) for f in sorted(folder(job['epoch'],job['task']).glob('episode_*.json'))]
        rows.append({'kind':'eval','task':job['task'],'epoch':job['epoch'],'state':job['state'],
            'successes':s['successes'] if s else sum(bool(e['success']) for e in episodes),
            'episodes':20 if s else len(episodes)})
    atomic_json(HERE/'checkpoint_index.json',snapshots)
    atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':state,'selection':selection,'jobs':jobs,'rows':rows})
    lines = ['# SimpleDP3：新训200轮，DDIM5，horizon16','',
        '五任务从头训练seed0，50示教（40训练/10验证），batch128；观测2、执行间隔8。',
        '沿用1000轮学习率计划，在完成50/100/200轮各存独立快照，本次最多训练200轮。',
        'pick-place按200→100→50轮依次完整测20局，首个低于18/20的轮次用于五任务；三项均≥18/20则报告无满足阈值快照。',
        '按结果选权重的探索评测，保留所有筛选和原基线。','',f'状态：{state}；筛选状态：{selection["state"]}；选中轮次：{selection["selected_epoch"]}。','',
        '| 类型 | 任务 | 当前epoch/权重轮次 | 进度 | 状态 |','|---|---|---:|---|---|']
    for row in rows:
        progress = '目标200轮' if row['kind']=='train' else f"{row['successes']}/{row['episodes']}成功"
        lines.append(f"| {row['kind']} | {row['task']} | {row['epoch']} | {progress} | {row['state']} |")
    lines += ['','筛选历史：']+[f"- {h['epoch']}轮pick-place：{h['successes']}/20（{h['success_rate']:.0%}）" for h in selection['history']]
    (HERE/'progress.md').write_text('\n'.join(lines)+'\n')


def finalize(p, selection):
    if selection['state'] == 'exhausted':
        lines = ['# SimpleDP3：筛选未找到低于90%的快照','', '| 权重轮次 | pick-place成功次数 | 成功率 |','|---|---:|---:|']
        for h in selection['history']:
            lines.append(f"| {h['epoch']} | {h['successes']}/20 | {h['success_rate']:.0%} |")
        lines += ['','五任务均已完成200轮训练，并保存50/100/200轮权重。所有筛选数据保留；没有符合条件的统一轮次，因此未编造五任务表，也未增加搜索。']
        (HERE/'summary.md').write_text('\n'.join(lines)+'\n')
        atomic_json(HERE/'complete.json',{'state':'screening_exhausted','training_tasks_complete':5,'screen_episodes':60,'selected_epoch':None})
        return
    epoch, records, rows = selection['selected_epoch'], [], []
    for task in TASKS:
        s = result(p,epoch,task)
        assert s
        records += audit_group(p,epoch,task)
        rows.append({'task':task,'checkpoint_epoch':epoch,'successes':s['successes'],'episodes':20,'success_rate':s['success_rate']})
    atomic_json(HERE/'summary.json',{'classification':p['classification'],'selection':selection,'results':rows})
    atomic_json(HERE/'successful_episodes.json',[r for r in records if r['success']])
    stream = io.StringIO()
    w = csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'summary.csv').write_text(stream.getvalue())
    lines = [f'# SimpleDP3：新训200轮，选中{epoch}轮，DDIM5/horizon16','',
        '五任务统一所选轮次；观测2、执行间隔8；各20局、统一seed。按结果选轮次的探索实验，原结果全部保留。','',
        '| 任务 | 轮次 | 成功次数 | 成功率 |','|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['task']} | {epoch} | {r['successes']}/20 | {r['success_rate']:.0%} |")
    lines += ['',f"五项平均：{sum(r['successes'] for r in rows)}%。",'','## 筛选记录','']
    lines += [f"- {h['epoch']}轮pick-place：{h['successes']}/20（{h['success_rate']:.0%}）" for h in selection['history']]
    (HERE/'summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json(HERE/'complete.json',{'state':'complete','selected_epoch':epoch,'selected_episodes':100,
        'unique_episodes':100+20*(len(selection['history'])-1),'training_tasks_complete':5,'audited':True})


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def eval_job(epoch, task):
    return {'key':f'eval_{epoch}_{task}','kind':'eval','epoch':epoch,'task':task,'state':'pending','attempt':0,'failures':0,'not_before':0}


def main():
    p = plan();verify(p)
    if len(sys.argv)>1:
        assert sys.argv[1]=='--eval'
        evaluate_worker(p,int(sys.argv[2]),sys.argv[3]);return
    lock = (HERE/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def handler(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time()})
    order = ['pick-place']+[t for t in TASKS if t!='pick-place']
    jobs = [{'key':'train_'+t,'kind':'train','task':t,'state':'complete' if training_done(t) else 'pending',
             'attempt':max([int(q.name.split('_')[-1]) for q in (HERE/'training'/t).glob('attempt_*')]+[0]),
             'failures':0,'not_before':0} for t in order]
    active = {}
    try:
        while True:
            free = {int(g):int(f) for g,f in (line.split(',') for line in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
            for key,item in list(active.items()):
                proc,j = item['proc'],item['job']
                out = training_output(j['task']) if j['kind']=='train' else folder(j['epoch'],j['task'])
                last = max([j['started'],Path(j['log']).stat().st_mtime]+[x.stat().st_mtime for x in out.rglob('heartbeat.json')])
                reason = None
                if proc.poll() is None and (free[j['gpu']]<3500 or time.time()-last>900):
                    reason = 'low_free_memory' if free[j['gpu']]<3500 else 'no_progress_900s';stop(proc)
                if proc.poll() is not None:
                    item['log'].close()
                    done = training_done(j['task']) if j['kind']=='train' else result(p,j['epoch'],j['task'])
                    if proc.returncode==0 and done:
                        if j['kind']=='eval':audit_group(p,j['epoch'],j['task'])
                        j['state']='complete'
                    else:
                        j['failures'] += reason!='low_free_memory'
                        j['state']='pending' if j['failures']<3 else 'error';j['not_before']=time.time()+30
                        atomic_json(HERE/f'alert_{key}_attempt{j["attempt"]}.json',{'time':time.time(),'job':j,'reason':reason or 'worker_error','exit_code':proc.returncode})
                    del active[key]
            selection = selection_state(p)
            wanted = []
            if selection['state']=='screening' and training_done('pick-place'):
                wanted=[(selection['next_epoch'],'pick-place')]
            elif selection['state']=='selected':
                wanted=[(selection['selected_epoch'],task) for task in TASKS]
            for epoch,task in wanted:
                key=f'eval_{epoch}_{task}'
                if not any(j['key']==key for j in jobs):
                    j=eval_job(epoch,task)
                    if result(p,epoch,task):j['state']='complete'
                    jobs.append(j)
            # Inference and training have separate per-GPU slots and explicit shared memory reservations.
            reserved={gpu:0 for gpu in free}
            for j in sorted(jobs,key=lambda j:0 if j['kind']=='eval' else 1):
                if j['state'] not in ['pending','waiting_checkpoint'] or time.time()<j['not_before']:continue
                kind=j['kind'];same=[x['job'] for x in active.values() if x['job']['kind']==kind]
                if len(same)>=p['max_parallel_'+kind]:continue
                if kind=='eval' and target(j['epoch'],j['task']) is None:
                    j['state']='waiting_checkpoint';continue
                budget=p[kind+'_budget_mib'];occupied={x['gpu'] for x in same}
                candidates=[(f-reserved[g],g) for g,f in free.items() if g not in occupied and f-reserved[g]>=budget+p['reserve_mib']]
                if not candidates:continue
                _,gpu=max(candidates);j.update(state='running',attempt=j['attempt']+1,gpu=gpu,started=time.time())
                logpath=HERE/'logs'/f'{j["key"]}_attempt{j["attempt"]}.log';logpath.parent.mkdir(exist_ok=True)
                log=logpath.open('a');j['log']=str(logpath)
                if kind=='train':argv=[str(PYTHON),'-B',str(HERE/'train200.py'),j['task'],str(j['attempt'])]
                else:argv=[str(PYTHON),'-B',str(HERE/'run.py'),'--eval',str(j['epoch']),j['task']]
                proc=subprocess.Popen(argv,cwd=BASE,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                j['pid']=proc.pid;active[j['key']]={'proc':proc,'job':j,'log':log};reserved[gpu]+=budget
            report(p,jobs,selection)
            if any(j['state']=='error' for j in jobs) and not active:
                report(p,jobs,selection,'partial_failure');return
            all_trained=all(training_done(t) for t in TASKS)
            all_evaluated=selection['state']=='selected' and all(result(p,selection['selected_epoch'],t) for t in TASKS)
            if all_trained and (selection['state']=='exhausted' or all_evaluated) and not active:
                finalize(p,selection);report(p,jobs,selection,'complete' if all_evaluated else 'screening_exhausted');return
            time.sleep(5)
    finally:
        for item in active.values():stop(item['proc']);item['log'].close()


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(HERE/'supervisor_error.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
