"""Continue a finite grid after a zero 20-episode result. Scores are selection validation."""
import csv
import importlib.util
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
SOURCE=HERE.parent/'metaworld_zero_success_search_20261019/search_followup.py'
sys.path.insert(0,str(BASE))
from runtime import PYTHON,atomic_json,bootstrap,environment,sha256


def audit_case(case,plan):
    import numpy as np
    stats=None
    if plan['row']['method']=='ACT':
        with (Path(plan['row']['output'])/'dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
    count=0
    for p in case.glob('**/episode_*.json'):
        r=json.loads(p.read_text());n=r['steps']
        assert n==len(r['infos'])==200
        assert r['success']==any(bool(i['success']) for i in r['infos'])
        with np.load(p.with_suffix('.npz')) as d:
            assert len(d['actions'])==200 and np.isfinite(d['actions']).all() and np.isfinite(d['predicted_chunks']).all()
            assert abs(float(d['rewards'].sum())-r['return'])<1e-8
            if not r['identity']['expert_positive_control'] and not r['identity']['config'].get('temporal_agg',False):
                q=np.searchsorted(d['query_steps'],np.arange(200),side='right')-1
                offsets=np.arange(200)-d['query_steps'][q]
                expected=d['predicted_chunks'][q,offsets]
                if stats is not None:expected=expected*stats['action_std']+stats['action_mean']
                assert np.allclose(expected,d['actions'],rtol=1e-6,atol=1e-6)
        assert p.with_suffix('.mp4').stat().st_size>1000
        count+=1
    for p in case.glob('**/summary.json'):
        s=json.loads(p.read_text());records=[json.loads(r.read_text()) for r in p.parent.glob('episode_*.json')]
        assert len(records)==s['episodes'] and sum(r['success'] for r in records)==s['successes']
    atomic_json(case/'artifact_audit.json',{'passed':True,'new_episodes':count,'scores_recomputed':True,'actions_checked':True})


def worker(case):
    plan=json.loads((case/'plan.json').read_text());row=plan['row']
    assert sha256(SOURCE)==plan['engine_sha256']
    spec=importlib.util.spec_from_file_location('continuation_engine',SOURCE)
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);engine.HERE=case
    bootstrap()
    import torch
    torch.set_num_threads(2)
    policy,stats,checkpoint=engine.load_model(row)
    original=Path(plan['source_case'])
    rejected={json.dumps(plan['previous_config'],sort_keys=True)}
    trials=[];validations=[];winner=None
    for block in range(plan['blocks']):
        for index,cfg in enumerate(plan['grid']):
            key=json.dumps(cfg,sort_keys=True)
            if key in rejected:continue
            trial=block*len(plan['grid'])+index
            seeds=list(range(plan['search_seed_start']+3*trial,plan['search_seed_start']+3*trial+3))
            previous=original/'search'/f'trial_{trial:03d}'/'summary.json'
            if previous.exists():
                result=json.loads(previous.read_text())
                assert result['identity']['config']==cfg and result['seeds']==seeds
                location=previous
            else:
                folder=case/'screening'/f'trial_{trial:03d}'
                result=engine.rollout(policy,stats,row,cfg,seeds,folder,'continuation_screening3')
                location=folder/'summary.json'
            trials.append({'trial':trial,'config':cfg,'result':result,'source':str(location),'reused':previous.exists()})
            atomic_json(case/'screening_index.json',trials)
            if result['successes']==0:continue
            folder=case/'validation20'/f'config_{index:02d}'
            value=engine.rollout(policy,stats,row,cfg,plan['validation_seeds'],folder,'nonzero_goal_selection_validation20')
            for i in range(20):
                a=json.loads((original/'holdout20'/f'episode_{i:02d}.json').read_text())
                b=json.loads((folder/f'episode_{i:02d}.json').read_text())
                assert a['initial_state_sha256']==b['initial_state_sha256'], (i,'fixed validation initial states changed')
            entry={'config':cfg,'summary':str(folder/'summary.json'),'successes':value['successes'],
                'episodes':20,'success_rate':value['success_rate'],'screening_source':str(location)}
            validations.append(entry);atomic_json(case/'validation_index.json',validations)
            if value['successes']>0:
                winner=entry;break
            rejected.add(key)
        if winner:break
    if winner is None:
        engine.rollout(policy,stats,row,plan['grid'][0],plan['validation_seeds'][:3],case/'expert_positive_control','expert_control',expert=True)
    assert sha256(checkpoint)==row['checkpoint_sha256']
    audit_case(case,plan)
    result={'time':time.time(),'goal_achieved':winner is not None,'winner':winner,
        'previous_zero20':str(original/'holdout20/summary.json'),'validations':validations,
        'classification':'Selected on a fixed validation suite for nonzero success; not an independent test estimate.',
        'checkpoint_unchanged':True,'audit_passed':True,'finite_grid_exhausted':winner is None}
    atomic_json(case/'complete.json',result)
    lines=['# 非零目标的后续参数验证','','保留首轮0/20及所有后续轮次。对同一批固定20个seed验证，初始状态逐局对齐。分数参与了配置选择，因此属于搜索后验证成绩，不能当作独立测试成绩。','',
        '| 配置 | 20局成功数 | 结果路径 |','|---|---:|---|',
        f"| 首轮 {plan['previous_config']} | 0/20 | {original/'holdout20/summary.json'} |"]
    for r in validations:lines.append(f"| {r['config']} | {r['successes']}/20 | {r['summary']} |")
    lines += ['', '已取得真实非零验证结果。' if winner else '剩余预定开发网格已穷尽，目标未达；原生专家正对照与全部模型输出保留。','']
    (case/'summary.md').write_text('\n'.join(lines))


def discover():
    roots=[HERE.parent/'metaworld_act_zero_grid_20261020']
    roots+=list(HERE.parent.glob('metaworld_zero_success_followup_*_20261019'))
    found=[]
    for root in roots:
        p=root/'plan.json'
        if not p.exists():continue
        source_plan=json.loads(p.read_text())
        for row in source_plan['targets']:
            original=root/(row['method']+'_'+row['task']);cp=original/'complete.json'
            if not cp.exists():continue
            done=json.loads(cp.read_text())
            if done['state']!='holdout_complete' or done['holdout']['successes']!=0:continue
            assert done['holdout']['episodes']==20
            key=row['method']+'_'+row['task']+'_'+row['checkpoint_sha256'][:8]
            case=HERE/'cases'/key;case.mkdir(parents=True,exist_ok=True)
            if not (case/'plan.json').exists():
                lock=json.loads((original/'locked.json').read_text())
                atomic_json(case/'plan.json',{'created_at':time.time(),'row':row,'source_root':str(root),
                    'source_case':str(original),'source_complete_sha256':sha256(cp),
                    'previous_config':lock['config'],'grid':source_plan['grids'][row['method']],
                    'blocks':source_plan['blocks'],'search_seed_start':source_plan['search_seed_start'],
                    'validation_seeds':done['holdout']['seeds'],'engine_sha256':sha256(SOURCE),
                    'classification':'Adaptive nonzero-goal validation. Same fixed20 pool for all candidates; report every round. Original baseline remains separate.'})
            found.append({'key':key,'method':row['method'],'task':row['task'],'root':str(case),'source_root':str(root)})
    return found


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(Path(sys.argv[2]));return
    atomic_json(HERE/'supervisor_pid.json',{'pid':os.getpid(),'started':time.time()})
    active={};attempts={}
    while True:
        cases=discover()
        for key,item in list(active.items()):
            proc=item['proc'];case=Path(item['case']['root']);hb=case/'heartbeat.json'
            last=max(item['started'],hb.stat().st_mtime if hb.exists() else item['started'],item['log_path'].stat().st_mtime)
            if proc.poll() is None and time.time()-last>600:
                atomic_json(case/f'alert_attempt_{attempts[key]}.json',{'time':time.time(),'reason':'600s without progress'})
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill()
            if proc.poll() is not None:
                item['log'].close()
                if proc.returncode:
                    atomic_json(case/f'error_attempt_{attempts[key]}.json',{'time':time.time(),'exit_code':proc.returncode,'log':str(item['log_path'])})
                del active[key]
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(g):int(f) for g,f in (line.split(',') for line in out.splitlines())}
        occupied={v['gpu'] for v in active.values()}
        for entry in cases:
            key=entry['key'];case=Path(entry['root'])
            if key in active or (case/'complete.json').exists() or attempts.get(key,0)>=3 or len(active)>=2:continue
            options=[(f,g) for g,f in free.items() if g not in occupied and f>=6144]
            if not options:continue
            _,gpu=max(options);attempts[key]=attempts.get(key,0)+1
            p=case/f'attempt_{attempts[key]:02d}.log';log=p.open('a')
            proc=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--worker',str(case)],env=environment(gpu),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]={'proc':proc,'case':entry,'gpu':gpu,'log':log,'log_path':p,'started':time.time()};occupied.add(gpu)
        entries=[]
        for e in cases:
            result=json.loads((Path(e['root'])/'complete.json').read_text()) if (Path(e['root'])/'complete.json').exists() else None
            entries.append({**e,'state':'complete' if result else 'running' if e['key'] in active else 'infrastructure_error' if attempts.get(e['key'],0)>=3 else 'pending','result':result})
        atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':'monitoring_and_running','cases':entries})
        lines=['# 非零目标续接进度','','只续接首个命中配置20局仍0的组。后续20局参与参数选择，按“搜索后验证”报告，每一轮保留。','']
        for e in entries:lines.append(f"- {e['method']} / {e['task']}: {e['state']}；{e['root']}/summary.md")
        (HERE/'progress.md').write_text('\n'.join(lines)+'\n')
        baseline=list(csv.DictReader((BASE/'report/results.csv').open()))
        future=HERE.parent/'metaworld_zero_success_search_20261019/followup_status.json'
        future_done=future.exists() and json.loads(future.read_text())['state']=='complete'
        grid_done=(HERE.parent/'metaworld_act_zero_grid_20261020/finalized.json').exists()
        if len(baseline)==15 and future_done and grid_done and not active and all(e['state'] in ['complete','infrastructure_error'] for e in entries):break
        time.sleep(10)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(HERE/'supervisor_error.json',{'time':time.time(),'traceback':traceback.format_exc()})
        traceback.print_exc();raise
