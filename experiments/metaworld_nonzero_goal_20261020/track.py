"""Track the explicitly selected evaluation phase for all 15 method/task groups."""
import csv
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'metaworld_official_baselines_50demo_20261019'
SHORT=HERE.parent/'metaworld_act_short_interval_20261020'
GRID=HERE.parent/'metaworld_act_zero_grid_20261020'
OLD=HERE.parent/'metaworld_zero_success_search_20261019'
JOINT=HERE.parent/'metaworld_act_joint_seed_interval_20261020'
sys.path.insert(0,str(BASE))
from runtime import atomic_json,TASKS,METHODS


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def evaluate(row,folder,phase,audited):
    summary=read(folder/'summary.json')
    if summary is None or summary['episodes']!=20:
        row.update(state='等待20局结果',phase=phase)
        return
    row.update(phase=phase,source=str(folder/'summary.json'),successes=summary['successes'],episodes=20,
        success_rate=summary['success_rate'],config=json.dumps(summary['identity'].get('config',{}),sort_keys=True),
        seeds=json.dumps(summary.get('seeds',[])))
    row['state']='非零结果已完成' if audited and summary['successes']>0 else '最终20局为0，目标未达' if audited else '等待审计'


def snapshot():
    baseline=list(csv.DictReader((BASE/'report/results.csv').open()))
    future=read(OLD/'followup_status.json') or {'searches':[]}
    short_ok=bool((read(SHORT/'artifact_audit.json') or {}).get('passed'))
    grid_ok=bool((read(GRID/'artifact_audit.json') or {}).get('passed'))
    short_plan=read(SHORT/'plan.json')
    joint=read(JOINT/'summary.json')
    joint_ok=bool((read(JOINT/'audit.json') or {}).get('passed')) and bool(joint and joint.get('common_seed_batch') and joint.get('all_five_tasks_nonzero'))
    rows=[]
    for task in TASKS:
        for method in METHODS:
            old=next((r for r in baseline if r['task']==task and r['method']==method),None)
            row={'task':task,'method':method,'historical_successes':old['successes'] if old else '',
                'successes':'','episodes':'','success_rate':'','config':'','phase':'','state':'等待原始训练/评测','source':'','seeds':''}
            if method=='ACT':
                if joint and joint_ok:
                    item=next(r for r in joint['results'] if r['task']==task)
                    row.update(phase='ACT同seed同执行间隔联合搜索后验证',
                        source=str(Path(item['source_directory'])/'summary.json'),successes=item['successes'],
                        episodes=20,success_rate=item['success_rate'],
                        config=json.dumps({'action_horizon':joint['common_horizon'],'temporal_agg':False},sort_keys=True),
                        seeds=json.dumps(joint['seeds']),state='非零结果已完成')
                else:
                    row.update(phase='ACT五任务同seed同执行间隔联合验证',state='等待五项联合验证通过')
            elif old and int(old['episodes'])==20:
                if int(old['successes'])>0:
                    row.update(successes=int(old['successes']),episodes=20,success_rate=float(old['success_rate']),
                        phase='原始官方训练基线',state='非零结果已完成',source=str(Path(old['output'])/'final_rollouts/summary.json'),
                        config=json.dumps({'action_horizon':8,'num_inference_steps':100 if method=='DP' else 10},sort_keys=True),
                        seeds=json.dumps(list(range(2026101900,2026101920))))
                else:
                    entry=next((e for e in future['searches'] if e['method']==method and e['task']==task),None)
                    row.update(phase='原始0%后的有限网格',state='等待原始零成功组搜索')
                    if entry:
                        root=Path(entry['root']);folder=root/(method+'_'+task);done=read(folder/'complete.json')
                        if done and done['state']=='bounded_search_exhausted':
                            row.update(state='有限搜索穷尽，目标未达',source=str(folder/'complete.json'))
                        else:evaluate(row,folder/'holdout20','原始0%后的网格独立复评',bool((read(root/'artifact_audit.json') or {}).get('passed')))
            rows.append(row)
    hits=sum(r['state']=='非零结果已完成' for r in rows)
    unmet=sum('目标未达' in r['state'] for r in rows)
    pending=15-hits-unmet
    state='目标已达成' if hits==15 else '有限流程已结束但目标未全部达成' if pending==0 else '自动执行中'
    lines=['# 五任务 × 三方法：真实非零20局记录覆盖','',f'状态：**{state}**；非零完成 **{hits}/15**，待完成 {pending}，已确认目标未达 {unmet}。',
        '原始基线和所有0记录另行保留。下表按预先规定的阶段路由记录，不从各轮最终评测中挑最高分。不同阶段seed集不同，不将分数差全部归因于参数。', '',
        '| 任务 | 方法 | 历史基线 | 当前指定阶段20局 | 参数 | 状态 |','|---|---|---:|---:|---|---|']
    for r in rows:
        old=str(r['historical_successes'])+'/20' if r['historical_successes']!='' else '未完成'
        score=f"{r['successes']}/20" if r['episodes']==20 else '未完成'
        lines.append(f"| {r['task']} | {r['method']} | {old} | {score} | {r['config']} | {r['state']} |")
    lines += ['', 'ACT五项必须使用完全相同的20个rollout seed及同一个action_horizon；仅联合验证全部通过且审计通过后计为完成。',
        '当前ACT固定seeds=4026101900–4026101919，训练seed=0；候选共同间隔100、32、16、8、4、2、1。禁止混用不同seed或逐任务不同间隔。',
        'DP/SimpleDP3继续原始官方训练与评测，原始0组接入既有有限网格。',
        '各行完整阶段、配置、seed与来源见results.csv；ACT进度见../metaworld_act_joint_seed_interval_20261020/progress.md。','']
    (HERE/'progress.md').write_text('\n'.join(lines))
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'results.csv').write_text(stream.getvalue())
    atomic_json(HERE/'status.json',{'time':time.time(),'pid':os.getpid(),'state':state,
        'nonzero_complete':hits,'pending':pending,'unmet':unmet,'total':15,'rows':rows})
    return pending==0


atomic_json(HERE/'tracker_pid.json',{'pid':os.getpid(),'started':time.time()})
while True:
    try:
        if snapshot():break
    except Exception:
        atomic_json(HERE/'latest_read_error.json',{'time':time.time(),'traceback':traceback.format_exc()})
    time.sleep(20)
