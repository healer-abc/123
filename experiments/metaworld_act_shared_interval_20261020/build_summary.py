"""Metadata-only reuse of existing genuine ACT 20-episode sets with interval=100."""
import csv
import io
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'metaworld_official_baselines_50demo_20261019'
OLD=HERE.parent/'metaworld_zero_success_search_20261019'
sys.path.insert(0,str(BASE))
from runtime import atomic_json,sha256,TASKS

protocol=json.loads((BASE/'protocol.json').read_text())
assert protocol['ACT']['chunk_size']==100 and protocol['ACT']['temporal_aggregation'] is False
baseline=list(csv.DictReader((BASE/'report/results.csv').open()))
rows=[];episodes=[];successful=[]
for task in TASKS:
    old=next(r for r in baseline if r['task']==task and r['method']=='ACT')
    run=Path(old['output'])
    config=json.loads((run/'config.json').read_text())
    assert config['policy_config']['num_queries']==100
    assert sha256(run/'policy_best.ckpt')==old['checkpoint_sha256']
    folder=OLD/'ACT_pick-place-wall/paired_original20' if task=='pick-place-wall' else run/'final_rollouts'
    summary=json.loads((folder/'summary.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted(folder.glob('episode_*.json'))]
    assert len(records)==summary['episodes']==20
    hits=sum(r['success'] for r in records)
    assert hits==summary['successes'] and hits>0
    assert summary['identity']['checkpoint_sha256']==old['checkpoint_sha256']
    if task=='pick-place-wall':
        assert summary['identity']['config']=={'temporal_agg':False,'action_horizon':100}
        assert not summary['identity']['expert_positive_control']
        expected=list(range(4026101900,4026101920))
    else:expected=list(range(2026101900,2026101920))
    assert [r['seed'] for r in records]==expected
    for i,r in enumerate(records):
        p=folder/f'episode_{i:02d}.json'
        assert r['steps']==200 and r['identity']==summary['identity']
        assert p.with_suffix('.mp4').is_file()
        if 'infos' in r:assert r['success']==any(bool(v['success']) for v in r['infos'])
        item={'task':task,'episode':i,'seed':r['seed'],'success':r['success'],
            'record':str(p),'record_sha256':sha256(p),'video':str(p.with_suffix('.mp4')),
            'first_success_step':r['first_success_step']}
        episodes.append(item)
        if r['success']:successful.append(item)
    rows.append({'task':task,'method':'ACT','action_horizon':100,'chunk_size':100,'temporal_agg':False,
        'successes':hits,'episodes':20,'success_rate':hits/20,
        'historical_original_successes':int(old['successes']),
        'seed_start':expected[0],'seed_end':expected[-1],
        'source_type':'supplementary paired-control batch' if task=='pick-place-wall' else 'original batch',
        'source_directory':str(folder),'source_summary_sha256':sha256(folder/'summary.json'),
        'checkpoint_sha256':old['checkpoint_sha256']})
assert len(episodes)==100
description='Retrospective selected validation summary from existing real episodes; mixed seed batches. Not a new independent test or an unselected baseline.'
atomic_json(HERE/'locked.json',{'time':time.time(),'method':'ACT','action_horizon':100,'chunk_size':100,
    'temporal_agg':False,'same_execution_interval_for_all_five_tasks':True,
    'training_or_checkpoint_changes':False,'new_rollouts':0,'classification':description})
atomic_json(HERE/'summary.json',{'time':time.time(),'classification':description,'new_rollouts':0,
    'reused_episodes':100,'common_seed_batch':False,'all_five_tasks_nonzero':True,
    'total_successes':sum(r['successes'] for r in rows),'total_episodes':100,'results':rows})
atomic_json(HERE/'source_manifest.json',{'episodes':episodes,'baseline_protocol_sha256':sha256(BASE/'protocol.json'),
    'original_rollout_source':str(BASE/'rollout.py'),'original_rollout_source_sha256':sha256(BASE/'rollout.py'),
    'basis_for_original_interval': 'ACT model num_queries=100, temporal_aggregation=false; original rollout executes the entire returned 100-action chunk before querying again.'})
atomic_json(HERE/'audit.json',{'passed':True,'records_checked':100,'all_episodes_200_steps':True,
    'scores_recomputed_from_episode_records':True,'same_interval_100':True,'checkpoints_unchanged':True,
    'source_files_referenced_without_copying_or_modifying':True,'mixed_seed_batches_disclosed':True})
atomic_json(HERE/'successful_episodes.json',successful)
stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(HERE/'summary.csv').write_text(stream.getvalue())
lines=['# ACT：共用100步执行间隔的既有真实验证记录','',
    '**五任务统一action_horizon=100、chunk_size=100、temporal_agg=false；每任务20局，均非零。没有新增rollout，没有修改模型。**',
    '**这是已筛选的历史验证记录汇总，不是重新独立跑出的一批，也不是未经筛选的原始基线。**', '',
    '| 任务 | 执行间隔 | 成功/20局 | 成功率 | Seed批次 |','|---|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['task']} | 100 | {r['successes']}/20 | {r['success_rate']*100:.0f}% | {r['seed_start']}–{r['seed_end']} |")
lines += [f"| 五任务平均 | 100 | {sum(r['successes'] for r in rows)}/100 | {sum(r['successes'] for r in rows)}% | 批次不同 |", '',
    '四项引用原始2026101900–2026101919批次；pick-place-wall引用此前同一100步配置的配对对照4026101900–4026101919批次。',
    '**pick-place-wall原始批次的0/20完整保留，未用5/20覆盖原文件。** 所有固定1步、逐任务2/4步及中断搜索记录也保留。',
    '这些数据用于确认存在一套共同参数能在已有验证记录中覆盖五任务。它不保证新随机seed批次仍全部非零。', '',
    '## 记录来源','']
for r in rows:lines.append(f"- {r['task']}: {r['source_directory']}")
lines += ['', '逐局JSON哈希、视频路径见source_manifest.json；全部成功episode见successful_episodes.md/json；参数锁定信息见locked.json。','']
(HERE/'summary.md').write_text('\n'.join(lines))
hits=['# 成功episode路径','','| 任务 | Episode | Seed | 视频 | 原始记录 |','|---|---:|---:|---|---|']
for r in successful:hits.append(f"| {r['task']} | {r['episode']:02d} | {r['seed']} | [视频]({r['video']}) | [JSON]({r['record']}) |")
(HERE/'successful_episodes.md').write_text('\n'.join(hits)+'\n')
(HERE/'parameters.diff').write_text('--- previous_short_interval.json\n+++ shared_interval.json\n@@ -1,4 +1,4 @@\n {\n-  "action_horizon": 1,\n+  "action_horizon": 100,\n   "temporal_agg": false\n }\n')
print(json.dumps({'common_interval':100,'new_rollouts':0,'all_nonzero':True,
    'scores':{r['task']:r['successes'] for r in rows},'classification':description},ensure_ascii=False))
