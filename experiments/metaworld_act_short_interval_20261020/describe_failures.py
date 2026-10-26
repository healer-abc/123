"""Describe zero-success groups under the single fixed protocol; no further selection."""
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
while not (HERE/'complete.json').exists():
    time.sleep(10)
complete=json.loads((HERE/'complete.json').read_text())
assert complete['audit_passed']
lines=['# 本轮固定配置的零成功记录统计','','这里只评测了执行间隔1、时间聚合关闭的固定配置，未声称穷尽所有推理参数。没有因最终20局失败而改参数或换seed。','']
details=[]
for row in complete['results']:
    if row['short_interval_successes']!=0:
        continue
    folder=Path(row['rollout_directory'])
    records=[json.loads(p.read_text()) for p in sorted(folder.glob('episode_*.json'))]
    assert len(records)==20 and not any(r['success'] for r in records)
    info_keys=set().union(*(i.keys() for r in records for i in r['infos']))
    counters={k:sum(any(bool(i.get(k,0)) for i in r['infos']) for r in records)
              for k in ['near_object','grasp_success'] if k in info_keys}
    distances=[float(i['obj_to_target']) for r in records for i in r['infos'] if 'obj_to_target' in i]
    detail={'task':row['task'],'episodes':20,'successes':0,'historical_baseline_successes':row['historical_successes'],
        'any_step_diagnostic_counts':counters,
        'nonfinite_action_components':sum(r['nonfinite_actions'] for r in records),
        'mean_fraction_action_components_outside_original_env_bounds':sum(r['fraction_components_outside_env_bounds'] for r in records)/20,
        'return_min':min(r['return'] for r in records),'return_max':max(r['return'] for r in records),
        'minimum_recorded_obj_to_target':min(distances) if distances else None,
        'checkpoint_sha256':row['checkpoint_sha256'],'records':str(folder)}
    details.append(detail)
    lines += [f"## {row['task']}",'',f"固定配置0/20；历史原始配置为{row['historical_successes']}/20，但两轮seed不同。",
        f"动作非有限数分量：{detail['nonfinite_action_components']}；原生环境边界外动作分量平均比例：{detail['mean_fraction_action_components_outside_original_env_bounds']:.6f}。",
        f"每步原生info统计（至少出现一次的episode数）：{counters}。",
        f"回报范围：{detail['return_min']:.6f}–{detail['return_max']:.6f}；最小记录obj_to_target：{detail['minimum_recorded_obj_to_target']}。",
        f"完整官方info、视频和原始模型输出：{folder}",'']
lines += ['## 判定边界','','100局数据审计已核对：checkpoint与归一化统计未变，每步执行动作可由保存的模型输出重建，成功率由原生info.success逐步重算。未降低环境成功阈值，未加入专家动作。',
    '这些检查排除了本轮记录中的NaN/Inf和动作还原错误；不能证明模型在所有输入上正确，也不能仅凭0/20认定真实成功概率为零。',
    '单独缩短执行间隔不能保证所有任务非零。若以后继续调参，应另用开发集选择配置，再固定新评测集，保留本轮结果。','']
(HERE/'fixed_protocol_zero_groups.json').write_text(json.dumps(details,indent=2,ensure_ascii=False))
(HERE/'fixed_protocol_zero_groups.md').write_text('\n'.join(lines))
