"""Wait for all evaluations and the replay check, audit, and write the delivery report."""
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
PYTHON = '/home/zhsu/dp3_ablation_v52/env/bin/python'
required = [HERE/'paired_control_complete.json', HERE/'baseline_parity_audit.json',
            HERE/'ACT_pick-place-wall/complete.json']
while not all(p.exists() for p in required):
    time.sleep(10)
parity = json.loads((HERE/'baseline_parity_audit.json').read_text())
assert parity['passed'], parity
subprocess.run([PYTHON,str(HERE/'audit.py')],check=True)
target = HERE/'ACT_pick-place-wall'
result = json.loads((target/'complete.json').read_text())
pair = json.loads((target/'paired_comparison.json').read_text())
audit = json.loads((HERE/'artifact_audit.json').read_text())
lock = json.loads((target/'locked.json').read_text())
holdout, original = result['holdout'], pair['original']
lines = ['# 最终交付：ACT pick-place-wall', '',
    '| 评测 | 成功/局数 | 成功率 |', '|---|---:|---:|', '| 历史原始基线 | 0/20 | 0% |',
    f"| 新seed上的原始配置配对对照 | {original['successes']}/20 | {original['success_rate']*100:.0f}% |",
    f"| 锁定配置独立20局 | {holdout['successes']}/20 | {holdout['success_rate']*100:.0f}% |", '',
    '仅将ACT执行间隔100改为1；temporal_agg保持false，预测块长保持100。没有重新训练或修改模型。',
    '50条示教预算、40训练/10验证；使用同一个原有最优验证checkpoint。',
    f"Checkpoint SHA256：`{lock['checkpoint_sha256']}`。", '',
    '## 筛选记录', '',
    '先登记有限空间。时间聚合开启组0/3；关闭聚合、每步重查询组1/3，首次命中后立即锁定。',
    '筛选共6局，全部记录保留。命中seed3026101905，第96步首次成功，官方success=1、obj_to_target=0.0659001627（阈值0.07）。',
    '独立20局固定seed4026101900–4026101919，未用于挑选参数。原始配置的配对对照使用同一批seed，20对初始状态哈希逐局相同。',
    '结果属于搜索后评测，不能替换或冒充未调参官方基线。', '',
    '## 验证与错误修复', '',
    f"审计通过，共检查{audit['episodes_checked']}局，所有200步记录均从官方逐步info重算成功判定，动作和预测均为有限值。",
    f"其中{audit['episodes_with_actions_reconstructed_from_model_predictions']}局未聚合ACT动作已从保存的模型预测块和原始归一化统计逐步重建核对。",
    '原始配置回放同一历史seed，初始状态、成功判定、步数和回报一致。',
    'CPU固定示教观测诊断结果见offline_model_diagnostic.json；未执行优化器步骤。',
    '补充配对对照曾因动态库路径在Python启动后设置而触发MuJoCo重编译，继而缺少X11头文件。已改为携带完整原始环境重启进程；不改环境物理或成功阈值。失败日志保留，主重评测未受影响。', '',
    '## 交付文件', '',
    '- 全部方法的汇总表：../summary.md、../summary.csv。未完成训练组明确标记未完成。',
    '- 参数位置：../PARAMETERS.md；参数差异：parameters.diff、parameter_diff.json。',
    '- 完整筛选记录：../search_trials.csv；锁定信息：locked.json。',
    '- 配对对照：paired_comparison.json。',
    '- 数据审计：../artifact_audit.json；原始配置回放审计：../baseline_parity_audit.json。', '',
    '## 独立20局中的全部成功episode', '', '| Episode | Seed | 首次成功步 | 视频 | 原始轨迹 |', '|---|---:|---:|---|---|']
for p in sorted((target/'holdout20').glob('episode_*.json')):
    r = json.loads(p.read_text())
    if r['success']:
        lines.append(f"| {r['episode']:02d} | {r['seed']} | {r['first_success_step']} | [{p.stem}.mp4]({r['video']}) | [{p.stem}.npz]({r['trajectory']}) |")
lines += ['', 'DP仍在等待训练资源，SimpleDP3正在训练；原始训练看门狗继续运行。后续新增0/20组由独立watch_future.py监控接续处理，状态见../followup_index.md。',
          'stick-pull-v2不在现有五任务中，本次没有对应基线checkpoint，未冒用stick-push替代。', '']
(target/'FINAL_REPORT.md').write_text('\n'.join(lines))
(HERE/'finalized.json').write_text(json.dumps({'time':time.time(),'passed':True,
    'holdout_successes':holdout['successes'],'holdout_episodes':20,
    'paired_original_successes':original['successes'],'report':str(target/'FINAL_REPORT.md')},indent=2))
print((HERE/'finalized.json').read_text(),flush=True)
