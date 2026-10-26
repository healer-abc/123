# 本轮固定配置的零成功记录统计

这里只评测了执行间隔1、时间聚合关闭的固定配置，未声称穷尽所有推理参数。没有因最终20局失败而改参数或换seed。

## shelf-place

固定配置0/20；历史原始配置为1/20，但两轮seed不同。
动作非有限数分量：0；原生环境边界外动作分量平均比例：0.266437。
每步原生info统计（至少出现一次的episode数）：{'near_object': 4, 'grasp_success': 1}。
回报范围：0.000000–3.375247；最小记录obj_to_target：0.35125842229412335。
完整官方info、视频和原始模型输出：/home/zhsu/jyluo/experiments/metaworld_act_short_interval_20261020/runs/shelf-place/rollouts20

## pick-place

固定配置0/20；历史原始配置为3/20，但两轮seed不同。
动作非有限数分量：0；原生环境边界外动作分量平均比例：0.098750。
每步原生info统计（至少出现一次的episode数）：{'near_object': 4, 'grasp_success': 1}。
回报范围：2.694142–694.489524；最小记录obj_to_target：0.09081014018924531。
完整官方info、视频和原始模型输出：/home/zhsu/jyluo/experiments/metaworld_act_short_interval_20261020/runs/pick-place/rollouts20

## 判定边界

100局数据审计已核对：checkpoint与归一化统计未变，每步执行动作可由保存的模型输出重建，成功率由原生info.success逐步重算。未降低环境成功阈值，未加入专家动作。
这些检查排除了本轮记录中的NaN/Inf和动作还原错误；不能证明模型在所有输入上正确，也不能仅凭0/20认定真实成功概率为零。
单独缩短执行间隔不能保证所有任务非零。若以后继续调参，应另用开发集选择配置，再固定新评测集，保留本轮结果。
