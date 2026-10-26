# 五任务 × 三方法：真实非零20局记录覆盖

状态：**自动执行中**；非零完成 **10/15**，待完成 5，已确认目标未达 0。
原始基线和所有0记录另行保留。下表按预先规定的阶段路由记录，不从各轮最终评测中挑最高分。不同阶段seed集不同，不将分数差全部归因于参数。

| 任务 | 方法 | 历史基线 | 当前指定阶段20局 | 参数 | 状态 |
|---|---|---:|---:|---|---|
| shelf-place | DP | 未完成 | 未完成 |  | 等待原始训练/评测 |
| shelf-place | ACT | 1/20 | 3/20 | {"action_horizon": 100, "temporal_agg": false} | 非零结果已完成 |
| shelf-place | SimpleDP3 | 16/20 | 16/20 | {"action_horizon": 8, "num_inference_steps": 10} | 非零结果已完成 |
| pick-place-wall | DP | 未完成 | 未完成 |  | 等待原始训练/评测 |
| pick-place-wall | ACT | 0/20 | 5/20 | {"action_horizon": 100, "temporal_agg": false} | 非零结果已完成 |
| pick-place-wall | SimpleDP3 | 19/20 | 19/20 | {"action_horizon": 8, "num_inference_steps": 10} | 非零结果已完成 |
| pick-place | DP | 未完成 | 未完成 |  | 等待原始训练/评测 |
| pick-place | ACT | 3/20 | 1/20 | {"action_horizon": 100, "temporal_agg": false} | 非零结果已完成 |
| pick-place | SimpleDP3 | 19/20 | 19/20 | {"action_horizon": 8, "num_inference_steps": 10} | 非零结果已完成 |
| stick-push | DP | 未完成 | 未完成 |  | 等待原始训练/评测 |
| stick-push | ACT | 20/20 | 20/20 | {"action_horizon": 100, "temporal_agg": false} | 非零结果已完成 |
| stick-push | SimpleDP3 | 20/20 | 20/20 | {"action_horizon": 8, "num_inference_steps": 10} | 非零结果已完成 |
| door-open | DP | 未完成 | 未完成 |  | 等待原始训练/评测 |
| door-open | ACT | 17/20 | 18/20 | {"action_horizon": 100, "temporal_agg": false} | 非零结果已完成 |
| door-open | SimpleDP3 | 20/20 | 20/20 | {"action_horizon": 8, "num_inference_steps": 10} | 非零结果已完成 |

ACT五项必须使用完全相同的20个rollout seed及同一个action_horizon；仅联合验证全部通过且审计通过后计为完成。
当前ACT固定seeds=4026101900–4026101919，训练seed=0；候选共同间隔100、32、16、8、4、2、1。禁止混用不同seed或逐任务不同间隔。
DP/SimpleDP3继续原始官方训练与评测，原始0组接入既有有限网格。
各行完整阶段、配置、seed与来源见results.csv；ACT进度见../metaworld_act_joint_seed_interval_20261020/progress.md。
