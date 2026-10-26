# ACT 同seed、同执行间隔五任务结果

所有任务训练seed=0；rollout seeds=4026101900–4026101919，顺序完全相同。
共同action_horizon=100，chunk_size=100，temporal_agg=false。每任务20局，每局200步。
这是固定seed集合上的共同参数搜索后验证；原始基线和失败记录保留。

| 任务 | 成功数 | 成功率 |
|---|---:|---:|
| shelf-place | 3/20 | 15% |
| pick-place-wall | 5/20 | 25% |
| pick-place | 1/20 | 5% |
| stick-push | 20/20 | 100% |
| door-open | 18/20 | 90% |

五项平均：47%。100局审计通过。

完整成功episode路径见successful_episodes.json；原始记录及哈希见source_manifest.json。
