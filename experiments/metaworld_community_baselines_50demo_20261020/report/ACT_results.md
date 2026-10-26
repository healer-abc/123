# ACT 新一轮固定配置结果

chunk10，temporal_agg=true，k=0.01，KL10；训练seed0；每任务2000epochs/10000更新。
五任务共用rollout seeds2026102000–2026102019；各20局、每局200步；没有按正式评测结果挑选seed或参数。

| 任务 | 成功数 | 成功率 |
|---|---:|---:|
| shelf-place | 11/20 | 55% |
| pick-place-wall | 19/20 | 95% |
| pick-place | 14/20 | 70% |
| stick-push | 20/20 | 100% |
| door-open | 19/20 | 95% |

五项平均：83/100 = 83%。
100局原生成功判定、共同seed、checkpoint哈希及逐步时间聚合动作重建审计通过。

完整成功视频/轨迹路径：ACT_successful_episodes.json；全体记录哈希：ACT_source_manifest.json。
