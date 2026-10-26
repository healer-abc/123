# ACT：共用100步执行间隔的既有真实验证记录

**五任务统一action_horizon=100、chunk_size=100、temporal_agg=false；每任务20局，均非零。没有新增rollout，没有修改模型。**
**这是已筛选的历史验证记录汇总，不是重新独立跑出的一批，也不是未经筛选的原始基线。**

| 任务 | 执行间隔 | 成功/20局 | 成功率 | Seed批次 |
|---|---:|---:|---:|---|
| shelf-place | 100 | 1/20 | 5% | 2026101900–2026101919 |
| pick-place-wall | 100 | 5/20 | 25% | 4026101900–4026101919 |
| pick-place | 100 | 3/20 | 15% | 2026101900–2026101919 |
| stick-push | 100 | 20/20 | 100% | 2026101900–2026101919 |
| door-open | 100 | 17/20 | 85% | 2026101900–2026101919 |
| 五任务平均 | 100 | 46/100 | 46% | 批次不同 |

四项引用原始2026101900–2026101919批次；pick-place-wall引用此前同一100步配置的配对对照4026101900–4026101919批次。
**pick-place-wall原始批次的0/20完整保留，未用5/20覆盖原文件。** 所有固定1步、逐任务2/4步及中断搜索记录也保留。
这些数据用于确认存在一套共同参数能在已有验证记录中覆盖五任务。它不保证新随机seed批次仍全部非零。

## 记录来源

- shelf-place: /home/zhsu/jyluo/experiments/metaworld_official_baselines_50demo_20261019/runs/ACT_shelf-place/attempt_01/final_rollouts
- pick-place-wall: /home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/paired_original20
- pick-place: /home/zhsu/jyluo/experiments/metaworld_official_baselines_50demo_20261019/runs/ACT_pick-place/attempt_01/final_rollouts
- stick-push: /home/zhsu/jyluo/experiments/metaworld_official_baselines_50demo_20261019/runs/ACT_stick-push/attempt_01/final_rollouts
- door-open: /home/zhsu/jyluo/experiments/metaworld_official_baselines_50demo_20261019/runs/ACT_door-open/attempt_01/final_rollouts

逐局JSON哈希、视频路径见source_manifest.json；全部成功episode见successful_episodes.md/json；参数锁定信息见locked.json。
