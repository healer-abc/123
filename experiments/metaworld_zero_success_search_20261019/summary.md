# MetaWorld 推理参数搜索与独立重评测

原始基线保留。搜索命中后的 20 局使用预先固定的独立 seed，不将筛选用 3 局混入成功率。
这些结果属于参数与 seed 筛选后的评测，不能当作未调参的官方基线。训练和 checkpoint 未改动。

| 任务 | 方法 | 原始基线 | 筛选 | 独立20局 | 状态 |
|---|---|---:|---:|---:|---|
| shelf-place | DP | 未完成 | — | — | 基线未完成 |
| shelf-place | ACT | 1/20 (5%) | — | — | 未调参 |
| shelf-place | SimpleDP3 | 未完成 | — | — | 基线未完成 |
| pick-place-wall | DP | 未完成 | — | — | 基线未完成 |
| pick-place-wall | ACT | 0/20 (0%) | 1/3 | 8/20 (40%) | 搜索后评测完成 |
| pick-place-wall | SimpleDP3 | 未完成 | — | — | 基线未完成 |
| pick-place | DP | 未完成 | — | — | 基线未完成 |
| pick-place | ACT | 3/20 (15%) | — | — | 未调参 |
| pick-place | SimpleDP3 | 未完成 | — | — | 基线未完成 |
| stick-push | DP | 未完成 | — | — | 基线未完成 |
| stick-push | ACT | 20/20 (100%) | — | — | 未调参 |
| stick-push | SimpleDP3 | 未完成 | — | — | 基线未完成 |
| door-open | DP | 未完成 | — | — | 基线未完成 |
| door-open | ACT | 17/20 (85%) | — | — | 未调参 |
| door-open | SimpleDP3 | 未完成 | — | — | 基线未完成 |

## 范围与记录

stick-pull-v2 is not in the existing five-task campaign; no matching baseline checkpoint is available in this campaign.

每个方法预先固定 3 个 seed 轮次，每个参数组合 3 局；首个命中即锁定。
ACT 使用原始 100 步预测块，action_horizon 仅控制执行多少步后重新查询。时间聚合严格采用官方 k=0.01 和原有权重方向。
DP 的 action_horizon 对应 policy.n_action_steps；训练预测 horizon=16 保持不变。
所有尝试的 episode JSON、MP4、动作/模型输出 NPZ 均保留在各任务 search/ 中。
独立20局 seeds: 4026101900–4026101919。成功定义与基线一致：200步内任意 official info.success。

### ACT / pick-place-wall 成功 episode

/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_01.mp4
/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_03.mp4
/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_07.mp4
/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_08.mp4
/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_10.mp4
/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_13.mp4
/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_15.mp4
/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_16.mp4

## 同checkpoint、同seed配对对照

| 任务 | 方法 | 原始执行100步 | 锁定执行1步 | 差值 |
|---|---|---:|---:|---:|
| pick-place-wall | ACT | 5/20 (25%) | 8/20 (40%) | +15个百分点 |

两组20个初始状态哈希逐局相同。配对对照不参与参数筛选。历史基线仍保留原始seed上的0/20。
