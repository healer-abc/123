# SimpleDP3：新训200轮，DDIM5，horizon16

五任务从头训练seed0，50示教（40训练/10验证），batch128；观测2、执行间隔8。
沿用1000轮学习率计划，在完成50/100/200轮各存独立快照，本次最多训练200轮。
pick-place按200→100→50轮依次完整测20局，首个低于18/20的轮次用于五任务；三项均≥18/20则报告无满足阈值快照。
按结果选权重的探索评测，保留所有筛选和原基线。

状态：screening_exhausted；筛选状态：exhausted；选中轮次：None。

| 类型 | 任务 | 当前epoch/权重轮次 | 进度 | 状态 |
|---|---|---:|---|---|
| train | shelf-place | 199.0 | 目标200轮 | complete |
| train | pick-place-wall | 199.0 | 目标200轮 | complete |
| train | pick-place | 199.0 | 目标200轮 | complete |
| train | stick-push | 199.0 | 目标200轮 | complete |
| train | door-open | 199.0 | 目标200轮 | complete |
| eval | pick-place | 200 | 19/20成功 | complete |
| eval | pick-place | 100 | 18/20成功 | complete |
| eval | pick-place | 50 | 18/20成功 | complete |

筛选历史：
- 200轮pick-place：19/20（95%）
- 100轮pick-place：18/20（90%）
- 50轮pick-place：18/20（90%）
