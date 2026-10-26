# SimpleDP3：固定50轮，五任务评测

用户明确指定50轮；DDIM5、horizon16、obs2、action8；各20局，统一种子2026102000–2026102019。
pick-place原筛选20局复用（18/20=90%），其他四项新测；旧筛选未满足<90%条件的结论保留。
本轮是事后固定checkpoint的探索结果，不替换原基线。

状态：complete。

| 任务 | 权重轮次 | 当前成功/已测 | 状态 |
|---|---:|---:|---|
| shelf-place | 50 | 8/20 | complete |
| pick-place-wall | 50 | 17/20 | complete |
| pick-place | 50 | 18/20 | complete |
| stick-push | 50 | 16/20 | complete |
| door-open | 50 | 20/20 | complete |
