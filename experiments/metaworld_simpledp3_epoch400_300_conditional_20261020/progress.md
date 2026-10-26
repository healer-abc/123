# SimpleDP3：400轮筛选与条件评测

先测400轮pick-place20局；成功少于18局→五任务均用400轮，否则五任务均用300轮。
DDIM10、观测2、动作间隔8、训练seed0、统一评测seed2026102000–2026102019，每局200步。
按观测结果选轮次的探索实验；保留全部筛选与原500/1000轮结果，不替换固定基线。

状态：complete；选择：300。

| 阶段 | 任务 | 训练轮数 | 当前成功/已测 | 状态 |
|---|---|---:|---:|---|
| screen400_pick-place | pick-place | 400 | 19/20 | complete |
| epoch300_shelf-place | shelf-place | 300 | 13/20 | complete |
| epoch300_pick-place-wall | pick-place-wall | 300 | 19/20 | complete |
| epoch300_pick-place | pick-place | 300 | 19/20 | complete |
| epoch300_stick-push | stick-push | 300 | 20/20 | complete |
| epoch300_door-open | door-open | 300 | 20/20 | complete |
