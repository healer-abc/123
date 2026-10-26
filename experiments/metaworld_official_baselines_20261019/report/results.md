# MetaWorld 官方方法 baseline

每格为成功次数 / 20（成功率）。未完成实验记为“待完成”；训练中的评测不计入本表。

| 场景 | DP | ACT | SimpleDP3 |
|---|---:|---:|---:|
| shelf-place | 待完成 | 0/20（0%） | 待完成 |
| pick-place-wall | 待完成 | 0/20（0%） | 待完成 |
| pick-place | 待完成 | 待完成 | 待完成 |
| stick-push | 待完成 | 待完成 | 待完成 |
| door-open | 待完成 | 待完成 | 待完成 |

训练与接口适配、固定 checkpoint 选择及单训练 seed 限制见 ../PROTOCOL.md。
最终评测用独立固定 seeds；成功定义为官方 info.success 在 200 步内曾为真。
本表不会用开发评测的最高成功率或截图中的历史结果补值。
