> 用户最新指令：旧DP训练进程及看门狗已结束；下表为结束前的历史进度，不再自动更新。已有权重和结果保留。新DP五任务从头训练500轮，并保存300/400/500轮及续训状态；500轮后等待用户指令才继续训练。
> 新进度：[DP500](../metaworld_dp_epoch500_20261020/progress.md)。归档及终止记录见新目录的preserved_baseline。

# 新一轮 MetaWorld 规范基线进度

训练seed统一0；50条示教预算=40训练+10验证；每组20局，测试seed统一2026102000–2026102019。
ACT chunk10/聚合开启/KL10，2000epochs；DP/SimpleDP3 DDIM10/sample，1000epochs、batch128。
所有模型从头训练；配置和测试seed冻结；0%如实报告，不触发调参搜索。

最终评测已完成：11/15组。GPU空闲MiB：{0: 19317, 1: 9892}。

| 任务 | 方法 | 阶段 | epoch/总数 | 已用小时 | 训练预计剩余小时 |
|---|---|---|---:|---:|---:|
| shelf-place | DP | complete | 999/1000 | 5.98 | 0.0 |
| shelf-place | ACT | complete | 2000/2000 | — | 测量中 |
| shelf-place | SimpleDP3 | complete | 999/1000 | 5.98 | 0.0 |
| pick-place-wall | DP | running | 957/1000 | 5.23 | 0.2 |
| pick-place-wall | ACT | complete | 2000/2000 | 5.98 | 0.0 |
| pick-place-wall | SimpleDP3 | complete | 999/1000 | 4.57 | 0.0 |
| pick-place | DP | running | 187/1000 | 0.88 | 3.8 |
| pick-place | ACT | complete | 2000/2000 | 5.41 | 0.0 |
| pick-place | SimpleDP3 | complete | 999/1000 | 3.57 | 0.0 |
| stick-push | DP | running | 38/1000 | 0.33 | 8.4 |
| stick-push | ACT | complete | 2000/2000 | 4.80 | 0.0 |
| stick-push | SimpleDP3 | complete | 999/1000 | 3.18 | 0.0 |
| door-open | DP | pending | —/1000 | — | 测量中 |
| door-open | ACT | complete | 2000/2000 | 4.18 | 0.0 |
| door-open | SimpleDP3 | complete | 999/1000 | 2.00 | 0.0 |

训练剩余时间按当前平均吞吐估计，排队与最终评测另计。共享GPU负载变化会影响估计。

结果表见report/results.md；协议见PROTOCOL.md；异常见logs/latest_alert.json。
