# SimpleDP3：epoch1000与epoch500配对结果

相同20个seed、执行间隔8、观测2、去噪10步。500轮模型来自同配置重放；学习率计划仍为1000轮。原始1000轮基线保留。

| 任务 | epoch1000 | epoch500 | 差值（百分点） |
|---|---:|---:|---:|
| shelf-place | 14/20（70%） | 13/20（65%） | -5 |
| pick-place-wall | 19/20（95%） | 19/20（95%） | +0 |
| pick-place | 20/20（100%） | 19/20（95%） | -5 |
| stick-push | 20/20（100%） | 20/20（100%） | +0 |
| door-open | 20/20（100%） | 20/20（100%） | +0 |

五项平均：epoch1000=93%，epoch500=91%。100局审计通过。

参数差异见parameters.diff；成功视频/轨迹路径见successful_episodes.json；逐seed配对结果见paired_episodes。
