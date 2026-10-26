# SimpleDP3：DDIM10与DDIM5配对结果

同模型、同20个seed、同执行间隔8、同观测2；仅去噪步数10→5。原始基线保留，此表为事后敏感性实验。

| 任务 | DDIM10 | DDIM5 | 差值（百分点） |
|---|---:|---:|---:|
| shelf-place | 14/20（70%） | 14/20（70%） | +0 |
| pick-place-wall | 19/20（95%） | 19/20（95%） | +0 |
| pick-place | 20/20（100%） | 20/20（100%） | +0 |
| stick-push | 20/20（100%） | 20/20（100%） | +0 |
| door-open | 20/20（100%） | 20/20（100%） | +0 |

五项平均：DDIM10=93%，DDIM5=93%。100局审计通过。

参数差异见parameters.diff；成功视频/轨迹路径见successful_episodes.json；逐seed配对结果见paired_episodes。
