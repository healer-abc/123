# ACT 五任务短执行间隔补评测

用户明确授权现在补跑。只改变推理执行间隔，不训练，不修改checkpoint或旧结果。

- 固定配置：`action_horizon=1`，`temporal_agg=false`；模型预测块长仍为100。
- 每任务20局，最长200步，任意一步原生`info.success`为真即记成功。
- 五任务统一seeds：4026101900–4026101919。
- 新增任务：shelf-place、pick-place、stick-push、door-open，共80局。
- pick-place-wall原20局直接引用：`../metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20`，8/20。没有复制为新采样，没有重复计入新增80局。
- 50条示教预算（40训练/10验证）；同原ACT最佳验证checkpoint与原始归一化统计。

## 运行与监控

`run_eval.py`为独立监督程序，进程身份见`supervisor_pid.json`，状态见`status.json`。

两张GPU各最多一个补评测进程；启动前要求至少6GiB空闲，给2GiB推理预算之外预留4GiB余量。与已有训练共同运行，不终止其他项目或原始训练进程。

每任务独立心跳在`progress/<task>/heartbeat.json`。600秒无进展告警并重启；最多3次基础设施重试。已提交episode在同配置同seed下复用；未提交的部分文件移入orphaned目录保留。失败或0%不触发换参数或换seed。

## 输出与验证

- `summary.md`、`summary.csv`每5秒更新；未跑完的分母和成功次数明确标为未定稿。
- `plan.json`在首局前冻结完整配置、种子、checkpoint/统计/config哈希、复用来源及评测引擎哈希。
- `runs/<task>/rollouts20/episode_XX.json`包含逐步原生info；同名NPZ保存原始动作、模型预测块、查询时刻、初始状态；MP4保存视频。
- 全部结束后自动审计100局（80新增+20引用），由每步info重算成功率，核对200步、有限输出、每步查询及从模型预测重建动作，检查权重和归一化统计未变化。
- 审计通过后生成`complete.json`、`artifact_audit.json`和`successful_episodes.md/json`。
- 参数变更见`parameters.diff`；旧基线全部保留在原目录。

## 报告口径

执行间隔1的配置在此前pick-place-wall搜索中确定；其余四任务使用固定配置直接评测，不再筛选。

历史原始ACT基线使用2026101900–2026101919，因此五任务旧/新列不属于完整的同seed配对比较。pick-place-wall另有新seed上的原始100步配对对照5/20；短间隔为8/20。

汇总不得将旧/新差值全部归因于执行间隔，不得覆盖失败记录或选择表现更好的参数填表。
