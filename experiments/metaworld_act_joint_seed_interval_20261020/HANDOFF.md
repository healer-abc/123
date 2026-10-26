# ACT 五任务同seed、同执行间隔

用户最新要求：五项使用同一seed协议、同一动作执行间隔，每任务20局且真实成功率均大于0。此前混用seed的汇总不满足要求。

- 固定训练seed=0，现有官方训练checkpoint保持不变。
- 固定rollout seed序列4026101900–4026101919，五任务完全一致，候选之间也不换seed。
- 仅搜索共同action_horizon：100、32、16、8、4、2、1；temporal_agg=false，模型chunk_size=100。
- 一个候选必须五任务各20局且均有成功才能接受。任何一项0/20就淘汰共同候选，跳过未启动任务；已启动任务跑满20局，保留全部原始数据。
- 优先验证shelf-place、pick-place、pick-place-wall；通过后再跑door-open、stick-push。最多两个推理进程，每GPU一个，要求至少6144MiB空闲；600秒无进展报警，基础设施异常最多重试三次。
- 只复用checkpoint、参数和seed序列完全匹配的既有20局记录；不复用旧20261019xx批次。1步的五项记录和100步wall记录符合要求。
- run.py、plan.json和评测engine校验SHA；运行期间不可修改。JSON为episode提交标记，保留视频、NPZ模型chunk/动作、原生info。最终100局逐条重算成功，并检查动作、相同seed的初始状态与checkpoint。
- 此固定seed集合参与参数选择，结果属于搜索后验证。原始基线及失败结果保留，不声称是未参与选择的独立测试。

入口：`run.py`。进度：`progress.md`、`status.json`。最终：`summary.md/csv/json`、`audit.json`、`parameters.diff`、`successful_episodes.json`。

原始DP/SimpleDP3训练和已有watch_future.py监控继续运行。全局状态由../metaworld_nonzero_goal_20261020/track.py读取本目录联合审计结果；联合验证通过前ACT不记为完成。
