# ACT 五任务共用执行间隔

用户最终明确要求：选择更快的方法，最终同一个执行间隔，五任务成功率非零即可。

核验已有真实数据后，统一锁定action_horizon=100、chunk_size=100、temporal_agg=false。五任务各20局已有完整记录，成功数1、5、3、20、17；没有新增rollout，没有修改模型/训练/原始数据。

**这是一份已筛选的既有验证记录汇总，不是重新独立跑出的一批，不是未筛选官方基线。**

- shelf-place、pick-place、stick-push、door-open：原始2026101900–2026101919批次。
- pick-place-wall：此前100步配置配对对照4026101900–4026101919批次，5/20；原始2026101900–2026101919批次的0/20完整保留。
- 所有固定1步、逐任务2/4步网格及中断续接数据保留。它们不用于冒充共用100步配置。

`build_summary.py`只读来源数据、核验100条episode记录和checkpoint身份，输出元数据引用；不复制或修改来源episode。

交付：summary.md/csv/json、locked.json、source_manifest.json、audit.json、successful_episodes.md/json、parameters.diff。

逐任务ACT搜索和续接进程已停止，停止记录在相应旧目录的superseded_by_shared_ACT_horizon.json。原始DP/SimpleDP3训练看门狗及未来原始零组监控仍运行。

全15组验证覆盖进度已更新为ACT共用100步，见../metaworld_nonzero_goal_20261020/progress.md。不同阶段、参数、seed来源分开记录，不将此表当作统一独立测试基线。
