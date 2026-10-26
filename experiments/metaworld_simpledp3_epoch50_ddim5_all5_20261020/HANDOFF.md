# 用户指定：SimpleDP3统一50轮五任务评测

用户在200→100→50筛选结束后明确指令“用50的跑全五个任务”。此前筛选结果为95%、90%、90%，不满足原先严格小于90%的条件；原结论及全部数据保留。本轮是新的明确指定50轮的探索评测，不修改旧筛选记录或原基线。

- 使用上一轮从头训练200轮期间保存的准确完成50轮权重（epoch50、global_step3050）。不重新训练，也不改模型权重。
- 五任务统一50轮、DDIM5、horizon16、n_obs_steps2、n_action_steps8、训练seed0。
- 每任务20局，统一seed2026102000–2026102019，每局200步，原生info.success曾为真作为成功。
- pick-place直接复用已有20局，18/20=90%；保留原文件、身份和SHA，不重写为新实验身份。其他四任务补测80局，总表100局。
- 无论后续成功率如何，都完成五任务并如实汇总，不追加checkpoint/seed搜索。
- 独立目录`metaworld_simpledp3_epoch50_ddim5_all5_20261020`；原筛选、训练快照和所有历史基线保持不变。DP训练继续运行。
- `run.py`在本进程内复用原`metaworld_simpledp3_epoch200_ddim5_select_20261020/run.py`中经过核验的权重加载和逐局审计函数，明确重定向本轮目录、目标与复用结果校验；不修改原源码。依赖源码及协议SHA在plan.json冻结。
- 原pick-place记录必须匹配旧plan SHA；新四项必须匹配新plan SHA。权重SHA、config SHA、轮次、20个种子均校验。
- 每局保存视频、原生infos、动作和预测chunk。逐组核验原始环境成功判定、200步、25个8步chunk、125次实际去噪调用、相同seed初始状态。
- 两GPU最多并行两个评测worker；按实际剩余显存调度并留4096MiB。错误保留并写alert，最多三次基础设施重试，只停止本轮自己的worker。
- 冻结代码修改前必须先停本轮supervisor及workers，并保留版本和原始错误记录。
- 实时状态`progress.md`和`status.json`；完成后`summary.md/csv/json`、`audit.json`、`successful_episodes.md/json`、`episode_manifest.json`、`parameters.diff`。


## 已完成

五项50轮统一权重评测结束，全部100局核验通过（新测80局，pick-place复用20局）。结果：shelf-place8/20=40%，pick-place-wall17/20=85%，pick-place18/20=90%，stick-push16/20=80%，door-open20/20=100%，平均79%。无报错、无重试。最终代码/依赖/原筛选结果SHA核验通过；所有原始结果保留。summary与成功视频索引已生成。
