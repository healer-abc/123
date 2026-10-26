# SimpleDP3：400轮pick-place筛选，条件选择400或300轮

用户指令：先用400轮权重测pick-place；成功率小于90%时，用400轮权重跑五任务；成功率大于等于90%时，用300轮权重跑五任务。

- 方法按当前对话指代SimpleDP3；不修改DP训练。
- 预先记录规则：完整20局筛选，成功0–17局→400轮；成功18–20局→300轮。禁止提前停止筛选或事后再改分支。
- 所有评测统一DDIM10、horizon16、n_obs_steps2、action_horizon8、训练seed0、20个评测seed2026102000–2026102019、每局200步。
- 权重只读加载自同配置SimpleDP3重放训练中准确完成300或400轮的独立快照；不重新训练，不改模型或动作。
- 本轮为**依据结果选择checkpoint轮次的探索评测**，不能当作事先固定配置的独立baseline。原1000轮基线、固定500轮实验及全部筛选数据保留，不覆盖、不删除。
- 若选择400轮，筛选的20局即作为400轮pick-place的20局结果，其余四任务各跑20局，总共100局；若选择300轮，另跑五任务各20局，加筛选20局共120局。
- 分支一旦决定就保存`selection.json`（含筛选结果、摘要SHA、plan SHA），不得因后续结果改变选择。
- 所有快照完成写入并生成receipt后才绑定；尚未就绪的任务等待checkpoint，不读取正在写的文件。
- 每局保留原生success全部infos、动作和预测chunk、初始状态、视频；核验初始状态与原1000轮同seed一致，20局完整，250次实际去噪调用/局。
- `run.py`为独立watchdog，最多两任务、每GPU一个；依据实际显存并保留4096MiB调度；错误写`alert_*.json`、最多三次基础设施重试。只停止本轮自己的worker。
- `plan.json`启动时冻结代码及基础协议SHA。修复代码前必须停止本轮supervisor和worker，并记录版本变更，不能修改运行中冻结代码。
- 最终交付`summary.md/csv/json`、`selection.json`、`parameters.diff`、`successful_episodes.json`和逐组审计`audits/`；实时状态`progress.md`、`status.json`。


## 已完成

400轮pick-place完整20局：19/20=95%，按既定>=90%分支锁定300轮。300轮五任务结果依次13/20、19/20、19/20、20/20、20/20，平均91%。共120局全部核验通过，执行代码SHA未变，无报错/重试。summary.md/csv/json、selection.json、audit.json和successful_episodes.md/json已写入。300轮pick-place仍为95%，不再追加参数或种子搜索。
