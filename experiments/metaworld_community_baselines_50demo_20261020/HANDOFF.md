# 当前有效实验：MetaWorld 社区/文献配置新基线

用户已授权重新开始，后续询问seed、执行间隔与早停属于当前实验的配置核验，未授权据此中途改变配置。

- 入口：baseline_watchdog.py；当前监督PID以logs/watchdog_pid.json为准。
- 15组从头训练：DP、ACT、SimpleDP3各五任务。共享50条示教（40训练/10验证）；训练seed0；每组20局，统一新评测seeds2026102000–2026102019。
- DP：horizon16，n_obs_steps2，n_action_steps/action_horizon8，DDIM10，sample预测；batch128，1000epochs，lr1e-4，固定terminal EMA。
- SimpleDP3：同上时间参数、采样与训练预算；保留官方SimpleDP3架构。
- ACT：chunk10，单观测，时间聚合开启，逐步重新查询（执行间隔1），k0.01，KL10，batch8，2000epochs；官方best-validation checkpoint。
- 三种算法均不采用训练早停；验证损失选checkpoint与提前结束训练不同。看门狗超时/OOM处理属于基础设施恢复，不属于按指标早停。若将来明确要求早停，须另定预先冻结的开发集规则和新实验协议，不能使用最终20局的成功率触发停止。
- 预检三方法均通过。DP batch128峰值allocated约5969.76MiB；ACT约1491.83MiB；SimpleDP3约1550.48MiB。五数据集各611文件全部通过收据哈希核验，见dataset_integrity_audit.json。
- 最多4进程，每GPU最多2个，保留4096MiB余量；实际共享GPU显存限制当前主要为DP+ACT运行，后续自动调度。调度显存预留在logs/resource_admission_revision_*.json留有修订记录，模型、batch和正式协议没有因此改变。
- 代码/协议冻结在source_manifest.json；运行中不能直接改冻结文件。资源调整若需改监督程序，先核对PID并停止该监督程序及其本轮子进程，再记录修订并恢复。其他项目的GPU进程禁止操作。
- 所有手动资源调整造成的停止尝试均保留，基础设施失败次数另计；最多三次异常重试，日志/心跳1200秒无进展报警。
- 当前正式训练为DP_shelf-place attempt01、ACT_shelf-place attempt04，随后以logs/status.json为准。此前ACT/SimpleDP3的短暂训练尝试只因资源校准停止，不计为完成结果。
- 初步耗时24–48小时，取决于共享GPU负载。DP最初60次更新中位0.270秒/步，约61000次更新对应4.6 GPU小时/任务；五DP合计约23 GPU小时，未含验证、开发/最终rollout、排队与负载变化。不能把这个初始吞吐当作保证完成时间。
- 进度：progress.md、logs/status.json；最终汇总：report/results.md/csv；异常：logs/latest_alert.json。

此前旧50demo训练监督、零成功搜索监督与探索结果跟踪均已停止；原始目录和所有0%结果保留。旧实验/破0结果不计入当前新主表。本轮0%照实统计，不进行正式测试seed或参数搜索。

## 2026-10-20 07:45 故障修复

ACT shelf-place已完成11/20=55%。监督因SimpleDP3在epoch200保存top-K缺少test_mean_score而停止约57分钟，DP也出现同源故障。已将不参与最终评测选择的top-K k设为0，保留每200epoch latest保存和固定terminal EMA；官方manager回归复现并验证12个缺失metric情况通过。模型、数据、正式协议hash、预算和seed不变。未完成训练按原seed从头重跑，不用缺少优化器状态的ACT权重冒充准确续训。所有失败尝试保留。单任务基础设施失败隔离，避免停止其他健康任务。修订详见audits/checkpoint_repair_20261020_0745。当前监督PID以logs/watchdog_pid.json为准。


## 用户指令更新：结束旧DP并从头训练500轮

旧DP supervisor2774422及三个worker进程组已结束，不自动恢复；现有磁盘权重和结果保留。新DP500独立目录为../metaworld_dp_epoch500_20261020，supervisor2942600。完成300/400/500轮各保存checkpoint及续训辅助状态。500轮后停止训练；后续续训等待用户明确指令。旧进度表是历史快照，暂停内存状态已经随进程结束而失效。详见新目录HANDOFF.md。
