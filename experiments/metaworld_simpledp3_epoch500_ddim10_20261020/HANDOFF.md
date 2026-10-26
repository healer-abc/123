# SimpleDP3 500轮快照重建与DDIM10评测，额外保存300/400轮

用户依次要求：用训练500轮模型测评、去噪恢复10；询问现有保存轮数；要求中途保存300和400轮权重。以上任务合并执行，未取消500轮评测。

## 数据与旧权重实况

原新基线五任务均完成1000epochs。各自final.ckpt内部epoch1000/global_step61000；latest.ckpt内部epoch800/global_step48860，是零起编号800结束的保存点，实际完成801轮。没有统一保存的500轮权重，不能将400/800/1000模型改名冒充。个别失败尝试有epoch200保存点，不用作本轮500模型。

## 本轮固定流程

- 五任务同50条示教、40/10划分、训练seed0、相同官方SimpleDP3模型与优化器，从头重放原训练前缀。
- 原训练学习率计划仍为1000epochs、batch128、warmup500steps。只将官方run函数训练循环固定为500次，即完成epoch0–499；不能把scheduler总计划也改成500，否则同时改变学习率。
- 完成300轮和400轮后，独立保存checkpoints/epoch_0300.ckpt、epoch_0400.ckpt；保存发生于epoch/global_step计数递增之后，分别对应18300、24400次更新，避免零起编号混淆。包含model、ema_model、optimizer状态，另存哈希收据，不被latest覆盖。
- 完成500轮共30500次更新，保存final.ckpt；完整流程结束后生成同内容的epoch_0500.ckpt硬链接。500模型使用DDIM10、horizon16、观测2、执行8，各任务20局、每局200步；评测seeds2026102000–2026102019，全部失败照实保留。
- 300/400轮仅保存权重，当前没有要求额外rollout。最终20局不用于挑选轮数、seed或是否保留结果。
- 每任务记录原官方run源码与最小循环/保存钩子diff；模型、损失、优化器、EMA数学均不改。前20次训练更新的损失/LR与原1000轮日志核对，支持同初始化与数据路径，不声称500轮全程逐位一致。
- 本轮是观察主表后的训练轮数消融；模型为同配置重新训练的前缀，不是原先遗留的500轮checkpoint。原1000轮主表和DDIM5结果均保留，不替换。

## 运行和交付

run.py负责监督，replay500.py调用原基线训练适配器并在进程内设置固定500轮终点；rollout500.py是独立评测副本。源码/协议hash冻结于plan.json。运行期间不能直接修改冻结源码；修改前核对身份并停止本轮相关进程。

最多两个训练/评测进程，每GPU一个；按空闲显存8192MiB门槛接入，保留余量。心跳或日志超过600秒无进展报警；低显存只让出本实验进程，基础设施错误最多三次重试。原基线DP训练继续，不操作其他项目进程。

进度progress.md/status.json；权重索引checkpoint_index.json；模型路径runs/<task>/attempt_XX/checkpoints/；300/400收据snapshot_receipts；前缀核验prefix_audit.json；完整重建核验replay_audit.json；最终summary.md/csv/json、audit.json、successful_episodes.json。监督PID见supervisor_pid.json。

预计五任务重放及100局评测2–4小时，受共享GPU负载影响。最终模型、300/400快照和每局原始视频/NPZ/原生info均保留。
