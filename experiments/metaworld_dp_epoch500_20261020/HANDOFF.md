# DP：重新从头训练500轮，等待后续续训指令

用户最新指令：结束旧DP进程；新实验训练500轮；完成300、400、500轮各保存一次快照；后续是否从500轮继续训练，等用户指令。

## 旧实验保留情况

- 原DP1000看门狗2774422及训练进程组2792130、2917972、2936068已按用户指令结束，不自动恢复。
- 原始实验目录及全部已有结果、日志、权重保留不覆盖。
- 独立复制的已有磁盘checkpoint在`preserved_baseline/DP_*/`，SHA见`preserved_baseline/receipt.json`。
- `preserved_baseline/termination_receipt.json`覆盖此前暂停状态：旧进程已经结束，内存现场不再存在。归档的是当时已有的周期checkpoint，不是终止瞬间新存的权重。
- 原shelf-place已训练1000轮并评测20局，3/20；其结果原样保留，不以本轮结果替换。
- SimpleDP3的独立500轮实验继续运行；不终止其他GPU任务。

## 本轮固定协议

- 五任务：shelf-place、pick-place-wall、pick-place、stick-push、door-open。
- 每项从头训练，训练seed0；同一50条示教、40训练/10验证划分；batch128。
- 官方DP图像UNet及原优化器、EMA、训练损失、验证/采样逻辑不变。
- 训练循环在完成500轮后返回；保留原1000轮学习率计划，便于与原训练前缀比较和未来继续训练。
- horizon16、n_obs_steps2、action_horizon8、DDIM去噪10步。
- 500轮EMA按原固定20个评测seed评测，每局200步。所有结果保留，不根据成功率选权重或种子。
- 本轮是独立的训练时长敏感性实验，不替换原DP1000基线。

## 快照与后续续训

- 在epoch和global_step计数递增后同步保存`epoch_0300.ckpt`、`epoch_0400.ckpt`、`epoch_0500.ckpt`，对应18300、24400、30500次更新。
- checkpoint包含官方workspace的模型、EMA、优化器和计数器。
- 每份快照另存`epoch_XXXX_resume.pt`：学习率调度状态、EMA更新计数、Python/NumPy/Torch CPU/CUDA随机数状态、训练采样诊断批次、runner轮次和分数、显式loader generator状态。
- 对应路径及SHA在`snapshot_receipts/epoch_XXXX.json`；汇总由看门狗写入`checkpoint_index.json`。
- snapshot完成后继续当前训练，到500轮后仅评测，不继续训练。
- 将来续训必须恢复上述额外状态；原官方workspace仅resume模型/优化器不够。重建persistent dataloader时，应避免新worker base seed额外消耗已恢复的全局随机数。当前数据集getitem没有随机增强，但不承诺跨进程逐比特复现。
- 续训脚本须在收到新用户指令后具体实现与验证；当前没有自动超过500轮的路径。

## 运行与审计

- `run.py`独立看门狗：每GPU至多一个本轮DP训练进程、最多两个；依据实际剩余显存调度，至少保留3500MiB。
- 任务错误保留原attempt，最多三次重试；错误记录`alert_*.json`。不得改动已启动任务的冻结代码；修复前先停止本轮supervisor及其worker，保留原文件并记录修订。
- 启动时冻结`plan.json`中的脚本、官方训练源码和基础协议SHA。
- 已有参考日志的四项任务比较前20次更新的loss和LR；door-open原未启动，不伪造参考。
- `training_audit.json`在500轮后核验300/400/500快照、优化器更新计数、学习率和EMA续训状态。
- 每任务完成后写`complete.json`；五任务均完成后逐episode核验原生success、20个seed、200步、250次实际去噪调用、动作轨迹与视频，再写summary表。
- 进度见`progress.md`和`status.json`，启动身份见`supervisor_pid.json`。


## 最新运行覆盖：剩余任务不再要求中途快照

用户取消stick-push和door-open的中途快照要求。新看门狗入口operations/no_intermediate_snapshots/supervisor.py，PID3016031，已接管原pick-place与stick-push进程，原worker PID不变。未启动door-open使用replay_terminal_only.py，仅保存500轮最终文件和续训状态；已运行stick-push的旧分支仍会自动保存300/400轮，以避免为改保存逻辑而丢失训练进度。原科学配置与plan SHA不变。完整说明、SHA和接管记录见operations/no_intermediate_snapshots/README.md及manifest.json；勿同时启动原run.py。


## 最终完成（北京时间2026-10-20T20:28:22.042110+08:00）

五任务均完成500轮训练（30500更新）和20局正式评测，100局数据核验通过。结果：shelf-place5/20=25%，pick-place-wall2/20=10%，pick-place3/20=15%，stick-push6/20=30%，door-open20/20=100%，平均36%。500轮后没有续训，所有本轮worker和supervisor均已退出。所有最终模型及续训辅助状态均存在；door-open仅500轮终点快照，其余四项保存300/400/500。summary.md/csv/json、audit.json和successful_episodes.md/json已生成。原科学源码及操作覆盖源码SHA最终检查通过，无报错/重试。
