# S1 / S2 LoRA SFT（2026-10-25）

用户已选择共同 π0 基础权重、每组 10,000 次更新，并授权根据实测显存并发运行。
两组已于 **2026-10-25 09:18:44（北京时间）** 启动，监督进程 PID 为 1262382。
实时阶段及完成步数见
[sft-experiments-state.json](sft-experiments-state.json)。

两组现已完成各 10,000 次更新，退出码均为 0；合并导出于
**2026-10-26 09:08:19（北京时间）** 前全部完成。
632 个冻结基座 tensor 逐元素保持不变，权重合并检查的最大误差均为 1.67e-6，
模型与归一化文件完整；验收见 [sft-completion-check.json](sft-completion-check.json)。
S1 / S2 最近 100 步平均 loss 为 0.00999 / 0.00978；这些数值不代表任务成功率。
独立 Clean / Randomized 评测尚未启动。

| 项目 | S1 | S2 |
| --- | --- | --- |
| 数据 | 200 Clean | 100 Clean + 100 Randomized |
| 训练帧 | 31,185 | 30,740 |
| LeRobot repo | `local/robotwin_pi0_clean200` | `local/robotwin_pi0_mixed200` |
| 配置 | `../configs/s1_clean.yaml` | `../configs/s2_mixed.yaml` |
| 初始化 | 同一份原始 π0 基座 | 同 S1 |
| 更新次数 / global batch | 10,000 / 16 | 同 S1 |
| micro batch / 累积次数 | 每 rank 4 / 2 | 同 S1 |
| LoRA | rank 8、alpha 8、dropout 0、all-linear | 同 S1 |
| 学习率 | 2.5e-5，warmup 500，cosine 到 2.5e-6 | 同 S1 |
| seed | 0 | 同 S1 |
| 保存步数 | 5,000、10,000 | 同 S1 |

混合组从两个已验收的数据池各均匀无放回抽 100 条；选择 seed 为 20261025，
原始 episode 编号固定在 [sft-mixed200-selection.json](sft-mixed200-selection.json)。
两组共用 **Clean200 训练池统计**，原文件为 `clean200_norm_stats/norm_stats.json`。
这份统计来自 S1 的全部 Clean 训练数据，不是 S2 单独估计的统计，也不含评测观测。
训练数据与官方 150 个 Clean 评测 seed 的交集为 0。
完整计划见 [sft-experiment-plan.json](sft-experiment-plan.json)。
混合组的所有状态/动作与原始 HDF5 逐帧核对、首中末三相机像素核对以及
使用共同统计的实际 OpenPI batch 检查均通过，见
[mixed200-converted-check.json](mixed200-converted-check.json) 和
[mixed200-loader-check.json](mixed200-loader-check.json)。

基础权重来自 [Physical Intelligence 官方 π0 基座](https://github.com/Physical-Intelligence/openpi#base-models)：
`gs://openpi-assets/checkpoints/pi0_base`。
保留 33 个原始 GCS 对象的 generation / MD5，并逐文件完成校验。
使用当前固定 RLinf 的 `jax_to_openpi` 转换为原生 PyTorch Pi0，632 个 tensor，
3,238,048,528 个参数；FP32 权重，horizon 50、模型动作维度 32、token 长度 48。
归一化文件独立指定，不将其宣称为基座原始预训练统计。
模型位于 `models/robotwin_pi0/pi0_base_openpi/`，
来源与模型 SHA256 见 [pi0-base-source.json](pi0-base-source.json)。

两组各使用 2 个 FSDP rank，**共同使用 GPU 0、1**，合计 4 个训练 worker。
每组都有独立 Ray 集群、临时目录、进程组、通信与日志目录，未复用同名 actor。
保持已验收的 BF16 计算、FP32 参数存储与指定 FP32 模块边界，启用梯度检查点。

micro-batch 4 的并发压测中，两组均完成 3 次更新、原生 DCP 保存，退出码均为 0；
实际读取保存的 adapter，并确认 202 个 LoRA B tensor 非零。
两组同时运行时每卡总显存峰值分别为 **39,441 / 38,919 MiB**（约 38.5 / 38.0 GiB）。
稳定阶段每组每步训练约 8.6–9.9 秒，按此估计两组并发训练约需 25 小时；
该估计不包含完整保存/导出耗时，实际以正式运行日志为准。
证据见 [sft-memory-profile.json](sft-memory-profile.json)。
第一轮 micro-batch 2 压测关闭了保存，只能验证更新；实际保存验收由第二轮完成。
资源压测使用现有任务 checkpoint，只测资源和工程行为，不计入 S1/S2。
压测的临时大 checkpoint 已在验证后移除，日志、统计和 checkpoint 元数据仍保留。

每次正式保存使用原生 DCP，包含模型、优化器、调度器及 RNG 状态；
不额外保存同内容的 `full_weights.pt`。
官方 OpenPI 数据加载器不提供本项目已验证的采样顺序恢复，因此中断续训不承诺数据顺序逐比特一致。
结束后依次在 CPU 读取第 10,000 步模型，核对冻结参数与共同基座逐元素相同、
训练 tensor 有限、LoRA 合并线性等价，并导出 632 个 tensor 的原生 Pi0、独立 adapter、
归一化统计与训练来源到 `models/robotwin_pi0/s{1_clean,2_mixed}_<时间戳>/`。
正式 DCP 的中间与最终 checkpoint 均保留；没有删除此前实验产物。

每个运行目录保存 `input-config.yaml`、`data-source.json`、`run.log`、`gpu.csv`、
TensorBoard 和 `exit-code.txt`。来源清单固定了模型、配置、数据 manifest 与统计文件哈希。
监督进程定时记录步数和有限性，单组失败会留下日志，另一组可继续；
使用独占锁避免重复启动。

本次运行目录：

- S1：`../runs/s1_clean_20261025_091844/`
- S2：`../runs/s2_mixed_20261025_091844/`

启动后首次检查已观察到 S1 / S2 分别完成 8 / 7 次正式更新，loss 和梯度范数均有限，
每卡合计峰值为 38,967 / 39,367 MiB。该快照仅确认当时正常启动；完整 10,000 步现已完成。
证据见 [sft-initial-updates-check.json](sft-initial-updates-check.json)。

查看状态：

```bash
cat experiments/robotwin_pi0/report/sft-experiments-state.json
```

查看 loss、显存和成本汇总（不会改变训练）：

```bash
# 任意目录执行，无需先激活训练环境；默认每 30 秒刷新
bash /home/zhsu/jyluo/experiments/robotwin_pi0/scripts/watch_sft_metrics.sh
# 仅查看一次
bash /home/zhsu/jyluo/experiments/robotwin_pi0/scripts/watch_sft_metrics.sh 0
```

查看脚本自行定位项目 Python，并仅对查看子进程清除 `LD_LIBRARY_PATH` / `LD_PRELOAD`。
已复现训练工具链的 `LD_LIBRARY_PATH` 会导致系统 `watch` 在初始化终端时崩溃，
因此采用普通循环刷新；不会修改训练进程的动态库环境。`Ctrl+C` 退出查看。

`Loss` 为最新一步，`Mean(100)` 为最近最多 100 步均值。
TensorBoard 已在服务器 `127.0.0.1:6006` 启动，仅载入本次 S1/S2 的正式日志；
远程 IDE 可转发 6006 端口后访问 `http://127.0.0.1:6006`。
在 Scalars 中查看 `train/loss`、`train/grad_norm`、`train/learning_rate` 和 `time/training`。

显存来自每秒采样的 `gpu.csv`，报告 GPU 0 / 1 的整卡占用峰值（两组同时运行的合计）。
它无法恢复每组独立或采样间隔内瞬时的峰值，不应将同一整卡数值分别填为 S1、S2 的独立峰值。

GPU-hours 按每张物理 GPU 的监测时段并集累计，包含初始化、训练与保存，
不含监测结束后的 CPU 导出；两张卡同时运行 1 小时合计为 2 GPU-hours。
两组在同两张卡上并发，不再乘以实验组数。
汇总中的 `GPU-h*` 按重叠时段均分给两组，是成本分摊估计，不能视为实测的逐进程 GPU 执行时间。
`sft-metrics-latest.json` 是运行汇总命令时的快照；训练监督的状态文件仍每 30 秒自动更新。

正式训练完成后，还需要在固定的 Clean / Randomized 评测条件下比较独立成功率；
本阶段不把训练 loss 或资源压测当作效果结论，也不自动启动 PPO。
