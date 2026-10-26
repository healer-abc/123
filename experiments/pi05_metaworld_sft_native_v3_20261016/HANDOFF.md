# π0.5 MetaWorld SFT 原生 v3 评测交接

用户要求调查 2026 年 4 月“官方成功率约 90%”修复，并以新版 MetaWorld 环境重跑 MetaWorld 微调 π0.5。已确认该修复属于独立的 [MTBench 项目](https://github.com/Viraj-Joshi/MTBench#updates)，2026-04-19 修改 Isaac Gym/Franka 任务奖励，对应 PPO 长程训练，不是 Farama MuJoCo/Sawyer v3 或 π0.5 SFT 权重的补丁。不能以本次结果验证该 90% 声称。

本轮选择与上轮相同的已完整 SHA256 验证 `RLinf/RLinf-Pi05-MetaWorld-SFT` 权重（commit `60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e`），只改用 Farama MetaWorld 3.1.1 的五个原生 v3 任务。保留 assembly、peg-insert-side、disassemble、peg-unplug-side、stick-pull，每项 20 个 rollout。协议见 `protocol.json`，评测入口为 `launch.py`/`rollout.py`。依赖复用已有隔离环境，模型未再下载，未训练。加载权重 missing/unexpected 均为空。

正式评测用 RLinf 官方设定：15 个零动作稳定 + 最多 160 个策略步，动作块 5，去噪步 5；曾出现任一 `info.success` 即记成功，另外记录最后一步是否成功。前轮 v2 是 500 步上限且首次成功立刻停止，故仅能并列展示，不能做纯环境版本 A/B 因果比较。

GPU0 在本轮启动时剩余约 18GiB，GPU1 仅约 7.5GiB；另一个用户的 `pjwang` 训练占用两张卡各约 66.7GiB，不能中断。一个 assembly-v3 smoke 已完成，`smoke_results/assembly-v3/episode_00.json`，峰值约 7.3GiB，未计入正式结果。第一次 smoke 因依赖路径先选到 v2 而失败，已修改路径顺序；第二次成功加载 v3 环境。无正式 episode 被错误版本污染。

2026-10-16 08:18 CST：正式 100 回合已开始，使用单 GPU0 顺序执行五任务。终端 session id `29024`，进程退出则按顺序补跑（代码跳过已有 episode JSON）。每条 JSON、NPZ 和 MP4 在 `results/`、`videos/results/`；心跳在 `logs/`。结束后运行：

```bash
projects/openpi/.venv/bin/python experiments/pi05_metaworld_sft_native_v3_20261016/aggregate.py
```

该聚合器验证所有 episode 的哈希、逐步 success 和轨迹，再导出 `results/report.md`、`summary.json/csv`、`episodes.csv`。正式结果必须完整 100/100 才可作为最终成功率。不要误用旧 π0 v3 结果（38/100，含 shelf-place）或旧 π0.5 v2 结果（29/100，500 步）。

2026-10-16 08:29 CST 中途校验：60/100 已完成且逐回合完整性通过。assembly-v3 2/20、peg-insert-side-v3 3/20、disassemble-v3 7/20；后两项尚未完成。disassemble 中若干回合仅短暂满足成功，主要指标按 `any(success)`，不要误用末帧成功。单 GPU0 工作进程仍在按任务顺序运行，当前切换至 peg-unplug-side-v3。不能把 12/60 中途成功率当作最终结果。

2026-10-16 08:35 CST 最终：正式 100/100 完成，顺序工作进程正常以 exit code 0 退出，GPU0 显存回落到空闲约 18GiB。结果 assembly 2/20、peg-insert-side 3/20、disassemble 7/20、peg-unplug-side 5/20、stick-pull 5/20，总计 **22/100** 曾成功；末帧成功 **8/100**。`aggregate.py` 对 100 个 JSON/NPZ/MP4、逐步成功、累计奖励、种子和协议校验通过。`replay.py` 在新建的原生 v3 仿真器中独立重放全部 100 回合，每步完整观测、奖励和成功标志都一致；审计记录在 `results/replay_validation.json`。权重及 MetaWorld 3.1.1 wheel 的 SHA256 重新计算均符合协议；权重严格加载无缺失/意外参数。

旧 v2 同权重评测为 29/100，但最多 500 步；从原轨迹离线截到 160 步是 23/100。旧 v2 无 15 步稳定阶段，场景实例和物理版本均不同，故 23 与新 v3 的 22 只可描述性并列，不能视为严格配对的性能差。`results/report.md` 提供逐项和版本对照；不要把本实验误称为验证 MTBench 的约 90% PPO 成功率。没有下载或使用任何声称“2026-04 MTBench 修复版”的 π0.5 权重，因为该修复并不存在于 RLinf π0.5/Farama MetaWorld 产品线。
