\/ # 实验推进交接

> **恢复指令（2026-10-18）**：用户已明确恢复 E1/E2/E3 主线，并授权按实际空闲显存并发、后台看门狗检测与限次恢复。先完成 E1/E2 的同清单开发评测，再决定唯一 E3。

最后更新：2026-10-18 13:24 CST（E3 约 3.72k/30k 正常训练；成本汇总钩子故障已修复，恢复 E1 数据并启动 E2 短测）。

## 当前状态

- E0 全量微调完成，最终目录 `29999/`；历史四套评测 1917/2000，95.85%。
- E1 LoRA 已完成 30k，最终 checkpoint 为 `projects/openpi/checkpoints/pi05_libero_lora/pi05_libero_lora_gpu0/29999`。
- E2 冻结视觉 LoRA 已完成 30k，最终 checkpoint 为 `projects/openpi/checkpoints/pi05_libero_lora_frozen_vision/pi05_libero_lora_frozen_vision_s42/29999`。
- 不要重复训练或覆盖 E1/E2。10 月 16 日评测曾因 `uv` 联网失败而退出，随后按用户要求暂停；10 月 18 日恢复，直接使用现成虚拟环境，GPU0/E1、GPU1/E2 同时评测，每模型两个任务分片客户端。
- 已建立实验协议、决策日志、runs/dataset/exposure manifests。
- E2 配置已添加并通过静态审计与独立 10 步验证；正式训练已完成。
- E1/E2 最终记录 loss 为 0.0171/0.0162，300 条指标均无 NaN/Inf；解析结果在 `metrics/e1/` 与 `metrics/e2/`。
- `manifests/eval_manifest.csv` 已冻结：开发 400、确认 1600，SHA-256 `6f52e79c4e194e04a57d40a283812078db54bef4117a86a5f02c52e31c17311b`。
- E1/E2 开发评测及比较于 2026-10-18 09:37:41 CST 完成：各 377/400（94.25%）。Spatial：E1 189/200（94.5%），E2 196/200（98.0%）；Long：E1 188/200（94.0%），E2 181/200（90.5%）。
- 配对分层 bootstrap 95% 区间为 E2−E1 `[-3,+3]` 个百分点，开发筛选通过；不替代未暴露确认集验证。Long task 2、9 各下降 20 个百分点，需要优先审阅。
- 两组各 400 次尝试全部正常完成、没有基础设施重试或看门狗告警，各有 23 个失败视频；E1/E2 评测服务均已退出。
- 动作审计完成：真实控制器裁剪运动输入、夹爪取符号，已直接调用验证。环境成功率两组均 94.25%；严格按原始动作任一维超限计失败则两组均 0/400，分别保留，不覆盖旧记录。审计与固定 40 个失败接触表审阅见 `report/development_review/`，不声称完成逐帧根因诊断。
- 唯一 E3 已登记：`pi05_libero_action_only` / `pi05_libero_action_only_s42`。非 LoRA 基础结构，从原始 pi05_base 训练；仅动作 expert 与动作/时间投影更新，视觉、语言冻结。完整 LR schedule 复制 E0，EMA 关闭；batch 32、seed 42、30k。
- E3 实际可训练参数 430,098,464 / 3,353,433,872（12.8256%）。10 步验证通过：32 个冻结叶零变化、19 个训练叶全部变化、两次重载推理输出差为 0（atol=1e-5）。证据为 `audits/e3_smoke_verification.json`。
- E3 正式训练于 10:08:50 CST 启动，GPU0，看门狗 PID `289075`，训练 PID `289079`。step 0 loss=0.0932、grad_norm=0.3847；尚无告警。状态以 `logs/e3_training_status.json` 为准。
- 10:11 CST 现场核验：E3 进度条约 34/30000，早期稳态约 3.1 s/step，纯训练暂估约 26 小时（不是完成承诺）；GPU0 进程约 33388 MiB。GPU1 E1 短测已到 52/220，正常。
- 13:23 CST 现场核验：E3 约 3720/30000（约 12.4%），最近 step 3600 loss=0.0313，checkpoint 3000 已 finalize，未发生训练中断或重启。稳态约 3.1 s/step，预计训练剩余约 23 小时，约 10 月 19 日中午结束，之后自动开发评测；不是完成承诺。
- GPU1 按 E1→E2→E3 顺序做相同设备的独立 220 步成本短测（20 预热、200 测量、覆盖保存）。首次看门狗 PID `285588` 因汇总断言退出，已替换为 `cost_profiles_watchdog.py`，新 PID `520688`；当前 E2 profiler PID `520767`。状态仍为 `logs/training_profiles_status.json`。GPU1 原有常驻进程约 10.4 GiB，保留共享干扰限制，不宣称独占测量。
- 已修复成本计时故障：W&B 初始化重绑定 `wandb.log`，原先提前挂接的 observer 被覆盖。`profile_hooks.py` 改在初始化后安装，回归测试已通过。E1 实际 220 步与保存完整，从原始离线 W&B 文件恢复全部 220 条记录，未重跑或覆盖原始文件；E1 平均 3.9279 s/step，采样进程峰值 33394 MiB。三组统一用 W&B 前端同步指标后的 wall-clock 时间戳计时，保留单调 observer 作为后续交叉检查；明确恢复来源与计时限制。
- 13:24 CST 真实恢复核验：E2 新 observer 已实际写出 steps 0–4（不是仅单元测试），短测继续正常；E3 约 3.74k/30k，step 3700 loss=0.0307、grad_norm=0.1369，训练未中断。

## 已完成核验

- E0 每套 500 episode，共 2000；Spatial 97.4%、Object 99.2%、Goal 97.0%、Long 89.8%。
- 未发现 tyro 参数、连接或中途退出错误；末尾 EGL 清理异常不影响已记录结果。
- 旧 `episodes.csv` 缺 init state、seed、checkpoint，不能直接用于严格配对 bootstrap。
- E0/E1 实际 argv 已从 W&B metadata 固定；源码关键文件和工具脚本已记录 SHA-256。
- 数据元信息：1693 episodes、273465 frames、40 tasks；norm stats 哈希已记录。
- 参数审计：E0 为 3,353,433,872/3,353,433,872 可训练；E1 为
  466,957,072/3,403,421,456（13.7202%）；E2 为
  52,153,376/3,403,421,456（1.5324%）。E1 中视觉编码器占 414,803,696 个可训练参数。
- E2 与 E1 的训练叶差异恰好只有 23 个 `PaliGemma/img/**` 叶。
- E2 smoke：10/10 步完成，step 0 loss 0.0936、grad_norm 0.1992；checkpoint `9/` 完整。
- 逐叶检查：23 个视觉叶和 20 个其他冻结叶全部零变化；20 个 LoRA 叶和 8 个投影叶全部更新；checkpoint 双重加载逐叶一致。

## 下一步（严格按顺序）

1. E3 后台完成 30k；正常结束且 checkpoint `29999` finalize 后自动启动 E3 开发集 400 episode，使用同一 manifest、seed 与两客户端分片。
2. GPU1 顺序完成 E1/E2/E3 成本短测；结果写入 `metrics/profiles/`，每秒采样进程显存，步骤计时在同步指标读取后记录。
3. `complete_e3_analysis.py --wait` 已后台等待 E3 开发集完成，随后输出 `evaluations/e3_development_comparison.json`，分别与 E1/E2 做配对 bootstrap，并报告环境/严格边界两种成功数。
4. 审阅 E3 失败与成本结果后冻结候选，再按原计划做 E0/E1/候选的共同确认清单；不在候选冻结前查看确认集，不再开启其他 E3。

## 禁止事项

- 不改正在执行的脚本文件，不使用 `--overwrite` 指向现有实验，不干扰其他用户进程。
- 不把 E0/E1 写成单变量 LoRA 对照；不把预分配不同的显存截图写成节省结论。
- 不为补齐字段伪造旧 episode 的 state ID 或 seed。
- 不得在 E1/E2 开发结果出现前自动接续 E3，也不得因 loss 较低选择模型。

## 关键文件

- 参数审计：`audits/parameter_audit_*.csv`
- 冻结验证：`audits/frozen_parameter_checks.json`
- E2 smoke 日志：`e2_smoke.log`
- E2 smoke checkpoint：`projects/openpi/checkpoints/pi05_libero_lora_frozen_vision/pi05_libero_lora_frozen_vision_smoke_s42/9`
- 开发评测入口：`run_development_pipeline.sh`（调用同一带互斥锁的看门狗）；看门狗：`development_watchdog.py`。
- 实时状态：`logs/watchdog_status.json`；主日志：`logs/development_watchdog.log`；完整事件：`logs/watchdog_events.jsonl`。
- 告警：`logs/alerts.jsonl`、`logs/latest_alert.json`；各模型/分片原始日志独立保留。
- 各模型全局汇总：`evaluations/{E1-s42,E2-s42}/development/summary.json`；分片心跳及汇总在同目录；`attempts/` 保存每个 episode 的原始尝试，最多两次。
- 显式 seed 服务：`seeded_policy_server.py`；逐 episode 客户端：`eval_manifest.py`
- E3 看门狗：`training_watchdog.py --mode e3`（运行中的文件未修改）；成本看门狗改为 `cost_profiles_watchdog.py --gpu 1`，不要再用旧 `--mode profiles` 入口；成本包装：`profile_training.py` / `profile_hooks.py`；E3 验证：`verify_e3_smoke.py`。
- 当前 E3 状态/日志：`logs/e3_training_status.json`、`logs/e3_training_watchdog.log`、`logs/E3_train_attempt_1.log`；正式训练资源：`metrics/e3/resources_attempt_1.csv`。
- 成本状态：`logs/training_profiles_status.json`；汇总：`metrics/profiles/summary.json`（三个短测全部结束后生成）。设备、精度与共享限制：`audits/resource_environment.json`。
- E3 后处理等待器日志：`logs/e3_analysis.log`、PID 文件 `logs/e3_analysis.pid`。

## 恢复后的活动进程与看门狗

- 2026-10-16 09:12 CST 的首次队列 PID `2128934` 已退出；原因是 GPU 空闲后 `uv run` 尝试联网解析依赖而失败，不是模型或环境 rollout 失败。
- 修复后的队列曾于 2026-10-16 11:46 CST 启动，PID `2205740`；已按用户要求主动停止。`logs/development_pipeline.pid` 仅保留历史 PID，不代表活动进程。
- 2026-10-18 09:05 CST 启动的新看门狗 PID `224923` 已于 09:37 正常退出，最终状态 `completed_awaiting_e3_review`；PID 文件只保留历史值，不能据此认定仍在运行。
- 模型服务首次 PID：E1 `224927` / GPU0 / 端口 8101；E2 `224928` / GPU1 / 端口 8102。客户端首次 PID：E1 `225903,225904`，E2 `225905,225906`；自动恢复后 PID 会变，以状态文件为准。
- 09:06:47 CST 实际 rollout 核验：E1、E2 各完成 13/400，均为 13 次成功；无告警。只覆盖少量 Spatial 任务，不能据此推断完整成功率。
- 每模型启动预算 24576 MiB，额外余量 8192 MiB；每 15 秒检查进程、心跳和显存，低于 4096 MiB 时释放本项目进程。服务器启动超时 1200 秒，客户端停滞阈值 900 秒。
- 基础设施错误限次恢复，保留所有尝试；模型输出异常、记录身份不匹配或重试耗尽会告警并暂停。自动恢复包括重启服务/客户端及跳过已验证 episode，不自动修改科学协议或训练配置。
- E1/E2 开发集比较已完成并释放服务；以上旧 PID 均为历史值。当前活动任务为 E3 正式训练和成本短测，见“当前状态”。
- 查看进度：`tail -f experiments/pi05_libero_adaptation/logs/development_watchdog.log`；完整进度：`cat experiments/pi05_libero_adaptation/logs/watchdog_status.json`。
- 停止本队列：读取并核对 `logs/development_watchdog.pid` 中进程后发送 SIGTERM；看门狗会清理自己启动的服务与客户端。
- 检查通过：6 项 CPU 单元测试（分片覆盖/互斥、显存预约与余量、聚合断点身份检查、异常分类、限次重试及模型异常暂停），以及两个虚拟环境的导入、脚本语法、重复启动互斥和真实 rollout。
- 后续管线累计 12 项单元测试通过，覆盖 E3 白名单、checkpoint finalize 门槛、仅统计本进程显存、离线数据环境及成对 seed 身份验证；E3 smoke 的逐叶检查和重载推理也已实际通过。
- 成本恢复后累计 15 项测试通过，新增 W&B 初始化重绑定回归测试、精确 200 个测量间隔校验及缺失/nonfinite/时钟回拨拒绝检查。10:22 的原始告警保留，13:23 的 `profile_recovered` 事件和当前状态标明该故障已恢复。
