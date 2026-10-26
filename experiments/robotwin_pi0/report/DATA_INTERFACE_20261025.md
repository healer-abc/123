# 自采数据接口验收与扩量（2026-10-25）

自采数据 → LeRobot v2.1 → 归一化/实际 OpenPI batch → LoRA SFT 更新与恢复 →
权重合并 → RoboTwin 评测 → PPO 更新与恢复，均已实际通过。
总验收：[data-interface-check.json](data-interface-check.json)。
扩量已完成：Clean / Randomized 各 **200 条**，共 **62,116 个训练帧**。
两组均已转换、核对、计算归一化统计，并通过实际 OpenPI batch 检查。
最终验收见 [expanded-data-check.json](expanded-data-check.json)，
任务状态见 [expanded-collection-state.json](expanded-collection-state.json)。

## 数据与动作协议

- 保持 RoboTwin `RLinf_support`，未切换 main；使用必要的旧 HDF5 → LeRobot 适配器。
- Pilot 两组各 5 条 HDF5 和对应语言指令均已采完，各取一条用于接口验收。
- Clean seed 0 原始 144 帧，形成 143 组输入/目标；Randomized seed 2 原始 150 帧，
  形成 149 组，共 **292 帧**的 LeRobot 接口数据。
- 严格采用 `state = raw[:-1]`、`action = raw[1:]`，不为没有下一帧目标的末帧补标签。
  官方成品数据第 0 条的前 142 个 action 与下一帧 state 逐元素一致；
  其末帧 action 与当前 state 的最大差约 3.60e-4，不能据此假定 terminal hold。
  证据见 [official-action-alignment-check.json](official-action-alignment-check.json)。
- 14 维顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。存储绝对关节目标，
  delta 变换、32 维补齐和图像缩放交给当前官方 OpenPI 管线。
- 三相机：`head_camera → cam_high`、`left_camera → cam_left_wrist`、
  `right_camera → cam_right_wrist`。采用官方 image-bit codec，不额外交换 RGB 通道；
  旧版和带新标记 JPEG 的颜色对照均通过。来源及哈希见 `image-codec-source.json`。
- 转换后所有状态/动作逐帧核对，每条轨迹首、中、末帧三相机逐像素核对均通过。
  见 [converted-dataset-check.json](converted-dataset-check.json) 和 `pilot_rgb_audit.png`。
- 语言只使用原始 `seen` 指令，未把 `unseen` 指令混入训练。
- LeRobot 名义 FPS 沿用官方成品数据的 50，用于加载器的帧索引。
  它不表示实际记录严格间隔 20ms：原始 timestep=1/250、save_freq=15，且包含段末帧。
  清单保留原始采样配置，未伪造物理时间戳或重采样。

最初的 294 帧 terminal-hold 原型已另存为
`local/robotwin_pi0_interface_terminal_hold_prototype`，对应模型位于
`models/robotwin_pi0/pilot_lora_merged_terminal_hold_prototype/`，报告在
`terminal_hold_prototype/`。发现末帧假设不成立后，已重新执行本报告中的全部验收。
原始 HDF5 始终保留。

## 训练、恢复与导出

| 环节 | `../runs/` 下的运行目录 | 结果 |
| --- | --- | --- |
| LoRA SFT | `sft_smoke_20261025_051831` | 第 2 步，退出 0 |
| SFT 恢复 | `sft_smoke_20261025_052223` | 第 3 步，退出 0 |
| 合并权重评测 | `eval_smoke_20261025_052425` | 2 条调试轨迹，1 条成功，退出 0 |
| PPO 更新 | `ppo_smoke_20261025_052544` | 第 1 步，退出 0 |
| PPO 恢复 | `ppo_smoke_20261025_052844` | 第 2 步，退出 0 |

SFT 使用双卡、rank 8、alpha 8、dropout 0、all-linear，micro/global batch 为 1/2。
原始模型来自官方任务 SFT checkpoint；这些小运行用于工程验收，不用于效果结论。
损失及梯度范数均有限；恢复后 adapter 实际更新，冻结基础权重保持不变。
每个 SFT 运行的 `data-source.json` 记录了实际 292 帧数据版本及 manifest 哈希。
单卡对照曾在 FSDP 状态导出时报错；最终采用已通过的双卡配置。
恢复检查覆盖模型/优化器及继续更新，没有验证中断前后数据采样顺序逐比特一致。

导出第 2 步 SFT 权重，保留独立 `adapter_weights.pt`，合并为原生 Pi0 的 632 个 tensor：
`models/robotwin_pi0/pilot_lora_merged/`。线性层等价检查通过；
在相同观测、相同噪声、FP32 且关闭 TF32 的完整 5 步 Euler 推理中，
`1×50×32` 动作窗口最大误差 **1.31e-06**，
平均误差 **1.26e-07**。
见 [lora-action-parity.json](lora-action-parity.json)。
生产 BF16 路径另已通过实际评测及 PPO。两条评测只用于接口检查，不能作为正式成功率。

PPO 恢复后动作专家和 value head 权重变化，冻结 VLM 权重不变。
SFT、合并、评测、PPO 使用同一份固定官方归一化统计，文件内容完全一致。
pilot 数据统计的计算与加载也已通过，但不把两条轨迹估计的统计作为正式统计。
两组 200 条正式数据池现已分别计算自身训练统计并验证加载。

## 环境与采集后端

MPLib 场景步进能通过，但专家采集长时间没有成功种子，有限时复验也未通过。
现采用官方 CuRobo v0.7.8，在本机 CUDA 12.8.93 / GCC 13.4 上编译，架构 SM 120，
`--maxrregcount=64`；参考 [NVIDIA issue #716](https://github.com/NVlabs/curobo/issues/716)。
源归档 SHA256：`702415790acf030aaf48c49f3381b8963d24b80bc4ab40d4a10808244ad5ee19`。
在线策略控制仍沿用 RLinf 环境中的 MPLib 配置，不执行专家规划。

渲染显式固定 `ROBOTWIN_DENOISER=none`，保留光追 32 spp / depth 8。
采集、评测和 PPO 使用同一协议；场景检查通过且不再有 OIDN 错误。
正式比较仍需按该协议固定初始化、训练预算、数据划分及充分的评测种子。

PEFT 所需配置接口由项目扩展 `compat/robotwin_pi0_ext.py` 补充；
LoRA adapter、projection FP32 islands 和 FSDP 边界在本地 SFT 配置中明确设置。
RLinf 上游源码未修改。339 个 Python 包的依赖检查通过。

## 扩量与续跑

两组已依次完成 10、20、30、40、50、100、150、200 条批次，采集和后处理均已退出。
最终任务完成时间为 **2026-10-25 08:43:41（北京时间）**。

| 数据池 | 成功轨迹 | 原始帧 | LeRobot 训练帧 | 转换/加载检查 |
| --- | --- | --- | --- | --- |
| Clean | 200 | 31,385 | 31,185 | 均通过 |
| Randomized | 200 | 31,131 | 30,931 | 均通过 |

每条原始轨迹去掉没有下一帧目标的末帧输入，原始帧与训练帧的差值为每组 200。
转换检查覆盖所有状态/动作，以及每条轨迹首、中、末帧的三相机像素。
最终汇总另核对了 400 个原始 HDF5 的 SHA256、唯一 seed、语言指令、配置快照和统计文件。
没有删除原始数据，也没有把接口用的两条轨迹追加计入 400 条。

`expanded-collection.pid` 保留历史启动 PID，任务完成后不代表活动进程。
采集脚本只调整官方采集器的数量，保留原有专家、seed 搜索、
失败过滤和成功回放判定。每个数据池保留配置快照、代码/运行库哈希和每批尝试记录，
后续协议发生变化时停止续采，避免混入同一数据池。

- 进度：`expanded-collection-state.json`；`completed_episodes` 表示完成整批核对的数量，
  当前批次逐条进度见日志。
- 日志：`../logs/collection/demo_{clean,randomized}_pi0_train200.log`。
- 原始数据：`data/robotwin_pi0/raw/{clean,randomized}/adjust_bottle/demo_*_pi0_train200/`。
- LeRobot：`data/robotwin_pi0/lerobot/local/robotwin_pi0_{clean,randomized}200/`。
- 统计：`{clean,randomized}200_norm_stats/norm_stats.json`。
- 验证证据：`{clean,randomized}200-converted-check.json`、
  `{clean,randomized}200-loader-check.json`、`expanded-data-check.json`。

自动转换、逐帧/图像核对、训练统计计算及真实 OpenPI batch 检查均已完成。
正式 S1（200 Clean）和 S2（100 Clean + 100 Randomized）已按用户选择于
2026-10-25 09:18:44 启动，各训练 10,000 步。
另已下载并转换原始共同 π0 基座，固定共同 Clean 训练池统计及训练预算，
训练数据与官方 150 个评测 seed 无交集。接口测试使用的官方任务 checkpoint 不用于这两组初始化。
训练记录和实时状态分别见 [SFT_EXPERIMENTS_20261025.md](SFT_EXPERIMENTS_20261025.md) 与
[sft-experiments-state.json](sft-experiments-state.json)。正式 PPO 尚未启动。

重新核验现有数据（工作区根目录）：

```bash
source experiments/robotwin_pi0/env.sh
python "$ROBOTWIN_EXPERIMENT_ROOT/scripts/verify_expanded_data.py"
```

需要从保留产物恢复采集/后处理时使用：

```bash
source experiments/robotwin_pi0/env.sh
setsid -f bash "$ROBOTWIN_EXPERIMENT_ROOT/scripts/run_expanded_collection.sh" \
  > "$ROBOTWIN_EXPERIMENT_ROOT/logs/expanded-collection-launch.log" 2>&1 < /dev/null
```

脚本通过独占锁避免重复启动，跳过已完成批次，保留中断转换的临时产物。
