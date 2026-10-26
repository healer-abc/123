# Fetch 项目交接

## 2026-10-23 面试题逐题核对补充

- [153 道题详细答案](report/INTERVIEW_153_ANSWERS_20261023.md) 保留原题编号，区分已核实、解释假设和建议实验；原题归档于 `source_materials/interview_153_questions_20261023.txt`。
- 从已有 CSV 复核 A 三个种子最终距离都与零动作逐回合一致；补充初始未成功94场景的条件成功数，C没有破坏6个初始成功场景。B seed0验证94万步90%、最终55%，最终测试53%。均为事后分析，不替换正式指标。
- 修正 `report/INTERVIEW_QA.md` 的 SAC 更新顺序与代码来源表述；未修改冻结 runner。
- 新增 `tools/audit_interview_evidence.py`、`tools/probe_interview_runtime.py` 和 `learning/interview_eval_100.py`、`learning/interview_clip_observer.py`，用于证据读取及面试练习。运行时核对31维输入、−4目标熵、奖励阈值及网络参数量。
- 结果与文档核验记录：`checks/interview_evidence_20261023.json`、`checks/interview_runtime_20261023.json`、`checks/interview_document_acceptance_20261023.json`。无新增训练，旧正式结果、模型和视频不变；新评估练习仅作独立校验，不并入3900回合。

## 2026-10-23 官方入口操作手册补充

更新：2026-10-23T17:14:41.569758+08:00。详细手册：`report/OFFICIAL_README_STYLE_20261023.md`；配套 `official_readme/`。

- 采用官方 RL Zoo v2.7.0 原始 train/enjoy/record_video 入口；官方源码 tag commit `ab4aadb57c6c42abcf1016318c2bebe35c4c1270`，逐文件核对 312 个原文件未改。
- A–E YAML、增益环境和逐回合统计属于新增复现材料，与既有 run.py 结果区分。原正式模型、数据、报告和视频哈希均保持不变。
- Zoo 附加依赖安装在项目 conda 环境之上的 `official_readme/.venv` 隔离层，未升级原 conda 核心依赖。
- 五组短训练均实际更新；官方回放、模型+回放恢复、软件录像、抽样配对评估和文档最小 API 例子通过。验收见 `official_readme/checks/acceptance.json`。
- 未启动新百万步矩阵；短验证不是新增正式结果。可选绘图命令只核对源码、未安装 extras 执行。

## 2026-10-23 最终交付（优先于全部历史记录）

后台流程完成：**2026-10-23 11:17（北京时间）**；交付复核：2026-10-23T14:36:42.946061+08:00。

五组×三个训练种子全部完成：15 个最终模型、3,900 个最终测试回合，未使用 GPU。目前无本项目训练任务。

- SAC+HER 正常成功率 98.7±1.2%，较无 HER 稀疏组 +92.7 个百分点。
- 增益随机化正常成功率 97.0±3.6%；范围外 D=89.8±1.9%、E=86.7±8.5%，差值 -3.2 个百分点，未观察到整体收益。
- [完整结果](PROJECT_RESULTS_20261023.md)、[六页 PDF](REPORT.pdf)、[简历](RESUME.md)、[90 秒视频](../videos/project_demo_90s.mp4) 已交付。
- E 凌晨中断记录和旧数据保留；07:30 相同参数从零补跑，最终模型满足原定预算。
- 视频为六个真实测试案例；成功示例初始距离大于 8 cm 且有实际物体移动。展示案例选择不改变全部测试统计。

本项目流程已完成，无需为追求正面结果追加训练。旧阶段文档为历史记录，最终定量表述以 REPORT.md、results.csv 和 RESUME.md 为准。


工作入口：`source env.sh`；完整报告重建：`python tools/summarize_results.py`。视频生成入口 `tools/make_case_video.py` 对已有完整视频验证模型身份、成功示例初始条件和帧数，不覆盖。不要重启训练矩阵。

核验：15 个模型身份、预算、完成标记、SAC 回放及独立评估一致，3,900 回合测试齐全；六页 PDF；视频 960×720、25 fps、2,250 帧即 90 秒。中断原因仍未知，不推断 OOM。最终记录见 `checks/final_delivery_20261023.json`。

## 2026-10-23 最新结果与恢复（优先于下方历史记录）

核验时间：2026-10-23T07:33:13.855456+08:00。

- A/B/C/D 四组×三个种子共 12 个最终模型及独立评估已完成，共 2,400 个最终测试回合。
- 正常成功率均值±样本 SD：A 6.0±0.0%，B 21.7±27.1%，C 6.0±0.0%，D 98.7±1.2%。C→D 差值 +92.7 个百分点。
- 当前成果汇总：[PROJECT_RESULTS_20261023.md](report/PROJECT_RESULTS_20261023.md)；四页可分享 PDF：[PROJECT_RESULTS_20261023.pdf](report/PROJECT_RESULTS_20261023.pdf)；当前简历条目：[RESUME_ABCD_20261023.md](report/RESUME_ABCD_20261023.md)。
- A–D 学习曲线、A/B 耗时曲线、D 单组鲁棒性图已从真实数据导出。D/E 对照与最终六页报告、90 秒视频仍待 E，不能标记全项目完成。
- E 三个进程于 00:12 左右同时中断，停在 197,200/152,600/181,200 步；无可用模型/回放检查点。原因尚未确认，未观察到主机重启；不可断言 OOM。
- 07:30 从零重新启动 E 三个相同种子，保留相同学习参数、冻结 runner 和百万步预算。仅 E 保存间隔改为 100,000 步；旧协议、日志和数据保留在 `archive/E_interruption_20261023_073019/`。结构化记录：`report/recovery_20261023.json`。
- 重启后已确认步数增长：{0: 21800, 1: 20800, 2: 21800}。这是当前快照，实际以各 `progress.csv` 为准。仍使用 CPU，每进程两线程。
- 协调器现在使用项目独立 tmux socket：`.local/fetch-experiment.sock`，会话 `fetch-recovery-20261023`；日志 `logs/full_experiment_recovery_20261023.log`。不要重复启动。
- 零动作事后诊断同样得到 6%，A seed0 距离逐回合与零动作完全一致；该基线成功率不能证明学会推物。诊断未用于调整超参数或选模型。

```bash
source /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/env.sh
cat report/STATUS.md
tmux -S "$PWD/.local/fetch-experiment.sock" list-sessions
tail -f logs/full_experiment_recovery_20261023.log
```

后续自动执行 E 最终五档增益评估、真实案例视频和完整五组报告；中断输出和已完成模型不覆盖。四组阶段汇总可用 `python tools/summarize_completed_abcd.py` 重建；完整汇总入口仍拒绝缺失 E 的结果。

更新日期：2026-10-22 12:07（Asia/Shanghai）。预实验已通过，正式三种子矩阵已启动，正式结果尚未完成。

## 最新目标与约定

用户补充完整实验设计，要求不按原时间表推进、准备工作动态调整、保持简单，尽快形成简历项目。保留已有单文件训练入口，正式内容为五组×三个训练种子×每模型 100 万步；不扩展算法或任务。

唯一实验配置：`configs/experiment.json`。执行规则与产物标准：`EXPERIMENT.md`。工作目录仍为 `~/jyluo/projects/mujoco-fetch-push-rl/`，使用已有 conda 环境。

## 当前实际运行

- seed99 预实验在 **654,410 步**提前保存，检查点为 `runs/push_pilot_s99_1m_20261022/interrupted/`，模型、回放和完成标记齐全。
- 最近验证 62/64 万步成功率为 85%/95%，最近两次均值 90%；已超过原学习门槛。
- 重新加载最终保存点独立验证 20 回合（10000–10019）：**19/20=95%，平均最终距离 0.02398 m**；目录 `evaluations/pilot_s99_admission_20261022/`。
- 按用户要求动态缩短准备：不再等无必要的 100 万步预实验，较继续跑满预计省约 80 分钟。只提前结束预实验，正式五组、三个种子和每模型 100 万步预算均未改。
- 正式 D 组 seed0、seed1、seed2 已同时从零启动，每进程两线程，只使用 CPU；初期交互步数已确认增长。
- 后续每个种子按 D→C→B→A→E 自动执行，完成后独立测试与生成报告、图表、视频和简历条目。
- 协调器 tmux：`fetch-complete-20261022`；总日志：`logs/full_experiment.log`；各组日志如 `logs/formal_D_seed0.log`。
- 实时准确状态见 `report/STATUS.md` / `report/status.json`。预实验结论只属于 seed99 的验证集，不是最终三种子测试结果。

提前结束和复测记录：`report/pilot_acceptance_20261022.json`。旧预实验进程是主动 SIGTERM 后正常保存，旧协调器日志中的退出码 130 是该次主动停止，不是数值训练故障。协调器退出后才修改接续逻辑并重新启动；旧协议归档在 `source_materials/protocol_before_pilot_acceptance_20261022.json`。运行中的冻结训练源码未改，第一项目未被操作。

## 已补齐的交付工作

- 完整设计原文归档：`source_materials/full_experiment_design_20261022.txt`。
- 冻结训练源码：`source_materials/runner_frozen_20261022.py`，哈希在协议中。
- `learning/her_relabel_demo.py` 用真实 50 步轨迹生成目标重标记样本，CSV/JSON 已生成；正式 HER 仍使用 SB3。
- 三张统计图、结果表、六页中文 PDF、简历条目和 90 秒案例视频的生成程序已实现；正式结果不全时拒绝生成最终结论。
- 字体、六页 PDF 版式、真实预实验图及视频标题导出已验收；`checks/report_layout_smoke.pdf` 只是版式测试，不是实验报告。
- 15 次训练的原始配置和代码哈希、缺失结果拒绝机制已检查；完整端到端导出还需等待真实正式结果。

## 如何查看

```bash
source /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/env.sh
cat report/STATUS.md
tail -f logs/full_experiment.log
```

完整协调器已在 tmux 中运行，不要再启动第二个矩阵。进程锁拒绝重复协调器；若错误退出，先看状态和日志。重新运行只跳过身份匹配的完整结果；不完整目录需用原 `--resume` 入口显式恢复至新目录，不能覆盖。

完成后的预期产物：`report/results.csv`、`report/results_per_seed.csv`、`figures/learning_steps.png`、`figures/learning_time.png`、`figures/robustness.png`、`report/REPORT.md`、`report/REPORT.pdf`、`report/RESUME.md`、`videos/project_demo_90s.mp4`。当前这些正式产物尚未产生，不要将 smoke/pilot 数据写成正式结果。

## 2026-10-18 CPU 验收记录

更新日期：2026-10-18（Asia/Shanghai）。CPU 阶段验收完成。

项目统一位于 `~/jyluo/projects/mujoco-fetch-push-rl/`，使用专用 conda 环境。用户要求不沿用原说明的时间表，先做不依赖 GPU 的流程。

## 环境与入口

```bash
source /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/env.sh
```

- conda 环境：`envs/fetch-rl/`，Python 3.11.17、CPU PyTorch 2.7.1+cpu。
- SB3 2.7.0、Gymnasium 1.2.0、Gymnasium-Robotics 1.4.0、MuJoCo 3.3.4、NumPy 2.2.6，与原包核心版本一致；pip 依赖检查通过。
- 环境复现：`environment.yml`、`conda-explicit.txt`、`requirements-lock.txt`。激活入口限制线程为 2、隐藏 GPU、进入项目并设置本地缓存。
- 本地 tmux 3.4：`.local/tmux/`；激活后可直接使用 `tmux`，创建和关闭独立测试会话已通过。没有更改系统包或全局 conda 配置。
- 临时 uv 环境及误建的 `~/lyluo` 目录已清理；先前结果保留于 `archive/preconda/`，历史绝对路径映射见 `docs/path_relocations.json`。

## 来源与修复

原包位于 `source_materials/fetch-server-kit.zip`，原始哈希清单见 `source_materials/archive_manifest.json`。原 `run.py` 的 `ActionGain.step` 返回 NumPy 标量，SB3 拒绝该奖励类型；已将该处返回值转为 Python `float`，奖励数值和向量化 `compute_reward` 不变。修改前后哈希及原因见 `source_materials/local_changes.json`。其他原包文件未改。

## 已完成验收

下列短训练和运行时验收已在最终 conda 环境通过：

| 项目 | 结果与位置 |
|---|---|
| Reach/Push 稀疏与密集环境 | 四种均通过，覆盖 observation/action、奖励重算及批量计算、50 步截断和相同 seed 重放；`checks/cpu_environment.json` |
| Reach D 组短训练 | 3000 步、2900 次更新；`runs/reach_smoke/final/` |
| 独立加载评估 | 10 回合，seed 10000–10009；`evaluations/reach_smoke/` |
| SAC＋HER 恢复 | 从 3000 累计到 5000 步，只新增 2000；模型、回放、HER 采样及恢复后更新通过；`runs/reach_resume/` |
| 周期检查点 | 1000、2000、3000、4000、5000 步均可加载；`checks/acceptance.json` |
| Push A–E 短训练 | A 2048 步，B/C/D/E 各 300 步，均实际更新且参数有限；`runs/smoke_*_push/` |
| PPO 恢复 | 2048→4096 步，更新计数 10→20，参数确实更新；`runs/smoke_A_resume/` |
| 动作增益 | D/E 五档评估；相同 seed 的场景初态配对一致；E 组增益按回合采样且回合内固定 |
| SIGTERM 中断恢复 | 本测试子进程在 211 步保存后恢复至 511 步；`checks/interruption.json` |
| CPU 软件渲染与视频 | Mesa llvmpipe、480×480，51 帧/25 fps；`videos/reach_smoke_cpu.mp4` |

5 万步 D 组 Push CPU 测速已完成：**513.738 秒，97.326 步/秒**，4 万次网络更新，保存了完整模型和回放。数据在 `runs/speed_cpu/`。该次计时在先前 uv 隔离环境完成，核心软件版本与最终 conda 环境逐项相同；基准检查点已在 conda 中重新加载并成功更新。GPU 尚未测速，不能据此完成 CPU/GPU 设备选择。

短训练和基准只验收流程，没有验证策略收敛。Reach 短评估为 0/10，Push 基准两次验证成功率为 0；后续仍需长预实验检验学习趋势。

最终机器可读汇总：`checks/cpu_stage_summary.json`。日志：`logs/`。原 `run_matrix.sh` 和新增脚本语法检查通过。各训练与评估入口拒绝覆盖已有输出；无需为了查看结果重跑验收。

## 当前活动与后续

本轮训练、评估、录像及测试 tmux 会话均已结束，没有后台训练队列，未使用 GPU。第一项目的训练、评测、交接文件未被操作。

后续顺序：获得 GPU 使用条件后完成同配置 GPU 测速，再选择训练设备；随后按 seed 99 做 Push 单种子预实验，检验学习趋势并冻结配置，通过后再考虑 A–E 正式多种子矩阵。没有启动长预实验、正式矩阵或 GPU 排队，不按原日程自动推进。
