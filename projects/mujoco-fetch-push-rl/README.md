# MuJoCo Fetch 强化学习项目

项目目录：`/home/zhsu/jyluo/projects/mujoco-fetch-push-rl`。按用户要求，不采用原手册的日程安排，先完成 CPU 准备和流程验收。


## 完整实验已完成（2026-10-23）

工程操作手册：[按官方 README 与 Zoo 命令执行 A–E 实验](report/OFFICIAL_README_STYLE_20261023.md)。包含安装、修改位置、完整配置与命令、恢复、评估和录像；配套源码在 `official_readme/`。

五组×三个训练种子，共 **15 个百万步模型、3,900 个最终测试回合**，CPU 完成。后台流程于北京时间 11:17 结束，当前无训练任务。

SAC+HER 稀疏奖励正常成功率 **98.7±1.2%**，较无 HER 稀疏组提高 **92.7 个百分点**。增益随机化正常成功率 **97.0±3.6%**；范围外平均成功率 E−D 为 **-3.2 个百分点**，如实保留负结果。

- [完整结果汇总](report/PROJECT_RESULTS_20261023.md) 与 [六页 PDF](report/REPORT.pdf)
- [简历条目](report/RESUME.md) 与 [面试问答](report/INTERVIEW_QA.md)
- [153 道面试题详细答案与源码、数据核对](report/INTERVIEW_153_ANSWERS_20261023.md)
- [90 秒案例视频](videos/project_demo_90s.mp4) 与 [案例元数据](videos/cases.json)
- [汇总 CSV](report/results.csv)、[分种子 CSV](report/results_per_seed.csv)、[交接记录](HANDOFF.md)

原 A–D 阶段汇总保留在 `archive/stage_summary_before_final_20261023/`。E 凌晨中断后从零补跑，相同参数和预算；中断、保存间隔变化和日志均已归档。增益随机化未观察到整体鲁棒性收益。

```bash
source /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/env.sh
cat report/REPORT.md
```

## 文件组织

- `run.py`、`run_matrix.sh`、`requirements.txt`、`SERVER_GUIDE.md`：用户上传的服务器脚本。
- `envs/fetch-rl/`：本项目的 **conda 环境**；`.cache/`：依赖和绘图库缓存；`.local/tmux/`：本地 tmux。
- `source_materials/`：原始压缩包、原说明、原始 SHA-256 清单和本地修改记录。
- `tools/`：CPU 验收、软件录像、中断恢复与汇总工具；`checks/`：结构化验收结果。
- `runs/`：训练配置、日志、TensorBoard 数据、模型、经验回放和检查点。
- `evaluations/`：独立评估；`videos/`：录像；`logs/`：命令日志；`docs/`：资源、路径迁移和工具记录。
- `archive/preconda/`：改用 conda 之前的验收结果，保留历史路径和运行记录。
- `HANDOFF.md`：完成情况与后续入口；`environment.yml`、`conda-explicit.txt`、`requirements-lock.txt`：环境复现文件。

## 激活环境

```bash
source /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/env.sh
```

该入口激活 conda 环境，进入项目目录，把缓存留在项目内，设置两个 CPU 线程并隐藏 GPU，同时提供本地 tmux 命令。也可单独激活：

```bash
conda activate /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/envs/fetch-rl
```

Python 3.11.17，CPU PyTorch 2.7.1+cpu；核心版本对齐原包：SB3 2.7.0、Gymnasium 1.2.0、Gymnasium-Robotics 1.4.0、MuJoCo 3.3.4、NumPy 2.2.6。此环境不作为 CUDA 已通过的环境使用。

全新目录中可按以下步骤复现，已有环境无需重建：

```bash
conda env create --prefix ./envs/fetch-rl --file environment.yml
conda activate ./envs/fetch-rl
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m pip check
```

Linux 下的 conda 基础依赖还提供 `conda-explicit.txt` 精确版本清单；pip 软件包使用锁文件安装。

## 原包与修复

原包保留在 `source_materials/fetch-server-kit.zip`，SHA-256 为 `018cbcb2e0d601b93b3e40a7a13fcbc5b2f731b34c33359a090ff52da0edecbc`。

`run.py` 的 `ActionGain.step` 对标量奖励增加 `float()` 转换，以通过 SB3 的返回类型检查；奖励数值和批量 `compute_reward` 保持原定义。修改前后哈希见 `source_materials/local_changes.json`，其他原包文件未改。原手册保留原文，实际路径、环境、时间安排及设备限制以本 README 和 HANDOFF 为准。

## CPU 验收

```bash
bash tools/run_cpu_preflight.sh
python tools/check_interruption.py
python tools/check_cpu_rendering.py
bash tools/record_cpu_video.sh --checkpoint runs/reach_smoke/final --episodes 1 --out videos/reach_smoke_cpu.mp4
```

验收包含 Reach/Push 的稀疏与密集环境、Reach 3000 步训练、10 回合独立评估、恢复到累计 5000 步、Push A–E 各组短训练、D/E 五档动作增益评估、周期检查点、HER 恢复及 SIGTERM 中断后继续训练。另验证 PPO 恢复到 4096 步。命令拒绝覆盖既有训练、评估及视频输出；结果已存在时直接查阅，不必重复。

软件录像使用 Mesa llvmpipe，限定只加载 Mesa EGL 驱动，不需要 GPU。本地 tmux 由 Ubuntu 包解包到项目中，依赖库仅由 tmux 启动包装脚本设置，没有安装到系统目录。

5 万步 D 组 Push CPU 基准位于 `runs/speed_cpu/`。该次基准在先前 uv 隔离环境完成，核心软件版本与最终 conda 环境相同；保存的模型和回放在 conda 中重新加载并验证。短训练、基准和示例视频仅验收流程，不能代表策略收敛效果。

早期 CPU 流程验收汇总见 `checks/cpu_stage_summary.json`。2026-10-22 起已按完整实验设计进入预实验和正式矩阵准备，最新规则以 `EXPERIMENT.md` 为准，当前状态见 `report/STATUS.md`。恢复会重置仿真，不承诺与连续运行逐步一致。

## 参考

- [Farama FetchPush 环境说明](https://robotics.farama.org/envs/fetch/push/)
- [SB3 HER 保存加载接口](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)
