# π0 + RoboTwin 环境

本项目采用独立 **uv / Python 3.11** 环境，训练依赖安装在
`experiments/robotwin_pi0/.venv`。`toolchain` 是单独的 Conda 工具环境，
只提供 Python、Git、CMake 等命令；不承载训练依赖。

安装及基础验收已完成，详见 [环境验收记录](report/ENVIRONMENT_SETUP_20261024.md)。
已通过双卡 BF16、NCCL/FSDP、Ray 双卡调度，以及 Clean/Randomized 任务步进检查。
官方仿真资产和 π0 任务 checkpoint 已准备；PPO 验收进展见下文。

2026-10-24 后续进展：用户已下载官方任务 checkpoint，双卡、两个环境的官方
推理入口已跑通，2 条调试轨迹均记录成功。详见
[阶段 0 推理检查](report/PHASE0_EVAL_SMOKE_20261024.md)。

2026-10-25：首次 PPO 在 FSDP 参数广播时触发 NCCL 2.26.2 的非法内存访问。
项目现预加载独立的 NCCL 2.31.2，已通过同机大消息广播复现检查。
实际 PPO 单步更新、checkpoint 保存、恢复并继续至第 2 步均已通过，两个运行退出码
均为 0。结果见 [PPO 失败诊断与修复](report/PPO_FAILURE_20261025.md)。

```bash
# 重跑小规模推理检查
bash experiments/robotwin_pi0/scripts/run_eval_smoke.sh
# 一次 PPO 更新的冒烟配置
bash experiments/robotwin_pi0/scripts/run_ppo_smoke.sh
```

目录沿用工作区现有结构：

- `projects/RLinf`：官方 RLinf 源码。
- `projects/RoboTwin`：官方 `RLinf_support` 分支及 `assets/` 仿真资源。
- `experiments/robotwin_pi0`：环境、脚本、配置、安装日志和验证报告。
- `data/robotwin_pi0`：后续采集和 LeRobot 数据。
- `models/robotwin_pi0`：后续 π0 基座和任务 checkpoint。

## 激活与检查

```bash
cd /home/zhsu/jyluo
source experiments/robotwin_pi0/env.sh
which python
python --version
python experiments/robotwin_pi0/scripts/check_env.py
torchrun --standalone --nproc-per-node=2 experiments/robotwin_pi0/scripts/check_nccl.py
torchrun --standalone --nproc-per-node=2 experiments/robotwin_pi0/scripts/check_broadcast.py
```

环境激活脚本仅影响当前 shell，将 Hugging Face、uv、Torch、Triton、
等缓存指向项目目录，并设置 `ROBOT_PLATFORM=ALOHA` 和 NVIDIA Vulkan 驱动。
Ray 临时目录使用 `/tmp/robotwin-pi0-ray-<uid>` 以满足 Unix socket 路径长度限制。
新开终端即可回到原来的 shell 环境。

PyTorch 固定为 `2.7.1+cu128`，与 RLinf OpenPI 依赖匹配，并支持 RTX 6000D
的 `sm_120`。专家采集使用已验证的 CuRobo v0.7.8；在线 π0 控制的环境仍沿用
RLinf 配置中的 MPLib（不调用专家规划）。FlashAttention 2 和 PyTorch3D 未安装，
π0 使用 PyTorch 注意力实现。采集配置已链接到 RoboTwin 的 `task_config/`。
CuRobo 使用项目工具链中的 CUDA 12.8 / GCC 13 编译，脚本为 `scripts/build_curobo.sh`。
本机 MPLib 的专家规划未通过，不能用场景步进检查代替专家采集验收。

`env.sh` 通过 `LD_PRELOAD` 使用 `runtime/nccl-2.31.2/` 的项目专用通信库，
修复本机 Blackwell 双卡的大消息广播崩溃。Python 包元数据仍保留 PyTorch
精确依赖的 NCCL 2.26.2；诊断脚本直接查询实际运行库版本。

RTX 6000D 上默认 OIDN GPU 去噪不兼容，项目现通过 `ROBOTWIN_DENOISER=none`
显式固定无去噪光追协议，采集、评测和 PPO 使用一致设置，场景检查通过。
后续评测需重新在该协议下建立基线，不能直接当作官方去噪基准。

## 重新安装

```bash
bash experiments/robotwin_pi0/scripts/install_env.sh
```

脚本复用 RLinf 官方安装器，仅适配项目路径、CUDA 12.8 wheel 来源和 MPLib
依赖安装。官方 `pyproject.toml` 的临时版本覆盖由安装器退出时还原。
不要在 RLinf 目录直接运行无参数的 `uv sync` 或 `uv run`，它们会按上游默认
版本重新解析依赖；激活后使用 `python` 和 `torchrun`。

安装日志在 `logs/env_install.log`，实际依赖版本在 `report/requirements-lock.txt`。
本机下载源直连速度更快，激活脚本仅为对应下载域名设置 `NO_PROXY`。

## 仿真资产与模型

```bash
source experiments/robotwin_pi0/env.sh
python experiments/robotwin_pi0/scripts/download_assets.py
python experiments/robotwin_pi0/scripts/download_model.py
```

资产来自 `TianxingChen/RoboTwin2.0`，三个压缩包共约 14 GiB，解压后占用更多空间。
任务模型为 `RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`。
这两条命令均不启动训练。

小规模 LoRA SFT → 权重合并 → 评测 → PPO 更新及恢复均已验收；
正式 S1/S2 对照训练各 10,000 步及模型导出均已完成。
独立评测已于 2026-10-26 完成四组各 150 个请求 seed。
Clean 成功率 S1 44.67%、S2 62.00%；Randomized 为 S1 2.52%、S2 74.79%。
Randomized 两组均排除相同的 31 个初始化回退请求，按共同有效的 119 个场景统计。
完整结果与范围说明见 [最终评测报告](report/SFT_EVAL_RESULTS_20261026.md)。
协议与空间预算见 [SFT 评测记录](report/EVAL_SFT_20261026.md)，
实时阶段见 [eval-experiments-state.json](report/eval-experiments-state.json)。

## 小规模专家采集

阶段 0 的 PPO 更新和恢复已通过，可先采集 Clean / Randomized 各 5 条。

```bash
cd /home/zhsu/jyluo
source experiments/robotwin_pi0/env.sh
cd "$ROBOTWIN_PATH"
bash collect_data.sh adjust_bottle demo_clean_pi0 0
bash collect_data.sh adjust_bottle demo_randomized_pi0 0
```

最后一个参数 `0` 是 GPU 编号。建议先完成 Clean 并检查输出，再运行 Randomized。
配置文件中的 `save_path` 使用相对于 RoboTwin 根目录的路径，以兼容上游语言
指令生成器的路径拼接；`projects/RoboTwin/data/adjust_bottle/demo_*_pi0` 已链接到
实际输出目录，使其固定输出的 `instructions/` 与 HDF5 位于同一数据集内。
未修改上游采集或语言生成代码。

实际目录：

- `data/robotwin_pi0/pilot/clean/adjust_bottle/demo_clean_pi0/`
- `data/robotwin_pi0/pilot/randomized/adjust_bottle/demo_randomized_pi0/`

每组应核对 `seed.txt`、`scene_info.json`、5 个 `data/episode*.hdf5` 和对应的
`instructions/episode*.json`，并回放检查图像及动作。官方 shell 脚本没有严格传播
Python 子程序的失败状态，不能仅凭命令退出码判断采集完整。
两组 pilot 已完成，各有 5 条 HDF5 和对应语言指令；各一条已转换为
`data/robotwin_pi0/lerobot/local/robotwin_pi0_interface`，共 292 帧。
原始状态/动作逐帧核对、三相机抽样像素核对、实际 OpenPI batch 及统计计算均通过。
已先用每组一条跑通 HDF5 → LeRobot 和 π0 DataLoader，再完成各 200 条扩量。
最新 SFT、权重合并、PPO 与扩量进展见 [数据接口记录](report/DATA_INTERFACE_20261025.md)。

自采数据的 LoRA SFT 更新/恢复、权重合并、实际评测及 PPO 更新/恢复均已通过。
完整验收结果为 [data-interface-check.json](report/data-interface-check.json)。
扩量已完成：Clean **200 条 / 31,185 帧**，Randomized **200 条 / 30,931 帧**。
任务状态见
[expanded-collection-state.json](report/expanded-collection-state.json)。
两组均已转换成 LeRobot、核对全部状态/动作和抽样图像、计算各自训练统计，
并通过实际 OpenPI 数据加载器检查。
最终验收见 [expanded-data-check.json](report/expanded-data-check.json)。
原始数据位于 `data/robotwin_pi0/raw/{clean,randomized}/adjust_bottle/demo_*_pi0_train200/`，
训练数据位于 `data/robotwin_pi0/lerobot/local/robotwin_pi0_{clean,randomized}200/`，
归一化统计位于 `report/{clean,randomized}200_norm_stats/norm_stats.json`。
正式 S1（200 Clean）和 S2（100 Clean + 100 Randomized）已使用同一原始 π0 基座、
同一 Clean 训练池统计、同一训练预算（各 10,000 步 / global batch 16 / rank 8）完成训练。
两组采用双卡 FSDP 分片并发运行，共享 GPU 0、1；资源压测中每卡合计峰值约 38–39 GiB。
配置、资源测试和日志位置见 [正式 SFT 记录](report/SFT_EXPERIMENTS_20261025.md)，
实时步数与阶段见 [sft-experiments-state.json](report/sft-experiments-state.json)。
训练与导出验收见 [sft-completion-check.json](report/sft-completion-check.json)。

官方说明：[RLinf RoboTwin 安装](https://github.com/RLinf/RLinf/blob/main/docs/source-zh/rst_source/examples/embodied/robotwin.rst)。
项目初版方案保存在 `source_materials/PROJECT_PLAN.md`。
