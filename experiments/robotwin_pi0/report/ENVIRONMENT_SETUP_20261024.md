# π0 + RoboTwin 环境验收（2026-10-24）

训练依赖安装在独立 uv 环境 `/home/zhsu/jyluo/experiments/robotwin_pi0/.venv`。
Python 为 3.11.17，现有 OpenPI/LIBERO/Fetch 等实验环境未修改。
另有项目独立 Conda `toolchain`，用于 Git、CMake、FFmpeg 等系统工具。

| 组件 | 验收版本 |
| --- | --- |
| PyTorch / CUDA runtime | 2.7.1+cu128 / 12.8 |
| RLinf / RLinf OpenPI | 0.4.0 / 0.1.1 |
| Transformers / OpenPI fork | 均为 4.53.2，共享模块目录最后由官方 OpenPI fork 安装 |
| PEFT / tokenizers | 0.17.1 / 0.21.4 |
| SAPIEN / MPLib | 3.0.1 / 0.2.1 |
| Ray / LeRobot | 2.59.0 / 0.3.3 |
| TorchCodec / Gymnasium | 0.5 / 0.29.1 |

`requirements-lock.txt` 保存实际 Python 依赖；`toolchain-explicit.txt` 保存工具环境清单。
官方源码以压缩包下载，校验值在 `source-archive-sha256.txt`。
下载后查询到的分支引用为 RLinf `main: 30e0236981c6a1dc19202de64116834c12673828`、
RoboTwin `RLinf_support: dca9ec682688821c42944eacd7f5dd9bfb15396f`，原始查询结果位于 `source_materials/*-ref.txt`。

## 已完成的检查

- 两张 RTX 6000D 均识别为计算能力 12.0，PyTorch 构建包含 `sm_120`。
- 两卡分别执行 BF16 反向传播和 AdamW 参数更新。
- 两个 rank 完成 NCCL all-reduce，以及 FSDP 反向传播和优化器更新。
- Ray 启动独立本地实例，两个 GPU worker 分别获得 GPU 0/1，均使用本项目 `.venv`。
- RLinf、OpenPI、PEFT、LeRobot、TorchCodec、RoboTwin 任务等模块导入成功。
- SAPIEN 使用 NVIDIA Vulkan 完成无头渲染。
- 官方三份 RoboTwin 资产压缩包全部通过与 Hugging Face LFS SHA-256 的比对，已解压并更新机器人资产路径。
- `adjust_bottle` 在 Clean 和 Randomized 场景均成功创建、重置、读取三路相机和 14 维状态，并执行一次保持当前姿态的 action chunk。

场景检查使用官方首个评测 seed `100100000`，仅检查环境接口，不计算任务成功率。
图像为 `240 × 320 × 3`，相机数组具有非零方差；截图位于
`adjust_bottle_clean.png`、`adjust_bottle_randomized.png`。
机器可读结果在 `environment-check.json`、`ray-check.json`、`task-check-*.json`。
双卡通信日志为 `../logs/nccl-check-cumem.log`。

## 本机兼容设置

使用官方 `adjust_bottle` 默认的 MPLib 规划后端。未安装 CuRobo、PyTorch3D 和
FlashAttention 2；RoboTwin 导入时会打印前两项缺失的提示，这是其可选依赖探测，
上述场景检查使用 MPLib 成功完成。

默认双卡 NCCL 初始化未在观察期间完成；使用以下设置后通信与 FSDP 检查通过：
`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`、`NCCL_CUMEM_HOST_ENABLE=0`、
`NCCL_CUMEM_ENABLE=0`、`NCCL_SOCKET_IFNAME=lo`。已写入本项目 `env.sh`，
这些结果不等于已定位唯一根因，也不代表通信吞吐已优化。首次成功的 NCCL
初始化日志记录约 66 秒 kernel 初始化时间。

版本安装顺序已适配：先运行官方依赖安装，再将 stock Transformers 固定到
4.53.2，最后重新安装同版本 OpenPI fork，避免共享目录与 tokenizers 版本冲突。
FFmpeg 动态库从项目 `toolchain/lib` 加载。大型缓存、数据及模型路径均指向工作区；
Ray 的临时运行目录使用较短的 `/tmp/robotwin-pi0-ray-<uid>`，避免 Unix socket 路径过长。

## 仍需单独验收

RoboTwin 上游代码固定启用 OIDN GPU 去噪，而该版本在 RTX 6000D 上报告
`unsupported device type: CUDA`。当前已获得有效的未去噪光追图像并完成环境步进；
正式采集、训练和与官方基准比较前，需要验证兼容的去噪方案，并固定一致的渲染协议。
本次保留上游代码，未静默替换渲染方式。

本次未下载 π0 任务 checkpoint，未启动正式采集、SFT 或 PPO。
小网络 FSDP 检查不等于完成一次 π0 PPO 更新，checkpoint 保存/恢复、
双卡 PPO 的规模与峰值显存，以及 LoRA SFT → PPO 的权重衔接均未验收。

## 使用

```bash
cd /home/zhsu/jyluo
source experiments/robotwin_pi0/env.sh
python experiments/robotwin_pi0/scripts/check_env.py
```

已准备 `configs/demo_clean_pi0.yml` 和 `configs/demo_randomized_pi0.yml`，各为 5 条
pilot 采集配置，并链接到 RoboTwin 的 `task_config/`。尚未启动采集。
模型下载入口为 `scripts/download_model.py`。

## 2026-10-25 补充

小网络通信检查遗漏了大消息路径：首次实际 PPO 在 FSDP 参数广播时触发
NCCL 2.26.2 非法内存访问。已用独立广播脚本复现，并为该项目预加载
NCCL 2.31.2 修复；已完成一次真实 PPO 更新和 checkpoint 保存。
环境的 320 个 Python 包依赖检查仍通过。完整对照证据及恢复检查结果见
[PPO 失败诊断与修复](PPO_FAILURE_20261025.md)。上文“未验收”为初次环境检查时的状态。

数据接口阶段进一步补装 CuRobo v0.7.8，并固定 CUDA 12.8 / GCC 13 的项目工具链。
MPLib 场景步进可用，但专家规划未通过；实际专家采集改用已验证的 CuRobo。
采集、评测、PPO 显式统一为 `ROBOTWIN_DENOISER=none`，原 OIDN 提示已通过
明确的渲染协议处理。自采数据 → LoRA SFT → 合并权重 → PPO 的真实更新与恢复
均已通过，见 [数据接口验收](DATA_INTERFACE_20261025.md)。
