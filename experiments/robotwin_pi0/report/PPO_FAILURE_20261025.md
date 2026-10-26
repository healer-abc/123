# PPO 首次运行失败诊断（2026-10-25）

首次失败目录：`../runs/ppo_smoke_20261025_032149/`。
失败发生在 actor 初始化时，FSDP 用 `dist._broadcast_coalesced` 同步模型参数，
CUDA 报 `an illegal memory access was encountered`，随后 NCCL watchdog 退出。
尚未进入 PPO 采样或优化器更新。可选 CuRobo 导入提示不是这次退出的原因。

## 最小复现与修复

`../scripts/check_broadcast.py` 不加载模型或仿真，只在两张 RTX 6000D 上广播
FP32/BF16 tensor。原 NCCL 2.26.2 在 1024 元素通过，在 1,048,576 元素失败。
`NCCL_PROTO=Simple` 仍失败；`NCCL_PROTO=LL` 能通过。对应日志：
`../logs/broadcast-check.log`、`broadcast-check-simple.log`、`broadcast-check-ll.log`。

NVIDIA 官方仓库有同类 Blackwell 双卡问题及新版修复记录：
[NCCL issue #2418](https://github.com/NVIDIA/nccl/issues/2418)。
本机对照测试也确认，预加载 NCCL 2.31.2 后，三个规模均通过，最大规模为
每卡同时广播 33,554,432 个 FP32 元素和同数量 BF16 元素（192 MiB）。
修复后日志：`../logs/broadcast-check-fixed.log`。

新版库独立安装在 `../runtime/nccl-2.31.2/`，`../env.sh` 通过 `LD_PRELOAD`
仅在激活该项目的进程中使用它。保留 PyTorch 2.7.1+cu128 及其原有包依赖；
`../scripts/install_env.sh` 已固定该额外运行库版本。

注意：`torch.cuda.nccl.version()` 返回 PyTorch 的构建版本 `(2, 26, 2)`，
不能据此判断预加载是否生效。测试脚本调用实际库的 `ncclGetVersion`，
确认当前运行版本为 `23102`（2.31.2）。

```bash
cd /home/zhsu/jyluo
source experiments/robotwin_pi0/env.sh
torchrun --standalone --nproc-per-node=2 experiments/robotwin_pi0/scripts/check_broadcast.py
bash experiments/robotwin_pi0/scripts/run_ppo_smoke.sh
```

此前小模型通信验收没有覆盖大消息广播路径，已补充上述检查。

## 实际 PPO 复验

运行：`../runs/ppo_smoke_20261025_033525/`，退出码 **0**。
保留官方算法设置，双卡、8 个环境、micro/global batch 2/32，完成一轮 PPO 更新。

- `Global Step: 1/1`，8 条训练轨迹，训练循环记录 70.255 秒（不含模型初始化）。
- `actor/total_loss=0.012`、`critic/value_loss=0.130`、`actor/grad_norm=210.8`，均有限。
- 每秒采样的显存峰值：GPU 0 为 17,669 MiB，GPU 1 为 18,391 MiB；
  这是离散采样结果，不是分配器精确峰值。
- `robotwin_pi0_ppo_smoke_2gpu/checkpoints/global_step_1/actor/` 已保存
  DCP 的两卡模型/优化器分片与 `.metadata`，以及 `model_state_dict/full_weights.pt`。
  完整权重文件可读取，共 640 个 tensor、6,487,745,737 字节。

恢复验证运行：`../runs/ppo_smoke_20261025_033830/`，从上述 `global_step_1`
加载，并设置 `runner.max_epochs=2`。退出码 **0**，完成 `Global Step: 2/2`，
保存 `global_step_2` 的 DCP 模型/优化器和完整权重。
`actor/total_loss=0.0026`、`critic/value_loss=0.065`、`actor/grad_norm=176.1`。
恢复运行每秒采样的显存峰值分别为 19,643 和 20,757 MiB。

权重检查确认实际更新：

- 原始 checkpoint → 第 1 步：`value_head.mlp.0.weight` 的 524,288 个 FP32 元素
  发生变化，最大绝对差约 `1.10e-4`。
- 第 1 步 → 第 2 步：同层 524,287 个元素变化；动作专家的
  `llm.layers.0.attn.q_proj.1.weight` 有 58,374 个 BF16 元素变化。
- 对照的冻结 VLM 层 `llm.layers.0.attn.q_proj.0.weight` 保持一致。
- 两次运行 TensorBoard 中已记录的标量均为有限值；机器可读结果见
  [ppo-smoke-check.json](ppo-smoke-check.json)。

恢复命令（从工作区根目录运行）：

```bash
bash experiments/robotwin_pi0/scripts/run_ppo_smoke.sh \
  runner.max_epochs=2 \
  runner.resume_dir=/home/zhsu/jyluo/experiments/robotwin_pi0/runs/ppo_smoke_20261025_033525/robotwin_pi0_ppo_smoke_2gpu/checkpoints/global_step_1
```

至此已完成本配置下的 PPO 更新、保存、恢复与继续更新的工程验收。

本次为训练管线检查，未启用 LoRA，训练轨迹的成功率不能当作正式评测。
此前记录的 OIDN 去噪兼容问题仍存在。
