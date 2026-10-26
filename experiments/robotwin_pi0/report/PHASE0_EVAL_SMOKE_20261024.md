# 阶段 0：官方 π0 checkpoint 推理检查

2026-10-24 已检查用户下载的 `RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`。
两个 safetensors 分片均可读取，索引中的 786 个 tensor 与实际分片键完全匹配。
这是分片结构检查，不是重新计算整个 checkpoint 的下载哈希。
机器可读记录见 `checkpoint-check.json`。

本包的归一化文件位于 `physical-intelligence/robotwin/norm_stats.json`，
本地配置已明确引用此路径；当前官方评测模板中带 `adjust_bottle/` 的默认路径
在本下载包中不存在。

## 已执行的推理检查

运行目录：`../runs/eval_smoke_20261024_230806/`。
入口为官方 `evaluations/eval_embodied_agent.py`，本项目配置为
`../configs/eval_smoke_2gpu.yaml`，重跑脚本为 `../scripts/run_eval_smoke.sh`。

- 两张 RTX 6000D，两个并行环境，每张卡一个 rollout worker 和一个 env worker。
- 沿用官方 π0 的 ALOHA、三相机、Clean 场景、50 步 action chunk、200 步上限。
- 5 次 Euler 推理步；为首次连通性检查显式关闭 `torch_compile`。
- 两个 worker 均成功加载约 3.24B 参数的模型，并正确加载归一化统计。
- 官方加载器在内存中转换旧 OpenPI checkpoint 键；纯推理模型按设计忽略 8 个 value-head tensor。
- 退出码 0，`num_trajectories=2`、`episode_len=200`、`success_once=1.0`、`success_at_end=1.0`。
- 评测循环记录 16.526 秒，不含 Ray 启动和模型加载。
- 日志、TensorBoard 和两段视频均保存在运行目录。

**这里只是 2 条轨迹的连通性检查，不能作为官方 S0 的正式成功率。**
完整评测仍需要核对 150 个种子的实际覆盖和重复情况。
此前记录的 OIDN GPU 去噪兼容问题尚未处理，本次保留上游渲染代码，
未将该小规模运行认定为正式可比较的基线。

## 下一步：双卡 PPO 冒烟运行

已准备 `../configs/ppo_smoke_2gpu.yaml` 和 `../scripts/run_ppo_smoke.sh`。
配置基于本地官方 `robotwin_adjust_bottle_ppo_openpi.yaml`，保留 PPO/GAE、
Flow-SDE、任务及模型结构，仅设置本机路径并缩小资源规模：

| 字段 | 候选值 |
| --- | --- |
| GPU | 0、1，actor/env/rollout 共置 |
| 训练环境数 | 8 |
| rollout_epoch / update_epoch | 1 / 1 |
| micro / global batch | 2 / 32 |
| max_epochs | 1 |
| save_interval | 1 |
| 中途评测 | 关闭 |
| 梯度检查点 / offload | 开启 |
| torch_compile | 关闭 |

配置已完成 Hydra 组合和插值解析，结果在 `ppo-smoke-resolved.yaml`；
2026-10-24 时尚未执行实际训练；2026-10-25 已修复首次运行暴露的 NCCL
大消息广播崩溃，并完成实际 PPO 更新、checkpoint 保存，以及恢复后继续更新至第 2 步。
最新证据及恢复检查见 [PPO 运行记录](PPO_FAILURE_20261025.md)。
该配置沿用官方动作专家训练设置，未启用 LoRA；本步骤是官方模型的工程连通性验收。

```bash
cd /home/zhsu/jyluo
bash experiments/robotwin_pi0/scripts/run_ppo_smoke.sh
```

脚本调用官方训练入口，运行目录写入 `latest-ppo-smoke.txt`。
关口 A 需要有限的训练指标、实际 optimizer update、checkpoint 保存，以及
`runner.resume_dir` 恢复后继续更新的证据。以最新运行记录为准，完成后再启动
正式专家数据采集。
