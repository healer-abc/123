# MetaWorld 版本与 2026 年 4 月修复核查

- 上轮使用的 Farama 原生 MuJoCo/Sawyer v2 固定在 commit `83ac03ca3207c0060112bfc101393ca794ebf1bd`（2024-05-01）。因此它不可能含有 2026-04-19 的代码；更关键的是，用户引用的修复根本不在 Farama 仓库。
- [MTBench 2026-04-19 提交](https://github.com/Viraj-Joshi/MTBench/commit/aa15bff0f011bc8807c4df985be423389a36aaab)更改的是另一个项目中 Isaac Gym/Franka 版 Meta-World 的奖励、初始位置和部分资产。[项目 README](https://github.com/Viraj-Joshi/MTBench#updates)写的 Stick Pull 约 90% / 200M frames 是该训练体系的结果；不能推断冻结 π0 在 Farama MuJoCo/Sawyer 环境下应达到 90%。
- [Farama 迁移提交](https://github.com/Farama-Foundation/Metaworld/commit/f1a106bf3e43f2c24617b92f5d6f9ea9bf34298e)使旧 v2 文件不再存在；[当前 PyPI 稳定版 3.1.1](https://pypi.org/project/metaworld/3.1.1/)只含 v3 任务。其 wheel SHA256 为 `99e25d9baf97f2f1a19ca89c0d2ca9c6a2b396d9a9be81729de64eedc683f0d3`，tag commit 为 `7ea2b501c4a698c8533cdc55a396fe2734e2649d`。
- 因此本次按用户选择，在隔离环境中运行 Farama 3.1.1 v3 + MuJoCo 3.3.0 + Gymnasium 1.1.1，沿用同一冻结 π0 SFT 10k 权重、种子序列、20 次/任务、160 步和 success_once 指标。v2 与 v3 的任务实例和动力学不同，比较表只供并列参考。
- stick-pull-v3 本次在第 4、5、14 回合（种子 46、47、56）达到 success；初次成功策略步分别为 83、95、92。三条轨迹的逐步 success 和仿真状态均通过独立重放；其他 97 条也通过重放。
