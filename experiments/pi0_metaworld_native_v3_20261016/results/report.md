# 冻结 π0 MetaWorld SFT 在 Farama MetaWorld 3.1.1 v3 上的评测

| 场景 | 已完成 / 计划 | 曾成功次数 | 曾成功率 | 最后一步成功次数 | 状态 |
|---|---:|---:|---:|---:|---|
| assembly-v3 | 20 / 20 | 8 | 40.0% | 7 | complete |
| shelf-place-v3 | 20 / 20 | 8 | 40.0% | 5 | complete |
| disassemble-v3 | 20 / 20 | 5 | 25.0% | 4 | complete |
| peg-unplug-side-v3 | 20 / 20 | 14 | 70.0% | 1 | complete |
| stick-pull-v3 | 20 / 20 | 3 | 15.0% | 2 | complete |
| 总计 | 100 / 100 | 38 | 38.0% | 19 | complete |

使用 RLinf 在 Hugging Face 发布的 π0 MetaWorld SFT 10,000 步检查点，冻结权重，无额外训练或模型改进。
每次先执行 15 个零动作等待物体稳定，再最多执行 160 个策略控制步；动作块长度 5，去噪步数 5。
主要指标为策略控制期间至少一次 `info['success']` 为真；另记录最后一步成功状态。运行错误不计为失败。
每场景使用固定种子 42–61，独立选择 20 个不同的 MT1 初始化配置。

Farama 3.1.1 的原生 reset(seed=...) 仍忽略 seed 参数，因此显式调用 env.seed(seed)；模拟器内部异常立即报错，不能算作任务失败。
每条结果校验 NPZ 中的逐步 success、最后一步、首个成功步、累计奖励、完整观测与视频/轨迹 SHA256。
完整观测仅用于离线审计，策略输入仍只有原四维 state 与相机图像。
本轮使用 Farama MetaWorld 3.1.1、MuJoCo 3.3.0、Gymnasium 1.1.1 和 v3 任务；前轮为 Farama 2024 年 v2。模型、种子序列、控制步数与成功口径相同，但场景实例和环境动力学已变，不构成逐场景配对比较。

2026 年 4 月约 90% / 200M frames 来自另一个项目 MTBench 的 Isaac Gym/Franka 环境和强化学习奖励修复，不是本 Farama MuJoCo/Sawyer 环境或冻结 π0 检查点的成绩。

检查点：Hugging Face `RLinf/RLinf-Pi0-MetaWorld-SFT`，revision `d0b8fbe41877a8105e2fd91e750e44542e6855bd`。
检查点 SHA256：`c606538b48414b00af1966dd8cf27fe0a87893f390619cac6bf0920a7cf06e2d`。
MetaWorld 版本：3.1.1，tag commit `7ea2b501c4a698c8533cdc55a396fe2734e2649d`，wheel SHA256 `99e25d9baf97f2f1a19ca89c0d2ca9c6a2b396d9a9be81729de64eedc683f0d3`。
协议 SHA256：`01ef3f9b236a8f4d73508a1d4ad705f5a0376f38684852567bb1a02b0782c3da`。

文件：`protocol.json`、`results/summary.csv`、`results/episodes.csv`、逐次 JSON/NPZ、`videos/results/` 和 `logs/`。

参考：[RLinf MetaWorld 文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/metaworld.html)、[RLinf 官方评测脚本](https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py)、[Farama 3.1.1](https://pypi.org/project/metaworld/3.1.1/)、[MTBench 更新说明](https://github.com/Viraj-Joshi/MTBench)。

100/100 回合通过独立模拟器逐步重放；完整版本对照见 `results/version_comparison.md`。
版本来源、2026 年 4 月 MTBench 修复与 stick-pull-v3 三次成功回合的核查见 `diagnostics/version_investigation.md`。
