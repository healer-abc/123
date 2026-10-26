# 冻结 π0 MetaWorld RL FlowNoise 在 Farama MetaWorld 3.1.1 v3 上的评测

| 场景 | 已完成 / 计划 | 曾成功次数 | 曾成功率 | 最后一步成功次数 | 状态 |
|---|---:|---:|---:|---:|---|
| assembly-v3 | 20 / 20 | 14 | 70.0% | 14 | complete |
| shelf-place-v3 | 20 / 20 | 17 | 85.0% | 10 | complete |
| disassemble-v3 | 20 / 20 | 17 | 85.0% | 11 | complete |
| peg-unplug-side-v3 | 20 / 20 | 19 | 95.0% | 3 | complete |
| stick-pull-v3 | 20 / 20 | 15 | 75.0% | 1 | complete |
| 总计 | 100 / 100 | 82 | 82.0% | 39 | complete |

使用 RLinf 在 Hugging Face 发布的 π0 MetaWorld PPO FlowNoise 微调检查点，冻结权重，无本地训练或模型改进。
每次先执行 15 个零动作等待物体稳定，再最多执行 160 个策略控制步；动作块长度 5，去噪步数 5。
主要指标为策略控制期间至少一次 `info['success']` 为真；另记录最后一步成功状态。运行错误不计为失败。
每场景使用固定种子 42–61，独立选择 20 个不同的 MT1 初始化配置。

Farama 3.1.1 的原生 reset(seed=...) 仍忽略 seed 参数，因此显式调用 env.seed(seed)；模拟器内部异常立即报错，不能算作任务失败。
每条结果校验 NPZ 中的逐步 success、最后一步、首个成功步、累计奖励、完整观测与视频/轨迹 SHA256。
完整观测仅用于离线审计，策略输入仍只有原四维 state 与相机图像。
本轮与上一轮均使用 Farama MetaWorld 3.1.1、MuJoCo 3.3.0、Gymnasium 1.1.1 和 v3 任务；种子、场景实例、控制步数与成功口径相同，仅检查点不同，详见配对比较。

2026 年 4 月约 90% / 200M frames 来自另一个项目 MTBench 的 Isaac Gym/Franka 环境和强化学习奖励修复，不是本 Farama MuJoCo/Sawyer 环境或冻结 π0 检查点的成绩。

检查点：Hugging Face `RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise`，revision `047f7806847cf7f6f74c769cd7af42a4235961ea`。
检查点 SHA256：`c0cb3acfeac49f7d6197f09ecd0f3fc22566c86fb45953c75d4ccb5f2408f9a4`。
MetaWorld 版本：3.1.1，tag commit `7ea2b501c4a698c8533cdc55a396fe2734e2649d`，wheel SHA256 `99e25d9baf97f2f1a19ca89c0d2ca9c6a2b396d9a9be81729de64eedc683f0d3`。
协议 SHA256：`c9876d0ab2a8e729546642b17211356f0a7bae07080f2958ba078c7e587b9d3c`。

文件：`protocol.json`、`results/summary.csv`、`results/episodes.csv`、逐次 JSON/NPZ、`videos/results/` 和 `logs/`。

参考：[RLinf MetaWorld 文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/metaworld.html)、[RLinf 官方评测脚本](https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py)、[Farama 3.1.1](https://pypi.org/project/metaworld/3.1.1/)、[MTBench 更新说明](https://github.com/Viraj-Joshi/MTBench)。

100/100 回合通过独立模拟器逐步重放；与 SFT 的配对对照见 `results/comparison.md`。

注意：`success_once` 表示回合中曾成功；本轮末步成功为 39/100。RLinf 文档把 `env/success_once` 列为关键训练指标，但其单进程独立评测脚本覆盖每一步的 success，实际输出为末步口径。两种指标均给出，避免误将短暂成功视为最终稳定完成。
