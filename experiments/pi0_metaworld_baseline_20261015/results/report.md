> 本页为原始100回合结果，已完成审计与修复后重跑。请查看[最新重跑报告](../../pi0_metaworld_baseline_recheck_20261015/results/report.md)和[新旧对照](../../pi0_metaworld_baseline_recheck_20261015/results/comparison.md)。原始记录与视频保留。

# π0 MetaWorld v2 baseline

| 场景 | 已完成 / 计划 | 成功次数 | 成功率 | 状态 |
|---|---:|---:|---:|---|
| assembly-v2 | 20 / 20 | 12 | 60.0% | complete |
| shelf-place-v2 | 20 / 20 | 8 | 40.0% | complete |
| disassemble-v2 | 20 / 20 | 4 | 20.0% | complete |
| peg-unplug-side-v2 | 20 / 20 | 7 | 35.0% | complete |
| stick-pull-v2 | 20 / 20 | 0 | 0.0% | complete |
| 总计 | 100 / 100 | 31 | 31.0% | complete |

使用 RLinf 官方发布的 π0 MetaWorld SFT 检查点，冻结权重，无额外训练或模型改进。
每次先执行 15 个零动作等待物体稳定，再最多执行 160 个策略控制步；动作块长度 5，去噪步数 5。
主要指标为策略控制期间至少一次 `info['success']` 为真；另记录最后一步成功状态。运行错误不计为失败。
每场景使用固定种子 42–61，独立选择 20 个不同的 MT1 初始化配置。

环境固定为用户指定的 v2；参考脚本当前使用 v3，因此这些结果不能直接当作官方 MT50 v3 分数的复现。

检查点：ModelScope `RLinf/RLinf-Pi0-MetaWorld`，revision `e63a613aca1d2c2e7f728dda99e2b87e0ed0fe8e`。
检查点 SHA256：`bef6bacc0a7075a3b2da7a88abd49a0fa2b927d9dce3020780d92afd5ae0f535`。
MetaWorld commit：`83ac03ca3207c0060112bfc101393ca794ebf1bd`。
协议 SHA256：`e82dda98f3db4aba6802eae33b2e2c16ffc2063ecc7759fff8a3271936a127f7`。

文件：`protocol.json`、`results/summary.csv`、`results/episodes.csv`、逐次 JSON/NPZ、`videos/results/` 和 `logs/`。

参考：[RLinf MetaWorld 文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/metaworld.html)、[官方评测脚本](https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py)。
