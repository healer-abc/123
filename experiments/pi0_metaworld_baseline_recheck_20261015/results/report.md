# π0 MetaWorld v2 baseline：审计修复后重跑

| 场景 | 已完成 / 计划 | 曾成功次数 | 曾成功率 | 最后一步成功次数 | 状态 |
|---|---:|---:|---:|---:|---|
| assembly-v2 | 20 / 20 | 11 | 55.0% | 11 | complete |
| shelf-place-v2 | 20 / 20 | 8 | 40.0% | 6 | complete |
| disassemble-v2 | 20 / 20 | 4 | 20.0% | 3 | complete |
| peg-unplug-side-v2 | 20 / 20 | 6 | 30.0% | 2 | complete |
| stick-pull-v2 | 20 / 20 | 0 | 0.0% | 0 | complete |
| 总计 | 100 / 100 | 29 | 29.0% | 22 | complete |

使用 RLinf 官方发布的 π0 MetaWorld SFT 检查点，冻结权重，无额外训练或模型改进。
每次先执行 15 个零动作等待物体稳定，再最多执行 160 个策略控制步；动作块长度 5，去噪步数 5。
主要指标为策略控制期间至少一次 `info['success']` 为真；另记录最后一步成功状态。运行错误不计为失败。
每场景使用固定种子 42–61，独立选择 20 个不同的 MT1 初始化配置。

修复：原生 v2 忽略 reset(seed=...)，现显式调用 env.seed(seed)；模拟器内部异常立即报错，不能算作任务失败。
每条结果校验 NPZ 中的逐步 success、最后一步、首个成功步、累计奖励、完整观测与视频/轨迹 SHA256。
完整观测仅用于离线审计，策略输入仍只有原四维 state 与相机图像。
旧轮100条轨迹的独立动作重放全部一致；未发现旧31/100的汇总错误。修复复现性不保证提高成功率。

环境固定为用户指定的 v2；参考脚本当前使用 v3，因此这些结果不能直接当作官方 MT50 v3 分数的复现。

检查点：ModelScope `RLinf/RLinf-Pi0-MetaWorld`，revision `e63a613aca1d2c2e7f728dda99e2b87e0ed0fe8e`。
检查点 SHA256：`bef6bacc0a7075a3b2da7a88abd49a0fa2b927d9dce3020780d92afd5ae0f535`。
MetaWorld commit：`83ac03ca3207c0060112bfc101393ca794ebf1bd`。
协议 SHA256：`25085b464bd50784b37ea9c3a99e264216a68409cfbec14fa3b4d0ceb2b46104`。

文件：`protocol.json`、`results/summary.csv`、`results/episodes.csv`、逐次 JSON/NPZ、`videos/results/` 和 `logs/`。

参考：[RLinf MetaWorld 文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/metaworld.html)、[官方评测脚本](https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py)。

复核完成：100/100 条新轨迹经独立模拟器重放，完整观测、奖励和成功标志一致。

[排查与修复说明](../DIAGNOSIS.md) · [旧结果与重跑对照](comparison.md) · [独立重放验证](../diagnostics/final_validation.json)
