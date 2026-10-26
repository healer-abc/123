# π0 MetaWorld SFT（Hugging Face 官方权重）v2 评测

| 场景 | 已完成 / 计划 | 曾成功次数 | 曾成功率 | 最后一步成功次数 | 状态 |
|---|---:|---:|---:|---:|---|
| assembly-v2 | 20 / 20 | 10 | 50.0% | 7 | complete |
| shelf-place-v2 | 20 / 20 | 8 | 40.0% | 3 | complete |
| disassemble-v2 | 20 / 20 | 6 | 30.0% | 3 | complete |
| peg-unplug-side-v2 | 20 / 20 | 13 | 65.0% | 2 | complete |
| stick-pull-v2 | 20 / 20 | 0 | 0.0% | 0 | complete |
| 总计 | 100 / 100 | 37 | 37.0% | 15 | complete |

使用 RLinf 在 Hugging Face 发布的 π0 MetaWorld SFT 检查点，冻结权重，无额外训练或模型改进。
每次先执行 15 个零动作等待物体稳定，再最多执行 160 个策略控制步；动作块长度 5，去噪步数 5。
主要指标为策略控制期间至少一次 `info['success']` 为真；另记录最后一步成功状态。运行错误不计为失败。
每场景使用固定种子 42–61，独立选择 20 个不同的 MT1 初始化配置。

修复：原生 v2 忽略 reset(seed=...)，现显式调用 env.seed(seed)；模拟器内部异常立即报错，不能算作任务失败。
每条结果校验 NPZ 中的逐步 success、最后一步、首个成功步、累计奖励、完整观测与视频/轨迹 SHA256。
完整观测仅用于离线审计，策略输入仍只有原四维 state 与相机图像。
与上轮的 ModelScope 权重使用相同20组场景与种子、相同接口和指标；两个检查点文件 SHA256 不同。

环境固定为用户指定的 v2；参考脚本当前使用 v3，因此这些结果不能直接当作官方 MT50 v3 分数的复现。

检查点：Hugging Face `RLinf/RLinf-Pi0-MetaWorld-SFT`，revision `d0b8fbe41877a8105e2fd91e750e44542e6855bd`。
检查点 SHA256：`c606538b48414b00af1966dd8cf27fe0a87893f390619cac6bf0920a7cf06e2d`。
MetaWorld commit：`83ac03ca3207c0060112bfc101393ca794ebf1bd`。
协议 SHA256：`1a82c8c0954fc1649e7fba58e56e390460549804cf34592e4fa5f05f52abdce4`。

文件：`protocol.json`、`results/summary.csv`、`results/episodes.csv`、逐次 JSON/NPZ、`videos/results/` 和 `logs/`。

参考：[RLinf MetaWorld 文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/metaworld.html)、[官方评测脚本](https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py)。

检查点训练元数据：旧 ModelScope 5,000 步；新 Hugging Face 10,000 步；均为 MetaWorld SFT。
逐回合独立模拟器重放及同场景配对检查均为 100/100，详见 `diagnostics/final_validation.json` 与 `results/comparison.md`。
