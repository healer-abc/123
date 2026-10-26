# Farama MetaWorld v3: π0 SFT 与 π0 RL FlowNoise 配对对照

| 场景 | 旧 SFT 成功 | 新 RL FlowNoise 成功 | 新成功率 | 新末步成功 |
|---|---:|---:|---:|---:|
| assembly-v3 | 8/20 | 14/20 | 70% | 14/20 |
| shelf-place-v3 | 8/20 | 17/20 | 85% | 10/20 |
| disassemble-v3 | 5/20 | 17/20 | 85% | 11/20 |
| peg-unplug-side-v3 | 14/20 | 19/20 | 95% | 3/20 |
| stick-pull-v3 | 3/20 | 15/20 | 75% | 1/20 |
| 总计 | 38/100 | 82/100 | 82% | 39/100 |

每一对回合的 seed、MT1 场景 SHA 一致，初始四维状态一致；模型权重 SHA 不同。主指标均为160步内 success_once。
两组均在 Farama MetaWorld 3.1.1 v3 上执行，分别为 RLinf 官方 SFT π0 与 PPO FlowNoise 微调 π0。评测中冻结权重，无本地训练或模型调整。
新100回合另由独立模拟器重放验证，详见 diagnostics/new_trajectory_replay.json。
推理为 BF16、固定随机种子；未强制所有 GPU 算子逐位确定性。

补充口径：新模型末步成功 39/100，旧模型末步成功 19/100。`success_once` 衡量回合中任一时刻达成，末步成功衡量 160 步结束时是否仍达成。
