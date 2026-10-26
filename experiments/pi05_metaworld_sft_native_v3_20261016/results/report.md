# π0.5 MetaWorld SFT 在 Farama MetaWorld 3.1.1 v3 的评测

| 任务 | 成功/完成 | 成功率 | 最后一步成功 |
|---|---:|---:|---:|
| assembly-v3 | 2/20 | 10.0% | 2 |
| peg-insert-side-v3 | 3/20 | 15.0% | 2 |
| disassemble-v3 | 7/20 | 35.0% | 3 |
| peg-unplug-side-v3 | 5/20 | 25.0% | 1 |
| stick-pull-v3 | 5/20 | 25.0% | 0 |
| 总计 | 22/100 | 22.0% | 8 |

模型：RLinf/RLinf-Pi05-MetaWorld-SFT，冻结权重，严格加载，使用发布者的分位数归一化统计与原生 4D MetaWorld 动作。
环境：Farama MetaWorld 3.1.1 / v3，MuJoCo 3.3.0。五任务每项 20 个 MT1 配置，种子 42–61。
按 RLinf 官方评测脚本设定先做 15 个零动作稳定，再至多 160 个策略步；动作块长度 5、去噪步数 5。
成功口径：策略控制期间至少一次 info.success 为真，不要求最后一帧仍为真；另列最后一帧成功数。
完整观测和逐步 success 仅用于审计；策略只看到相机图像和前四维手部状态。每条轨迹和视频均校验 SHA256。
前轮 v2 使用同一权重，但环境版本、初始化实例和每回合最大步数（500）不同，不能把差异归因于单一因素。

| 任务 | 旧 v2 ≤160 步 | 旧 v2 ≤500 步 | 新 v3 ≤160 步 |
|---|---:|---:|---:|
| assembly | 1/20 | 1/20 | 2/20 |
| peg-insert-side | 2/20 | 2/20 | 3/20 |
| disassemble | 7/20 | 8/20 | 7/20 |
| peg-unplug-side | 6/20 | 11/20 | 5/20 |
| stick-pull | 7/20 | 7/20 | 5/20 |
| 总计 | 23/100 | 29/100 | 22/100 |

旧 v2 的 ≤160 步列仅是原 500 步轨迹的离线截断；v3 有 15 步稳定阶段，而且场景和动力学不同，不能视为严格配对对照。

2026-04-19 的约 90% 属于 MTBench 的 Isaac Gym/Franka PPO 奖励修复；不适用于此 Farama MuJoCo/Sawyer 环境或本冻结 π0.5 权重。
模型 commit：60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e；权重 SHA256：6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221。
MetaWorld wheel SHA256：99e25d9baf97f2f1a19ca89c0d2ca9c6a2b396d9a9be81729de64eedc683f0d3；协议 SHA256：2aa8e8466ad910079f62660874aa73351813b4b8e6e221a0b8859382515bba52。

参考：[MTBench 更新](https://github.com/Viraj-Joshi/MTBench#updates)、[RLinf MetaWorld 权重](https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT)、[RLinf 官方评测脚本](https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py)。

100/100 回合通过独立 MetaWorld v3 仿真器逐步重放；每步完整观测、奖励与成功标志一致。
