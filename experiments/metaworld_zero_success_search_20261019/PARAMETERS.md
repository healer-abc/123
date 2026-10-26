# 参数位置与本轮边界

原始训练代码、训练配置、checkpoint 和原始 20 局结果均不修改。此目录只做推理参数搜索；不能用搜索结果覆盖未调参基线。

## DP

相对于 `../metaworld_official_baselines_50demo_20261019/`：

- `sources/dp/diffusion_policy-main/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml:54`：`policy.num_inference_steps: 100`。
- 同文件第 14 行：`n_action_steps: 8`；第 51 行 `horizon` 为 16，第 52 行将执行步数传给 policy。
- `sources/dp/diffusion_policy-main/diffusion_policy/policy/diffusion_unet_image_policy.py:80` 保存 `self.num_inference_steps`，第 100 行 `scheduler.set_timesteps(...)` 使用它。
- 同文件第 73 行保存 `self.n_action_steps`，第 178–180 行用 `start = n_obs_steps - 1; end = start + n_action_steps` 截取执行动作。

用户所说 `action_horizon` 在该官方实现中对应 `n_action_steps`。搜索只更改执行步数为 1/2/8，保持训练预测 `horizon=16`。DP 预登记推理步数为 100/50/20/10；SimpleDP3 为 10/20/50。当前没有已完成且成功率为零的 DP/SimpleDP3 50 示教基线，因此不冒用未训练或训练中 checkpoint 生成“最终基线”。

## ACT

- `sources/act/act-main/imitate_episodes.py:433`：官方 `--temporal_agg` 命令行开关。
- 同文件第 192–194 行：开启聚合后 `query_frequency=1`。
- 同文件第 250–259 行：保存所有历史块、取当前步候选，按 `exp(-0.01 * arange(n))` 加权；保留官方由旧到新的权重方向和全分量非零筛选。
- 原基线 `rollout.py` 的 ACT 分支每次查询执行完整 100 步；本目录 `search.py` 独立实现官方可选时间聚合和短执行间隔。

ACT 首先测试 `temporal_agg=true, action_horizon=1, aggregation_k=0.01`；然后测试不聚合、执行间隔 1/2/4/8/16/32/100。始终加载同一个 100 查询输出的已训练模型，不改变模型块长、结构、损失或训练轮数。

## 筛选和评测

具体有限空间、顺序、所有 seed 和 checkpoint 哈希见 `plan.json`。每组完整跑 3 局；首个成功组锁定；之后仅执行一次独立 20 局。筛选 seed 与 holdout seed 分离。搜索前已写入计划，完成前不根据结果追加组合或 seed。

有命中时生成 `ACT_pick-place-wall/locked.json` 和 `parameter_diff.json`；无命中则保留全部失败记录，执行官方专家正对照并生成 `diagnostic_report.md`，按停止规则结束。

`stick-pull-v2` 与现有任务 `stick-push` 不同。本轮五任务中没有 stick-pull 的模型和原始评测，未替换任务，也未声称它已经完成。
