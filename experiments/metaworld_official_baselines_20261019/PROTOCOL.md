# MetaWorld 官方方法 baseline

用户要求：五个任务分别用 DP、ACT、SimpleDP3 训练并最终 rollout 20 次，汇总成功率；不得改进模型。OpenPI 主线保持暂停。

任务：shelf-place、pick-place-wall、pick-place、stick-push、door-open。15 组模型，最终 300 个 episode。截图只用于识别任务，不作为实验结果。

## 官方来源与训练

- DP：https://github.com/real-stanford/diffusion_policy ，使用 `train_diffusion_unet_image_workspace.yaml`：8000 epochs，batch 64，原版 ResNet18 + diffusion U-Net、DDPM 100 步、EMA、AdamW 与 cosine schedule。
- SimpleDP3：https://github.com/YanjieZe/3D-Diffusion-Policy ，使用 `simple_dp3.yaml` 与各 MetaWorld task config：3000 epochs，batch 128，原版 point encoder、DDIM 10 步、EMA、AdamW 与 cosine schedule。
- ACT：https://github.com/tonyzhaozh/act ，README 的仿真训练命令：2000 epochs，batch 8，chunk 100，KL 10，hidden 512，feedforward 3200，lr 1e-5；官方 4 层 encoder、7 层 decoder、8 heads、预训练 ResNet18。不启用可选 temporal aggregation。直接调用官方 policy 与通过 AST 原样加载的 `train_bc`、`forward_pass` 等函数，避免导入 ALOHA 专用仿真环境。

固定训练 seed 0，数据采集 seed 42。每个场景通过官方 DP3 MetaWorld expert generator 生成 10 条成功示教，保留原版“成功且至少 5 步 success”的筛选；三种方法共用这些动作与时刻对齐的视觉/机器人状态。原版 ACT 用 80/20 episode split，DP/SimpleDP3 按 MetaWorld config 用 2% validation（10 条中 1 条）。不为了统一而改写方法的原生训练划分。

## 必要适配与复现边界

DP、ACT 的官方仓库没有这五个 MetaWorld 任务的现成训练配方。因此本实验是**官方算法在 MetaWorld 的接口适配 baseline**，不能声称“官方已发布的 MetaWorld 结果复现”。

DP 使用新增 MetaWorld RGB dataset/runner：保持官方采样、normalizer、图像 encoder、优化与训练循环；输入为官方 DP3 相机 corner2 的原生 128×128 RGB 和 9 维 robot state，输出 4 维动作。ACT 仅把原版硬编码的 14 维机器人输入/动作接口分别改成 9/4，完整 diff 存在 `act_metaworld_dimensions.patch`。共享数据无损转为其原生 HDF5 格式；qvel 只是未使用字段的零占位，不作为模型输入。

MuJoCo 2.1 / MetaWorld v2 来自官方 DP3 仓库，使用现有兼容 Python3.10 / PyTorch2.7.1 CUDA12.8 运行环境；不修改共享环境。软件 EGL 设备运行时探测，避免默认设备权限错误；模型与 FPS 仍用分配的 CUDA GPU。渲染设备修复不改变相机、观测、控制或物理参数。源码归档与执行文件 SHA256 保存在 `source_manifest.json`；原版代码保存在 `sources/`，ACT 维度适配副本在 `runtime_act/`。

## 最终评测

每个方法×场景恰好 20 个完整 episode，200 环境步上限，成功定义为官方 `info['success']` 曾为真，不增加成功阈值、不接专家、不加残差控制、不修改动作。使用显式播种的底层 MetaWorld v2 环境，因为原版 wrapper 的 seed 方法为空实现。

最终 seeds 固定为 2026101900–2026101919，与训练中的开发 rollout seeds（1026101900–1026101919）分开。三种方法逐 task/episode 验证初始 simulator state 的 hash 一致，失败则停止汇总。保存每个 episode 的 seed、初态 hash、checkpoint hash、成功、首次成功步、长度、回报及视频。基础设施错误保留原始尝试；策略失败不能重试刷分。

DP/SimpleDP3 固定使用完整训练结束的 terminal EMA checkpoint；ACT 按官方命令使用最小 validation loss checkpoint。此规则在任何最终结果产生前冻结。不根据最终成功率挑选 checkpoint、不重复最终评测选择最好一轮。官方训练内的周期性开发 rollout 依然保留，但不会进入最终 20 次表格；DP3 的最大 K 次开发成功率指标另存，不当成本实验的最终成功率。

单个训练 seed、20 次 rollout 的结果用于本地 baseline，不表示多训练种子的均值/标准差，也不与截图中的旧结果直接作提升比较。

## 看门狗

`baseline_watchdog.py` 独占本实验锁，每 15 秒检查子进程、日志/心跳与 GPU 余量。初始最多 4 个 worker，每 GPU 最多 3 个，按实测 smoke 和显存预算准入，保留 10000 MiB 余量。余量低于 3500 MiB 或 1200 秒无进展会终止本实验自己的进程组。不会杀其他实验进程。

基础设施故障最多 2 次尝试，每次独立目录；OOM 降低并发，不更改 batch/model/科学配置。失败数据隔离保存，完成数据不覆盖。非有限值等模型/科学异常直接告警暂停。没有代码自动生成器：未知故障写入告警并停止，需诊断后修复；不能承诺后台自动修改任意错误代码。

状态：`logs/status.json`；事件：`logs/events.jsonl`；告警：`logs/alerts.jsonl` 与 `logs/latest_alert.json`；原始日志：`runs/*/attempt_*/worker.log`。最终表格自动写到 `report/results.md`、`report/results.csv`。未完成格子标“待完成”，不填估计值。
