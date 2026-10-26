# π0 + RoboTwin 2.0：数据工程、LoRA SFT 与 PPO 强化学习实验方案（官方代码优先版）

最终项目定位：基于 RoboTwin 2.0 的机器人操作数据质量优化与 π0 在线强化学习。

我建议把工作重点分为两条：第一条是自己构建、清洗和增强机器人专家数据；第二条是使用这些数据适配 π0，并通过在线 PPO 进一步优化策略。第一个项目已经研究了 π0.5 的微调方法，因此第二个项目不再重复 LoRA rank、冻结策略等消融，LoRA 在这里主要是节省资源的训练手段。

最终希望完成以下闭环：

RoboTwin：生成 Clean / Randomized 专家示范

复用官方脚本专家与环境随机化

数据审计、异常筛选、LeRobot 转换

复用格式工具，自己实现质量判断

RLinf：π0 LoRA SFT 对照

比较同等数据量的原始与增强数据

RLinf：PPO 在线强化学习

环境交互 → 奖励和优势估计 → 更新策略

标准场景、随机化场景与训练效率评测

正式任务固定为 RoboTwin 的 `adjust_bottle`。RLinf 已提供这个任务的 π0 SFT、PPO 和评测配置，因此不需要自己搭建新任务环境。

## 一、先看整个项目要做什么

| 阶段 | 你要完成什么             | 官方代码复用程度 | 主要交付物                 |
| -- | ------------------ | -------- | --------------------- |
| 0  | 安装环境、验证双卡 PPO      | 高        | 最小训练闭环                |
| 1  | 专家数据自动采集           | 高        | Clean / Randomized 轨迹 |
| 2  | 数据审计与清洗            | 中        | 质量报告、有效数据清单           |
| 3  | 转换 LeRobot、计算归一化统计 | 中，存在版本适配 | π0 可读取的数据集            |
| 4  | π0 LoRA SFT 对照     | 高        | 两套 SFT 模型             |
| 5  | π0 PPO 在线强化学习      | 高，但需双卡适配 | PPO checkpoint、学习曲线   |
| 6  | 标准评测、泛化与效率分析       | 高        | 完整对照表与失败案例            |

你只需要熟悉 RoboTwin、RLinf 和 LeRobot 的数据接口，并不需要自己实现机器人运动规划、π0 网络或 PPO 算法。

有两个地方我会特别谨慎：

第一，RLinf 官方要求使用 RoboTwin 的 `RLinf_support` 分支，而 RoboTwin 当前主线已经更新了数据格式，不能混用两个版本的命令。

[image](https://www.google.com/s2/favicons?domain=https://robotwin-platform.github.io\&sz=32)

RoboTwin 2.0 Official Document



第二，LoRA SFT 产生的 adapter 能否不经额外适配直接接入 PPO，需要提前验证。我会把它设成正式训练前的验收项，而不是假设框架会自动衔接。

## 二、阶段 0：环境安装和双卡可行性验证

正式数据采集前必须完成

你的服务器有两张 RTX 6000D，每卡约 83.6 GiB 显存、计算能力 12.0，系统内存也比较充足。等两张卡空闲时，可以尝试缩小规模的 PPO，但目前没有经过验证的同型号双卡配置。

这一阶段先不训练自己的数据。 直接加载官方 π0 SFT checkpoint，在 `adjust_bottle` 上完成一次最小 PPO 更新。这样可以提前排除最贵的环境与算力风险。

### 要看的官方文件

| 阅读顺序 | 文件或文档                                                                                                                                             | 主要看什么                    |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| 1    | [RLinf RoboTwin 训练指南](https://github.com/RLinf/RLinf/blob/main/docs/source-zh/rst_source/examples/embodied/robotwin.rst?utm_source=chatgpt.com)   | 官方安装、RoboTwin 适配分支、资产路径  |
| 2    | [RLinf requirements/install.sh](https://github.com/RLinf/RLinf/blob/main/requirements/install.sh?utm_source=chatgpt.com)                          | 环境安装入口，不需要逐行读懂           |
| 3    | [PPO 主配置 YAML](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml?utm_source=chatgpt.com) | GPU 分配、rollout、actor、优化器 |
| 4    | [π0 模型默认配置](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/model/pi0.yaml?utm_source=chatgpt.com)                            | 模型结构、训练模式、LoRA 默认值       |
| 5    | [run_embodiment.sh](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/run_embodiment.sh?utm_source=chatgpt.com)                        | PPO 启动入口                 |

按官方方式，项目可以放在：

```
~/jyluo/
├── projects/
│   ├── RLinf/
│   └── RoboTwin/
├── data/
│   └── robotwin_pi0/
├── models/
│   └── robotwin_pi0/
└── checkpoints/
    └── robotwin_pi0/
```

目录只是组织建议，尤其要避免将大型训练数据默认写到系统 `~/.cache`。如果使用官方虚拟环境，RLinf 的 `.venv` 可以独立于已有 Conda 环境存在。

### 具体实验

先验证环境加载和官方 checkpoint 评测，再创建一份自己的 `robotwin_ppo_2gpu.yaml`。不要直接覆盖官方 YAML。

官方配置使用 8 卡资源放置、256 个训练环境、较大的 batch。你的冒烟测试可以先尝试 4～8 个训练环境、micro-batch 2～4、global batch 32～64、开启相应 offload，再根据显存调整。这些是待验证的初始值，不是官方保证可运行的双卡参数。

验收不应止于“训练程序启动成功”，而要同时满足：模型能正常推理、仿真环境能执行动作、PPO 完成一次反向传播和优化器更新、checkpoint 能保存且能够继续加载。

阶段完成标志：成功记录一次完整 PPO 更新，以及两张卡的峰值显存和单轮训练时间。

## 三、阶段 1：利用 RoboTwin 自动采集专家示范

这一阶段以官方脚本为主，你的工作主要是设置采集条件和记录数据分布，不需要写机械臂控制器。

### 1.1 需要阅读的文件

| 文件                                                                                                                                                         | 用途        | 阅读重点                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------------------------------------- |
| [collect_data.sh](https://github.com/RoboTwin-Platform/RoboTwin/blob/RLinf_support/collect_data.sh?utm_source=chatgpt.com)                                 | 官方采集入口    | 三个命令参数如何传递                               |
| [script/collect_data.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/RLinf_support/script/collect_data.py?utm_source=chatgpt.com)                   | 专家数据采集核心  | seed 搜索、成功判定、轨迹保存                        |
| [envs/adjust_bottle.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/RLinf_support/envs/adjust_bottle.py?utm_source=chatgpt.com)                     | 具体任务与脚本专家 | `setup_demo`、`play_once`、`check_success` |
| [task_config/demo_clean.yml](https://github.com/RoboTwin-Platform/RoboTwin/blob/RLinf_support/task_config/demo_clean.yml?utm_source=chatgpt.com)           | 普通场景配置    | `episode_num`、相机、保存路径                    |
| [task_config/demo_randomized.yml](https://github.com/RoboTwin-Platform/RoboTwin/blob/RLinf_support/task_config/demo_randomized.yml?utm_source=chatgpt.com) | 随机场景配置    | `domain_randomization`                   |

另外读两份说明即可：[Collect Data](https://robotwin-platform.github.io/doc/usage/collect-data.html?utm_source=chatgpt.com)、[Configurations](https://robotwin-platform.github.io/doc/usage/configurations.html?utm_source=chatgpt.com)。这两份主要介绍采集参数；涉及新版 XPolicyLab 文件布局时，以你实际使用的 `RLinf_support` 分支源码为准。

### 1.2 先弄懂脚本专家怎么生成数据

RoboTwin 的专家采集大致分为两个过程。

首先，脚本通过任务内置的动作规划器尝试不同初始化条件，只有当规划执行成功并且 `check_success()` 为真时，才保留该 seed 和对应轨迹信息。

随后，程序按照已筛选的 seed 重放动作并保存观测数据。旧分支会生成 HDF5 和视频等文件，同时写入 `seed.txt`、`scene_info.json` 等元数据。

因此，你不用再写自己的抓取策略。

你需要理解的是：成功的专家轨迹从哪里来、包含什么数据，以及为什么有些 seed 被舍弃。

### 1.3 正式采集设计

建议先采集两个数据池：

| 数据池        | 数量（目标）    | 环境条件            | 用途             |
| ---------- | --------- | --------------- | -------------- |
| Clean      | 200 条成功轨迹 | 普通背景、普通光照       | SFT 基线         |
| Randomized | 200 条成功轨迹 | 随机背景、光照、桌面杂物、高度 | SFT 增强         |
| Pilot      | 各 5 条     | 两种条件            | 检查数据格式，不作为正式评测 |

从官方配置复制出自己的版本，例如 `demo_clean_pi0.yml`、`demo_randomized_pi0.yml`，避免修改原文件。初期只改 `episode_num`、保存路径和随机化参数。

官方的调用形式如下，第三个参数是 GPU 编号：

```
# 在 RoboTwin 根目录执行
bash collect_data.sh adjust_bottle demo_clean_pi0 0
bash collect_data.sh adjust_bottle demo_randomized_pi0 0
```

这里 `demo_clean_pi0` 等文件是你从官方模板复制后建立的配置名，并不是仓库自带文件。

建议先按默认的 50 条或更少进行验证，再增加至 200 条，而不是第一次就长时间采集全部数据。

### 1.4 需要自己补什么？

主要补充一份数据采集统计表，而不是改动专家算法。可以在采集日志基础上记录：尝试次数、成功次数、失败原因、轨迹长度、随机种子、左右臂选择、瓶子模型和生成耗时。

`adjust_bottle.py` 已经实现左右臂选择和物体模型设置，因此这些字段可以从任务及场景元数据中获得或统计，不必重新推断机器人正在使用哪一侧机械臂。

验收标准： 能稳定生成两组专家数据，每条轨迹有可追溯的 seed 和采集配置，能够回放并通过任务成功判定。

## 四、阶段 2：数据检查、清洗与质量报告

这是最值得你亲自编程的一部分。 官方已经解决了大部分数据采集与保存问题，但没有替你完成适合这项研究的数据质量分析。

### 2.1 主要参考文件

| 官方文件                                                                                                                                         | 用来解决什么问题               |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| [envs/utils/pkl2hdf5.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/RLinf_support/envs/utils/pkl2hdf5.py?utm_source=chatgpt.com)     | 理解原始观测如何写入 HDF5、图像如何压缩 |
| [LeRobot Dataset Tools](https://github.com/huggingface/lerobot/blob/main/docs/source/using_dataset_tools.mdx?utm_source=chatgpt.com)         | 数据查看、分割、删除和合并          |
| [lerobot_dataset_viz.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_dataset_viz.py?utm_source=chatgpt.com) | 图像、状态和动作的同步可视化         |
| [RLinf 数据可视化](https://github.com/RLinf/RLinf/blob/main/toolkits/lerobot/visualize_lerobot_dataset.py?utm_source=chatgpt.com)                 | LeRobot 数据检查的现成示例      |

这些 LeRobot 工具主要用于转换之后的数据。原始旧版 HDF5 在转换前仍需要用 `h5py` 检查。

### 2.2 建议自己编写的唯一主要清洗脚本

`audit_dataset.py`

功能控制在四项：

| 审计内容  | 检查方法                | 结果         |
| ----- | ------------------- | ---------- |
| 文件完整性 | HDF5 能否读取、字段是否存在    | 标记损坏文件     |
| 时序完整性 | 检查图像、状态、动作长度及时间关系   | 标记缺失或不同步轨迹 |
| 数值合法性 | NaN/Inf、动作异常跳变、状态维度 | 标记数值异常     |
| 数据分布  | 轨迹长度、左右臂、物体模型、场景条件  | 生成统计图      |

每条轨迹输出一行记录，形成 `dataset_audit.csv`，并保存异常原因。如果发现数值异常，不应直接删除所有极端动作；需要结合单位、动作定义及视频判断。

这一点很关键：RoboTwin 的采集程序已经筛选成功轨迹，因此你自己的清洗程序未必会剔除很多数据。 如果最终没有明显坏数据，也完全正常。不能为了体现清洗效果而故意删除正常轨迹，或者没有实测就声称清洗提升了成功率。

你的贡献可以是完成一套自动数据验证流水线，同时测量其异常检出率、数据分布和质量控制成本。

### 2.3 不用自己写的部分

转成 LeRobot 后，删除指定 episode、数据集分割与合并可以复用 LeRobot 的编辑工具。RLinf 也有 [merge_lerobot_datasets.py](https://github.com/RLinf/RLinf/blob/main/toolkits/lerobot/merge_lerobot_datasets.py?utm_source=chatgpt.com)，不需要重新实现一套存储格式。

验收标准： 保留原始数据不覆盖，得到清洗后轨迹清单、拒绝原因表、数据分布图和少量人工审核视频。

## 五、阶段 3：将数据转换为 π0 可以读取的 LeRobot 格式

这一步和数据清洗不同，它是整个项目的接口验收阶段。

### 3.1 先用官方成品数据验证训练读取

RLinf 已经公开了一个适用于 `adjust_bottle` 的 [50 条示范的 LeRobot 数据集](https://huggingface.co/datasets/RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data?utm_source=chatgpt.com)。

RLinf 官方示例数据

LeRobot

## 50

episodes

## 7,188

数据行

## 1.69 GB

文件体积

用于验证数据加载与训练接口，不代替你自己的数据采集实验。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Datasets at Hugging Face

我建议先用它跑通 RLinf 的 DataLoader，再转换你自己采集的两条 HDF5 轨迹，检查转换后各字段是否一致。

这样可以明确区分：程序加载失败究竟是 RLinf 环境问题，还是你的数据转换问题。

### 3.2 数据格式的真正标准

重点阅读：

[robotwin_aloha_dataconfig.py](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py?utm_source=chatgpt.com)

这个文件比泛读 LeRobot 文档更重要，因为它明确规定了 π0 在 RoboTwin Aloha 任务中需要哪些输入：

```
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state
action
prompt
```

其中 Aloha 的动作是 14 维：左右臂各 6 个关节动作和 1 个夹爪动作。模型内部会按对应 OpenPI 配置处理维度及动作变换，因此不要在原始数据里自行将动作补到 32 维，也不要提前重复执行 delta-action 转换。

### 3.3 官方转换工具与需要注意的地方

你有两套参考：

[XPolicyLab：transform_lerobot_v21_format.py](https://github.com/XPolicyLab/XPolicyLab/blob/main/scripts/transform_lerobot_v21_format.py?utm_source=chatgpt.com)[RoboTwin：旧版数据迁移和 LeRobot 转换说明](https://robotwin-platform.github.io/doc/usage/collect-data.html?utm_source=chatgpt.com)

XPolicyLab 已有将新版 XPolicyLab 结构的 HDF5 转换成 LeRobot v2.1 的脚本，能减少大量重复开发。

但需要特别说明：`RLinf_support` 分支保存的是旧版 RoboTwin HDF5。RoboTwin 新版文档提到过 `scripts/process_data_xpolicylab.py`，但我没有在当前核对的仓库版本中确认到这一脚本文件。因此，不能把文档中的旧版迁移命令视为已经核实可直接执行的命令。

这里我建议坚持两个原则：RoboTwin 仿真与 PPO 使用 `RLinf_support`，保证环境协议一致；数据转换尽可能复用 XPolicyLab 官方 v2.1 转换器，如果原始字段布局不兼容，只编写一个必要的旧版 HDF5 适配层。

这个适配层是整个计划中尚未完全核实的工程工作。完成两条轨迹的转换测试之前，不应大规模采集数据。

### 3.4 归一化统计也有官方工具

不需要自己实现统计计算。使用：

[calculate_norm_stats.py](https://github.com/RLinf/RLinf/blob/main/toolkits/lerobot/calculate_norm_stats.py?utm_source=chatgpt.com)

它会调用 OpenPI 的数据变换过程，计算训练所需的状态和动作归一化统计。

例如，使用的配置名应是 `pi0_aloha_robotwin`，而不是你第一个项目中的 LIBERO 配置。

务必保存每个实验对应的 `norm_stats.json`，并确保 SFT、PPO、推理评测使用与该策略匹配的统计。 RLinf 官方特别提醒：SFT checkpoint 的权重文件不会自动携带归一化统计。

[image](https://www.google.com/s2/favicons?domain=https://rlinf.readthedocs.io\&sz=32)

RLinf 文档



验收标准： 官方 50 条示范和自己转换的 2 条示范，都能被 RLinf 数据加载器正确读取；检查图像、状态、动作和语言字段之后，再执行批量转换。

## 六、阶段 4：π0 LoRA SFT——研究数据质量与多样性

这一阶段直接用 RLinf 官方训练代码。你不需要改 π0 的 Flow Matching loss，也不需要自己写训练循环。

### 4.1 先看三个文件

| 重要性 | 官方文件                                                                                                                                                                 | 重点                    |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| 必读  | [robotwin_adjust_bottle_sft_openpi.yaml](https://github.com/RLinf/RLinf/blob/main/examples/sft/config/robotwin_adjust_bottle_sft_openpi.yaml?utm_source=chatgpt.com) | 训练数据路径、模型路径、优化器与 FSDP |
| 必读  | [model/pi0.yaml](https://github.com/RLinf/RLinf/blob/main/examples/sft/config/model/pi0.yaml?utm_source=chatgpt.com)                                                 | LoRA、动作维度、网络配置        |
| 参考  | [run_vla_sft.sh](https://github.com/RLinf/RLinf/blob/main/examples/sft/run_vla_sft.sh?utm_source=chatgpt.com)                                                        | 官方 SFT 启动入口           |

完整说明：[RLinf OpenPI 监督微调](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/sft_openpi.html?utm_source=chatgpt.com)。

官方配置的启动方式为：

```
# 在 RLinf 根目录执行，需先修改配置中的路径
bash examples/sft/run_vla_sft.sh robotwin_adjust_bottle_sft_openpi
```

训练使用 `pi0_aloha_robotwin`，其网络 Action Horizon 为 50，环境动作维度为 14。官方配置还包含专门的 Aloha 数据变换，不要直接套用 LIBERO 的配置。

[image](https://www.google.com/s2/favicons?domain=https://rlinf.readthedocs.io\&sz=32)

RLinf 文档



### 4.2 实验对照

我建议首先做两组主要对照，增加一个官方 checkpoint 作为背景参照。

| 编号 | 初始权重                       | 训练数据                       | 目的            |
| -- | -------------------------- | -------------------------- | ------------- |
| S0 | 官方 RoboTwin SFT checkpoint | 不继续训练                      | 参考模型性能        |
| S1 | 同一 π0 基础权重                 | 200 条 Clean                | 检验普通专家示范的适配能力 |
| S2 | 与 S1 相同的初始权重               | 100 Clean + 100 Randomized | 检验数据多样性收益     |

S1、S2 应保持 LoRA rank、优化步数、batch size、训练 seed 和模型结构一致。更严谨的做法是同时固定归一化方案，或明确报告分别计算统计量对结果的潜在影响。

S0 只是官方性能参照，不是用来直接证明你的清洗方法有效的严格消融组。 因为 S0 的数据来源和原始训练预算与 S1、S2 不同。

开始时可以采用 LoRA rank 8 作为候选值。如果训练显存仍然偏高，优先检查模型精度、激活内存、FSDP 和 batch size，而不是只通过进一步降低 rank 来解决。

### 4.3 你要改哪些参数？

主要维护两份自己复制的配置：

```
configs/
├── sft_clean.yaml
└── sft_randomized.yaml
```

其中关键参数包括：

```
data:
  train_data_paths: /path/to/your/lerobot_dataset

actor:
  model:
    is_lora: true
    lora_rank: 8
    model_path: /path/to/pi0_base
    openpi_data:
      norm_stats_path: /path/to/norm_stats.json
```

这是局部字段示例，不是一份能够单独启动的完整 YAML。需要保留官方 Hydra defaults、OpenPI 模型设置以及训练、保存、FSDP 等字段。

额外注意：官方 RoboTwin SFT 配方当前使用较高精度设置，不能简单认为启用 LoRA 后显存就会很低。应测量一次真实峰值显存，再决定 micro-batch 与 FSDP 策略。

### 4.4 SFT 的训练结果要看什么？

主要记录训练 loss、GPU-hours、峰值显存、训练数据量和独立评测成功率。

这一阶段最核心的不是训练 loss 降低多少，而是：

S2 是否比 S1 在相同训练预算下，面对随机化场景时表现更好？

如果 S2 的 Clean 成功率略低，但 Randomized 成功率显著更高，这也是合理且值得分析的结果。

验收标准：得到可加载的 S1、S2 两套模型，分别通过完整 Clean 评测与 Randomized 评测。

## 七、阶段 5：将 LoRA SFT 模型接入 PPO

这是全项目最重要的工程衔接点。

### 5.1 需要看的源码

| 文件                                                                                                                                                                          | 必须理解什么                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| [robotwin_adjust_bottle_ppo_openpi.yaml](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml?utm_source=chatgpt.com) | PPO 训练配方                  |
| [model/pi0.yaml](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/model/pi0.yaml?utm_source=chatgpt.com)                                                 | RL 训练模式、Flow-SDE、冻结策略     |
| [rlinf/models/\_\_init\_\_.py](https://github.com/RLinf/RLinf/blob/main/rlinf/models/__init__.py?utm_source=chatgpt.com)                                                    | PEFT LoRA 是如何注入模型的        |
| [openpi/tasks/rl.py](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/tasks/rl.py?utm_source=chatgpt.com)                                            | π0 RL 模型与 Actor-Critic 接口 |
| [OpenPI Checkpoint Converter](https://github.com/RLinf/RLinf/blob/main/rlinf/utils/ckpt_convertor/openpi/README.md?utm_source=chatgpt.com)                                  | SFT 权重转换、归一化统计文件处理        |

前两个需要理解配置，后面三个只需要定位相关逻辑，不用从头读完所有代码。

### 5.2 先解决 SFT checkpoint 衔接

这里不能直接照搬官方 PPO YAML，因为它默认加载的是官方已经整理好的 SFT checkpoint，而你的 S1、S2 是 LoRA SFT 的结果。

需要先验证：

1. 保存的 LoRA adapter 能否与对应 π0 基础权重一起正确加载。
2. SFT checkpoint 在独立评测程序中能否复现训练后成功率。
3. PPO 的 Actor 和 Rollout 是否都加载了正确的 adapter，并能正常同步更新后的权重。
4. Value head 是否正常初始化、参与训练和保存。
5. 同一份动作归一化统计是否被正确加载。

官方提供 `sft_to_openpi` 等转换模式，但我没有核实到该转换器能够直接处理你这条 LoRA SFT 流程的所有 adapter 合并情况。如果涉及 PEFT adapter 合并，必须单独做数值一致性测试，不能假定 checkpoint 转换完成就等于模型一致。

这一步通过之后，才开始正式 PPO。

### 5.3 PPO 训练本身用官方实现

算法层面不需要你修改 PPO、GAE 或 Critic。

在官方配置上调整两张 GPU 的资源分配、训练环境数量、batch size 和显存卸载选项，继续保留 `adjust_bottle` 的奖励与动作设置。

我建议依次完成三种运行：

| 运行             | 用途       | 成功条件                    |
| -------------- | -------- | ----------------------- |
| PPO Smoke Test | 验证训练流程   | 完成少量更新且无显存错误            |
| PPO Short Run  | 验证优化是否有效 | 训练和评测指标能够正常记录           |
| PPO Formal Run | 正式强化学习实验 | 有稳定学习曲线、checkpoint、独立评测 |

不能仅因为训练成功率上涨就判断强化学习有效，因为 PPO 使用的数据来自不断变化的策略和训练环境。必须用独立评测集比较训练前后结果。

### 5.4 正式 PPO 对照

最小实验只需要一组：

| 编号 | 初始化策略 | 在线训练      |
| -- | ----- | --------- |
| R0 | S2    | 无 PPO     |
| R1 | S2    | RLinf PPO |

这样可以隔离 PPO 的收益。

如果资源允许，再增加：

| 编号 | 初始化策略 | 在线训练      |
| -- | ----- | --------- |
| R2 | S1    | RLinf PPO |

R1 和 R2 的价值在于：检验前期增加的数据多样性，是否会影响后续 PPO 的训练效率和泛化表现。

不过这属于增强实验，不必一开始就做。先把 S2 → PPO 完整跑通，比设计更多实验组更重要。

如果 LoRA PPO 在框架衔接上无法顺利运行，可以考虑保留 LoRA SFT，在 PPO 阶段使用冻结视觉语言骨干、只优化动作专家等受限参数方案。这依然不是整个模型的全量微调，但需要在实验报告中明确写出 PPO 阶段更新的参数范围，不应称为 LoRA PPO。

### 5.5 强化学习部分必须掌握的知识

建议认真阅读 [RLinf π0 在线 RL 教程](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/pi0.html?utm_source=chatgpt.com) 和 [πRL 论文](https://arxiv.org/abs/2510.25889?utm_source=chatgpt.com)。

结合你已有的 CS285 基础，重点复习 PPO 的 Clipped Objective、Value Function、GAE、Importance Sampling，以及 Flow-SDE 如何为 Flow Matching 策略提供可优化的动作概率。

不需要为了这个项目重新学习一遍完整的强化学习课程。

验收标准： PPO 模型能保存、恢复、独立评测，并能报告相对 SFT 的成功率变化及单位环境交互量带来的性能提升。

## 八、阶段 6：标准评测、泛化能力与资源效率

### 6.1 直接复用官方评测

必读文件：

[robotwin_adjust_bottle_openpi_eval.yaml](https://github.com/RLinf/RLinf/blob/main/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml?utm_source=chatgpt.com)[eval_seeds.json](https://github.com/RLinf/RLinf/blob/main/rlinf/envs/sim/robotwin/seeds/eval_seeds.json?utm_source=chatgpt.com)

说明文档：

[RoboTwin 官方评测指南](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/evaluations/guides/robotwin.html?utm_source=chatgpt.com)

官方定义的核心指标是 `eval/success_once`，即 episode 内至少一次满足成功条件的比例。

注意一个与你的双卡服务器直接相关的细节：降低评测并行环境数量不等于自动完成了全部测试。

官方 `adjust_bottle` 共有 150 个预设成功种子，默认 128 环境的单轮配置也不一定覆盖全部种子。RLinf 文档提供了动态种子与自动重置模式，可以在有限并行环境下逐步评测更多 seed。你应使用这一方式，并核对实际跑过的 seed 数量与重复情况。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 6.2 正式评测分成四个条件

| 评测条件                    | 用来检验             |
| ----------------------- | ---------------- |
| Clean                   | 标准任务成功率          |
| Randomized              | 光照、背景、杂物变化下的泛化能力 |
| Background/Lighting（可选） | 单独的视觉域变化         |
| Clutter（可选）             | 杂物干扰下的操作稳定性      |

Clean 评测用 RLinf 官方 π0 协议，Randomized 则是你额外引入的扩展场景。不要把随机化评测结果直接与官方 Clean 排名混为一谈。

尤其需要保证评测期间模型不再训练，所有被比较的策略使用相同的测试条件、相同动作执行长度和一致的成功判定。

### 6.3 最终模型比较表

| 模型                     | Clean SR | Randomized SR | GPU-hours | PPO 环境步数 |
| ---------------------- | -------- | ------------- | --------- | -------- |
| 官方 SFT 参考              | 待测       | 待测            | —         | 0        |
| S1：Clean LoRA SFT      | 待测       | 待测            | 待测        | 0        |
| S2：Mixed LoRA SFT      | 待测       | 待测            | 待测        | 0        |
| R1：Mixed SFT + PPO     | 待测       | 待测            | 待测        | 待测       |
| R2：Clean SFT + PPO（可选） | 待测       | 待测            | 待测        | 待测       |

SR 指任务成功率。正式报告时建议补充成功数/episode 总数和 Wilson 95% 置信区间，并保存每个模型在对应 seed 上的结果，以便进行配对比较。

学习曲线和日志可以直接使用 RLinf 的 TensorBoard；[analyze_logs.py](https://github.com/RLinf/RLinf/blob/main/tests/parity_tests/analyze_logs.py?utm_source=chatgpt.com) 可作为读取、绘制成功率曲线的参考，但不是专门为这套 RoboTwin 对照实验准备的分析程序。

你只需要自己编写一个 `analyze_results.py`，集中完成成功率汇总、置信区间、训练成本和失败案例统计。

## 九、你最终真正需要维护的代码

尽量让仓库保持轻量：

```
pi0-robotwin-experiments/
├── configs/
│   ├── demo_clean_pi0.yml
│   ├── demo_randomized_pi0.yml
│   ├── sft_clean.yaml
│   ├── sft_randomized.yaml
│   ├── ppo_2gpu.yaml
│   └── eval_randomized.yaml
│
├── scripts/
│   ├── audit_dataset.py
│   ├── legacy_hdf5_adapter.py    # 仅在格式不兼容时需要
│   └── analyze_results.py
│
├── manifests/
│   ├── collection_manifest.csv
│   ├── dataset_audit.csv
│   └── experiment_manifest.csv
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── videos/
│
└── README.md
```

其中：

- `audit_dataset.py` 是你最值得独立完成的数据工程代码。
- `legacy_hdf5_adapter.py` 属于可能需要的兼容性代码，先查官方转换工具能否覆盖，不要提前开发。
- `analyze_results.py` 用于实验结果分析。
- YAML 基于官方模板复制并修改，不需要从零设计。
- PPO、SFT、仿真采集和评测的核心代码都沿用官方实现。

## 十、推荐的阅读顺序与实际推进顺序

我建议阅读内容和实施任务一一对应，而不是先把几十个源码文件全部读完。

项目执行检查表

0/7 已完成

1\. 环境与 PPO 连通性

读：RLinf RoboTwin 文档、PPO YAML、π0 模型 YAML

验收：两卡完成一次 PPO 更新

2\. 专家数据生成

读：collect_data.py、adjust_bottle.py、demo_clean / demo_randomized

验收：能生成并回放两种环境的轨迹

3\. 数据质量审计

读：pkl2hdf5.py、LeRobot Dataset Tools

验收：完成 audit_dataset.py 和质量报告

4\. LeRobot 数据接口

读：robotwin_aloha_dataconfig.py、v2.1 转换工具、calculate_norm_stats.py

验收：自己的数据可被 RLinf DataLoader 正确读取

5\. LoRA SFT

读：robotwin_adjust_bottle_sft_openpi.yaml、SFT 文档

验收：S1、S2 两个模型和对照评测

6\. SFT → PPO 衔接

读：models/\_\_init\_\_.py、Checkpoint Converter、PPO YAML

验收：训练权重与归一化统计正确加载

7\. 正式 PPO 与评测

读：rl.py、RoboTwin Evaluation、eval_seeds.json

验收：学习曲线、最终成功率与失败分析

## 十一、最后要明确的三个验收关口

关口 A：双卡 PPO 训练可用。 不只是模型加载，必须能够执行完整优化器更新。

关口 B：自己采集的数据成功进入 π0 训练。 旧版 HDF5 转换成 LeRobot 后，图像、状态、动作与语言必须正确匹配。

关口 C：LoRA SFT 权重能正确接入 PPO。 Adapter、Actor/Rollout 权重、Value head 与归一化统计都要经过测试。

这三个关口都通过后，整个项目基本就不存在需要重写大型框架的问题，后面主要是采集数据、组织对照实验和分析结果。

最终你应该真正学会的，不是怎样把四个开源仓库的命令复制一遍，而是怎样检查机器人数据是否可靠、保证模型与数据接口一致、完成一个在线 RL 训练闭环，并用可复现的实验说明数据质量与 PPO 分别产生了什么作用。 这才是第二个项目区别于第一个 π0.5 微调项目的核心。