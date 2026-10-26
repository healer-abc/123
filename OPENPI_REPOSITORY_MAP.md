# openpi／π0.5 项目仓库地图

快照日期：2026-10-23。工作区：`/home/zhsu/jyluo`（也就是 `~/jyluo`）。范围已确认：**openpi／π0.5，含 LIBERO 主线、MetaWorld VLA 探索和相关基线**。Gemma、LLaMA-Factory、Fetch-Push 只标记边界，不混成一个项目。

这是一份按实际文件系统扫描建立的地图，不是重新解释“仓库应该长什么样”。内容分两层：本页用于理解各目录及数据流；逐文件/逐目录清单覆盖大量权重分片、环境依赖与评测产物。不会把几十万个环境包文件伪装成几行目录树就算“全部列清”。

## 地图入口与逐文件索引

地图配套目录：[REPOSITORY_MAP_20261023](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/scan_summary.json)。

| 文件 | 内容 | 怎么用 |
|---|---|---|
| [SOURCE_FILES.md](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/SOURCE_FILES.md) | 项目关联源码、脚本、配置、文档逐文件说明，含外部源码副本 | 在编辑器中按文件名搜索；有用途、来源依据和本人审查点 |
| [UPSTREAM_LOCAL_CHANGES.md](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/UPSTREAM_LOCAL_CHANGES.md) | 本地上游路径的内容/换行差异列表 | 分清官方来源与本地修改，不把换行差异算算法创新 |
| [project_files.csv.gz](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/project_files.csv.gz) | 主线、MetaWorld、数据/模型/日志等关联文件全清单 | 每个文件一行；包括非源码、权重分片、视频和逐episode结果 |
| [dependencies_files.csv.gz](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/dependencies_files.csv.gz) | venv、site-packages、miniforge 等安装产物逐文件清单 | 查依赖边界，不把第三方安装包当作项目原创 |
| [adjacent_files.csv.gz](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/adjacent_files.csv.gz) | 相邻独立项目/来源待核实项及敏感配置的metadata | 防止误将 Gemma/Fetch-Push 纳入 openpi；不含凭据内容 |
| [directories.csv.gz](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/directories.csv.gz) | 扫描到的每个目录、文件数、逻辑字节数、目录用途 | 查任何深层目录，包括 task/state、环境包和checkpoint子目录 |
| [scan_summary.json](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/scan_summary.json) | 本次计数、Git索引基线、扫描错误和边界说明 | 确认覆盖范围；没有扫描错误才不应漏报权限失败 |
| [lookup.py](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/lookup.py) | 无需解压的只读检索工具 | 输入路径前缀，可查目录或任意文件的身份/用途 |
| [build_map.py](experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/build_map.py) | 本次地图生成器及分类规则 | 可审计“来源分类”依据；只写地图目录，不运行模型 |

**清单字段**：`path` 路径、`scope` 项目关系、`kind` 文件/链接、`bytes` 字节数、`origin` 来源分类、`evidence` 分类依据、`purpose` 用途、`review` 本人审查点、`created_by` 生成机制。批量重复产物按类型和路径身份解释，不声称读取过每个大权重内部张量。

只读检索示例（不占GPU、不训练、不解压大文件）：

```bash
cd /home/zhsu/jyluo

# 查核心训练配置这一个文件
python experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/lookup.py \
  projects/openpi/src/openpi/training/config.py

# 查主线目录，0表示打印全部匹配
python experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/lookup.py \
  experiments/pi05_libero_adaptation/manifests/ --limit 0

# 查某组完整权重目录及其每个分片
python experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/lookup.py \
  projects/openpi/checkpoints/pi05_libero_lora/pi05_libero_lora_gpu0/29999/ --limit 0

# 查目录而不是文件；不要拿环境包目录当源码入口
python experiments/pi05_libero_adaptation/report/REPOSITORY_MAP_20261023/lookup.py \
  projects/openpi/src/openpi/ --catalog directories --limit 0
```

## 1. 来源标记：怎样读这张地图

| 标记 | 意义 | 不能据此宣称的事情 |
|---|---|---|
| 上游来源、与本地Git索引一致 | 文件blob与该仓库本地索引相同 | 不是在线main最新版本认证；索引本身也可能有暂存变化 |
| 上游路径、仅换行不同 | CRLF→LF后与索引相同 | 不能将其写为本地算法修改 |
| 上游路径、本地内容修改 | 不只是换行，与本地索引不同 | 差异不自动证明具体个人作者 |
| 本地项目扩展 | 本项目的脚本、清单、审阅、报告、适配器等 | 不应称作官方openpi原生功能或本人全部独立手写 |
| 本会话可确认Agent新增 | 对话中有明确创建记录的文件 | 不证明用户本人已运行、已审阅或已熟悉 |
| 外部下载输入 | LIBERO数据、pi05_base、tokenizer、公开模型等 | 不是本项目自行采集/预训练/微调产生 |
| 训练器/跟踪自动产物 | checkpoint、Orbax、W&B | 自动生成不等于不需审阅，也不等于可随意删除 |
| 本地管线产物 | 自定义日志、身份JSON、NPZ、视频、接触表和统计 | 调用官方模型不意味着这些输出路径由官方训练器自行建立 |
| 来源待核实 | 不能从索引/路径/现有记录证明来源 | 不从mtime或文件名推断“是我/你/官方写的” |

本次只读了小型源码/说明的静态摘要、Git元数据以及文件系统metadata；没有读取认证凭据，没有全量哈希TB级数据与模型，没有执行任何训练/测评/历史调度器。索引比较不是完整Git提交历史重建。

## 2. 工作区总览：先分清十二个大区域

```text
~/jyluo/
├── projects/
│   ├── openpi/                       [主代码] 实际模型、训练、服务与示例
│   ├── offline-deps/                 [第三方输入] lerobot、dlimp源码依赖
│   ├── LLaMA-Factory/                [独立项目] Gemma等LLM训练框架
│   └── mujoco-fetch-push-rl/          [独立项目] Fetch-Push强化学习
├── experiments/
│   ├── pi05_libero_adaptation/       [主线] E0–E3协议、审计、结果、交付
│   ├── pi0*_metaworld_*/             [旁线] VLA模型/动作接口/环境探索
│   └── metaworld_*/                  [关联对照] DP/ACT/SimpleDP3等
├── dataset_root/                     [外部数据] LeRobot LIBERO示范
├── myopenpi/dataset/ckpt/             [预训练输入缓存] pi05_base、tokenizer
├── tools/                           [早期本地工具] 配置补丁/解析/监控/调度
├── eval_auto/                       [早期自动输出] 旧LIBERO四套测评
├── metrics/                         [早期自动输出] 全参训练日志解析
├── metrics_lora/                    [早期自动输出] LoRA训练日志解析
├── miniforge3/                      [共享基础设施] 编译器/conda/包与环境
├── gemma-exp/                       [独立项目] Gemma LoRA配置/输出
├── models/gemma-7b/                 [独立项目资产] Gemma权重
└── .codex/                         [敏感基础设施] 认证配置，非模型证据
```

`~/jyluo`不是单个Git仓库。openpi内部用Gemma家族组件，不代表它会读取根目录`models/gemma-7b`或`gemma-exp/train_gemma_lora.yaml`；本轮模型来自集成的pi05_base。

### 根目录每个已有普通文件

| 文件 | 身份/用途 | 本人注意 |
|---|---|---|
| `OPENPI_REPOSITORY_MAP.md` | 本次Agent新增的仓库地图入口 | 本文，不是官方README |
| `README_部署说明.md` | 早期部署记录/离线路径源方案 | 含历史删除/覆盖命令，不能当当前执行清单 |
| `openpi_env.sh` | 本地shell环境变量入口 | 只设缓存/编译器/镜像等，不保证当前Python来自venv；其中W&B offline曾是注释 |
| `pyproject.server.toml` | 离线服务器依赖模板 | 生效文件在`projects/openpi/pyproject.toml`，不是根目录模板 |
| `openpi.tar.gz` | 传入的openpi源码/子模块离线分发包 | 不等于官方未修改原件，也不等于当前工作树；禁止解压覆盖旧项目 |
| `offline-deps-bundle.tar.gz` | 离线依赖汇总分发包 | 当前引用的是解压后的`projects/offline-deps` |
| `lerobot-min.tar.gz` | 精简LeRobot源码包 | 外部依赖，不是本地算法原创 |
| `dlimp-min.tar.gz` | 精简dlimp源码包 | 同上；主要涉及另一数据装载路线 |
| `auto_eval.log` | 旧自动评测总日志 | shell/本地调度产生，不能证明现在还在跑 |
| `metrics_lora_steps.txt` | 历史训练指标文本摘录 | 派生摘要，严肃引用回到原日志/结构化表 |
| `bashrc.backup.before_revert` | shell配置历史备份 | 不是openpi训练配置，个人来源/内容不作本次作者判断 |
| `load_gemma_test.py` | 独立Gemma加载测试 | 不属于π0.5训练/测评入口 |

`.codex/auth.json`只记录其敏感文件身份，不展示内容、不纳入公开交付。地图清单不是授权打包整工作区。

## 3. projects/openpi：官方来源工作树与本地改动

### 3.1 当前可追溯到什么基线

本地Git元数据记录openpi HEAD为 `215abfb217dbac7d5f1273282331b9b1866c0479`；LIBERO子模块HEAD为 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`；ALOHA为 `d1dc83afd89ded4379851257fe5d85632d31d5ec`。这次读取Git元数据恢复了**当前检出基线**，不把它反填成所有历史训练已登记的commit；历史runs中commit为空的事实仍然保留。

相对本地索引，openpi主工作树实质内容差异集中在：

| 文件 | 本地变更含义 | 来源归属 |
|---|---|---|
| `pyproject.toml` | 使用本地lerobot/dlimp路径源及本机依赖适配 | 官方配置载体上的本地部署变更 |
| `src/openpi/training/config.py` | 添加π0.5 E1/E2/E3及相应freeze/LR/asset配方 | 官方注册表中的本地实验扩展 |
| `examples/libero/main.py` | 视频命名、成功视频保存选项等本地补丁 | 官方示例上的本地测评输出变更 |

大量其余文件只是CRLF/LF差异，不应被算成自研实现。完整逐项情况和未列入索引的路径见差异清单；没有Git可追溯基线的副本仍标待核实。

### 3.2 源码目录逐区地图

| 路径 | 作用 | 本项目是否重点阅读 |
|---|---|---|
| `scripts/train.py` | JAX/Flax训练入口，初始化、数据、求导/更新、记录、保存 | 是，理解真正的训练流程 |
| `scripts/train_pytorch.py` | 另一后端训练入口 | 可参考，不能当本轮实际训练器 |
| `scripts/serve_policy.py` | 普通WebSocket模型服务 | 是，和固定噪声服务区分 |
| `scripts/compute_norm_stats.py` | 计算训练输入/动作归一化统计 | 是，写资产会影响所有组 |
| `src/openpi/training/config.py` | config名→模型、数据、冻结、LR、预算、路径 | 第一审查入口，真实argv还会覆盖默认值 |
| `training/data_loader.py` | LeRobot/RLDS数据与batch/变换/shuffle | 审采样、episode边界、padding、归一化 |
| `training/optimizer.py` | LR与AdamW/梯度裁剪 | 审真实beta、weight_decay、warmup/decay |
| `training/checkpoints.py` | 保存、恢复、EMA推理参数导出 | 审是否finalized、恢复状态/EMA类型 |
| `training/sharding.py` | JAX mesh/数据与模型分片 | 两卡机器不等于单模型两卡训练 |
| `training/utils.py` | 训练状态与辅助操作 | 配合train/checkpoint追踪状态 |
| `training/droid_rlds_dataset.py` | DROID/RLDS数据路线 | 不要和本轮LeRobot LIBERO混用 |
| `models/pi0.py` | π0/π0.5实际前向、mask、flow loss和采样 | 核心算法阅读 |
| `models/pi0_config.py` | 模型变体/state/动作维度/冻结过滤器 | 冻结集合与训练集合的关系 |
| `models/gemma.py` | 语言/动作专家、attention/FFN/LoRA/remat | 实际矩阵位置与rank |
| `models/lora.py` | 低秩Einsum与FFN | 初始化/缩放以本地算子为准 |
| `models/siglip.py`、`models/vit.py` | 图像表征实现/组件 | img子树冻结范围 |
| `models/model.py` | 观测/动作结构和训练图像增强 | 相机、mask、dtype、空间增强 |
| `models/tokenizer.py` | 指令/state token化及tokenizer下载 | 当前state开关关闭，不代表默认都关闭 |
| `models/pi0_fast.py`、`gemma_fast.py` | FAST动作token路线 | 官方其他模型，不是本轮flow主线 |
| `models_pytorch/` | PyTorch模型与替换组件 | 非本轮主后端；不要盲复制替换包 |
| `policies/libero_policy.py` | LIBERO输入和32→7输出映射 | 最高优先级接口审查 |
| `policies/policy_config.py` | 从config/checkpoint/资产构造policy | checkpoint与统计/精度要匹配 |
| `policies/policy.py` | 前后变换/JIT/推理 | 显式noise和计时、权重dtype |
| `policies/{aloha,droid}_policy.py` | 其他机器人的接口 | 不等于能直接适配MetaWorld/Sawyer |
| `serving/websocket_policy_server.py` | 网络协议服务 | 模型身份、端口、错误传播 |
| `transforms.py` | 重打包、归一化、token化、padding等 | 数据链条审查 |
| `shared/normalize.py` | 统计读写和数值归一化组件 | 分位数与z-score不同 |
| `shared/download.py` | 资源下载/缓存 | 看当前实际代码与缓存，不推定为本地原创 |
| `shared/image_tools.py` | resize/pad/图像处理 | 与客户端相机处理配套 |
| `shared/nnx_utils.py` | 参数过滤、JIT等NNX工具 | 训练集合和冻结规则审计 |
| `shared/array_typing.py` | 数组类型与运行时形状约束 | 不等于数据语义已被验证 |
| `*_test.py`、`conftest.py` | 上游来源测试与fixture | 文件存在不等于当前环境已经跑过 |

源码中更深的`utils/`、PyTorch替换文件、测试、示例、CI等每个文件都在`SOURCE_FILES.md`中，不只列上表的主入口。

### 3.3 例子、子包和第三方

- `examples/libero/`：示例仿真客户端与Python3.8环境。`main.py`使用`--args.*`；会渲染环境、请求模型、保存视频。它不是项目固定manifest客户端。
- `examples/{aloha_sim,aloha_real,droid,ur5}/`：其他机器人演示/接入说明；目录存在不表示本项目都做过实机实验。
- `examples/simple_client/`：轻量调用演示。
- `packages/openpi-client/`：网络客户端与图像工具，能装进独立仿真环境；不是完整模型训练包。
- `third_party/libero/`：Lifelong-Robot-Learning/LIBERO子模块，环境、BDDL、初始状态、benchmark定义和资源。官方第三方代码，不是Agent创建的仿真器。
- `third_party/aloha/`：ALOHA子模块，非LIBERO实际评测器。
- `docs/{norm_stats,remote_inference,docker}.md`：上游来源说明；和本项目新写的报告不是同一来源。
- `.github/`、`.pre-commit-config.yaml`：开发/CI工具声明；不等于历史实验的执行日志。
- `.git/`：版本对象/索引/子模块元数据，不手改、不作为模型参数。
- `.vscode/`：编辑器工作区配置，不是训练超参数入口；可能有本地路径。

### 3.4 openpi根目录里并非都是官方源码

| 文件/目录 | 类别 | 谁/什么生成 |
|---|---|---|
| `README.md`、`LICENSE*`、`CONTRIBUTING.md` | 上游来源文档/许可 | 源码分发/checkout |
| `pyproject.toml` | 已本地适配的生效依赖配置 | 上游载体+本地部署修改 |
| `uv.lock` | 当前解析锁，具体追踪状态看索引 | uv解析/迁移；不是运行模型的参数 |
| `.python-version` | 项目Python选择提示 | 项目配置；绝对路径解释器仍优先判断 |
| `pyproject.toml.bak` | 部署前后历史备份 | 本地复制，不是当前生效配置 |
| `src/openpi/training/config.py.bak.20261014-114042` | 追加LoRA配置前备份 | 历史补丁工具/本地复制 |
| `train.log`、`train_resume.log` | 早期全参训练/恢复stdout记录 | shell重定向/训练启动方式决定文件名 |
| `train_lora.log` | 早期E1正式训练日志 | 同上，不是官方固定输出名称 |
| `train_e2_frozen_vision.log` | E2正式训练日志 | 同上 |
| `sync.log`、`libero_sync.log`、`hf_download.log` | 安装/数据下载阶段输出 | 本地shell/安装下载过程 |
| `echo`、`grep` | 当前存在的异常命名普通文件 | 来源/用途待核实；不是训练入口，不能凭名字当系统命令 |
| `.venv/` | 训练/服务Python3.11及依赖 | uv/pip安装产物，**不是train.py创建** |
| `examples/libero/.venv/` | 仿真Python3.8及依赖 | 另一套uv/pip环境 |
| `__pycache__/` | Python编译缓存 | import/compile自动生成 |
| `wandb/` | 离线运行历史/元信息/事件/媒体 | 训练器调用W&B自动创建 |
| `assets/` | 归一化输入/复制资产 | compute_norm_stats产生后复用；不是每次训练都重算 |
| `checkpoints/` | 模型、优化器和恢复状态/推理资产 | train.py调用保存逻辑与Orbax |

## 4. 权重目录地图：base、正式、smoke和profile不混淆

### 4.1 外部base和tokenizer

```text
myopenpi/dataset/ckpt/
├── openpi-assets/checkpoints/pi05_base/params/  外部预训练起点
└── big_vision/                                tokenizer资源缓存
```

`myopenpi`不是另一套源码仓库；`OPENPI_DATA_HOME`控制下载器缓存位置。这里的params不是E0/E1/E2/E3本地训练结果。分片文件并非可单独替换的模型。

### 4.2 本地训练权重

```text
projects/openpi/checkpoints/
├── pi05_libero/
│   └── pi05_libero_full_gpu1/                  E0正式（最终29999）
├── pi05_libero_lora/
│   ├── pi05_libero_lora_gpu0/                  E1正式（最终29999）
│   └── e1_profile_220_s42/                     E1独立成本短测
├── pi05_libero_lora_frozen_vision/
│   ├── pi05_libero_lora_frozen_vision_s42/      E2正式（最终29999）
│   ├── pi05_libero_lora_frozen_vision_smoke_s42/ E2十步验证（9）
│   └── e2_profile_220_s42/                     E2成本短测（219）
├── pi05_libero_action_only/
│   ├── pi05_libero_action_only_s42/            E3正式（最终29999）
│   ├── pi05_libero_action_only_smoke_s42/       E3十步验证（9）
│   └── e3_profile_220_s42/                     E3成本短测（219）
└── debug_pi05/debug_pi05/                      早期调试，不替代正式组
```

每个exp下面还有1000、5000等已保留训练步目录；实际所有文件/目录由清单列出，不只列29999。恢复应保持同一exp并用resume；新实验用新exp，不对旧目录overwrite。

### 4.3 每个checkpoint内部文件是什么

| 模式 | 含义 | 必审 |
|---|---|---|
| `<step>/_CHECKPOINT_METADATA` | Orbax checkpoint元信息/保存完成核验项 | 只有目录不代表保存完成 |
| `<step>/params/_METADATA`、`_sharding`、`array_metadatas/` | 张量结构、分片/数组元数据 | 与实际数据分片一起保留 |
| `manifest.ocdbt`、`d/<hash>`、`ocdbt.process_*/d/<hash>` | TensorStore/OCDBT索引和数据块 | hash样文件名不是随意命名的实验，不逐片手改 |
| `opt_state/`或checkpoint中其他状态树 | 优化器/训练恢复状态，具体以目录记录为准 | 推理权重与训练恢复全状态不同 |
| `assets/physical-intelligence/libero/norm_stats.json` | 本checkpoint配套的推理归一化资产 | 四组是否同源，不能用另一个机器人统计替换 |
| 临时/未finalize目录 | 保存过程的中间状态 | 不供测评，不为“修复”而手工补完成标记 |

E0推理导出采用EMA，E1/E2非EMA。文件布局/参数数量不能代替检查实际导出代码与manifest。

## 5. 训练数据每类文件：不是运行时采集的新数据

```text
dataset_root/physical-intelligence/libero/
├── meta/
│   ├── info.json        数据schema、fps、总episode/frame/task与路径模板
│   ├── episodes.jsonl   每个episode对应任务与长度
│   ├── tasks.jsonl      训练task_index与语言指令
│   └── stats.json       发布数据的统计元信息
├── data/
│   ├── chunk-000/episode_000000.parquet ...
│   └── chunk-001/episode_001000.parquet ...
└── .cache/              HF下载缓存/metadata/完成记录等
```

每个episode Parquet包含嵌入图像/腕部图像、state、actions、时间/索引等。元信息`total_videos=0`不表示没有视觉输入，而是没有独立MP4视频存储。训练ID与LIBERO套件task_id不是同一编号体系。

本轮使用1693条、273465帧、40任务的公开转换版数据，不是本人本轮新采集。本人应审查数据版本、指令、动作定义、采样/尾部padding、训练/推理变换和norm stats，而不是编辑episode里的二进制图像去“修成绩”。

`projects/offline-deps/lerobot/`提供本地LeRobot装载源码（含episode尾部clamp/pad行为）；`dlimp/`为另一装载生态依赖。二者来源于外部项目的精简离线副本，不能把其中全部文件计为本项目编写。

## 6. LIBERO主实验目录：每个入口、材料和结果

主目录：`experiments/pi05_libero_adaptation/`。这里主要是**项目本地扩展**，不是官方openpi源码附带的科学实验管理系统。

### 6.1 顶层程序逐文件

| 文件 | 用途 | 人工审查/执行边界 |
|---|---|---|
| `audit_parameters.py` | 按config展开逐叶计数/训练标记CSV | 在训练前审更新范围 |
| `verify_e2_smoke.py` | E2固定seed初始化与10步checkpoint差分/重载 | 专用配方，不是任意模型通用验证 |
| `verify_e3_smoke.py` | E3冻结/更新检查及重载推理 | 断言失败不能直接放宽容差掩盖 |
| `build_eval_manifest.py` | 生成固定开发/确认身份与seed/hash | 已冻结清单不可重生成覆盖 |
| `seeded_policy_server.py` | 接收显式高斯噪声的模型服务 | 与普通serve_policy不是同一个协议入口 |
| `eval_manifest.py` | 按清单评测，保存JSON/动作NPZ/失败MP4/attempt | 核对metadata、checkpoint、分母和异常 |
| `development_watchdog.py` | 资源准入/进程/心跳/锁/限次恢复与汇总 | 新run不应未经改造读取旧runs配置 |
| `training_watchdog.py` | 历史E3训练恢复与后续开发调度 | 带旧路径/预算；不是新的通用训练命令 |
| `run_development_pipeline.sh` | 旧开发看门狗入口外壳 | 不直接重复启动旧组 |
| `profile_training.py` | 每组独立220步成本短测 | 不把短测权重当正式训练起点 |
| `profile_hooks.py` | 在W&B初始化后挂observer | 监测修复，不改变模型算法 |
| `cost_profiles_watchdog.py` | 成本队列与历史记录恢复 | 旧队列/旧输出路径 |
| `compare_development.py` | 旧E1/E2固定400开发对比 | 不传新run仍会读取旧结果 |
| `complete_e3_analysis.py` | 等待旧E3开发并与参考比较 | 不会产生新训练 |
| `review_development.py` | E1/E2动作边界与失败接触表 | 自动生成接触表不等于完成本人审阅 |
| `review_final_development.py` | 终审数据验收/固定E3抽样 | 不等于全体逐帧机制标注 |
| `confirmation_gate.py` | 固定候选、清单、源码和权重核验 | 冻结后变更应拒绝，不绕断言 |
| `freeze_confirmation.py` | 原候选E2冻结记录 | 一次性原主线入口，不覆盖重做 |
| `run_confirmation_pipeline.py` | 原确认/E0补齐/分析串联 | 绑定旧E0/E1/E2，不给新run盲用 |
| `analyze_confirmation.py` | 记录验收、任务/套件等权、Wilson及配对bootstrap | 核实抽样单位/分母/区间局限 |
| `test_development_watchdog.py` | 调度、互斥、身份与恢复CPU回归测试 | 测试源码≠本轮测试已执行 |
| `test_e3_pipeline.py` | E3白名单/保存门槛等测试 | 同上 |
| `test_profile_recovery.py` | observer/计时恢复口径测试 | 同上 |
| `test_confirmation.py` | 确认隔离、等权和冻结核验测试 | 同上 |
| `HANDOFF.md` | 交接和阶段状态快照 | 必须审更新时间，当前文件含历史状态，不能直接当活动进程事实 |
| `protocol.md` | 原实验设计/固定预算/指标口径 | 真实科学依据之一，不事后静默改写 |
| `decision_log.md` | 选择/暂停/恢复/异常与口径决策 | 审时间顺序和依据 |
| `e2_smoke.log` | 历史E2验证stdout | 运行产物，不是配置源码 |

### 6.2 manifests：训练/测评前本人最该读的材料

| 文件 | 具体含义 | 是否可手改 |
|---|---|---|
| `dataset_manifest.json` | 所用数据/输入/动作维度/统计文件登记 | 有些文字是登记时快照，须结合后续审计；新实验写新版本 |
| `runs.json` | 各组config、真实argv、seed、检查点、恢复/状态记录 | 原结果引用依据；不能偷偷改变旧实验身份 |
| `eval_manifest.csv` | phase、suite、task、state/hash、环境/策略seed、执行规则 | 已冻结，不覆盖 |
| `exposure_log.csv` | 结果/状态何时被查看或确认启动等暴露记录 | 按事实追加，不能伪造未暴露历史 |
| `candidate_freeze.json` | 原E2候选、E0/E1参考、权重/源码/资产/规则哈希 | 冻结后不可手改 |
| `candidate_freeze.json.sha256` | 冻结文件自身内容哈希 | 不是随意“更新一下”使错误通过的文件 |

这些很多是本地工具生成的，但“机器生成”不代表可以不读。本人需要明确接受了哪些配置、计分与风险，不能把文件存在说成自己已经审过。

### 6.3 audits、metrics、logs

| 目录/文件模式 | 具体用途 | 生成机制/审查重点 |
|---|---|---|
| `audits/parameter_audit_pi05_*.csv` | 所有参数叶及总数/可训练标记 | 参数审计脚本自动输出；本人对账 |
| `audits/frozen_parameter_checks.json` | E2实际冻结/更新差分 | 比配置文本更强，但仍受验证范围限制 |
| `audits/e3_smoke_verification.json` | E3冻结与重载验收 | 不证明所有跨进程未来推理完全确定 |
| `audits/resource_environment.json` | 设备、dtype、预热/采样/共享约束 | 成本结论必须配套读 |
| `audits/final_delivery_audit_20261022.json` | 原6400条记录的交付验收 | 通过范围是身份/hash/有限性等，不是机制因果证明 |
| `metrics/e1/`、`e2/` | 训练steps/throughput/summary | 由早期日志解析，不是验证集loss |
| `metrics/e3/` | 训练各attempt资源与评测资源 | 区分中断/恢复窗口 |
| `metrics/profiles/` | 独立成本测量与恢复时间序列/资源CSV | E1原observer缺失与恢复来源要保留 |
| `logs/*train_attempt*.log` | 每次训练stdout/错误 | 不能拼接掩盖重启 |
| `logs/*.launch.json` | 启动argv、设备和环境等快照 | 可核对实际执行配置 |
| `logs/*status*.json`、heartbeat等 | 监督器最后一次写的状态 | 是历史文件，不等于进程现存 |
| `logs/alerts.jsonl`、`latest_alert.json` | 错误及报警 | 后续恢复不删除原告警 |
| `logs/*.pid`、`*.lock` | 历史PID、互斥相关材料 | 查真实PID归属，不照抄kill |

### 6.4 evaluations：一条评测的证据在哪里

```text
evaluations/<E0-s42|E1-s42|E2-s42|E3-s42>/<development|confirmation>/
├── episodes/<suite>/task_XX/state_YY.json   身份/seed/成功/动作边界/路径/hash
├── episodes/<suite>/task_XX/state_YY.npz    原始动作、控制器输入代理、推理耗时
├── videos/<suite>/*_failure.mp4            失败可视化（如该episode失败）
├── attempts/<suite>/task_XX/state_YY/      attempt编号、时间、完成/错误记录
├── summary.json                           总完成量、任务/套件成功率
├── summary.shard_N.json                   分片汇总（如使用分片）
└── heartbeat.shard_N.json / shard锁        最后状态与互斥记录
```

上述目录由**本地eval_manifest/调度系统**建立，不是跑官方`train.py`就会自动有。JSON和NPZ不是同一个用途；视频文件数量也不是总评测次数。state命名中的YY来自清单，不是随机新轨迹的编号。

`evaluations/development_comparison.json`、`e3_development_comparison.json`、`confirmation_comparison.json`是本地统计派生；逐任务分子、身份和计分应能回到episode记录。6400仅包含新协议主线：E0/E1/E2各2000，E3开发400；不混旧eval_auto或诊断24次。

### 6.5 report：报告不是数据源的替代品

| 文件/目录 | 用途与来源 | 本人要做什么 |
|---|---|---|
| `PROJECT_RESULTS_20261022.md` | 完整主线与MetaWorld历史附录，项目整理文档 | 主交付入口，审数字和边界 |
| `confirmation_report.md` | 本地统计生成的原确认报告 | 审分母不等时的suite等权 |
| `development_review/` | E1/E2接触表、样本、标签、动作边界与review_notes | 图表由工具生成；标签/结论需要本人核对 |
| `final_development_review/` | E3抽样、artifact audit、成败分歧、终审记录 | 不把抽样比例当总体机制发生率 |
| `INTERVIEW_TRAINING_EVAL_QA_20261022.md` | 本会话Agent创建的训练/测评面试总结 | 区分实测与通用原理 |
| `INTERVIEW_72_ANSWERS_20261023.md` | 本会话Agent创建，逐题及40任务附表 | 个人贡献按真实经历调整，不照抄虚构独立工作 |
| `OPENPI_ENGINEERING_RUNBOOK_20261023.md` | 本会话Agent创建，完整环境/tmux/四组命令手册 | 阅读不等于已安装/执行 |
| `OPENPI_README_STYLE_20261023.md` | 当前存在的另一版按官方入口讲解的操作说明 | 本次只识别用途；不凭路径认定本会话作者，也不要两套流程重复运行 |
| `REPOSITORY_MAP_20261023/` | 本次新增地图、生成器及索引 | 元数据盘点，不是新实验 |

`.jpg`接触表、`.csv`标签、`.json`审计分别是“可视化”“审阅判断”“机器检查”，三者不能混成“官方代码自动证明了结论”。

### 6.6 manual_workflow：本会话新增的学习辅助，不是官方代码

- `lab.sh`：定义四组配置选择、smoke、train/resume、serve、eval、profile/status等bash函数；source本身不启动训练。新run目录保护不等于完整自动看门狗。
- `results.py`：对新`manual_runs/manual_*`做结果验收、候选冻结、配对分析，复用原统计函数但不运行原硬编码main。
- `manual_runs/<tag>/`：只有真正执行新学习实验时才会出现对应日志/环境快照/冻结结果；本地图不因模板有路径就声称这些训练已经完成。

## 7. 事件加权拓展：已做诊断，没做加权训练

目录：`experiments/pi05_libero_adaptation/extensions/event_weighting_20261022/`。

| 文件/目录 | 身份与用途 |
|---|---|
| `PROTOCOL.md` | 先固定48示范配对、12身份×两模型重放与预算的预登记 |
| `MEASUREMENT_NOTES.md` | 测量定义、频率/精度/复现补充，不覆盖预登记 |
| `HANDOFF.md` | 完成状态与历史快照，旧PID不代表运行 |
| `DIAGNOSTIC_REPORT.md` | 最终诊断结论和未解决的跨进程数值复现边界 |
| `common.py` | 统一路径、SHA、持续命令事件窗口 |
| `prepare.py` | 元数据/episode事件扫描、固定样本和示范接触表 |
| `analyze_cpu.py` | 指关节响应校准与既有rollout统计 |
| `loss_probe.py` | 两模型原精度、共用noise/time配对前向，拆分运动/夹爪/padding loss |
| `summarize_loss.py` | loss汇总与探索性区间 |
| `gpu_diagnostics.py` | 有预算、显存准入的短前向/重放调度 |
| `replay.py` | 固定已暴露开发状态重放、另外保存成功和失败视频 |
| `prepare_rollout_review.py` | 同policy-step对齐接触表 |
| `render_failure_review.py` | 当前失败粗/细时间表 |
| `first_input_probe.py` | 原始输入/状态/noise保存与同进程重复/reset检查 |
| `fixed_input_reload.py` | 精确输入包跨新进程重载 |
| `run_reload_checks.py` | 原预算内两次重载跟进 |
| `finalize_diagnosis.py` | 诊断记录验收，不会启动模型训练 |
| `artifacts/` | 固定样本、统计、精确输入NPZ、标签、接触表、日志与状态 |
| `rollouts/{E1-s42,E2-s42}/` | 新增24次诊断重放的轨迹/视频/结果，独立于旧成绩 |

这些本地诊断工具不是官方训练器自带。`demo_visual_labels.csv`、`failure_visual_labels.csv`等标签尤其要本人核查；命令切换不等于真实接触。没有A/B/R加权训练权重，不应把目录存在写成拓展方法已验证提升。

## 8. MetaWorld VLA旁线：每个实验目录做什么

这些目录围绕openpi模型或接口探索，但**不是LIBERO主线的四组消融**。很多checkpoint是外部发布者训练后下载，本地只rollout；看见checkpoints目录不能宣称本地重新训练了该模型。

| 实验目录 | 地图定位 |
|---|---|
| `pi0_metaworld_baseline_20261015` | 早期π0接口/权重/仿真bootstrap、下载、rollout与汇总 |
| `pi0_metaworld_baseline_recheck_20261015` | 早期baseline的seed/接口/结果复查 |
| `pi05_metaworld_base_20261015` | 原pi05_base五任务基线尝试 |
| `pi05_metaworld_base_fixed_20261015` | 基础模型动作桥接修复、物理单位/控制器语义检查与分支重跑 |
| `pi05_metaworld_sft_20261016` | 外部π0.5 MetaWorld SFT权重下载与五任务推理 |
| `pi05_metaworld_sft_native_v3_20261016` | 对应原生动作接口/新版环境路线的冻结模型测评 |
| `pi0_metaworld_sft_hf_20261016` | HF发布π0 SFT模型的独立协议/结果 |
| `pi0_metaworld_native_v3_20261016` | π0与v3原生接口路线检查 |
| `pi0_metaworld_v3_rl_flownoise_20261016` | 外部RL FlowNoise模型旁线；与SFT、π0.5模型不能混称 |

这些目录顶层的本地`run.py/rollout.py/launch.py/adapter.py/check_interface.py/verify_results.py/download_checkpoint.py/watchdog.py/setup_runtime.py`逐个在源码索引中解释。不要因为`verify_results.py`名字一样就认为其验证规则相同。

常见子目录的身份：

| 子目录 | 意义 | 审查点 |
|---|---|---|
| `sources/`、`deps/` | 外部仓库/发布代码快照，可能含明确适配 | 看source_manifest/patch，不认定全部未改 |
| `checkpoints/` | 下载/准备的公开模型资产或运行时权重 | 读下载脚本、来源与模型类型，不认定本地训练 |
| `runtime/` | 隔离运行依赖/副本 | 与原openpi环境分开 |
| `manifests/`、`protocol.json` | 任务/seed/步数/模型/接口/成功口径登记 | 不把v2/v3或不同五任务集直接比较 |
| `diagnostics/`、`audit/` | 接口、动作、seed或oracle检查 | oracle环境检查结果不是模型成绩 |
| `smoke*`、`smoke_results/` | 小规模接入验证 | 不充当全任务成功率 |
| `results/`、`results_droid_units/`、`results_servo/` | 不同接口方案的独立结果 | 明确对应哪一套动作语义，不能挑最好混报 |
| `videos/`、`logs/` | 行为与错误记录 | 曾成功/最终成功、超时和异常分别审查 |

## 9. MetaWorld关联基线：不要和VLA主线归为同一模型

以下是当前工作区中另外19个关联实验目录。名字中的epoch/DDIM/horizon表明配置意图，**不等于已完整执行或达到某成功率**；以其HANDOFF/protocol/plan/audit/结果为准。

| 目录 | 地图定位 |
|---|---|
| `metaworld_official_baselines_20261019` | 公开DP/ACT/SimpleDP3基线接入、数据、外部源码、worker与监督 |
| `metaworld_official_baselines_50demo_20261019` | 50-demo设置的单独基线运行/证据 |
| `metaworld_community_baselines_50demo_20261020` | 社区实现路线与官方来源路线分开存档 |
| `metaworld_zero_success_search_20261019` | 零成功排查、配对控制状态与工况搜索 |
| `metaworld_nonzero_validation_20261020` | 非零结果复核案例 |
| `metaworld_nonzero_goal_20261020` | goal相关非零排查路线 |
| `metaworld_act_zero_grid_20261020` | ACT零成功配置搜索/进度分支 |
| `metaworld_act_short_interval_20261020` | ACT较短执行间隔对照 |
| `metaworld_act_shared_interval_20261020` | ACT共享间隔设置记录 |
| `metaworld_act_joint_seed_interval_20261020` | ACT种子/间隔联合设置与分任务进度 |
| `metaworld_dp_epoch500_20261020` | DP epoch500预算分支，保留旧baseline/审计 |
| `metaworld_dp_resume1000_20261020` | DP恢复到更长预算、数据/解析/恢复前置审计 |
| `metaworld_simpledp3_ddim5_20261020` | SimpleDP3采样步数路线及配对episode |
| `metaworld_simpledp3_epoch50_ddim5_all5_20261020` | epoch50、DDIM5、五任务配置/清单/审计 |
| `metaworld_simpledp3_epoch200_ddim5_select_20261020` | epoch200、DDIM5、选定任务/episode清单分支 |
| `metaworld_simpledp3_epoch300_ddim5_h8_20261020` | 名称标记epoch300/DDIM5/h8的分支；检查实际文件是否为空/完成 |
| `metaworld_simpledp3_epoch400_300_conditional_20261020` | 条件预算400/300的运行与审计 |
| `metaworld_simpledp3_epoch500_ddim10_20261020` | epoch500、DDIM10的目标/运行/配对episode |
| `metaworld_selected_results_20261021` | 被选中结果的整理/引用边界，不是全部搜索都成为独立确认 |

这些基线中`worker.py`可以调用外部官方训练入口，但`baseline_watchdog.py/prepare.py/runtime.py/rollout.py/*_adapter.py`仍是本地接入层。`sources/`内的算法不能算本人原创，`runs/`内的自动配置/权重也不能反过来算成官方仓库原始代码。

## 10. 早期工具和输出：保留历史，但不让它们冒充当前主线

| 路径 | 每类文件的作用 |
|---|---|
| `tools/add_pi05_lora_config.py` | 历史追加配置补丁；会写源码，不重复执行 |
| `tools/patch_libero_video_names.py` | 示例视频命名/保留方式补丁；会写源码 |
| `tools/auto_eval_after_train.sh` | 旧E0等待→服务→四套评测→解析；带旧run/GPU硬编码 |
| `tools/auto_eval_after_train.sh.v1.bak` | 旧版本备份，不是并行执行的第二个队列 |
| `tools/gpu_monitor.sh` | 设备采样，不是独占锁或抢占保护 |
| `tools/parse_train_log.py` | 从原训练文本抽loss/grad_norm/吞吐和摘要 |
| `tools/parse_eval_log.py` | 从早期eval文本抽episode/任务率/摘要 |
| `metrics/{steps,throughput}.csv`、`summary.json` | 早期全参日志解析产物 |
| `metrics_lora/{steps,throughput}.csv`、`summary.json` | 早期LoRA日志解析产物 |
| `eval_auto/{eval,server,auto_eval_tail}.log` | 旧服务/评测/监督输出 |
| `eval_auto/parsed/episodes.csv` | 旧episode解析，身份字段不如新manifest完整 |
| `eval_auto/parsed/task_rates.csv` | 旧逐task统计 |
| `eval_auto/parsed/eval_summary.json` | 旧套件汇总，不覆盖新协议成绩 |
| `eval_auto/videos/*.mp4` | 旧失败视频，文件名含task/episode和指令片段 |

例如旧E0 1917/2000与统一新协议1914/2000各自留档，不能为取更高成绩替换。旧结果的缺失身份不能靠现在猜一个seed补上。

## 11. 哪些是“运行官方代码就会有”，哪些不会

这里按**真实生成链路**分层，不能只看后缀：

| 触发动作 | 自动产生或复制 | 不会凭空产生 |
|---|---|---|
| Git/源码包迁移 | 工作树、Git元数据/子模块、原有README/测试 | 本地实验设计、审阅结论 |
| uv/pip/conda安装 | venv、site-packages、entrypoints、egg-info/dist-info、缓存 | 训练好的E1/E2模型 |
| 第一次加载base/tokenizer/数据 | 外部资产下载缓存/完成记录 | 本地微调成绩、本人采集轨迹 |
| `compute_norm_stats.py` | 指定assets路径中的统计文件 | 四组公平性的自动证明 |
| 官方来源`train.py` | 指定run的checkpoint/训练状态、W&B；stdout/stderr | 自定义manifests、候选冻结、成对测评、review_notes |
| `... > train_lora.log`或`tee` | 名为train_lora.log等的文本记录 | 官方固定命名规则（名字是shell选择） |
| 普通`serve_policy.py` | 服务进程与输出；显式record时才有相应记录 | 自动闭环成功率；客户端不运行就不测评 |
| 官方来源LIBERO示例 | 按示例/本地补丁保存视频、终端成功率输出 | 本项目严格身份化attempt/JSON/NPZ与冻结协议 |
| 本地manifest客户端/看门狗 | evaluations/attempts/heartbeat/结构化汇总 | 本人已经查看失败、接受风险、完成候选选择 |
| 本地参数/成本/统计工具 | 审计CSV/JSON、接触表、比较区间 | 原创算法、全任务机制因果结论 |
| 人工/Agent辅助审阅与写作 | 标签、决策、交接、报告、面试/操作说明 | 用户个人独立经历或实机验证事实 |

## 12. 本人在推进项目中应该审哪些位置

这不是泛泛检查列表，而是**对应文件的审查责任地图**。

| 阶段 | 必须打开的文件 | 应能确认 |
|---|---|---|
| 确定科学问题 | `protocol.md`、`decision_log.md`、`manifests/runs.json` | 比什么、只改什么、预算、预训练起点、哪些是配方混杂 |
| 确定数据 | dataset `meta/*.json*`、data_loader、libero_policy、assets统计 | 版本、动作语义、输入是否真的用state、帧采样/padding/归一化 |
| 决定更新范围 | config、pi0_config、gemma/lora、参数审计CSV | 冻结集合/训练集合、实际计数、LoRA不等于只训adapter |
| 训练前验收 | smoke JSON/日志、真实启动argv、环境快照 | 冻结叶确实不变、训练叶更新、能保存重载、GPU/解释器正确 |
| 训练中 | 当前attempt日志/资源CSV/状态JSON + 真实进程 | loss/grad是否持续异常、是数值还是资源问题、恢复是否留痕 |
| 测评前 | eval_manifest、服务metadata、checkpoint资产、成功规则 | 初始状态/seed、最终权重、步数/replan、环境成功vs严格边界 |
| 开发后 | episode JSON/动作NPZ、固定样本视频、failure_labels/review_notes | 哪些失败真实可见、uncertain和选样偏差、逐任务局部退化 |
| 确认前 | candidate_freeze及其hash、审阅记录、exposure记录 | 先选择后确认、不可事后改规则、旧任务暴露局限 |
| 交付 | confirmation_comparison、完整报告、成本summary与源记录 | 分子/分母闭合、置信区间含义、没做的实验/限制明确 |
| 简历/面试 | 训练与测评QA、72问、结果原始证据 | 不把Agent输出当本人已做，删证据不足的普适因果表述 |

自动检查和本人审查是互补的。`passed=true`只说明代码检查了其设定条件，不等于所有数据语义、实验公平性、解释与个人贡献都获认证。

## 13. 地图完整性与使用边界

- 全工作区遍历，按主线/旁线/共享输入、环境依赖、相邻项目分别建索引；准确计数见`scan_summary.json`，避免手工抄写后过期。
- 不跟随符号链接，链接本身单列，避免把外部目录重复算入；逻辑文件大小未做hardlink去重，不等于磁盘实际占用。
- 对同类重复产物逐路径登记，按生成器/目录语义解释；不是逐一解码所有Parquet、MP4、OCDBT或依赖包二进制。
- 来源依赖Git索引/已知创建记录/目录规则，置信依据逐行给出；没有在线拉取上游或验证第三方副本所有改动。无法确认的来源明确标待核实。
- 认证文件只做metadata盘点，不读取、导出或公开内容。运行日志/环境快照也可能含内部信息，分享时另做脱敏审查。
- 地图产物自身除生成器/查询脚本外不递归纳入扫描，以免每次生成自己扩大自己；这些清单/说明在本页逐项列出。
- 不因盘点结果判定某文件“可安全删除”。权重分片、锁、旧结果是否仍被引用需要单独分析；本次未删除/移动/重命名任何旧文件。

推荐阅读顺序：本页第2节掌握边界 → 第3节定位官方与本地变更 → 第6节掌握主实验 → 第4/5节连接数据和模型 → 第8/9节看旁线 → 第12节建立本人审查责任 → 用逐文件索引追任何未知路径。
