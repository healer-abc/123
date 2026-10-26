# openpi：按官方 README 跑通，再加入 E1/E2/E3

这版只使用官方的 `train.py`、`serve_policy.py`、`examples/libero/main.py`，不需要上一版的 `lab.sh`、`results.py`、manifest 服务或看门狗。

你只需抓住一句话：**实验配方在 `src/openpi/training/config.py` 里改，命令行只切换 config 名、实验名和 GPU；训练后按官方方式起服务、跑 LIBERO。**

文中是操作说明，本次没有修改真实模型配置或启动训练。当前仓库已存在 E1/E2/E3 配置，先读懂，**不要重复添加同名配置，也不要修改已冻结的旧实验来重写历史**。新写配方应在源码工作副本中进行。

## 1. 先按官方方式准备环境

依据：[openpi 主 README](https://github.com/Physical-Intelligence/openpi#installation)、[LIBERO README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)。以下选官方给出的非 Docker 路径，方便理解训练与仿真两个进程。

### 1.1 新机器才安装；本机已有环境跳过

先安装 uv（如未安装），并确保系统有 tmux：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/openpi-uv-install.sh
less /tmp/openpi-uv-install.sh
sh /tmp/openpi-uv-install.sh
export PATH="$HOME/.local/bin:$PATH"

# 需要 sudo 权限；没有权限请管理员安装
sudo apt-get update
sudo apt-get install -y git tmux ffmpeg libegl1 libgl1
```

在**新机器的新目录**获取官方代码，已有本机项目不要重新 clone 覆盖：

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

建立官方 LIBERO 客户端环境：

```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
deactivate
```

不要在现有 `.venv` 上重复创建或升级全部依赖。网络受限时，本机已将 lerobot/dlimp 改成本地源码依赖；这是安装适配，不是模型算法改动。官方全新 clone 不含下文自定义 E1/E2/E3，需要自行添加。

### 1.2 本机每个新终端都先做这几行

```bash
cd /home/zhsu/jyluo/projects/openpi
export OPENPI_DATA_HOME=/home/zhsu/jyluo/myopenpi/dataset/ckpt
export HF_LEROBOT_HOME=/home/zhsu/jyluo/dataset_root
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

后文训练/服务使用 `uv run --no-sync`：就是官方的 `uv run` 入口，额外禁止自动同步依赖，以免长任务意外联网改环境。已安装的环境必须正确；新环境安装完成后同样可用。

验收：

```bash
.venv/bin/python --version
examples/libero/.venv/bin/python --version
nvidia-smi
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c 'import jax; print(jax.devices())'
```

训练/服务应为 Python3.11，仿真客户端为 Python3.8。不要因终端写着 `(base)` 就以为实际使用的是 base Python。

## 2. 数据与归一化：沿用官方 LIBERO 示例

官方配置已经指定转换好的 `physical-intelligence/libero`，**不需要重新执行数据转换脚本**。本机已经下载好：

```bash
.venv/bin/python -m json.tool \
  /home/zhsu/jyluo/dataset_root/physical-intelligence/libero/meta/info.json
```

预期40个任务、1693条示范、273465帧。新机器联网使用时，配置会通过 LeRobot 数据加载流程获取所需数据；网络受限时提前迁移完整数据和缓存。

官方要求训练前计算归一化统计；本机已有则直接复用：

```bash
ls assets/pi05_libero/physical-intelligence/libero/norm_stats.json

# 只有该文件不存在时才执行，不要覆盖旧统计
if [ ! -f assets/pi05_libero/physical-intelligence/libero/norm_stats.json ]; then
  uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_libero
fi
```

基础权重使用 `pi05_base`，不是已经针对 LIBERO 微调的 `pi05_libero` 专家权重。配置会加载 base；本机缓存目录由 `OPENPI_DATA_HOME` 指定。首次缺少权重/tokenizer时需要下载，不是程序卡死。

## 3. 自己的实验到底改哪个文件、怎么改？

**只涉及这轮更新范围实验时，改 `src/openpi/training/config.py` 的 `_CONFIGS` 列表。** 不需要改 `scripts/train.py`、模型 loss、`serve_policy.py` 或 LIBERO 主评测脚本。

先定位：

```bash
rg -n 'name="pi05_libero' src/openpi/training/config.py
```

打开文件找到 `TrainConfig(name="pi05_libero", ...)`。下面按“官方 E0 → 自己的 E1 → 复制 E1 得到 E2 → 复制 E0 得到 E3”理解。当前文件里这四组已经写好，下列代码用于解释/在干净副本中复现，不是让你再粘贴一次。

### 3.1 E0：官方全量微调，不改模型代码

原配置关键内容：

```python
name="pi05_libero",
model=pi0_config.Pi0Config(
    pi05=True, action_horizon=10, discrete_state_input=False,
),
```

沿用其原学习率和 EMA。官方配置默认 batch256，本机统一在命令行覆盖为32；不必为了 batch 改源码。

### 3.2 E1：在 `_CONFIGS` 中增加一份 LoRA 配置

如果从官方干净副本开始，把下面完整 `TrainConfig(...)` 放进 `_CONFIGS = [...]` 列表中，与 E0 并列：

```python
TrainConfig(
    name="pi05_libero_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        assets=AssetsConfig(assets_dir="./assets/pi05_libero"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2e-4,
        decay_steps=30_000,
        decay_lr=2e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    num_train_steps=30_000,
),
```

这几处改动的含义：

- `paligemma_variant` 和 `action_expert_variant`：语言/动作两个分支使用已有 LoRA 变体，不是重写网络。
- `freeze_filter`：规定哪些权重不更新，必须与上面的 LoRA 变体匹配。
- `assets`：新 config 名默认会找新 assets 目录；这里显式复用 E0 的归一化统计。
- `lr_schedule`、`ema_decay`：本项目 E1 的实际配方与 E0 不同，所以 E0/E1 不只是 LoRA 一个变量。

**E1 仍训练完整视觉编码器，不是“只训练 LoRA”。** 具体可训练范围不要靠名字猜。

### 3.3 E2：复制 E1，只改名字和 `freeze_filter`

先复制 E1 的整个 `TrainConfig`，名字改成：

```python
name="pi05_libero_lora_frozen_vision",
```

其他内容保持 E1 不变，仅把原 `freeze_filter=...` 替换成：

```python
freeze_filter=nnx.Any(
    pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    nnx_utils.PathRegex("PaliGemma/img/.*"),
),
```

意思是：**原来冻结的继续冻结，再冻结视觉 `PaliGemma/img/**`**。因此 E1/E2 才是视觉是否训练的对照。若上游副本缺少导入，在文件顶部加入 `import openpi.shared.nnx_utils as nnx_utils`；`flax.nnx as nnx` 原文件通常已有。

### 3.4 E3：复制 E0，只训练动作侧

复制 E0 整个 `TrainConfig`，保留其非 LoRA model、LR schedule、base loader、optimizer 和30k预算。改动如下：

```python
name="pi05_libero_action_only",
```

在原 `data=LeRobotLiberoDataConfig(...)` 内加上同样的资产路径：

```python
assets=AssetsConfig(assets_dir="./assets/pi05_libero"),
```

关闭 EMA，并增加冻结过滤器：

```python
ema_decay=None,
freeze_filter=nnx.Not(nnx.Any(
    nnx_utils.PathRegex("PaliGemma/llm/.*_1/.*"),
    nnx_utils.PathRegex(
        "(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/.*"
    ),
)),
```

`freeze_filter` 的含义是“哪些冻结”，所以这里写“**除了动作分支与投影以外都冻结**”。`*_1` 是本地模型的动作分支参数命名；这是按实现路径选张量，不是适用于所有模型的通用命名。

E3 本机原配置将 base loader 写成了缓存的本地绝对路径；干净副本保留 E0 的 `gs://.../pi05_base/params` 也加载同一基座，只是取文件的方式不同。不要把它改成 E1/E2 checkpoint。

### 改完先检查是否注册成功

```bash
JAX_PLATFORMS=cpu uv run --no-sync scripts/train.py pi05_libero --help
JAX_PLATFORMS=cpu uv run --no-sync scripts/train.py pi05_libero_lora --help
JAX_PLATFORMS=cpu uv run --no-sync scripts/train.py pi05_libero_lora_frozen_vision --help
JAX_PLATFORMS=cpu uv run --no-sync scripts/train.py pi05_libero_action_only --help
```

报重复 name 就检查是否粘贴了第二份同名配置；报语法错误先检查逗号与括号。不要等到长训练时才发现配置根本没注册。

## 4. 用官方训练命令跑四组，tmux 只负责保活

### 4.1 先理解命令的四个位置

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/train.py pi05_libero_lora \
  --exp-name=readme_e1_20261023_v1 --batch-size=32 --num-train-steps=30000
```

`CUDA_VISIBLE_DEVICES`选物理GPU；`pi05_libero_lora`选代码配方；`exp-name`选本次结果目录；`batch/steps`覆盖运行预算。换 exp-name 不会变成新算法，换 config 才改变相应配方。

### 4.2 四组先各跑10步（顺序跑）

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/train.py pi05_libero \
  --exp-name=readme_e0_smoke_20261023_v1 --batch-size=32 --num-train-steps=10 --log-interval=1
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/train.py pi05_libero_lora \
  --exp-name=readme_e1_smoke_20261023_v1 --batch-size=32 --num-train-steps=10 --log-interval=1
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/train.py pi05_libero_lora_frozen_vision \
  --exp-name=readme_e2_smoke_20261023_v1 --batch-size=32 --num-train-steps=10 --log-interval=1
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/train.py pi05_libero_action_only \
  --exp-name=readme_e3_smoke_20261023_v1 --batch-size=32 --num-train-steps=10 --log-interval=1
```

检查 loss 有限、无 traceback、最后保存到各自 `.../<smoke名>/9/`。这只证明训练路径能跑通，不自动证明冻结叶正确或模型有效；严格验收仍应比较参数差分与实际 rollout。

### 4.3 正式训练：四组都有命令，最多按两卡分批执行

从普通终端开一个 tmux：

```bash
tmux new -s openpi-train
```

在 tmux 里先执行1.2节的目录/环境变量，再准备日志：

```bash
mkdir -p logs/readme_20261023_v1
set -o pipefail
nvidia-smi
```

**第一轮：E0用GPU1，E1用GPU0。以下两段分别放在两个tmux窗口运行。**

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/train.py pi05_libero \
  --exp-name=readme_e0_20261023_v1 --batch-size=32 --num-train-steps=30000 \
  2>&1 | tee logs/readme_20261023_v1/e0_train.log
```

Ctrl+b，c 新开窗口，重复1.2节与 `set -o pipefail` 后运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/train.py pi05_libero_lora \
  --exp-name=readme_e1_20261023_v1 --batch-size=32 --num-train-steps=30000 \
  2>&1 | tee logs/readme_20261023_v1/e1_train.log
```

两组退出、资源释放后，**第二轮：E2用GPU1，E3用GPU0**，同样分窗口：

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/train.py pi05_libero_lora_frozen_vision \
  --exp-name=readme_e2_20261023_v1 --batch-size=32 --num-train-steps=30000 \
  2>&1 | tee logs/readme_20261023_v1/e2_train.log
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/train.py pi05_libero_action_only \
  --exp-name=readme_e3_20261023_v1 --batch-size=32 --num-train-steps=30000 \
  2>&1 | tee logs/readme_20261023_v1/e3_train.log
```

共享卡有别人的任务就等待，tmux不会为你预约显存。E0–E3均从base新训，**不从10步smoke接着训**。实验名已存在就换v2，不使用官方示例中的 `--overwrite` 覆盖结果。

### 4.4 只记住这几个tmux操作

| 操作 | 命令/按键 |
|---|---|
| 离开但继续运行 | Ctrl+b，d |
| 重新连接 | `tmux attach -t openpi-train` |
| 看会话 | `tmux ls` |
| 新窗口 | Ctrl+b，c |
| 切窗口 | Ctrl+b，n / p |
| 停当前任务 | Ctrl+c，不是detach |

查看日志与显存：

```bash
tail -f logs/readme_20261023_v1/e2_train.log
rg 'Step [0-9]+:' logs/readme_20261023_v1/e2_train.log | tail -5
watch -n 5 nvidia-smi
```

中断后恢复E2，保持相同config、exp-name、预算并加 `--resume`，用新日志文件：

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/train.py pi05_libero_lora_frozen_vision \
  --exp-name=readme_e2_20261023_v1 --batch-size=32 --num-train-steps=30000 --resume \
  2>&1 | tee logs/readme_20261023_v1/e2_resume_01.log
```

其他三组同理，换各自config/exp-name；不要恢复时改LR或数据。完成30k后最终目录是 `29999`，不是 `30000`；必须等保存结束，确认 `_CHECKPOINT_METADATA` 存在。

## 5. 按官方方式测评：服务端一个终端，客户端另一个终端

### 5.1 服务端：四组只切换config和checkpoint

创建服务会话：

```bash
tmux new -s openpi-server
```

执行1.2节环境变量，然后按要评的组选择下面**一条**服务命令。同一端口只开一个服务，换组前Ctrl+c停旧服务。

```bash
# E0
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py --port=8000 \
  policy:checkpoint --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/readme_e0_20261023_v1/29999

# E1
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py --port=8000 \
  policy:checkpoint --policy.config=pi05_libero_lora \
  --policy.dir=checkpoints/pi05_libero_lora/readme_e1_20261023_v1/29999

# E2
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py --port=8000 \
  policy:checkpoint --policy.config=pi05_libero_lora_frozen_vision \
  --policy.dir=checkpoints/pi05_libero_lora_frozen_vision/readme_e2_20261023_v1/29999

# E3
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py --port=8000 \
  policy:checkpoint --policy.config=pi05_libero_action_only \
  --policy.dir=checkpoints/pi05_libero_action_only/readme_e3_20261023_v1/29999
```

不要只用 `--env LIBERO`：不指定checkpoint会选择默认已发布专家模型，未必是你刚训练的模型。

### 5.2 客户端：先少量跑通，再四套完整测

新建会话：

```bash
tmux new -s openpi-eval
```

使用官方Python3.8客户端环境，不用训练Python：

```bash
cd /home/zhsu/jyluo/projects/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH="$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p logs/readme_20261023_v1
set -o pipefail
```

先每任务1次，验证端到端；此处 Spatial10个任务，所以共10次，不是整套只rollout一次：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python examples/libero/main.py \
  --args.host=127.0.0.1 --args.port=8000 \
  --args.task-suite-name=libero_spatial --args.num-trials-per-task=1 \
  --args.video-out-path=data/libero/readme_20261023_v1/smoke_e0
```

这里假定服务端当前是E0；测其他组时把视频目录后缀改成对应组名。看到动作、成功/失败日志和视频后，再做完整50次/任务。

正式客户端命令只需要一份，各组完全相同；**每次先确认服务端确实换到了该组**：

```bash
# 第一次填e0；换服务测下一组时，依次改为e1、e2、e3
GROUP=e0

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python examples/libero/main.py \
    --args.host=127.0.0.1 --args.port=8000 \
    --args.task-suite-name="$SUITE" --args.num-trials-per-task=50 \
    --args.seed=7 --args.replan-steps=5 \
    --args.video-out-path="data/libero/readme_20261023_v1/$GROUP/$SUITE" \
    2>&1 | tee "logs/readme_20261023_v1/${GROUP}_${SUITE}.log"
  rc=$?
  if [ "$rc" -ne 0 ]; then break; fi
done
```

操作顺序：**E0服务→GROUP=e0完整测→停服务；E1服务→GROUP=e1完整测→停服务；E2、E3同理。** 若只做历史方案的筛选，E3开发不通过就不应为了凑表去做正式确认；本段全四套命令是普通完整基准路径，不是原项目开发/确认划分。

每组四套各500次，共2000次；视频和日志各自分目录。不要重复使用相同目录覆盖旧评测；重跑改v2。官方客户端日志里可能捕获异常后继续，因此还要检查日志，不能只看退出码。

```bash
rg 'Total success rate|Total episodes|Caught exception' logs/readme_20261023_v1/*.log
find data/libero/readme_20261023_v1 -name '*.mp4' | head
```

服务端的config/权重日志是防止“测错模型”的重要证据；`GROUP`只是客户端文件名，不会自动切换服务端模型。

## 6. 你应该如何复述这套流程

“我从官方LIBERO微调配置出发，在 `training/config.py` 中增加实验配置：E1为语言/动作LoRA并保留视觉训练，E2额外冻结视觉，E3只开放动作分支。训练入口完全沿用官方 `scripts/train.py`，通过config名选择配方，通过exp-name隔离输出。每组先短跑确认链路，再正式训练；训练完成后用官方 `serve_policy.py` 加载对应config和checkpoint，另一个Python3.8进程运行LIBERO客户端。tmux负责长任务保活，日志记录用于排查与核验。”

最后一个边界：**这版是贴近官方的运行/完整基准流程，不等于严格重建原项目6400条固定清单、显式策略噪声与冻结确认协议。** 官方客户端固定seed也不等于跨进程逐轨迹完全确定。先把这一层学会；只有需要做正式成对研究时，再加原项目的身份记录、参数差分、候选冻结和统计审计，不必一开始就背完全部工程脚本。
