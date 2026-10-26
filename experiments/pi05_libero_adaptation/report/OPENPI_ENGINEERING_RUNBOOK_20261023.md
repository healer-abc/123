# openpi 完整工程实操手册：环境 → E0/E1/E2/E3 → 测评 → 交付

日期：2026-10-23；适用：本机 `/home/zhsu/jyluo` 的 **LIBERO π0.5 E0–E3 主线**。不是 Fetch-Push RL 或 MetaWorld 项目的训练手册。

这是可按阶段操作的教程，不是一键全部执行的脚本。阅读本文件不会训练；下面的训练、下载、评测命令要由你自行决定何时执行。当前没有替你安装、训练或占用 GPU。已有 E0–E3 结果不必重跑；新一轮只用于学习/验证时，先完成 CPU 检查和 smoke，30k 正式训练耗时是数十小时。

## 先建立正确的工程模型

你负责六件事：数据是否正确、哪些参数更新、训练是否健康、权重是否真的保存、闭环如何计分、结论能否被证据支持。能背命令不是终点；每一步都应能回答“输入是什么、产出在哪里、怎样算通过、失败保留什么”。

目录分工：

```text
/home/zhsu/jyluo/
├── projects/openpi/                   模型源码、训练入口、两套 Python 环境
│   ├── .venv/                         Python 3.11：训练/模型服务
│   ├── examples/libero/.venv/          Python 3.8：仿真/评测客户端
│   ├── assets/pi05_libero/             原有训练归一化统计
│   └── checkpoints/<config>/<run>/     模型及恢复状态
├── dataset_root/physical-intelligence/libero/   训练数据
├── myopenpi/dataset/ckpt/              base 权重、tokenizer 等下载缓存
└── experiments/pi05_libero_adaptation/
    ├── manifests/eval_manifest.csv     固定测评身份清单，不能覆盖
    ├── evaluations/<run>/<phase>/      逐 episode 结果/动作/失败视频
    ├── manual_workflow/                本教程辅助命令，可打开阅读
    └── manual_runs/<tag>/              新一轮日志、环境快照、冻结与分析
```

官方入口：`scripts/train.py` / `scripts/serve_policy.py` / `examples/libero/main.py`。本项目扩展：四组配置、参数审计、固定 seed 服务、manifest 评测、冻结与配对统计。不要把自定义扩展说成官方默认功能。

### 四组实验，而不只是跑一个模型

| 组 | config | 更新范围 | 30k 配方 |
|---|---|---|---|
| E0 | `pi05_libero` | 全量，EMA 0.999 | batch32；warmup10k，LR5e-5 |
| E1 | `pi05_libero_lora` | 完整视觉 + 两分支 LoRA + 投影 | batch32；warmup1k，2e-4→2e-5 |
| E2 | `pi05_libero_lora_frozen_vision` | E1 再冻结视觉 | 与 E1 相同 |
| E3 | `pi05_libero_action_only` | 非 LoRA，只训练完整动作分支/投影 | batch32；warmup10k，LR5e-5，无 EMA |

四组均从同一个 `pi05_base` 起步，**不是 E0→E1→E2→E3 逐组续训**。E1/E2 是相对干净的视觉冻结消融；E0/E1、E2/E3 还有 LR/EMA/参数化等混杂。

## 1. 环境：先记录，再决定是否重建

### 1.1 机器与系统工具

```bash
cd /home/zhsu/jyluo
hostname
uname -a
nvidia-smi
lscpu
free -h
df -h /home/zhsu/jyluo
command -v uv
command -v tmux
command -v ffmpeg
```

`nvidia-smi` 中的 CUDA Version 是驱动支持能力，不等于你必须安装同版本系统 CUDA Toolkit。不要为了本教程重装共享服务器驱动。

如系统工具缺失，下面是管理员/有 sudo 权限时的**可选安装命令**，不是本次已经执行过的动作：

```bash
sudo apt-get update
sudo apt-get install -y tmux git curl ffmpeg libegl1 libgl1 libglib2.0-0
```

没有 sudo 权限就请管理员补工具，不要拿失败的 apt 命令继续往下猜。uv 如缺失可按 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/)安装；例如：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/openpi-uv-install.sh
less /tmp/openpi-uv-install.sh
sh /tmp/openpi-uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

下载后先看脚本再运行。这个临时文件只有安装器；不是写入/删除项目源码。

### 1.2 本机现有环境：直接用，不要重装

先定义运行命令，**每个新终端/tmux 窗口都要执行**：

```bash
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
```

该文件只设置变量/函数，不开始实验。最重要的变量：

```bash
printf '%s\n' "$LAB_TRAIN_PY" "$LAB_CLIENT_PY" "$OPENPI_DATA_HOME" "$HF_LEROBOT_HOME"
"$LAB_TRAIN_PY" --version
"$LAB_CLIENT_PY" --version
```

预期训练 Python 3.11，客户端 3.8。不要只看终端前的 `(base)`，**绝对路径解释器才决定运行环境**。

本次实查版本：训练 JAX/JAXlib 0.5.3、Flax 0.10.2、Orbax 0.11.13、Torch 2.7.1、NumPy 1.26.4；客户端 Torch 1.11.0+cu113、NumPy 1.22.4、robosuite 1.4.1、MuJoCo **3.2.3**。这些是 10 月 23 日查询值，不用旧交接里可能不同的版本号冒充当前环境。

保存环境快照（只写新 `manual_runs`，不修改安装）：

```bash
mkdir -p "$LAB_ROOT/environment"
uv pip freeze --python "$LAB_TRAIN_PY" > "$LAB_ROOT/environment/train.freeze.txt"
uv pip freeze --python "$LAB_CLIENT_PY" > "$LAB_ROOT/environment/client.freeze.txt"
nvidia-smi > "$LAB_ROOT/environment/nvidia-smi.txt"
cp -n "$LAB_OPENPI/pyproject.toml" "$LAB_ROOT/environment/pyproject.toml"
cp -n "$LAB_OPENPI/uv.lock" "$LAB_ROOT/environment/uv.lock"
```

freeze 文件中的 editable 本地依赖路径也要保留源码，不能以为只有 pip 包名就足够在另一台机器复原。

### 1.3 只有在新环境确实需要时才执行的重建流程

**不要对现有 `.venv` 做 uv sync / pip upgrade。新建目录验证，成功后再选择新解释器。**

先理解代码来源：官方仓库没有本项目完整的 E1/E2/E3 与固定测评工具。迁移工程应带上这份本地源码和实验脚本；只 clone 官方 main 不能得到完全相同的实验。当前项目的 `pyproject.toml` 已将 lerobot/dlimp 指向 `projects/offline-deps/`。

```bash
cd /home/zhsu/jyluo/projects/openpi
ls ../offline-deps/lerobot/pyproject.toml
ls ../offline-deps/dlimp/pyproject.toml
ls third_party/libero/libero/libero
rg -n 'offline-deps|jax|flax|torch' pyproject.toml
```

新机器无法访问 GitHub 时，在可联网机器准备源码包，再复制到**新目标目录**：

```bash
# 在能联网的机器：下面是官方起始代码，不含本项目自定义修改
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git openpi-upstream
tar -czf openpi-upstream.tar.gz openpi-upstream
# 传输目标主机/目录按你的实际情况填写
# scp openpi-upstream.tar.gz user@host:/path/to/new-workspace/
```

本机已经有 patched 源码，无需执行这段 clone/tar，也不要把上游 tar 解压覆盖本机项目。旧 `README_部署说明.md` 含历史清理命令，**本教程不需要其中任何 rm -rf 或删除锁文件的步骤**。

#### 重建训练环境（Python 3.11）

```bash
cd /home/zhsu/jyluo/projects/openpi
uv python install 3.11 3.8
export UV_HTTP_TIMEOUT=300

# 先确认该新目录不存在；若存在，换名字，不要删除
test ! -e .venv-rebuild-20261023
uv venv --python 3.11 .venv-rebuild-20261023

# 按现有 uv.lock 安装，不重新生成 lock；失败就检查路径/索引，不删除 lock
UV_PROJECT_ENVIRONMENT="$PWD/.venv-rebuild-20261023" \
  uv sync --frozen --no-dev

.venv-rebuild-20261023/bin/python -c 'import openpi, jax; print(jax.__version__, openpi.__file__)'
```

`test` 返回非零时**不要继续创建**。uv 遇到已有目录可能提示重建，不能顺手确认覆盖。需要镜像可使用可信的 PyPI 镜像；CUDA 专用 wheel 仍可能需要其官方索引，不能假定一个镜像提供全部包。CUDA JAX wheel 已在依赖中指定，不要额外安装一个不匹配的 CPU 版 JAX 覆盖它。

如要更严格重建本机当前已安装版本，在已有 freeze 快照的前提下，可在另一全新 venv 使用 `uv pip install --python <新python> -r <train.freeze.txt>`；这条路径与 uv.lock 重建是**两种选择，不是必须依次覆盖安装**。两者都要核对实际版本，网络/平台/本地源码缺失时不能保证一次成功。

#### 重建仿真客户端（Python 3.8）

```bash
cd /home/zhsu/jyluo/projects/openpi
test ! -e examples/libero/.venv-rebuild-20261023
uv venv --python 3.8 examples/libero/.venv-rebuild-20261023

uv pip sync --python examples/libero/.venv-rebuild-20261023/bin/python \
  examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

uv pip install --python examples/libero/.venv-rebuild-20261023/bin/python \
  -e packages/openpi-client -e third_party/libero
```

这里使用两个可信来源来满足旧 CUDA Torch wheel，`unsafe-best-match` 是显式的跨索引解析选择，不应随意添加未知索引。该流程依据 [openpi LIBERO 官方依赖说明](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)；PyTorch 3.11 训练环境不能直接替代旧仿真环境。

若选新建环境，在每个后续终端**先 override 再 source**：

```bash
export LAB_TRAIN_PY=/home/zhsu/jyluo/projects/openpi/.venv-rebuild-20261023/bin/python
export LAB_CLIENT_PY=/home/zhsu/jyluo/projects/openpi/examples/libero/.venv-rebuild-20261023/bin/python
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
```

只学现有实验则继续默认的现成解释器。`uv run` 可能检查/同步依赖，长任务直接用明确 Python 路径，避免断网时意外重新解析。参见 [uv 项目运行机制](https://docs.astral.sh/uv/guides/projects/)与[显式指定环境](https://docs.astral.sh/uv/pip/environments/)。

### 1.4 环境验收，不只是 import 一次

CPU 导入和仿真资源路径检查：

```bash
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" -c 'import openpi, flax, jax; print(openpi.__file__); print(jax.__version__)'
"$LAB_CLIENT_PY" - <<'PY'
import mujoco, robosuite
from pathlib import Path
from libero.libero import get_libero_path
print('mujoco', mujoco.__version__)
for key in ('bddl_files', 'init_states'):
    path = Path(get_libero_path(key))
    print(key, path)
    assert path.is_dir(), path
PY
```

LIBERO 首次 import 如要求创建配置，确认资源目录属于这份 `third_party/libero`，不要盲用另一份旧 clone 的绝对路径。

GPU 连接检查（会初始化一个 GPU 上下文，不是训练；先看卡是否可用）：

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 "$LAB_TRAIN_PY" -c 'import jax; print(jax.devices())'
```

无显示器仿真使用 `MUJOCO_GL=egl`；`PYTHONPATH` 已包含 `third_party/libero`。缺少路径会出现 `ModuleNotFoundError: libero`，这不等于模型坏了。可先运行客户端 `--help`，不加载模型：

```bash
"$LAB_CLIENT_PY" "$LAB_OPENPI/examples/libero/main.py" --help
"$LAB_CLIENT_PY" "$LAB_PROJECT/eval_manifest.py" --help
```

## 2. tmux：让长任务脱离 SSH 生命周期

tmux 是持久终端，**不是显存调度器，也不是故障看门狗**。

```bash
# 在普通 SSH 终端创建训练会话
tmux new-session -s pi05-e0

# 以后从普通 SSH 终端查看/重新连接
tmux ls
tmux attach-session -t pi05-e0
```

会话内常用按键（先按 Ctrl+b，松开，再按后一个键）：

| 按键 | 作用 |
|---|---|
| Ctrl+b，d | detach，只离开终端，不停止训练 |
| Ctrl+b，c | 新窗口；新窗口重新 source `lab.sh` |
| Ctrl+b，n / p | 下一/上一窗口 |
| Ctrl+b，, | 重命名窗口 |
| Ctrl+b，% / " | 左右 / 上下分屏 |
| Ctrl+b，[ | 进入滚动模式，q 退出 |
| Ctrl+c | 中断当前前台任务；不是 detach |

不要在训练中用 `exit` 或 `tmux kill-session` 代替 detach。进程被停不是“会话隐藏”。

建议四组训练分别用 `pi05-e0`、`pi05-e1`、`pi05-e2`、`pi05-e3`；模型服务用 `pi05-server`，客户端用 `pi05-eval`。两张卡也不能同时无条件起四个训练。

## 3. 新一轮命名与数据/权重准备

### 3.1 每个窗口必须一致的 tag

默认新一轮 tag 是 `manual_20261023_01`。重做第二轮先换 `02`，不要复用旧日志和旧评测输出。

```bash
export LAB_TAG=manual_20261023_01
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
mkdir -p "$LAB_ROOT/environment"
```

helper 默认产物在 `manual_runs/<tag>`，权重在原 checkpoint 根目录下的**新实验名**；不使用旧 `E1-s42` / `pi05_libero_lora_gpu0` 等名称。

### 3.2 数据检查

```bash
"$LAB_TRAIN_PY" -m json.tool "$HF_LEROBOT_HOME/physical-intelligence/libero/meta/info.json"
wc -l "$HF_LEROBOT_HOME/physical-intelligence/libero/meta/episodes.jsonl"
du -sh "$HF_LEROBOT_HOME/physical-intelligence/libero"
```

预期：40 tasks、1693 episodes、273465 frames、10Hz。1693 行 episode 记录不是 1693 个独立任务。

只有缺少数据时才下载到一个新的目标缓存/目录，不覆盖当前数据：

```bash
HF_HUB_OFFLINE=0 "$LAB_TRAIN_PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='physical-intelligence/libero', repo_type='dataset',
                  local_dir='/home/zhsu/jyluo/dataset_root_new/physical-intelligence/libero')
PY
```

这会联网下载真实数据；要使用新路径，之后显式 `export HF_LEROBOT_HOME=/home/zhsu/jyluo/dataset_root_new`。再次 source helper 会恢复其默认路径，注意覆盖顺序。

### 3.3 基座、tokenizer、归一化

```bash
ls "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"
ls "$LAB_OPENPI/assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
sha256sum "$LAB_OPENPI/assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
```

本项目原统计 SHA-256 为 `c0ee3c1a6ca5f1414d31180af3bc333dca1d78cf912f341943191b393fe68707`。

base 或 tokenizer 缺失时先联网准备，训练期间不要临时排网络问题：

```bash
HF_HUB_OFFLINE=0 JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" - <<'PY'
from openpi.shared import download
from openpi.models.tokenizer import PaligemmaTokenizer
print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))
tokenizer = PaligemmaTokenizer(max_len=200)
print('base and tokenizer ready')
PY
```

注意 `HF_HUB_OFFLINE` 只约束 HF，不约束 GCS；真正离线还要求上述资源本地完整。E3 当前 config 使用这个 base 的本地绝对路径，迁移机器时需要在**新源码工作副本**调整它。

仅当本配置统计不存在时才计算（会扫描数据，耗时，不要每组重算）：

```bash
cd "$LAB_OPENPI"
if [ ! -f assets/pi05_libero/physical-intelligence/libero/norm_stats.json ]; then
  JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" scripts/compute_norm_stats.py --config-name=pi05_libero
fi
```

E1/E2/E3 已显式复用 E0 的 assets。换新数据时，不能继续套用旧统计：先建新 config/独立 assets，重新统计并登记版本，不能把原统计文件覆盖后还称相同协议。

## 4. 看配置和做静态参数审计：不占 GPU

```bash
cd "$LAB_OPENPI"
rg -n 'name="pi05_libero|warmup_steps|peak_lr|ema_decay' src/openpi/training/config.py
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" scripts/train.py pi05_libero --help
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" scripts/train.py pi05_libero_lora --help
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" scripts/train.py pi05_libero_lora_frozen_vision --help
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" scripts/train.py pi05_libero_action_only --help

JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" "$LAB_PROJECT/audit_parameters.py" \
  pi05_libero pi05_libero_lora pi05_libero_lora_frozen_vision pi05_libero_action_only \
  --output-dir "$LAB_ROOT/parameter_audits"
```

预期可训练参数：E0 3,353,433,872；E1 466,957,072；E2 52,153,376；E3 430,098,464。静态正确不代表真的更新正确，所以还要 smoke。E0 config 默认 batch256，下面命令统一显式覆盖32。

## 5. 四组都做独立 10 步 smoke

**以下每段在 GPU 有资源时执行，最初建议顺序运行，别整段同时后台丢出去。** `lab_use` 选实验、物理 GPU 和服务端口；它不启动计算。`lab_smoke` 才开始训练。

```bash
lab_use E0 0 8120
lab_smoke

lab_use E1 0 8121
lab_smoke

lab_use E2 0 8122
lab_smoke

lab_use E3 0 8123
lab_smoke
```

每个 smoke 使用独立 `<run>_smoke/9`，不是正式训练起点。首次编译通常比后续 step 慢，不能用第一步估计整轮耗时。

`lab_smoke` 展开的核心命令就是：

```bash
cd "$LAB_OPENPI"
CUDA_VISIBLE_DEVICES="$LAB_GPU" "$LAB_TRAIN_PY" -u scripts/train.py "$LAB_CONFIG" \
  --exp-name="${LAB_RUN}_smoke" --batch-size=32 --seed=42 \
  --num-train-steps=10 --log-interval=1
```

封装额外做新目录检查、`pipefail`、日志保存及 checkpoint metadata 检查；源文件 [lab.sh](../manual_workflow/lab.sh) 可直接打开逐行理解。它没有自动抢 GPU、没有隐含更改模型配方。

### E2 / E3 的实际冻结与重载验收

现有两个 verifier 专门针对原配方和 seed42，不适合改成任意模型后照搬。CPU 运行可避免额外占 GPU，但大模型会用较多主机内存/时间：

```bash
lab_use E2 0 8122
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" "$LAB_PROJECT/verify_e2_smoke.py" \
  --checkpoint "$LAB_OPENPI/checkpoints/$LAB_CONFIG/${LAB_RUN}_smoke/9" \
  --output "$LAB_LOG/smoke_verification.json"

lab_use E3 0 8123
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" "$LAB_PROJECT/verify_e3_smoke.py" \
  --checkpoint "$LAB_OPENPI/checkpoints/$LAB_CONFIG/${LAB_RUN}_smoke/9" \
  --output "$LAB_LOG/smoke_verification.json"
```

E2 应看到冻结视觉/语言叶不变、LoRA/投影叶更新；E3 应看到冻结叶不变、动作侧更新，并做重载推理检查。若断言失败，先保留日志定位，不修改容差让它“绿灯通过”。E0/E1 本段只有 smoke 完成与静态范围验收；没有偷换为已完成逐叶更新差分，严格新实验还应补对应差分验收。

## 6. 四组正式 30k 训练：两卡分两轮

### 6.1 先查资源，不把 CUDA_VISIBLE_DEVICES 当资源锁

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

同一张卡可能已有别人的进程。E1/E3 短测约需33GiB、E2约25GiB，再留至少数GiB余量；E0 没有同口径峰值，先由 smoke 确认。JAX 关闭预分配不等于显存零开销，也不能靠一张空闲截图保证后续不被抢占。

### 6.2 第一轮：E0 / GPU0、E1 / GPU1

普通 SSH 终端启动第一会话：

```bash
tmux new-session -s pi05-e0
```

在会话里：

```bash
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E0 0 8120
lab_train
```

Ctrl+b，d 离开后，从普通终端再建：

```bash
tmux new-session -s pi05-e1
```

在会话里：

```bash
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E1 1 8121
lab_train
```

GPU1 不空闲就等待，不因教程安排而强行启动。可以只单卡顺序跑完全部四组。

### 6.3 第二轮：释放资源后 E2 / GPU0、E3 / GPU1

```bash
# 普通终端
tmux new-session -s pi05-e2
```

```bash
# e2 会话
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E2 0 8122
lab_train
```

```bash
# 普通终端
tmux new-session -s pi05-e3
```

```bash
# e3 会话
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E3 1 8123
lab_train
```

四组对应的原始训练核心指令（**与上面封装二选一，不要重复跑**）：

```bash
cd /home/zhsu/jyluo/projects/openpi
CUDA_VISIBLE_DEVICES=0 "$LAB_TRAIN_PY" -u scripts/train.py pi05_libero \
  --exp-name="${LAB_TAG}_E0_s42" --batch-size=32 --seed=42 --num-train-steps=30000
CUDA_VISIBLE_DEVICES=1 "$LAB_TRAIN_PY" -u scripts/train.py pi05_libero_lora \
  --exp-name="${LAB_TAG}_E1_s42" --batch-size=32 --seed=42 --num-train-steps=30000
CUDA_VISIBLE_DEVICES=0 "$LAB_TRAIN_PY" -u scripts/train.py pi05_libero_lora_frozen_vision \
  --exp-name="${LAB_TAG}_E2_s42" --batch-size=32 --seed=42 --num-train-steps=30000
CUDA_VISIBLE_DEVICES=1 "$LAB_TRAIN_PY" -u scripts/train.py pi05_libero_action_only \
  --exp-name="${LAB_TAG}_E3_s42" --batch-size=32 --seed=42 --num-train-steps=30000
```

这里按行在同一终端执行会串行；想并行要像前面那样分别开会话，不要无脑在每行末尾加 `&`。

E1/E2 历史正式训练约32.5/28.4小时，仅供估时，不承诺新一轮用时。完整重跑四组是昂贵实验，不是练习命令必须付出的代价。

## 7. 训练进度、停止、恢复

监控终端 source 后选择要看的组：

```bash
lab_use E2 0 8122
ls -lt "$LAB_LOG"
rg 'Step [0-9]+:' "$LAB_LOG"/train_*.log | tail -12
rg -n -i 'traceback|nan|inf|out of memory|resource exhausted' "$LAB_LOG"/train_*.log
watch -n 5 nvidia-smi
```

跟随某一个确切日志文件（把文件名换成 `ls` 输出的真实名字）：

```bash
# 示例结构，不是保证存在的真实文件名
# tail -f "$LAB_LOG/train_20261023_120000_12345.log"
```

正常结束还要核对 finalize，而不是只看到目录：

```bash
test -f "$LAB_CKPT/_CHECKPOINT_METADATA"
test -f "$LAB_CKPT/params/_METADATA"
ls "$LAB_CKPT/assets/physical-intelligence/libero/norm_stats.json"
```

30000 次更新最终目录 `29999`，不是 `30000`。保存中的 `.tmp` 等目录不能交给推理服务。

停止：回到正确 tmux 会话按 Ctrl+c；等程序退出/保存器收尾。想继续运行而自己下线，应该 Ctrl+b，d。不要使用 `pkill python`、`killall` 或按宽泛名称杀别人的训练。

恢复同一实验（**不是新建实验，也不加 --overwrite**）：

```bash
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E2 0 8122
lab_train resume
```

其他组恢复：

```bash
lab_use E0 0 8120   # 然后单独执行 lab_train resume
lab_use E1 1 8121   # 同上
lab_use E3 1 8123   # 同上
```

恢复核心是保持 config、exp-name、数据、batch、seed、LR schedule 一致并加 `--resume`；模型/优化器/step 恢复不保证原数据迭代器顺序精确重现。若没有可恢复 checkpoint，需要保留错误现场，以新 run 从头开始，不能装作成功续训。

## 8. 统一训练成本短测：E0/E1/E2/E3 都有命令

不是对正式训练接着改参数，而是独立 `<run>_profile`，每组220步，20步预热+200步测量。

```bash
lab_use E0 0 8120
lab_profile
lab_use E1 0 8121
lab_profile
lab_use E2 0 8122
lab_profile
lab_use E3 0 8123
lab_profile
```

为公平比较在同卡顺序跑、相近共享负载；不与自己的其他训练同卡混跑。原项目只完成 E1/E2/E3 同口径短测，这里 E0 命令是新一轮可选补测，不是已有 E0 实测数字。

另开监控窗口，在短测期间记录 GPU 每进程占用；开始前记住采样终端，结束后 Ctrl+c 停止采样：

```bash
mkdir -p "$LAB_ROOT/resources"
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,used_memory \
  --format=csv -l 1 > "$LAB_ROOT/resources/process_memory_$(date +%Y%m%d_%H%M%S).csv"
```

采样包括其他进程，分析时按本轮真实 Python PID 筛选，不能拿整卡总占用当作自己的显存。短测 JSONL 中有 step、计时、loss，和采样 CSV 配套报告；1Hz 观测峰值不是精确瞬时峰值。

## 9. 服务和测评是两个进程，运行在两种 Python 中

### 9.1 普通演示路径（可选，不用于本项目严格配对）

先在 `pi05-server` 会话用正式 E1 权重开普通服务：

```bash
tmux new-session -s pi05-server
```

```bash
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E1 0 8121
cd "$LAB_OPENPI"
CUDA_VISIBLE_DEVICES=0 "$LAB_TRAIN_PY" scripts/serve_policy.py --port=8121 \
  policy:checkpoint --policy.config="$LAB_CONFIG" --policy.dir="$LAB_CKPT"
```

另开客户端窗口，四套各任务试1次，共40次，不是“每套只试1次”：

```bash
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E1 0 8121
cd "$LAB_OPENPI"
for suite in libero_spatial libero_object libero_goal libero_10; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 "$LAB_CLIENT_PY" examples/libero/main.py \
    --args.host=127.0.0.1 --args.port=8121 --args.task-suite-name="$suite" \
    --args.num-trials-per-task=1 --args.replan-steps=5 \
    --args.video-out-path="$LAB_LOG/demo_videos/$suite"
done
```

官方 `main.py` 使用 `--args.xxx`，自定义 `eval_manifest.py` 使用 `--xxx`，不要混淆。看完演示在服务窗口 Ctrl+c，释放模型占用后再起下面的显式噪声服务。演示不并入正式成绩；同一 demo 视频路径重复跑可能覆盖，换 tag 或子目录。

### 9.2 固定清单开发测评：先 E0/E1，再 E2/E3，全部使用新 run ID

从普通终端创建服务会话和测评会话：

```bash
tmux new-session -s pi05-seeded-server
# detach 后从普通终端再建
tmux new-session -s pi05-eval
```

每次先在**服务窗口** source，然后选某组执行：

```bash
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E0 0 8120
lab_serve
```

等待服务开始监听、加载没有 traceback。在**客户端窗口**：

```bash
source /home/zhsu/jyluo/experiments/pi05_libero_adaptation/manual_workflow/lab.sh
lab_use E0 0 8120
lab_eval development
lab_status development
```

完成后在服务窗口 Ctrl+c，换下一组；四组的成对命令如下，**左右两列在两个不同窗口中运行**：

| 组 | 服务窗口 | 客户端窗口 |
|---|---|---|
| E0 | `lab_use E0 0 8120` 然后 `lab_serve` | `lab_use E0 0 8120` 然后 `lab_eval development` |
| E1 | `lab_use E1 0 8121` 然后 `lab_serve` | `lab_use E1 0 8121` 然后 `lab_eval development` |
| E2 | `lab_use E2 0 8122` 然后 `lab_serve` | `lab_use E2 0 8122` 然后 `lab_eval development` |
| E3 | `lab_use E3 0 8123` 然后 `lab_serve` | `lab_use E3 0 8123` 然后 `lab_eval development` |

预期每组400条有效开发记录（Spatial200+Long200），不是四套全部都在开发阶段。两张卡足够时可同时两个服务，分别配两个客户端；两边组名/端口/GPU对应一致，不根据“启动先后”猜连接到了谁。

服务真正展开的命令：

```bash
CUDA_VISIBLE_DEVICES="$LAB_GPU" "$LAB_TRAIN_PY" "$LAB_PROJECT/seeded_policy_server.py" \
  --config="$LAB_CONFIG" --checkpoint="$LAB_CKPT" --port="$LAB_PORT"
```

客户端真正展开的命令：

```bash
CUDA_VISIBLE_DEVICES="$LAB_GPU" MUJOCO_EGL_DEVICE_ID="$LAB_GPU" \
  "$LAB_CLIENT_PY" "$LAB_PROJECT/eval_manifest.py" \
  --run-id="$LAB_RUN" --config="$LAB_CONFIG" --checkpoint="$LAB_CKPT" \
  --phase=development --suites libero_spatial libero_10 \
  --host=127.0.0.1 --port="$LAB_PORT"
```

**普通客户端 ↔ 普通服务；manifest 客户端 ↔ 显式噪声服务。** 后者检查服务 metadata、checkpoint、config；连错应该报错，而不是绕过检查。

客户端中断后可在同权重、同协议下重跑同一个 `lab_eval development`，已有兼容记录会跳过；最多两次基础设施尝试，不能反复重试失败 episode 刷分。改变 checkpoint 却复用同 run ID 会被拒绝。

### 9.3 查看完整进度、视频、动作

```bash
lab_use E2 0 8122
lab_status development
find "$LAB_PROJECT/evaluations/$LAB_RUN/development/episodes" -name '*.json' | wc -l
find "$LAB_PROJECT/evaluations/$LAB_RUN/development/videos" -name '*.mp4' | head
ls "$LAB_PROJECT/evaluations/$LAB_RUN/development/heartbeat.shard_0.json"
```

失败视频由 VS Code 文件树打开下载，或复制到本地播放器播放。`videos/` 没文件可能是尚未出现失败，不代表客户端没保存动作；每次动作 NPZ 在 `episodes/<suite>/task_xx/`。

## 10. 汇总与公平配对，而不是手抄一个平均数

四组开发都完成后，CPU 验收并比较：

```bash
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" "$LAB_PROJECT/manual_workflow/results.py" summarize \
  --root "$LAB_ROOT" --phase development --arms E0 E1 E2 E3

"$LAB_TRAIN_PY" -m json.tool "$LAB_ROOT/development_comparison.json"
```

脚本要求完整身份集合、manifest字段、配置、checkpoint、轨迹/失败视频哈希和动作有限性匹配；再复用已有的套件等权和任务内成对 bootstrap 数学。文件使用独占创建，拒绝覆盖已有汇总。确实要重新生成，请先保留旧输出并换新报告路径/新 tag，不要删结果伪装第一次运行。

该工具只帮助审计，不自动选候选。你要实际审阅：逐任务成功率、失败视频、训练健康和成本。候选不是“loss更低就选”，更不能看到确认成绩后倒过来改变开发决策。

## 11. 冻结候选，再做确认；E3 不要求硬跑进确认

历史主线 E3 开发68.25%，未进入确认。新一轮需要按实际开发结果审阅，不能复制旧分数作为本轮审阅结论。下面以**本轮审阅后确实选择 E2**为示例：

```bash
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" "$LAB_PROJECT/manual_workflow/results.py" freeze \
  --root "$LAB_ROOT" --candidate E2 \
  --reason '填写你本轮实际审阅结论：逐任务、失败类型、成本和接受的退化，不要照抄旧成绩'
```

这将核验本轮 E0/E1/E2 开发完整，拒绝在其本轮确认目录已存在后冻结；对模型权重/资产、源码与manifest做哈希（大权重扫描会花时间），生成 `freeze.json` 和 `freeze.sha256`。**freeze 不等于封存显存，不启动服务。**

原 benchmark 已经暴露；新 run 的冻结只是防止再次边看结果边改配方，不能恢复成从未见过的独立测试。此工具不是历史 `candidate_freeze.json` 的替换品，也不覆盖原协议。

确认时，像开发阶段一样服务/客户端成对操作：

| 组 | 服务窗口 | 客户端窗口 |
|---|---|---|
| E0 | `lab_use E0 0 8120`；`lab_serve` | `lab_use E0 0 8120`；`lab_eval confirmation` |
| E1 | `lab_use E1 0 8121`；`lab_serve` | `lab_use E1 0 8121`；`lab_eval confirmation` |
| E2 | `lab_use E2 0 8122`；`lab_serve` | `lab_use E2 0 8122`；`lab_eval confirmation` |

每组1600条：Spatial/Long各300，Object/Goal各500。helper 在确认前重新核验冻结哈希和组别，不能冻结E2却偷偷评E3后挑结果。如果实际选择E3，将 freeze 的 candidate 改成E3，并用E3替换确认第三组；这不是照着旧实验筛选结论硬走。

确认全部完成后：

```bash
JAX_PLATFORMS=cpu "$LAB_TRAIN_PY" "$LAB_PROJECT/manual_workflow/results.py" summarize \
  --root "$LAB_ROOT" --phase confirmation --arms E0 E1 E2

"$LAB_TRAIN_PY" -m json.tool "$LAB_ROOT/confirmation_comparison.json"
```

确认套件样本数不同，主报告取四套等权，不能只用总成功数/1600代替。E0/E1/E2 开发+确认各2000，另加E3开发400，才是完整6400记录。E3不进入确认，并不等于“实验没做完”。

统计报告同时给环境成功与原始动作严格边界成功。原主线严格边界成功为0，不能说环境完成任务就等于原始动作安全合规。−5pp是预先声明的工程非劣界，不是看到差距后才临时设定的容忍值；单训练seed及跨进程推理波动不包含在普通rollout bootstrap里。

## 12. 如果想学习原项目看门狗，应该看什么，而不是什么都启动

阅读入口：

```bash
rg -n 'def |add_argument|budget|reserve|stall|retry|terminate' "$LAB_PROJECT/development_watchdog.py"
rg -n 'def |add_argument|resume|finalized|free' "$LAB_PROJECT/training_watchdog.py"
rg -n 'E0-s42|E1-s42|E2-s42|E3-s42|runs.json' "$LAB_PROJECT/run_confirmation_pipeline.py" "$LAB_PROJECT/compare_development.py"
```

历史 `training_watchdog.py --mode e3`、`cost_profiles_watchdog.py`、`run_confirmation_pipeline.py`、`freeze_confirmation.py`、`compare_development.py` 带有旧run/路径/筛选约束；**不是本轮新实验的通用一键入口，不要直接执行来练手**。

本教程 helper 是前台手动流程：有新目录保护、显式run、metadata检查、冻结校验和日志，不具有自动GPU调度、低显存抢占保护、自动心跳重启等完整看门狗能力。要扩展无人值守队列，应先让旧调度器读取本轮独立manifest/路径并加互斥与回归测试，不能仅把旧脚本启动就称新实验已经被保护。

人工排查命令：

```bash
tmux ls
pgrep -af 'scripts/train.py|seeded_policy_server.py|eval_manifest.py'
ss -ltnp | rg '8120|8121|8122|8123'
nvidia-smi
```

如果要用 `kill -TERM`，先 `ps -p <准确PID> -o pid,ppid,args` 核实归属，再仅停止本轮目标；不要把示例PID照抄。Ctrl+c停止自己的前台服务更直观，客户端结束不等于服务也退出。

## 13. 从报错到定位：常见工程题

| 现象 | 第一检查点 | 处理方向 |
|---|---|---|
| 没有日志文件 | 当前目录、LAB_TAG、LAB_ARM | 用绝对路径和 `lab_use`，别在openpi目录再重复加projects/openpi |
| 找不到libero | 客户端解释器/PYTHONPATH | 每个新窗口source，别用训练Python跑客户端 |
| JAX只有CPU | Python/JAX wheel/CUDA_VISIBLE_DEVICES | 检查GPU wheel与驱动；不是先升级全部包 |
| 首步几分钟不打印 | 编译/权重读取/CPU I/O | 看GPU/CPU和日志；别立刻判死锁，也不能无期限无心跳等待 |
| norm stats not found | 自定义config的assets路径 | E1/E2/E3复用已有资产，换数据则独立统计 |
| 输出异常、动作不动 | 相机方向、动作尺度、夹爪符号、归一化 | 先验接口，再看模型；不要直接扩大训练时长 |
| loss偶发上跳 | 同窗口grad_norm/LR、任务batch | 看后续是否回落，保存异常样本；单点震荡不等于发散 |
| OOM/Resource exhausted | 本进程+其他进程显存 | 停本轮、保存现场、等待余量、从checkpoint恢复，不杀他人 |
| checkpoint目录有但不能读 | 是否finalized | 等 `_CHECKPOINT_METADATA` 和保存器结束，不加载tmp |
| wrong server metadata | 服务config/checkpoint/端口 | 停错服务或改正确端口，不能去掉断言 |
| EGL错误 | 客户端GL依赖、设备id、无头配置 | 检查egl路径/选卡；不要修改训练数据方向来凑视频 |
| 推理跨进程不完全复现 | 原始输入、noise、dtype、编译/依赖 | 保存确切输入和版本；固定seed不是全部根因已排除 |

## 14. 收尾和工程经验内化

收尾逐项核验：

1. 训练结束与最终权重finalize，而非进度条截图。
2. 每组真实配置、argv、数据统计/哈希与环境快照。
3. 开发审阅→候选冻结→确认的时间顺序。
4. 逐任务结果、区间、两种成功口径、失败视频与动作记录。
5. 成本测量的设备、共享负载、预热与采样频率。
6. 恢复次数/数据顺序局限、没做的多seed/OOD/实机验证。
7. 客户端和所有模型服务退出；只释放本项目进程，不承诺整卡空闲。

保留本轮清单，检查结果而不再运行任务：

```bash
find "$LAB_ROOT" -maxdepth 3 -type f | sort
find "$LAB_OPENPI/checkpoints" -path "*/${LAB_TAG}_*/29999/_CHECKPOINT_METADATA"
pgrep -af 'scripts/train.py|seeded_policy_server.py|eval_manifest.py'
```

如何把经验变成自己的：先逐行展开理解`lab.sh`；能不看文档解释config和exp-name的区别；能画出服务/客户端的数据流；亲自读一条失败JSON和NPZ；指出一次loss抖动与一次资源中断的区别；最后用自己的话写出成本—性能结论和限制。未亲自执行过的部分，应诚实说“我读过并能解释流程”，不能包装为本人处理过的故障。

### 最短但完整的执行顺序速记

```text
环境/路径验收 → 资源与数据验收 → 四组参数审计
→ 四组10步smoke与冻结检查 → 两卡分批四组30k训练
→ 独立成本短测 → 四组开发400 → 审阅失败/成本
→ 冻结候选 → E0/E1/候选各1600确认 → 配对分析/完整基准
→ 保存证据、解释限制、停止自己的服务
```

所有本轮辅助文件在 [manual_workflow](../manual_workflow/lab.sh)，和旧实验的源码/证据分离。文中安装命令参考 [uv](https://docs.astral.sh/uv/)、[openpi LIBERO](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)，终端管理参考 [tmux 官方入门](https://github.com/tmux/tmux/wiki/Getting-Started)。新命令已做下述静态/CPU检查，不等于本轮四组训练或新环境重建已完成。

本次已完成的校验：`bash -n lab.sh`、Python 语法/CLI检查、source只定义函数的检查、训练/服务/两种客户端帮助参数核对、GPU采样字段核对、合成配对数据零差值与身份不匹配拒绝测试、本地链接存在性检查。**没有**运行安装器/apt/uv同步/数据下载，没有初始化新的训练或rollout，没有验证全套新环境安装及真实GPU端到端运行。首次实际使用时仍必须按验收门槛逐步检查。
