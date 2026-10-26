# openpi 服务器部署方案（离线 git 依赖 + 清华源）

## 背景

服务器 `aius-04` 网络情况：

| 目标 | 结果 | 结论 |
|---|---|---|
| GitHub | ❌ 完全不通 | git 源依赖必须离线解决 |
| PyPI | ⚠️ 10s 超时 | 换清华源 |
| 清华源 | ✅ 200（5.4s） | 可用但偏慢 |
| GCS (权重) | ✅ 400 / 0.4s | **模型权重可直接在服务器下载** |

服务器环境：
- OS: Ubuntu 24.04.5 LTS, x86_64
- GPU: 2× RTX 6000D (85G each)，驱动 595.91.07 / CUDA 13.2
- base Python: 3.14.7（太新，不可用）
- uv: 未安装

## 要传的文件

从本地传到服务器（建议放 `~/jyluo/`）：

1. `openpi.tar.gz`（524M）—— openpi 主仓库（含子模块，重新打包的干净版本）
2. `lerobot-min.tar.gz`（~1.5M）—— lerobot 源码，替换 git 依赖
3. `dlimp-min.tar.gz`（~100K）—— dlimp 源码，替换 git 依赖
4. `pyproject.server.toml` —— 把 git 源改成路径源的补丁版配置

---

## 服务器端步骤

### Step 1: 清理旧目录

```bash
# 确认旧 openpi 没有你的改动后，删除它（注意在 LLaMA-Factory 内部）
git -C ~/jyluo/projects/LLaMA-Factory/openpi status 2>/dev/null | head
rm -rf ~/jyluo/projects/LLaMA-Factory/openpi

# 删掉那个坏的空 conda 环境（Python 3.10、无 jax）
rm -rf ~/jyluo/miniforge3/envs/openpi
```

### Step 2: 解压 openpi 到正确位置

```bash
cd ~/jyluo/projects
tar xzf ~/jyluo/openpi.tar.gz          # 得到 ~/jyluo/projects/openpi
du -sh ~/jyluo/projects/openpi          # 应约 784M
```

### Step 3: 放置离线 git 依赖

```bash
# 解压 lerobot / dlimp 到 openpi 的 third_party 旁（或任意固定位置）
mkdir -p ~/jyluo/projects/offline-deps
tar xzf ~/jyluo/lerobot-min.tar.gz -C ~/jyluo/projects/offline-deps/
tar xzf ~/jyluo/dlimp-min.tar.gz   -C ~/jyluo/projects/offline-deps/
ls ~/jyluo/projects/offline-deps/   # 应有 lerobot/ 和 dlimp/
```

### Step 4: 应用 pyproject 补丁

```bash
cd ~/jyluo/projects/openpi
cp pyproject.toml pyproject.toml.bak        # 备份原始
cp ~/jyluo/pyproject.server.toml pyproject.toml
```

补丁内容（把 git 源 → 本地路径源）：

```toml
[tool.uv.sources]
openpi-client = { workspace = true }
lerobot = { path = "../offline-deps/lerobot", editable = true }
dlimp   = { path = "../offline-deps/dlimp", editable = true }
```

### Step 5: 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

> 若 astral.sh 也慢/不通，用清华镜像装：
> `pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple`
> 或直接下载 uv 二进制：https://github.com/astral-sh/uv/releases（走代理）

### Step 6: 配置清华源

```bash
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml <<'EOF'
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
EOF
```

### Step 7: 建 3.11 环境并同步依赖

```bash
cd ~/jyluo/projects/openpi

# 环境变量（按你的实际路径改）
export OPENPI_DATA_HOME=~/jyluo/myopenpi/dataset/ckpt
export HF_LEROBOT_HOME=~/jyluo/dataset_root
export UV_CACHE_DIR=/tmp/uv-cache
export WANDB_MODE=offline

# uv 自动下载 Python 3.11（走 python-build-standalone，可能需镜像）
uv python install 3.11

# 同步依赖（关键：--no-install-package 避免触发 git 源）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

> 注意：`uv sync` 会重建 `uv.lock`。若 lock 里 git 源冲突，
> 可删掉 uv.lock 让它按补丁后的 pyproject 重新解析：
> `rm uv.lock && uv sync`

### Step 8: 验证

```bash
uv run python -c "import jax; print('jax', jax.__version__); print('devices', jax.devices())"
uv run python -c "import openpi; print('openpi ok')"
```

---

## 模型权重（可选，服务器直接下）

GCS 在服务器上可通，且下载走 `gsutil` 或 gcsfs：

```bash
# 方式 A：用 openpi 自带下载（会缓存到 OPENPI_DATA_HOME）
uv run python -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
print(p)
"

# 方式 B：直接 gsutil（需先装 gcloud sdk）
# gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base ~/jyluo/myopenpi/dataset/ckpt/
```

可用权重列表（详见 README）：
- `pi0_base`, `pi0_fast_base`, `pi05_base`（微调基座）
- `pi05_libero`, `pi0_fast_droid`, `pi05_droid`, `pi0_aloha_towel` 等（专家模型）

---

## 常见问题

1. **uv sync 报 git 错误** → 确认 pyproject.toml 补丁已生效，且 offline-deps 路径正确
2. **jax 装不上** → 确认 Python 是 3.11（不是 3.14）
3. **CUDA 版本** → 驱动 595/CUDA13.2 向下兼容 cuda12 包，正常可用
4. **下载慢** → uv.toml 清华源已配；必要时加 `UV_HTTP_TIMEOUT=300`
