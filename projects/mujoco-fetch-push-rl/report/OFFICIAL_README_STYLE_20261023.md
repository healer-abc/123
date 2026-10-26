# FetchPush 工程操作手册：从官方 README 到 A–E 实验

这份手册对应第二个项目 `mujoco-fetch-push-rl`。主线是：安装官方环境和算法库 → 用官方 Zoo 命令训练 → 用配置实现 A/B/C/D → 增加动作增益环境实现 E → 官方回放、录像 → 按项目口径统计结果。

**训练入口使用 RL Baselines3 Zoo 的 `python train.py`，回放使用 `python enjoy.py`，录像使用 `python -m rl_zoo3.record_video`。** 自己的改动放在五个 YAML 和一个环境模块中，便于对照源码阅读。

先说明来源：已经完成的 15 个模型由本项目 `run.py` 调用 SB3 训练，结果见 `REPORT.md`。本手册新增的 `official_readme/` 是另一套可执行的官方入口复现材料，不能把旧结果写成“由 Zoo 命令跑出”。两套使用相同的主要实验参数，验证种子流程、回调和文件格式存在差异，不承诺重现完全相同的数值。本次没有重跑百万步矩阵。

## 1. “官方”具体指哪些仓库

| 来源 | 提供什么 | 本手册怎么使用 |
|---|---|---|
| [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics/tree/v1.4.0) | Fetch 机器人环境、目标、奖励和仿真接口 | 安装 `gymnasium-robotics==1.4.0`，调用 `FetchPush-v4` / `FetchPushDense-v4` |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/tree/v2.7.0) | PPO、SAC、HER 回放实现 | 安装 `stable-baselines3==2.7.0`，直接使用算法核心 |
| [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/tree/v2.7.0) | SB3 配套命令行训练、回放和录像 | 使用 v2.7.0 的原始入口，加载自己的 YAML |
| 本项目 `official_readme/` | 实验配置、动作增益、逐回合统计 | 属于自己的实验代码 |

SB3 本体的 README 以 Python API 为主，并没有本项目这种 `run.py train --group D` 命令。[SB3 HER 文档](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/her.html)展示了 `HerReplayBuffer` 的接入方式，并指向 Zoo 的命令行训练。下面沿用 [Zoo v2.7.0 README](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v2.7.0/README.md) 的入口；README 中 `hyperparameters/` 的文字在本版本实际源码里对应 `hyperparams/` 目录。

## 2. 本机目录与启动方式

材料已经放在项目中：

```text
projects/mujoco-fetch-push-rl/
├── run.py                         # 已完成实验的项目入口
├── source_materials/runner_frozen_20261022.py
├── runs/formal_20261022/           # 已完成模型；不要放教学运行输出
├── evaluations/formal_20261022/    # 已完成独立测试
└── official_readme/
    ├── activate.sh                # 本机环境、模块路径与工作目录
    ├── custom_envs.py             # 新增：ActionGain 与自定义环境注册
    ├── configs/A.yml ... E.yml    # 新增：官方 Zoo 格式的五份配方
    ├── evaluate_goals.py          # 新增：项目逐回合统计；调用 SB3 API
    ├── upstream/
    │   ├── source.json            # 官方版本、来源和压缩包哈希
    │   └── rl-baselines3-zoo/      # 官方 v2.7.0 源码
    ├── logs/                      # 按手册重新训练时的输出
    ├── evaluations/               # 新运行的逐回合测试
    └── checks/                    # 本手册的短流程验证
```

**本机每个新终端先执行：**

```bash
source /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/official_readme/activate.sh
export FETCH_KIT=/home/zhsu/jyluo/projects/mujoco-fetch-push-rl/official_readme
pwd
python -c 'import sys, stable_baselines3; print(sys.executable); print(stable_baselines3.__version__)'
python train.py --help
```

`pwd` 应落在 `official_readme/upstream/rl-baselines3-zoo/`，SB3 应为 2.7.0。

本机沿用已有 `envs/fetch-rl` conda 环境，并用 `official_readme/.venv` 的轻量隔离层安装 Zoo 新增依赖。Torch、MuJoCo 等仍来自原 conda 环境；原环境和已完成训练依赖没有被升级。`activate.sh` 会依次处理两层环境、隐藏 GPU、设置线程，并让官方入口能导入项目的 `custom_envs.py`。

## 3. 新机器才做：按官方方式安装

已有本机环境直接跳到第 4 节。本节是新目录的安装说明，命令中的项目路径按目标机器调整。

### 3.1 安装算法库和机器人环境

[机器人官方安装说明](https://robotics.farama.org/content/installation/)的基础命令是 `pip install gymnasium-robotics`。为复现本项目，在相同安装方式上固定版本：

```bash
source /home/zhsu/jyluo/miniforge3/etc/profile.d/conda.sh
cd /home/zhsu/jyluo/projects/mujoco-fetch-push-rl
conda env create --prefix ./envs/fetch-rl --file environment.yml
conda activate "$PWD/envs/fetch-rl"
python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m pip check
```

这里是 CPU 路线。新版本 MuJoCo Python 绑定随 `mujoco` 包安装，不走旧 `mujoco-py`、许可证或 `~/.mujoco` 安装路线。

### 3.2 获取 Zoo 官方源码

新机器上有 git 时：

```bash
cd /home/zhsu/jyluo/projects/mujoco-fetch-push-rl
mkdir -p official_readme/upstream
git clone --depth 1 --branch v2.7.0 \
  https://github.com/DLR-RM/rl-baselines3-zoo.git \
  official_readme/upstream/rl-baselines3-zoo
```

本机通过官方 tag 压缩包获取了同版源码，因为当时 shell 中没有 git；不需要再 clone 一份。本项目也没有使用 Zoo 的预训练模型，因而不要求下载 `rl-trained-agents` 子模块。

### 3.3 按 README 的最小安装方式安装 Zoo

官方支持 `pip install -e .` 或安装 `rl_zoo3` 包。本机采用隔离层安装包并从官方源码目录运行入口：

```bash
cd /home/zhsu/jyluo/projects/mujoco-fetch-push-rl
source env.sh
python -m venv --system-site-packages official_readme/.venv
official_readme/.venv/bin/python -m pip install \
  'rl_zoo3==2.7.0' 'sb3-contrib==2.7.0' 'moviepy>=2,<3'
source official_readme/activate.sh
export FETCH_KIT=/home/zhsu/jyluo/projects/mujoco-fetch-push-rl/official_readme
python -m pip check
```

只跑 Fetch 不需要为了 README 的 full installation 安装 Atari、PyBullet 等额外环境。MoviePy 用于官方视频模块。

## 4. 官方最小例子：先理解“环境 → 模型 → learn”

在当前终端直接运行这个短例子，它使用 SB3 官方 API；代码是本手册针对 Fetch 编写的示例，不是另一个官方训练脚本：

```bash
python - <<'PY'
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC, HerReplayBuffer

gym.register_envs(gymnasium_robotics)
env = gym.make("FetchPush-v4")
model = SAC(
    "MultiInputPolicy", env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    buffer_size=10000,
    learning_starts=100,
    batch_size=64,
    device="cpu",
    verbose=1,
)
model.learn(total_timesteps=200)
env.close()
PY
```

这只验证接口和更新，不检验收敛。后面的官方 `train.py` 完成的核心也是“读配置 → 创建 env → 创建 SAC/PPO → learn → save”，并加上日志、验证和保存。

观察是字典，因此用 `MultiInputPolicy`。HER 是回放类型，不是 `--algo her`；对应命令仍然是 `--algo sac`。环境在本机为 v4，不能直接照搬上游旧 YAML 里的 `FetchPush-v1`。

## 5. A–E 实验分别改哪个文件

| 组 | 算法与环境 | 主要改动位置 | 对照含义 |
|---|---|---|---|
| A | PPO / `FetchPushDense-v4` | `configs/A.yml` | 密集奖励 PPO |
| B | SAC / `FetchPushDense-v4` | `configs/B.yml` | A↔B 比算法 |
| C | SAC / `FetchPush-v4` | `configs/C.yml` | B↔C 比奖励密度 |
| D | SAC+HER / `FetchPush-v4` | `configs/D.yml` 的回放参数 | C↔D 比 HER |
| E | SAC+HER / `FetchPushGain-v0` | `configs/E.yml` + `custom_envs.py` | D↔E 比训练增益随机化 |

上表中的配置都位于 `official_readme/configs/`，不是 Zoo 自带的默认配方。Zoo 使用 `--conf-file` 原生参数加载它们。[官方训练参数源码](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v2.7.0/rl_zoo3/train.py)可查到该选项。

### 5.1 从官方 YAML 格式建立 SAC 配方

打开 `official_readme/configs/C.yml`。内容已经写好；从干净项目建立时可使用：

```yaml
FetchPush-v4:
  n_envs: 1
  n_timesteps: 1000000
  policy: MultiInputPolicy
  policy_kwargs: "dict(net_arch=[256, 256])"
  learning_rate: 0.0003
  gamma: 0.98
  batch_size: 256
  buffer_size: 1000000
  learning_starts: 10000
  train_freq: 1
  gradient_steps: 1
  tau: 0.005
  ent_coef: auto
  normalize: false
  monitor_kwargs: "dict(info_keywords=('is_success',))"
```

键名 `FetchPush-v4` 必须与命令的 `--env` 一致。`normalize: false` 对应本次实验使用原始状态和奖励；`monitor_kwargs` 记录回合末 `is_success`。这两个是 Zoo 配置字段，不是直接传给 `SAC(...)` 的参数。Zoo 的 [配置说明](https://rl-baselines3-zoo.readthedocs.io/en/v2.7.0/guide/config.html)介绍了这种 YAML 接口。

B 的参数保持相同，只把最上层环境键改为 `FetchPushDense-v4`，写入 `B.yml`。A 切换 PPO，配置使用 `n_steps=2048`、`n_epochs=10`、`clip_range=0.2`、`gae_lambda=0.95`，去掉 SAC 的回放和熵系数字段；完整内容见 `A.yml`。

### 5.2 C → D：只增加 HER 回放配置

在 C 的基础上，`D.yml` 增加：

```yaml
  replay_buffer_class: HerReplayBuffer
  replay_buffer_kwargs: "dict(n_sampled_goal=4, goal_selection_strategy='future')"
```

对应 [SB3 官方 HER 接口](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/her.html)。Zoo 的 `rl_zoo3/exp_manager.py` 会把 YAML 中的回放类和参数转换为对象，再传给 SAC。这里无需修改 `stable_baselines3/sac/sac.py` 或 `stable_baselines3/her/her_replay_buffer.py`。

### 5.3 D → E：增加 `custom_envs.py`

自己的代码在 `official_readme/custom_envs.py`，核心有三块。

第一块：`ActionGain.reset()` 每回合只采样一次增益；使用独立 RNG，避免改变物体和目标的初始采样：

```python
obs, info = self.env.reset(seed=seed, options=options)
if seed is not None:
    self.gain_rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
self.gain = float(self.gain_rng.uniform(self.gain_low, self.gain_high))
return obs, info
```

第二块：`ActionGain.step()` 只改变前三维平移动作，第四维保持原样；不修改目标和奖励定义：

```python
executed = np.asarray(action, dtype=np.float32).copy()
executed[:3] = np.clip(executed[:3] * self.gain, -1, 1)
obs, reward, terminated, truncated, info = self.env.step(executed)
info["action_gain"] = self.gain
return obs, float(reward), terminated, truncated, info
```

第三块：保留 HER 的向量化奖励重算，并注册可接收增益参数的环境：

```python
def compute_reward(self, achieved_goal, desired_goal, info):
    return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

def make_fetch_push_gain(gain_low=0.8, gain_high=1.2, **kwargs):
    return ActionGain(gym.make("FetchPush-v4", **kwargs), gain_low, gain_high)

gym.register(
    id="FetchPushGain-v0",
    entry_point="custom_envs:make_fetch_push_gain",
    max_episode_steps=50,
)
```

完整文件还包含导入和重复注册保护，直接使用现有文件。`FetchPushGain-v0` 是项目自定义名字，内部物理环境仍是官方 `FetchPush-v4`，不是上游新增版本。

E 的 YAML 与 D 相同，只将环境键改成 `FetchPushGain-v0`。训练范围由命令里的 `--env-kwargs` 给出。官方 `rl_zoo3/import_envs.py` 会尝试导入名为 `custom_envs` 的模块，所以把该文件所在目录加入 `PYTHONPATH` 后，训练、回放和录像均可加载它。训练和回放命令中再显式写 `--gym-packages custom_envs`，便于读懂导入来源。

### 5.4 为什么不能简单在训练和评估上共用随机增益 wrapper

如果 wrapper 在验证时仍采样 `[0.8,1.2]`，得到的是混合增益验证表现，不是正常条件 `g=1`。本手册通过自定义环境构造参数解决：训练给 `--env-kwargs`，验证给官方 `--eval-env-kwargs`。二者都由 Zoo 创建，训练主循环保持官方实现。

## 6. 正式运行命令：仍然调用官方 `train.py`

先设置本次种子：

```bash
source /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/official_readme/activate.sh
export FETCH_KIT=/home/zhsu/jyluo/projects/mujoco-fetch-push-rl/official_readme
training_seed=0
```

下面是完整预算命令，会真正启动训练。已有正式结果只需阅读命令；本次新增材料没有自动执行这 15 次长训练。日志写入 `official_readme/logs/`，不进入旧 `runs/formal_20261022/`。

### A：PPO，密集奖励

```bash
python train.py --algo ppo --env FetchPushDense-v4 \
  --gym-packages custom_envs --conf-file "$FETCH_KIT/configs/A.yml" \
  --seed "$training_seed" -n 1000000 --device cpu --num-threads 2 \
  --eval-freq 20000 --eval-episodes 20 --n-eval-envs 1 \
  --save-freq 250000 \
  -f "$FETCH_KIT/logs/A_seed${training_seed}"
```

### B：SAC，密集奖励

```bash
python train.py --algo sac --env FetchPushDense-v4 \
  --gym-packages custom_envs --conf-file "$FETCH_KIT/configs/B.yml" \
  --seed "$training_seed" -n 1000000 --device cpu --num-threads 2 \
  --eval-freq 20000 --eval-episodes 20 --n-eval-envs 1 \
  --save-freq 250000 --save-replay-buffer \
  -f "$FETCH_KIT/logs/B_seed${training_seed}"
```

### C：SAC，稀疏奖励

```bash
python train.py --algo sac --env FetchPush-v4 \
  --gym-packages custom_envs --conf-file "$FETCH_KIT/configs/C.yml" \
  --seed "$training_seed" -n 1000000 --device cpu --num-threads 2 \
  --eval-freq 20000 --eval-episodes 20 --n-eval-envs 1 \
  --save-freq 250000 --save-replay-buffer \
  -f "$FETCH_KIT/logs/C_seed${training_seed}"
```

### D：SAC＋HER，稀疏奖励

```bash
python train.py --algo sac --env FetchPush-v4 \
  --gym-packages custom_envs --conf-file "$FETCH_KIT/configs/D.yml" \
  --seed "$training_seed" -n 1000000 --device cpu --num-threads 2 \
  --eval-freq 20000 --eval-episodes 20 --n-eval-envs 1 \
  --save-freq 250000 --save-replay-buffer \
  -f "$FETCH_KIT/logs/D_seed${training_seed}"
```

### E：SAC＋HER，训练时加入增益随机化

```bash
python train.py --algo sac --env FetchPushGain-v0 \
  --gym-packages custom_envs --conf-file "$FETCH_KIT/configs/E.yml" \
  --env-kwargs gain_low:0.8 gain_high:1.2 \
  --eval-env-kwargs gain_low:1.0 gain_high:1.0 \
  --seed "$training_seed" -n 1000000 --device cpu --num-threads 2 \
  --eval-freq 20000 --eval-episodes 20 --n-eval-envs 1 \
  --save-freq 100000 --save-replay-buffer \
  -f "$FETCH_KIT/logs/E_seed${training_seed}"
```

五条命令中的 `--algo` 决定 PPO/SAC，`--conf-file` 决定参数与 HER，`--env` 决定稀疏/密集/增益环境。种子改为 1、2，输出目录会随变量变更；每组需要三次独立训练。PPO 按完整 rollout 更新，百万步预算会略超出，本项目原结果为 1,001,472 步。

### 6.1 先做短验证的方法

在 D 命令上将 `-n` 改为 300，并额外加 `--hyperparams learning_starts:100 buffer_size:1000`，使短过程真正采样 HER 并更新。短验证输出应改为独立目录。完整可执行示例：

```bash
python train.py --algo sac --env FetchPush-v4 \
  --gym-packages custom_envs --conf-file "$FETCH_KIT/configs/D.yml" \
  --seed 99 -n 300 --device cpu --num-threads 2 \
  --hyperparams learning_starts:100 buffer_size:1000 \
  --eval-freq 200 --eval-episodes 2 --n-eval-envs 1 \
  --save-freq 200 --save-replay-buffer \
  -f "$FETCH_KIT/logs/manual_smoke_D"
```

短验证临时改过 warmup 和 buffer，不能加入正式统计。正式训练回到第 6 节原命令。

## 7. 输出在哪里，选哪个模型

第一次执行 D seed0，典型布局是：

```text
official_readme/logs/D_seed0/sac/FetchPush-v4_1/
├── FetchPush-v4.zip          # 本次训练结束时的模型
├── best_model.zip            # 官方验证回调按平均回报保存的模型
├── replay_buffer.pkl         # --save-replay-buffer 在结束保存的回放
├── evaluations.npz           # 验证步数、回报、长度、成功信息
├── 0.monitor.csv             # 训练回合统计
├── rl_model_250000_steps.zip # --save-freq 生成的模型快照
└── FetchPush-v4/
    ├── args.yml              # 实际命令参数
    ├── config.yml            # 实际配置
    └── command.txt           # 训练命令
```

若在同一个日志根目录重跑，后缀会增加为 `_2` 等，不一定还是 `_1`。后续 `--exp-id` 和模型路径必须按实际目录选择，不能默默取另一轮的结果。

本项目比较固定预算结束模型，所以加载 `FetchPush-v4.zip`，回放不加 `--load-best`。官方自动生成 `best_model.zip` 并不等于项目要用它做最终测试。特别是稀疏奖励下，平均回报与“第 50 步成功率”不是同一个指标。

## 8. 按官方方式回放：`enjoy.py`

### 8.1 D 正常条件

```bash
python enjoy.py --algo sac --env FetchPush-v4 \
  --gym-packages custom_envs \
  --folder "$FETCH_KIT/logs/D_seed0" --exp-id 1 \
  -n 5000 --seed 20000 --device cpu --num-threads 2 \
  --deterministic --no-render
```

### 8.2 E 在固定增益下回放

```bash
python enjoy.py --algo sac --env FetchPushGain-v0 \
  --gym-packages custom_envs \
  --folder "$FETCH_KIT/logs/E_seed0" --exp-id 1 \
  --env-kwargs gain_low:0.6 gain_high:0.6 \
  -n 5000 --seed 20000 --device cpu --num-threads 2 \
  --deterministic --no-render
```

将两个 gain 都改为 0.8、1.0、1.2、1.4，即可检查其余增益。**E 回放要显式固定增益**，否则可能沿用保存的训练随机化参数。

这些参数对应 [官方 enjoy 源码](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v2.7.0/rl_zoo3/enjoy.py)。`-n` 表示控制步数，不是回合数。`--no-render` 适合服务器，仍会执行仿真与推理。

## 9. 项目最终测试：补充逐回合统计

官方 `enjoy.py` 适合回放和快速检查，但它的 `--seed 20000` 不是“每回合分别 reset 到 20000、20001……20099”。若要严格使用本项目的配对场景，需要一个短评估扩展。

文件是 `official_readme/evaluate_goals.py`。它直接使用 SB3 官方的 `PPO.load/SAC.load`、`model.predict(..., deterministic=True)` 和 Gymnasium 的 `reset/step`；这里的统计脚本属于项目代码，不能称为 Zoo 官方命令。[SB3 保存加载说明](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/save_format.html)介绍模型文件内容。

代码中需要读懂的两处：

```python
for scene in range(args.seed_base, args.seed_base + args.episodes):
    obs, _ = env.reset(seed=scene)
    # 每回合循环到 terminated 或 truncated，取最后一步的 is_success

model = model_class.load(args.model, env=env, device="cpu", **options)
action, _ = model.predict(obs, deterministic=True)
```

D 模型测试五档增益：

```bash
python "$FETCH_KIT/evaluate_goals.py" \
  --model "$FETCH_KIT/logs/D_seed0/sac/FetchPush-v4_1/FetchPush-v4.zip" \
  --group D --training-seed 0 --episodes 100 --seed-base 20000 \
  --gains 0.6 0.8 1.0 1.2 1.4 \
  --out "$FETCH_KIT/evaluations/D_seed0"
```

E 对应命令：

```bash
python "$FETCH_KIT/evaluate_goals.py" \
  --model "$FETCH_KIT/logs/E_seed0/sac/FetchPushGain-v0_1/FetchPushGain-v0.zip" \
  --group E --training-seed 0 --episodes 100 --seed-base 20000 \
  --gains 0.6 0.8 1.0 1.2 1.4 \
  --out "$FETCH_KIT/evaluations/E_seed0"
```

A/B/C 只给 `--gains 1.0`，修改组、模型和输出路径；种子 1、2 也分别测试。输出为 `episodes.csv` 和带模型 SHA-256 的 `summary.json`。评估不加载回放池；SAC 只创建一个占位小 buffer，避免无用的百万容量分配。

口径固定为：每模型每增益 100 回合，每回合 50 步；成功取最后一步，距离阈值 5 cm；不把 `ever_success` 代替最终成功。每组三种子的成功率先分别计算，再用 `np.std(values, ddof=1)` 得到训练种子标准差。不能把 300 个回合当成 300 次独立训练。

等 15 个新模型都评估完成后，用下面命令汇总新路线数据；它不会读取旧正式结果：

```bash
python - <<'PY'
import csv, json, os
from pathlib import Path
import numpy as np

root = Path(os.environ["FETCH_KIT"])
rows = []
for group in "ABCDE":
    seeds = []
    for seed in (0, 1, 2):
        record = json.loads((root / f"evaluations/{group}_seed{seed}/summary.json").read_text())
        assert record["group"] == group and record["training_seed"] == seed
        assert record["seed_base"] == 20000
        seeds.append({m["gain"]: m for m in record["metrics"]})
    gains = (0.6, 0.8, 1.0, 1.2, 1.4) if group in "DE" else (1.0,)
    for gain in gains:
        assert all(s[gain]["episodes"] == 100 for s in seeds)
        rates = [s[gain]["success_rate"] for s in seeds]
        rows.append(dict(group=group, gain=gain,
                         success_mean=float(np.mean(rates)),
                         success_sample_sd=float(np.std(rates, ddof=1))))
with (root / "evaluations/results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
PY
```

如果只有一个短模型，缺少其余种子的文件会报错；这表示数据未齐，不能补零或复制一个种子的结果凑成三次。

## 10. 官方录像命令

服务器不打开窗口，使用本机已验收的 Mesa 软件渲染：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export LP_NUM_THREADS=2
```

使用 [Zoo 官方录像模块](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v2.7.0/rl_zoo3/record_video.py)：

```bash
python -m rl_zoo3.record_video \
  --algo sac --env FetchPushGain-v0 \
  --folder "$FETCH_KIT/logs/E_seed0" --exp-id 1 \
  --env-kwargs gain_low:0.6 gain_high:0.6 \
  --seed 20003 --deterministic --no-render -n 150 \
  --output-folder "$FETCH_KIT/videos/E_seed0_g06"
```

该版本录像模块没有 `--gym-packages`、`--device`、`--num-threads` 参数，不能照训练命令硬加。模块通过官方 `import_envs` 自动发现 `custom_envs`；CPU 由激活脚本隐藏 CUDA 控制，线程环境变量也已设置。

`-n 150` 是连续录像 150 个控制步，涉及自动重置后的多个回合，不代表独立测试统计。现有正式 90 秒视频额外做了案例标注、重复回放标记和片尾统计，组合程序为项目 `tools/make_case_video.py`。

如果新机器缺少 Mesa/EGL，先补齐相应系统运行库；这里固定的 vendor 路径是本服务器路径。单纯训练和不渲染评估不需要这些录像环境变量。

## 11. 恢复训练：官方与现有入口的区别

Zoo 的 `-i` / `--trained-agent` 读取模型，若同目录存在 `replay_buffer.pkl`，会自动加载回放。源码位置是 [`rl_zoo3/exp_manager.py` 的 `_load_pretrained_agent()`](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v2.7.0/rl_zoo3/exp_manager.py)。

例如 D 从已有模型再训练 20 万步，放到新目录：

```bash
python train.py --algo sac --env FetchPush-v4 \
  --gym-packages custom_envs --conf-file "$FETCH_KIT/configs/D.yml" \
  -i "$FETCH_KIT/logs/D_seed0/sac/FetchPush-v4_1/FetchPush-v4.zip" \
  --seed 0 -n 200000 --device cpu --num-threads 2 \
  --eval-freq 20000 --eval-episodes 20 --n-eval-envs 1 \
  --save-freq 100000 --save-replay-buffer \
  -f "$FETCH_KIT/logs/D_seed0_continued"
```

这里 Zoo 的 `-n 200000` 是本次 `learn()` 的训练量，当前版本会默认重置步数计数，也会重新影响依赖步数的 warmup 和日志横轴；不能理解成“累计目标 20 万步”。本项目旧 `run.py --steps` 则是累计目标，会主动计算剩余步数，二者不能混用。

还要区分两种保存：v2.7.0 的 `--save-freq` 内置回调默认只存周期模型；`--save-replay-buffer` 在训练结束保存回放，不自动给每个周期检查点配套回放。若希望周期模型和回放一起保存，可在单独的恢复配方中使用官方 callback 配置：

```yaml
  callback:
    - stable_baselines3.common.callbacks.CheckpointCallback:
        save_freq: 100000
        save_path: /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/official_readme/checkpoints/D_seed0
        name_prefix: paired
        save_replay_buffer: true
```

字段来自 [SB3 官方 CheckpointCallback](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/callbacks.html)。本例只有一个环境；多环境时频率要按环境数换算。该 callback 的回放文件名带步数，而 Zoo 自动恢复寻找固定文件名 `replay_buffer.pkl`。恢复前将**同一步**的模型与回放复制到一个新的恢复目录，并将回放命名为 `replay_buffer.pkl`，再把 `-i` 指向该目录的模型。不能把较新的回放随意配到较早的模型上。

终端前台 Ctrl-C 的 `KeyboardInterrupt` 可被 Zoo 捕获并进入结束保存；硬杀进程没有这种保证。恢复也不恢复完整仿真和全部随机状态，不能承诺与连续训练逐步相同。

## 12. 官方画图与项目报告的范围

Zoo 可用 `scripts/all_plots.py` 读取验证结果。当前主线没有安装绘图 extras；需要该功能时，在隔离环境额外安装依赖：

```bash
python -m pip install seaborn scipy
MPLBACKEND=Agg python scripts/all_plots.py \
  -a sac -e FetchPush-v4 -f "$FETCH_KIT/logs/D_seed0" \
  -k successes --no-million --no-display -o "$FETCH_KIT/D_validation_plot"
```

官方 `-o` 会自行附加 `.pkl`，这里保存的是绘图中间数据，不是 PNG。需要导出图时继续运行：

```bash
MPLBACKEND=Agg python scripts/plot_from_file.py \
  -i "$FETCH_KIT/D_validation_plot.pkl" \
  -o "$FETCH_KIT/D_validation.png" --format png --no-million
```

它用于查看验证趋势，不能替代最终配对场景统计。A/B/C 的回报定义不同，跨组报告优先比较成功率，不把密集与稀疏累计回报放在一个纵轴作胜负比较。这部分是可选绘图功能，不属于训练所需安装项。

现有 `tools/summarize_results.py` 针对旧 `runs/formal_20261022/` 的配置、完成标记和评估格式做了校验；不能把 Zoo 输出直接塞进去。新路线若要重做完整报告，应读取本路线自己的 CSV，保留组、训练种子、增益、场景和模型哈希。

## 13. 对应回已有工程：实际代码在哪

在项目根目录执行以下只读定位命令：

```bash
cd /home/zhsu/jyluo/projects/mujoco-fetch-push-rl
rg -n 'GROUPS =|class ActionGain|def make_env|def assess|def save_bundle|def load_bundle|class ExperimentCallback|def train|def evaluate|def video' run.py
rg -n 'common =|HerReplayBuffer|model = PPO|model = SAC|remaining =|reset_num_timesteps' run.py
```

| 官方接口/本手册 | 已完成项目中的对应代码 | 作用 |
|---|---|---|
| Zoo 读取 A–E YAML | `run.py: GROUPS` 与 `train()` | 选算法、奖励和 HER |
| YAML 的 `policy_kwargs`、SAC/PPO 参数 | `train()` 的 `common` 与模型构造分支 | 创建 SB3 模型 |
| `custom_envs.ActionGain` | `run.py: ActionGain`、`make_env()` | 控制误差与环境创建 |
| 官方 `EvalCallback` | `ExperimentCallback._on_step()` | 原工程另外固定验证场景并保存恢复 RNG |
| `SAC.save`、`save_replay_buffer` | `save_bundle()` | 打包模型、回放、配置，最后写完成标记 |
| `SAC.load`、`load_replay_buffer` | `load_bundle()` | 评估加载或恢复训练 |
| 官方 `enjoy.py` + 项目逐回合统计 | `assess()`、`evaluate()` | 固定场景和五档增益，写 CSV |
| 官方录像 API | `video()` 与 `tools/make_case_video.py` | 原始回放与 90 秒组合视频 |

现有入口的等价 D 训练命令是项目命令，不是官方命令：

```bash
source /home/zhsu/jyluo/projects/mujoco-fetch-push-rl/env.sh
python run.py train --task Push --group D --seed 0 \
  --steps 1000000 --lr 0.0003 --buffer-size 1000000 --learning-starts 10000 \
  --eval-every 20000 --eval-episodes 20 --checkpoint-every 250000 \
  --device cpu --threads 2 --out runs/manual_project_D_seed0
```

旧正式结果具体执行的是同内容冻结文件 `source_materials/runner_frozen_20261022.py`；真实命令、配置和版本在 `logs/`、`runs/formal_20261022/*/config.json`、`system.json` 中。用户上传的服务器包已经含有 `run.py` 等实验代码；本地记录的原包修复是把 `ActionGain.step()` 的标量奖励转为 Python `float`，见 `source_materials/local_changes.json`。要区分“上游库提供的算法”“上传脚本已有的工程实现”和后续新增的评估、验收与交付，不把所有代码都称为从零编写。

## 14. 已完成结果如何如实表述

现有结果：D 正常成功率 98.7±1.2%，C 为 6.0%；E 正常成功率 97.0±3.6%；范围外 g=0.6/1.4 的均值 D=89.8±1.9%、E=86.7±8.5%。这里的 ± 是三个训练种子的样本标准差。

这些结果支持本配置下 HER 的效果，但未观察到随机化提高整体范围外成功率。6% 的弱基线也不能解释为学会推物：相同测试场景不施加动作已有 6% 成功率。新路线短验证仅说明官方入口与扩展能执行，不构成另一轮正式实验结果。

## 15. 阅读与动手顺序

先用第 2 节进入环境，读 `custom_envs.py` 和 `configs/C.yml` / `D.yml`；执行第 6.1 节短 D 验证，再用官方 `enjoy.py` 回放它；按 E 命令做短验证并确认训练/验证增益分开；最后再决定是否用第 6 节预算重做独立实验。已有 15 个正式模型可以直接用于理解第 9 节的评估 API，无需为了阅读手册重跑。

验收记录在 `official_readme/checks/`。这里没有引入新的训练协调器、模型服务或算法实现；训练与模型回放仍由官方 Zoo 入口完成。

### 本轮实际执行与验证范围

- A：官方入口执行 2,048 步并更新 PPO；B/C/D/E：各 300 步，短验证临时将 warmup 设为 100、buffer 设为 1,000，四组均实际更新。
- D/E 的回放对象为官方 `HerReplayBuffer`；E 使用项目增益环境，训练与验证构造参数分开保存。
- 官方 `enjoy.py` 已分别回放 D/E 短模型，各 100 个控制步。
- 官方 `record_video` 已通过自定义环境生成 51 帧、25 fps 的短视频。
- 新评估脚本重新加载已有正式 D seed0 模型，复测 2 档增益×3 场景；成功标签和距离与旧逐回合结果一致。
- 官方恢复命令已做短验证：加载模型和回放后再训练，核验更新继续、回放增加，以及本版本步数计数重置的行为。
- 官方源码与 tag 压缩包逐文件比对；项目扩展没有修改官方训练、回放或录像源文件。
- 所有 Bash 代码块检查语法；未执行 15 次百万步新矩阵，未执行需额外安装依赖的可选绘图命令，也未把短模型表现作为正式结果。

机器可读记录：`official_readme/checks/acceptance.json`；依赖快照：`official_readme/requirements-tested.txt`。详细日志和短模型仅用于验证手册命令。
