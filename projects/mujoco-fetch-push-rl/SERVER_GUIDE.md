# FetchPush 项目：共享服务器执行手册

适用信息：你提供的 GPU 名称为 RTX 6000D、显存约 85 GB，并且可以使用 conda 和 tmux。系统暂按 Linux x86_64；驱动、CPU、内存、磁盘配额与资源分配方式还需要实机确认。本包是起跑用的实验脚手架，不含已经训练好的模型或成功率结果。

执行顺序：检查资源 → 独立环境 → 无渲染仿真检查 → 短训练与恢复检查 → 速度测试 → Push 单种子预实验 → 正式多种子实验 → 独立评估与录像。

## 1. 放置文件、检查资源

使用你已经配置好的 SSH 客户端登录。地址、账号、端口尚未提供，因此这里不编造登录命令。将下载的 `fetch-server-kit.zip` 上传到自己的服务器目录并解压，把其中的 `fetch-server-kit` 目录改名或放置为 `~/projects/mujoco-fetch-push-rl`。不要覆盖已有同名项目。

在服务器终端运行：

```bash
cd ~/projects/mujoco-fetch-push-rl
pwd
ls
uname -m
cat /etc/os-release
nvidia-smi
lscpu
free -h
df -h .
quota -s
conda --version
tmux -V
command -v sbatch
```

`quota` 不存在或未配置不影响开发，但需确认个人可用空间。建议准备至少 30 GB 空闲磁盘、16 GB 系统内存；85 GB 显存不等于系统内存。单次实验使用一个环境、两个 PyTorch CPU 线程，先不并行跑多个实验。

如果服务器通过 Slurm 分配资源，在获配的计算节点上执行训练，沿用调度器设置的 GPU 可见范围；tmux 不能代替作业分配，也不能延长配额时间。下面 tmux 持续训练的命令默认是实验室允许直接运行作业的共享计算服务器。

多卡且没有调度器时，只选择分配给你的卡。先用 `nvidia-smi -L` 确认索引；例如仅当分给你的卡是物理 0 号卡时，使用：

```bash
export CUDA_VISIBLE_DEVICES=0
```

筛选后，程序中 `cuda:0` 指第一张可见卡，不一定是整机物理 0 号卡。

## 2. 建立项目专用环境

```bash
conda create -n fetch-rl python=3.11 pip -y
conda activate fetch-rl
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > requirements-lock.txt
```

这里固定 PyTorch 2.7.1 / CUDA 12.8 wheel，以及 SB3 2.7.0、Gymnasium 1.2.0、Gymnasium-Robotics 1.4.0、MuJoCo 3.3.4，目的是保持这一批实验一致。这是待服务器验证的依赖组合，不是“已在你的服务器测试通过”。PyTorch 安装命令来自官方历史版本页；驱动兼容性和 GPU 架构支持还需下面的实际运算检查。

```bash
python -c "import torch; print('torch:', torch.__version__); print('wheel CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0)); x=torch.randn(256,256,device='cuda'); print((x@x).mean().item()); torch.cuda.synchronize()"
```

必须看到 CUDA 可用、设备名称和正常运算结果。若失败，保留完整报错与 `nvidia-smi` 输出，按驱动版本重新选官方 wheel；先用 `--device cpu` 继续检查环境，无需先修改服务器驱动或安装全局 CUDA Toolkit。

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p logs
```

环境名已存在时先 `conda env list` 查看，不要直接删除或重建别人的环境。每次重新登录均需激活 `fetch-rl`，再进入项目目录。

## 3. 先确认环境、训练和恢复都能运行

训练默认不渲染，不需要窗口或 `DISPLAY`。先检查：

```bash
python run.py check --task Reach
python run.py check --task Push
```

Push 应显示 `observation: (25,)`、两个目标字段均为 `(3,)`、动作空间为 `(4,)`。环境检查会实际执行 reset/step 与 SB3 接口检查。

用 3000 步、较小回放池做 CPU 短跑。100 步后开始更新，确保覆盖 HER 完整回合采样与网络更新流程：

```bash
python -u run.py train --task Reach --group D --seed 99 --steps 3000 --buffer-size 20000 --learning-starts 100 --eval-every 1000 --eval-episodes 5 --device cpu --out runs/reach_smoke
python run.py eval --checkpoint runs/reach_smoke/final --episodes 10 --seed-base 10000 --out evaluations/reach_smoke
python -u run.py train --resume runs/reach_smoke/final --steps 5000 --eval-every 1000 --eval-episodes 5 --device cpu --out runs/reach_resume
```

这个阶段只检查流程；低成功率可以接受。`--steps` 始终代表累计目标步数，所以恢复到 5000 步是在已有 3000 步基础上再跑约 2000 步。恢复时继承原配置，包括回放容量和预热步数；因此 smoke 模型不用于正式实验。

确认 `final/model.zip`、`final/replay.pkl`、`final/complete.json`、`validation.csv` 和 `summary.json` 生成。输出目录已存在会报错，避免覆盖；再次尝试时明确换一个新目录名。

## 4. 检查 CPU / GPU 速度，再跑 Push 预实验

```bash
python -u run.py train --group D --seed 99 --steps 50000 --eval-episodes 5 --device cpu --out runs/speed_cpu
python -u run.py train --group D --seed 99 --steps 50000 --eval-episodes 5 --device cuda:0 --out runs/speed_gpu
cat runs/speed_cpu/summary.json
cat runs/speed_gpu/summary.json
```

比较 `session_steps_per_second`。其中包括预热、训练、评估和保存，所以适合估算真实用时；50000 步中前 10000 步是预热，估算百万步耗时还要参考后续较长预实验。小网络不保证 GPU 更快，MuJoCo 物理仿真仍在 CPU 运行。

以下预实验以 GPU 为例；若 CPU 更快，将 `--device cuda:0` 改为 `--device cpu`。

```bash
tmux new -s fetchpilot
```

进入 tmux 后，确认 `conda activate fetch-rl` 并进入项目目录，再运行：

```bash
set -o pipefail
python -u run.py train --group D --seed 99 --steps 300000 --device cuda:0 --out runs/push_pilot 2>&1 | tee logs/push_pilot.log
```

按 `Ctrl+B`，松开后按 `D`，可离开 tmux，训练继续。重新登录后用 `tmux attach -t fetchpilot` 回到会话。直接关闭 SSH 不会关闭 tmux 中的进程；服务器重启或调度器终止资源会停止进程。

另一终端可查看：

```bash
tail -n 30 logs/push_pilot.log
tail -n 8 runs/push_pilot/validation.csv
nvidia-smi
```

看实际步数是否增长、成功率和距离是否改善、损失是否出现 NaN。300000 步时成功率为 0 不足以断言算法失效，先确认确实有网络更新、目标与奖励正确、评估无误，再决定延长或调参。

需要延长到累计 100 万步时：

```bash
python -u run.py train --resume runs/push_pilot/final --steps 1000000 --device cuda:0 --out runs/push_pilot_extended 2>&1 | tee logs/push_pilot_extended.log
```

预实验固定 seed 99；正式实验用 0、1、2，不把调参种子的最好结果当正式结论。若修改学习率，可用 `--lr` 启动新的预实验；正式矩阵也必须同步修改参数，不能拿原默认值当作调整后的配置。现有默认参数未做任务专项优化。

## 5. 先完成核心四组，再做扩展组

| 组 | 方法 | 奖励 | 动作增益 |
|---|---|---|---|
| A | PPO | 密集 | 1.0 |
| B | SAC | 密集 | 1.0 |
| C | SAC | 稀疏 | 1.0 |
| D | SAC + HER | 稀疏 | 1.0 |
| E | SAC + HER | 稀疏 | 每回合从 [0.8,1.2] 抽取 |

单种子预实验通过、配置冻结后，新建一个 tmux 会话：

```bash
tmux new -s fetchcore
```

激活环境、进入项目目录后运行：

```bash
bash run_matrix.sh core cuda:0 1000000
```

此命令按 D、C、B、A 顺序执行，每组 seed 0、1、2，共 12 次训练、约 1200 万步，始终只有一个训练进程。A 组固定在 CPU 上跑；SAC 各组使用你指定的设备。若选择 CPU：

```bash
bash run_matrix.sh core cpu 1000000
```

上面两条择一执行。不要同时启动两套矩阵。PPO 会按完整 rollout 执行，实际步数可能略超 100 万，报告以实际步数为准。

第 3 周再执行：

```bash
bash run_matrix.sh extension cuda:0 1000000
```

E 组再增加 3 次训练、约 300 万步。五组总预算约 1500 万步，预实验另算。若实测 100 步/秒，总训练约 42 小时；300 步/秒约 14 小时，但不同算法速度不同，应分别估算。PPO/CPU 与 SAC/GPU 的时间对比应明确硬件差异，不能声称是纯算法效率差异。

矩阵遇到失败立即停止；已有同名目录也会停止，不会自动覆盖或跳过。修复后显式运行剩余组和种子；恢复任务放入新目录，汇总时将其与原训练阶段关联，不能当成新增独立种子。

## 6. 保存、停止和恢复

每 25 万步保存一次完整检查点，包含模型、SAC 回放池、配置和完成标记；正常结束另外保存 `final/`。程序收到一次 Ctrl+C 或 SIGTERM 后，在下一次回调停止并写入 `interrupted/`，保存可能需要等待。强制结束、断电或磁盘写满时不能依赖这次保存，应使用上一个带 `complete.json` 的目录。

例如 A–D 预实验中的 D 组在 25 万步后意外中止，检查完整性：

```bash
ls runs/push_pilot/checkpoints/step_250000
cat runs/push_pilot/checkpoints/step_250000/complete.json
```

恢复到累计 100 万步：

```bash
python -u run.py train --resume runs/push_pilot/checkpoints/step_250000 --steps 1000000 --device cuda:0 --out runs/push_pilot_recovered
```

恢复会重置仿真环境，并对 HER 未完成的回合做截断处理。它是带历史经验的继续训练，不是仿真状态和随机数逐位相同的精确续跑。正式结果记录恢复事件；论文式严格对照优先使用完整连续运行。

每个 SAC 完整回放池约为数百 MB，四个周期检查点加 final 会重复占用磁盘；15 次训练可能产生数十 GB 文件。每晚检查：

```bash
du -sh runs
df -h .
```

不要将回放池当模型上传到 Git；评估和录像只需要检查点中的 `model.zip`、`config.json` 和 `complete.json`，继续训练才需要 `replay.pkl`。

## 7. 最终评估、录像和结果下载

训练中验证默认每 2 万步、20 回合，重置种子固定为 10000–10019。最终评估使用 20000–20099，不用于选择参数。成功率按第 50 步的 `is_success` 统计，另存 `ever_success` 区分曾经到达目标后又推过头。

先评估一个模型：

```bash
python run.py eval --checkpoint runs/D_seed0/final --episodes 100 --gains 0.6 0.8 1.0 1.2 1.4 --out evaluations/D_seed0
```

核心四组的最终正常环境评估：

```bash
set -e
for group in A B C D; do
  for seed in 0 1 2; do
    python run.py eval --checkpoint "runs/${group}_seed${seed}/final" --episodes 100 --out "evaluations/${group}_seed${seed}_nominal"
  done
done
```

D 和 E 的鲁棒性对照：

```bash
set -e
for group in D E; do
  for seed in 0 1 2; do
    python run.py eval --checkpoint "runs/${group}_seed${seed}/final" --episodes 100 --gains 0.6 0.8 1.0 1.2 1.4 --out "evaluations/${group}_seed${seed}_robust"
  done
done
```

每个输出目录有逐回合 `episodes.csv` 和汇总 `summary.json`。先分别计算每个训练种子的成功率，再跨三个种子计算均值和标准差；不要把 300 个测试回合当作 300 次独立训练。

录制视频是单独步骤，仅此时启用离屏渲染：

```bash
MUJOCO_GL=egl python run.py video --checkpoint runs/D_seed0/final --episodes 3 --gain 1.0 --out videos/D_seed0_nominal.mp4
MUJOCO_GL=egl python run.py video --checkpoint runs/E_seed0/final --episodes 3 --gain 0.6 --out videos/E_seed0_gain06.mp4
```

EGL 失败先保留报错；训练不需要 EGL。若服务器具备 OSMesa 库，可将 `MUJOCO_GL=egl` 改为 `MUJOCO_GL=osmesa` 录制。渲染后端问题与策略训练问题分开处理，不用 `render_mode='human'` 在服务器上开窗口。

用已有 SSH 客户端下载代码、配置、锁文件、`validation.csv`、`summary.json`、`evaluations/` 和 `videos/`。图表至少包含学习曲线、实际耗时曲线、D/E 鲁棒性曲线。本包暂不包含绘图和报告生成器。

可选 TensorBoard：

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

使用你已有 SSH 客户端，将本地 6006 端口转发到服务器 `127.0.0.1:6006`，然后在本机浏览器打开 http://127.0.0.1:6006 。地址和端口转发信息尚不完整，因此这里不写含虚构服务器地址的 SSH 命令。

## 8. 30 天安排和停止条件

| 时间 | 操作 | 验收 |
|---|---|---|
| 第 1 天 | 硬件检查、独立环境、CUDA 运算、Reach/Push 接口检查 | 无接口和依赖错误 |
| 第 2 天 | Reach 3000 步训练、加载评估、恢复到 5000 步 | 模型和回放池可恢复 |
| 第 3 天 | CPU/GPU 各 50000 步测速 | 得到真实吞吐率和设备选择 |
| 第 4–7 天 | D 组 seed 99 的 Push 预实验；必要时延长和有限调参 | 成功率或距离有可重复改善 |
| 第 8–14 天 | 冻结设置，运行 A–D 共 12 次正式实验 | 曲线、配置与原始日志齐全 |
| 第 15–21 天 | 阅读并验证动作增益模块，跑 E 组，做 D/E 五档增益评估 | 鲁棒性对照 |
| 第 22–30 天 | 汇总三种子统计、录像、报告与面试准备 | 可复现交付 |

无服务器的 2–3 天可以先读脚本和课程对应概念；以上“第 1 天”从获得服务器使用权开始计算。第 7 天仍卡在基础训练时暂停正式矩阵；第 14 天尚无可重复学习趋势时延期 E 组，先完成核心诊断。高显存不能保证策略学会，预实验通过后再投入批量预算。

## 来源与验证范围

- 环境：[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
- 算法：[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- HER：[官方接口与保存加载说明](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)
- 安装：[PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)
- 仿真：[MuJoCo Python 说明](https://mujoco.readthedocs.io/en/stable/python.html)

生成时做了 Python 语法检查；本地没有 PyTorch、MuJoCo 或服务器连接，因此没有执行真实训练、CUDA 运算或离屏录像。服务器上的接口检查、短训练和恢复检查是必须完成的首次验收步骤。脚手架没有实现论文级完全确定性恢复、自动调参、绘图、作业调度器集成和最优模型选择；正式报告使用固定预算结束时的模型。
