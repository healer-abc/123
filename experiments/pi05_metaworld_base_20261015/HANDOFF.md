# MetaWorld 五任务基础模型评测

更新时间：2026-10-15 21:10 CST。

> **已完成**：正式100次全部有效，五任务均0/20，总成功率0%。完成时间21:07:13；逐轮一致性核验通过；PID1753382已退出，其约9GB显存已释放。其他用户任务及LIBERO E2仍运行，不能据此说整张卡空闲。没有启动额外训练或评测。

用户新指令：删除上架任务 shelf-place-v2，改成 peg-insert-side-v2；共五任务各 rollout 20 次，使用未做任务微调的基础权重，首次达到环境成功条件即成功，自行实现环境接入接口。

## 已实施

- 固定 protocol.json：assembly-v2、peg-insert-side-v2、disassemble-v2、peg-unplug-side-v2、stick-pull-v2，各 20，共 100。
- 基线选用本项目已缓存的原始 pi05_base；不加载 LIBERO LoRA/全参结果，也不加载 MetaWorld SFT。
- 已精确核对并停止旧 pi0_metaworld_baseline_20261015 的 watchdog、bootstrap、setup_runtime、download；旧 SFT 管线不得重启，旧文件与已下载分块保留。
- adapter.py 是本次接入：单相机 180° 旋转并缩放到224；末端 XYZ 按工作空间范围缩放，夹爪张开量映射为[-1,1]；补齐32维后编码进离散 state prompt。
- 输出：模型前4维对应 dx/dy/dz/gripper，限幅到动作空间。**这是一种固定、未训练的直接映射；没有验证过的 Sawyer 专用 decoder/norm stats，结果不能独立证明基础模型任务理解能力。**
- run.py 保存每轮 JSON/NPZ/MP4、reset参数哈希、seed、首次成功步数、限幅计数、异常尝试；成功立即结束；每轮最多500步。
- 5个环境全部通过 reset、render、4D action、success字段检查，100条 episode manifest 已生成。
- 冒烟阶段已通过：assembly-v2完整500步，无接口/推理异常，success=false；耗时22.74秒，含首次编译。日志 logs/policy_smoke.log；不计入正式100轮。
- 成功停止规则单元测试2项通过（首次成功立即停止、未成功跑满预算）。
- 正式评测已启动：PID/SID 1753382，PPID=1，启动约20:47:24，日志 logs/run.log。
- 正式100/100全部通过 verify_results.py：seed/reset清单、原始预测到执行动作一致性、成功即停计数规则、视频和轨迹文件全部通过检查，无基础设施缺失。
- 共50000个策略控制步；episode时间合计1173.58秒（19.56分钟），所有任务均跑满500步后失败。五任务各20次成功数都是0。
- 共200000个执行动作分量，其中32273个限幅（16.14%）。不能据此把低成功率归因于某一个模型机制。
- 逐任务Wilson95%区间为[0,16.11%]，20次样本不能证明总体成功率严格为0。
- 跨进程数值诊断：smoke和正式assembly seed42初始状态相同，但BF16预测第一块最大差0.00488，完整轨迹存在小幅差异；固定seed不等于跨进程逐bit一致。

## 后续执行

1. 本次请求已完成；结果入口 results/report.md、results/summary.csv、results/validation.json。
2. adapter.py和run.py为独立接口及可复跑驱动；源文件SHA256在manifests/source_hashes.json。
3. 不因0成功自动改动作映射、切换SFT权重或新增训练；如用户另要求诊断，先使用已存视频/轨迹。
4. LIBERO E1/E2 的后续评测仍暂停，不因这次 MetaWorld 授权恢复。

## 运行命令

```bash
cd /home/zhsu/jyluo
env CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 PYTHONUNBUFFERED=1 projects/openpi/.venv/bin/python -u experiments/pi05_metaworld_base_20261015/run.py
```

## 权重标识

- 路径：myopenpi/dataset/ckpt/openpi-assets/checkpoints/pi05_base/params
- manifest.ocdbt SHA256：78b6898ea1f8897a0225022b0ab799455cff0e15158a9ca349d8efbbe044aa55
- _METADATA SHA256：250d0ad0f6540a5e4bdd70442d6864cef729724f2d23e2dd2cc669563a668d70

旧 LIBERO 交接位于 ../pi05_libero_adaptation/HANDOFF.md。
