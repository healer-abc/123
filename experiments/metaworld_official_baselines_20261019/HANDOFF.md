# MetaWorld DP / ACT / SimpleDP3 baseline 交接

用户于 2026-10-19 授权：五个场景分别评测 DP、ACT、SimpleDP3，每组最终 rollout 20 次，汇总成功率。必须保持原版算法，不做任何模型改进。允许根据实际空闲显存并发；后台看门狗检测错误并保留告警。

## 协议与边界

任务：`shelf-place`、`pick-place-wall`、`pick-place`、`stick-push`、`door-open`。15 组，最终 300 次 rollout。截图中的数字不得作为本次结果。

详见 [PROTOCOL.md](PROTOCOL.md) 和冻结的 `protocol.json`。官方源归档与执行文件指纹：`source_manifest.json`。新实验完全独立于 `/home/zhsu/dp3_ablation_v52` 的改进工作区；只复用其兼容的 Python/CUDA/MuJoCo 运行依赖，不加载其改进模型或 checkpoint。`pi05_libero_adaptation` 继续暂停，不得恢复。

官方 DP3 脚本采集每场景 10 条成功示教，共享给三种方法。DP 使用官方 image U-Net 配置 8000 epochs/batch64，SimpleDP3 官方配置 3000 epochs/batch128，ACT 官方仿真示例 2000 epochs/batch8。seed0。不可提前停止，不可为省时改训练预算或 batch size。

DP/ACT 缺少官方 MetaWorld 配方，因此须准确称为“官方算法的 MetaWorld 接口适配 baseline”。DP 数据/runner 适配，ACT 9维robot state/4维action适配；完整差异见 `act_metaworld_dimensions.patch`。不能宣称是官方已发布 MetaWorld 数字的直接复现。

最终固定20个独立 seeds，200步、官方 success 曾为真即成功。DP/SimpleDP3 固定terminal EMA，ACT官方best validation checkpoint。开发周期评测保留但不进入最终表格；不以最终成功率挑 checkpoint 或重刷失败 episode。

## 已启动

- 后台看门狗首次 PID：`1971037`；修复后当前看门狗 PID：`1975066`。以 `logs/watchdog_pid.json` 与真实 `/proc` 命令行核对，不能只凭历史 PID 判断活动。
- 看门狗已接管预检采集 worker；最初四路采集分布于GPU0/1，按空闲显存自动补队列。正式训练受各方法 full-batch smoke 门槛约束。
- 首个 `shelf-place` 的10条官方示教已采集完成，并无损转换ACT HDF5格式。
- 已通过：官方DP workspace/ACT policy导入、离线W&B日志hook真实写入、3种方法的配对reset（simulator state、goal、full_state一致）、6项CPU管线测试。
- 本实验预检发现并修复：MuJoCo默认EGL硬件设备权限失败，运行时探测并选取唯一Mesa软件设备；本地 `watchdog.py` 命名曾遮蔽W&B依赖，已改为 `baseline_watchdog.py`；ACT原版DETR绝对导入需要额外`runtime_act/detr`路径。
- 官方ACT ImageNet ResNet18已下载并校验SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

## 查看进度与结果

```bash
cat experiments/metaworld_official_baselines_20261019/logs/status.json
tail -f experiments/metaworld_official_baselines_20261019/logs/watchdog.log
cat experiments/metaworld_official_baselines_20261019/report/results.md
```

状态：`logs/status.json`；事件：`logs/events.jsonl`；告警：`logs/alerts.jsonl` / `logs/latest_alert.json`。单任务日志、指标、模型、视频、逐episode原始记录都在 `runs/<job>/attempt_*/`。最后自动汇总到 `report/results.md` 与 `report/results.csv`，未完成格子明确标待完成。

不要改动正在执行的代码。需要修复时先停止本看门狗，让其清理自己启动的进程组，再修复、重新测试、运行 `prepare.py` 冻结新指纹并恢复。已完成任务不会自动重跑；失败尝试不能覆盖或删除。未知错误和非有限值会暂停等待诊断；后台不会凭空生成修复代码。

看门狗：初始全局最多4 worker，每GPU最多3；每15秒检查；预留10000MiB，低于3500MiB释放本实验自己的worker；1200秒无日志或心跳告警。基础设施错误最多2次独立尝试，OOM降低并发，不改模型或batch。不干扰其他用户进程。

安全停止：先核对 `logs/watchdog_pid.json` 指向的是本目录的 `baseline_watchdog.py`，再向该PID发SIGTERM，由它清理本实验worker。恢复入口为本目录 `baseline_watchdog.py`，需使用 `runtime.environment(0)` 启动指定 `/home/zhsu/dp3_ablation_v52/env/bin/python`。不使用其他实验的启动器。

## 2026-10-19 预检修复与正式训练启动

完整batch预检发现官方生成器磁盘图像为HWC，初版DP/ACT转换误按CHW。已暂停全队列后修复DP的HWC→CHW、ACT保留HWC；原始zarr未改变，40条完成示教的ACT图像/动作逐条对原始数据相等。旧错误转换目录保存在各collection attempt的`act_conversion_channels_failed/`。告警时间字段重复的异常已修复并加入回归测试。

DP/ACT/SimpleDP3的full-batch前向、反向、optimizer step和推理均真实通过：PyTorch峰值分配5731.8 / 1502.6 / 1550.5 MiB。证据`audits/full_batch_verification.json`。ACT_shelf-place、ACT_pick-place-wall、SimpleDP3_shelf-place已开始完整正式训练；DP等待队列槽位，door-open采集第二次尝试（第一次9条时被看门狗全队列清理，未完整数据已隔离）。四个场景示教已完整，告警最新状态resolved。所有修复均在正式训练开始前完成，未做模型改进。

## 2026-10-19 15:34:50 CST 当前核验

五场景50条成功示教及其ACT视图均完成。看门狗`1975066`运行；GPU0: ACT_shelf-place PID1975413、SimpleDP3_shelf-place PID1975414、DP_shelf-place PID1988030；GPU1: ACT_pick-place-wall PID1975415。DP和SimpleDP3已真实完成epoch0训练batch并进入官方周期开发rollout，ACT持续训练。其余组自动排队。最新告警resolved；0/15组完成最终评测，不能报告最终成功率。已真实验证重复启动互斥，未影响活动看门狗。

## 2026-10-19 16:41:13 CST 进度核验

最终评测完成2/15组（40/300）：ACT shelf-place、pick-place-wall均0/20。两组完整2000epoch及原版最小validation loss checkpoint记录完整。抽查各episode00的5个视频帧，可见真实渲染和机器人动作但未完成任务；仅抽样检查，不声称已排除全部接口问题，不调模型或刷分。接触表`audits/act_final_video_review.png`。

当前四worker：SimpleDP3 shelf-place约1000/3000epoch、pick-place-wall约840/3000、pick-place约600/3000；DP shelf-place约77/8000。其余九组排队。无新增告警，看门狗正常。GPU0其他实验占用增大，总占用约70GiB；DP最近一轮20次开发rollout中位每episode约102秒（初测约30秒）。此前总工期60–84小时的早期估计已不可靠，需要延后并按后续共享资源重估；此前ACT全场景1–2小时也未计入当前队列等待，不能继续当承诺。

## 2026-10-19 16:53:57 CST 用户要求核验已完成

只读核验通过并发现协议限制。报告`report/implementation_audit.md`；机器证据`audits/pipeline_verification/report.json`、`audits/control_flow_verification/report.json`。755个执行文件哈希全部一致、50条示教ACT转换逐像素/状态/动作一致、200个ACT数据时刻与15个DP序列时序/归一化通过；ACT两项最佳checkpoint逐tensor等于官方保存epoch，官方与适配推理最大输出差异<3e-6；两条原始专家动作完整真实回放成功，状态误差<3.4e-8；两任务各两个最终seed专家正向对照4/4成功；真实rollout循环的合成控制流/瞬时成功计数测试通过。未发现可将ACT 0/20归为加载/控制/评分错误的证据，保留当前记录；未定位唯一失败根因。

协议限制必须随结果披露：ACT train8/val2，DP/SimpleDP3 train9/val1；ACT当前10条示教×2000epoch仅2000更新，官方50条仿真示例约10000更新；chunk100来自ALOHA示例，非MetaWorld标准配方；旧截图改进实验数据/划分/初态身份未匹配。不能把本轮称为已证明与改进模型公平的正式对照或官方MetaWorld数字复现。不擅自改训练或重刷结果。当前配置和活动worker未修改。

## 已被50示教实验替代

用户明确要求每场景改为50条示教并允许灵活安排。旧看门狗和训练已停止；旧数据、模型和两项ACT评测全部保留，不计入新版结果。新工作目录：`../metaworld_official_baselines_50demo_20261019/`。不得自动恢复本10示教队列。
