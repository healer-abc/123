# MetaWorld 50 示教实验交接

用户最新授权：五个场景分别使用50条成功示教，并允许灵活安排；不加入模型改进。旧10示教轮已停止，数据/模型/结果保留，不能恢复旧队列或将其成绩填入新版表格。

当前根目录：`experiments/metaworld_official_baselines_50demo_20261019/`。旧根目录：`../metaworld_official_baselines_20261019/`。完整协议见 `PROTOCOL.md`、`protocol.json`；共享40-train/10-validation划分见 `manifests/dataset_split.json`。官方源码与ACT视觉骨干权重软链接复用旧轮已冻结产物，不复制或训练改进工作区模型。

## 已做

- 已核对并SIGTERM停止旧看门狗1975066，确认其训练worker均退出。旧状态保存在旧目录`logs/status_before_50demo_migration.json`。
- 建立新版独立源码、数据、模型和结果目录，生成新的执行文件哈希。
- 每场景复用已核验的10条，加官方专家新采40条（seed43），合并为50并检查每个像素/状态/动作、episode边界和初态唯一性。尚未完成采集时，不得声称已有250条。
- 统一40/10划分，与官方ACT 50示教原生划分一致。DP/SimpleDP3改用相同episode ID；不再有旧轮8-vs-9训练条数不一致。
- 全部15组重新初始化训练；ACT保留官方视觉预训练。DP8000epochs/batch64、SimpleDP33000/batch128、ACT2000/batch8。ACT现在有10000次optimizer更新。
- 减少开发评测：DP/SimpleDP3每500epochs ×5开发episodes；DP每20epochs验证，采样诊断每100epochs；最终仍每组20次。模型、loss、optimizer、LR schedule、完整训练epoch预算保持不变。
- 加入持久数据加载worker与开发评测RNG隔离；调度防止ACT饥饿，最多4并发、每GPU最多2，按实占和峰值预算预留8GiB左右。
- 9项CPU回归测试已通过，覆盖共享40/10划分、方法调度、拒绝10示教旧结果、最终记录/初态/checkpoint一致性与告警时间字段。

## 当前后台

新版看门狗首次PID：`2210046`，以`logs/watchdog_pid.json`和真实命令行为准。已启动数据采集队列；若共享GPU无足够显存，将等待，不杀其他任务。

看门狗入口`baseline_watchdog.py`；资源环境由`runtime.environment(gpu)`提供，使用`/home/zhsu/dp3_ablation_v52/env/bin/python`。不可直接使用系统Python运行训练，也不可恢复OpenPI/Libero暂停实验。

状态`logs/status.json`，事件`logs/events.jsonl`，告警`logs/latest_alert.json`，每任务`runs/<job>/attempt_*/worker.log`。数据完成后，三个方法分别通过真实full-batch smoke才可正式训练。自动将最终成绩写到`report/results.md`与CSV；旧ACT 0/20不计入本轮。

修改运行中执行文件前必须先停本轮看门狗；修复后测试并运行`prepare.py`重新冻结，保留所有原始尝试。未知错误停止并诊断，不擅自调模型刷分。

## 2026-10-19 17:08:07 CST 现场核验

看门狗2210046健康运行，无告警。受其他共享GPU任务占用影响，仅启动两路追加40条采集：shelf-place PID2210240、pick-place-wall PID2210241，均已接收至少8条新增成功示教（仍在采集中，未完成50条整库）；其余三场景等待资源。原10条全量数据和旧ACT结果保留，旧看门狗状态stopped_by_signal。

新增独立CPU端到端数据检查通过：10+40整episode无损合并、重复初态拒绝、DP/SimpleDP3实际sampler均严格采到相同40训练/10验证ID，证据`audits/merge_and_split_test.json`。正式训练尚未开始；每场景数据完成与方法full-batch smoke通过后由看门狗自动启动。

## 2026-10-19 18:35:57 CST 数据完成，正式训练启动

五个场景各50条、共250条成功示教已完成，均有合并凭证和ACT视图。18:33现场发现新版训练仍因8GiB显存预留过于保守而全部等待；已在无活动worker时安全停止旧看门狗2210046，保存变更前证据，按实测小模型峰值修改资源准入（RESERVE4096，ACT3584、SimpleDP34096、DP12000MiB）。10项CPU测试通过，重新prepare冻结，确认protocol.json科学配置完全不变；新看门狗2273945。

ACT和SimpleDP3已在真实50条数据上通过full-batch训练/推理smoke，峰值分配分别1502.6、1550.5MiB。正式ACT_shelf-place在GPU1/PID2274141（约20/2000epoch），SimpleDP3_shelf-place在GPU0/PID2274142（完成epoch0 batch，进入5次开发rollout）。DP仍等待足够显存。其余场景按方法平衡队列接续。当前无告警，最终0/15组；资源实际余量GPU0约5.1GiB、GPU1约6.2GiB。模型与数据协议未修改，旧10示教ACT成绩不混入本轮。

## 2026-10-19 22:29:00 CST 进展与工期核验

新版最终5/15组（100/300 episodes）完成，均为ACT：shelf-place1/20、pick-place-wall0/20、pick-place3/20、stick-push20/20、door-open17/20，平均41%。四个SimpleDP3同时训练：shelf-place约1659/3000，pick-place-wall1175/3000，pick-place952/3000，stick-push521/3000；door-open排队。GPU0近期约8.24s/epoch，GPU1约4.55–4.64s/epoch；结合剩余开发/最终评测与最后一个场景排队，SimpleDP3全部完成初估还需7–11小时（共享负载变化会改变该范围）。

DP的smoke和全部五任务仍未启动，等待空闲显存达到当前预算12000+4096MiB。现场余量GPU0约12757、GPU1约13209MiB，尚未满足准入；其他四个slot由SimpleDP3占用。没有当前50示教DP实际步速，也不知道外部显存何时释放，不能给全15组一个可靠完工时间。最初60–84小时估计已作废，不应继续转述。看门狗健康、无新增告警，不修改训练配置或干扰其他任务。
