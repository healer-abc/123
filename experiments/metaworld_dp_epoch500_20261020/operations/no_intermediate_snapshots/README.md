# 用户取消剩余任务的中间快照要求

用户询问GPU占用和训练速度，并表示剩余两个任务不需要保存快照。按stick-push、door-open处理，500轮最终模型和续训状态仍保留。

## 实际应用范围

- 新启动的stick-push/door-open使用独立`replay_terminal_only.py`：关闭官方周期latest和snapshot保存，仅完成500轮时同步保存一份模型/EMA/优化器及续训状态。`final.ckpt`是该文件的硬链接，避免重复写盘。
- 当前stick-push进程3000994的快照分支已在内存中编译，不会因新增文件而改变。为保留已完成的训练进度，没有重启它；其原先300/400轮自动保存仍会执行。若发生必须重试的基础设施错误，下一次attempt才采用terminal-only规则。
- door-open尚未启动，将直接采用terminal-only规则。
- pick-place进程2983254继续原配置。原已完成两项和所有已存数据均不覆盖。
- 原看门狗2942600已结束；新看门狗3016031从内存状态接管两个原worker，worker PID未变。只结束了本实验自己的旧supervisor，没有停止worker或其他实验。
- 新看门狗入口为本目录`supervisor.py`。不要另行运行根目录`run.py`，它的旧调度规则会与本覆盖重复。

## 为什么慢

核验时GPU0除本轮DP外还有六个其他MetaWorld消融训练，合计约38GiB显存；GPU1有另一位用户的VITRA训练，约50GiB显存。GPU利用率约90–100%，实际存在算力共享。没有把这些其他任务当成本轮任务终止。

DP近期每轮约14–17秒。已有pick-place任务的300/400轮快照保存前后日志间隔约11.05/10.36秒（含下一训练批次），中途快照不是数小时耗时的主要来源。取消新任务的这些保存主要减少磁盘写入，不能消除算力竞争。

## 审计与恢复

- 原`plan.json`及三份执行源码保持原SHA，原科学配置和评测seed完全不变。
- `manifest.json`另行冻结新终点保存driver和本目录supervisor的SHA；`terminal_driver.diff`记录纯保存行为差异。
- 接管前的状态保存在`status_at_handover.json`，新PID记录在`launch_receipt.json`及根目录`supervisor_pid.json`。
- 新driver同样核验最终500轮、30500次更新、模型/EMA/优化器和续训状态，最终评测仍20局、DDIM10、horizon16、obs2、action8。
- 总审计按各任务实际保存的epoch列表报告，不把terminal-only任务标成保存过300/400轮。
- 500轮后不自动续训；沿用用户既定要求。
