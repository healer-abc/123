> 新进度：用户已授权从500轮完整状态续训到1000轮，每100轮保存。[查看DP1000续训进度](../metaworld_dp_resume1000_20261020/progress.md)。下方为保留的500轮实验结果。

# DP独立500轮实验

从头训练seed0；50示教（40训练/10验证）；保留原1000轮学习率计划。
已运行任务保留原保存逻辑；新启动的stick-push/door-open仅保存500轮最终权重及续训状态，最终评测仍各20局。
原DP1000进程及看门狗已按用户指令结束，已有磁盘权重另存于preserved_baseline；SimpleDP3评测已完成。
各快照另存学习率调度、EMA计数、随机数等续训状态；500轮后停止训练，等待用户指令才续训。

状态：complete

| 任务 | 阶段 | 当前epoch（约） | 正式评测 |
|---|---|---:|---|
| shelf-place | complete | 499.0 | 5/20 |
| pick-place-wall | complete | 499.0 | 2/20 |
| pick-place | complete | 499.0 | 3/20 |
| stick-push | complete | 499.0 | 6/20 |
| door-open | complete | 499.0 | 20/20 |

## 已保存的独立快照

- shelf-place，完成300轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/shelf-place/attempt_01/checkpoints/epoch_0300.ckpt)
- shelf-place，完成400轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/shelf-place/attempt_01/checkpoints/epoch_0400.ckpt)
- shelf-place，完成500轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/shelf-place/attempt_01/checkpoints/epoch_0500.ckpt)
- pick-place-wall，完成300轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/pick-place-wall/attempt_01/checkpoints/epoch_0300.ckpt)
- pick-place-wall，完成400轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/pick-place-wall/attempt_01/checkpoints/epoch_0400.ckpt)
- pick-place-wall，完成500轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/pick-place-wall/attempt_01/checkpoints/epoch_0500.ckpt)
- pick-place，完成300轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/pick-place/attempt_01/checkpoints/epoch_0300.ckpt)
- pick-place，完成400轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/pick-place/attempt_01/checkpoints/epoch_0400.ckpt)
- pick-place，完成500轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/pick-place/attempt_01/checkpoints/epoch_0500.ckpt)
- stick-push，完成300轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/stick-push/attempt_01/checkpoints/epoch_0300.ckpt)
- stick-push，完成400轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/stick-push/attempt_01/checkpoints/epoch_0400.ckpt)
- stick-push，完成500轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/stick-push/attempt_01/checkpoints/epoch_0500.ckpt)
- door-open，完成500轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_epoch500_20261020/runs/door-open/attempt_01/checkpoints/epoch_0500.ckpt)
