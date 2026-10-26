# DP500→1000轮续训与评测

恢复500轮的模型、EMA、优化器、学习率、EMA计数、随机数和训练采样批次；原1000轮学习率计划。
累计600/700/800/900/1000轮保存独立快照；去噪10步、horizon16、obs2、action8、训练seed0。
1000轮固定终点EMA，每项20局，同一评测seed；500轮结果保留。

状态：complete。

| 任务 | 当前累计epoch（约） | 500轮结果 | 1000轮结果 | 状态 |
|---|---:|---:|---:|---|
| shelf-place | 1000 | 5/20 | 3/20 | complete |
| pick-place-wall | 1000 | 2/20 | 6/20 | complete |
| pick-place | 1000 | 3/20 | 4/20 | complete |
| stick-push | 1000 | 6/20 | 5/20 | complete |
| door-open | 1000 | 20/20 | 19/20 | complete |

## 已保存快照

- shelf-place，600轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_0600.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_0600_resume.pt)
- shelf-place，700轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_0700.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_0700_resume.pt)
- shelf-place，800轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_0800.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_0800_resume.pt)
- shelf-place，900轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_0900.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_0900_resume.pt)
- shelf-place，1000轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_1000.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/shelf-place/attempt_02/checkpoints/epoch_1000_resume.pt)
- pick-place-wall，600轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_0600.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_0600_resume.pt)
- pick-place-wall，700轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_0700.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_0700_resume.pt)
- pick-place-wall，800轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_0800.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_0800_resume.pt)
- pick-place-wall，900轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_0900.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_0900_resume.pt)
- pick-place-wall，1000轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_1000.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place-wall/attempt_02/checkpoints/epoch_1000_resume.pt)
- pick-place，600轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_0600.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_0600_resume.pt)
- pick-place，700轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_0700.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_0700_resume.pt)
- pick-place，800轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_0800.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_0800_resume.pt)
- pick-place，900轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_0900.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_0900_resume.pt)
- pick-place，1000轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_1000.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/pick-place/attempt_02/checkpoints/epoch_1000_resume.pt)
- stick-push，600轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_0600.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_0600_resume.pt)
- stick-push，700轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_0700.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_0700_resume.pt)
- stick-push，800轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_0800.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_0800_resume.pt)
- stick-push，900轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_0900.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_0900_resume.pt)
- stick-push，1000轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_1000.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/stick-push/attempt_02/checkpoints/epoch_1000_resume.pt)
- door-open，600轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_0600.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_0600_resume.pt)
- door-open，700轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_0700.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_0700_resume.pt)
- door-open，800轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_0800.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_0800_resume.pt)
- door-open，900轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_0900.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_0900_resume.pt)
- door-open，1000轮：[权重](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_1000.ckpt) / [续训状态](/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020/runs/door-open/attempt_02/checkpoints/epoch_1000_resume.pt)
