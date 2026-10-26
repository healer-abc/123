# SimpleDP3 epoch500：独立敏感性实验

训练前缀1000→500轮，保留1000轮学习率计划；去噪10步、horizon16、obs2、action8、训练seed0。
500轮模型为同配置重放训练所得；完成300/400轮另存独立权重，不用原800/1000轮权重冒充。
五任务各20局；统一评测seeds2026102000–2026102019，每局200步。
保留原1000轮基线；这是观察主表后的参数敏感性实验，不替换主表，也不声称独立未见测试。

状态：complete。

| 任务 | epoch1000基线 | epoch500当前 | 阶段 |
|---|---:|---:|---|
| shelf-place | 14/20 | 13/20 | complete |
| pick-place-wall | 19/20 | 19/20 | complete |
| pick-place | 20/20 | 19/20 | complete |
| stick-push | 20/20 | 20/20 | complete |
| door-open | 20/20 | 20/20 | complete |

## 已保存权重

- shelf-place 完成300轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/shelf-place/attempt_01/checkpoints/epoch_0300.ckpt)
- shelf-place 完成400轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/shelf-place/attempt_01/checkpoints/epoch_0400.ckpt)
- shelf-place 完成500轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/shelf-place/attempt_01/checkpoints/final.ckpt)
- pick-place-wall 完成300轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/pick-place-wall/attempt_01/checkpoints/epoch_0300.ckpt)
- pick-place-wall 完成400轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/pick-place-wall/attempt_01/checkpoints/epoch_0400.ckpt)
- pick-place-wall 完成500轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/pick-place-wall/attempt_01/checkpoints/final.ckpt)
- pick-place 完成300轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/pick-place/attempt_01/checkpoints/epoch_0300.ckpt)
- pick-place 完成400轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/pick-place/attempt_01/checkpoints/epoch_0400.ckpt)
- pick-place 完成500轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/pick-place/attempt_01/checkpoints/final.ckpt)
- stick-push 完成300轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/stick-push/attempt_01/checkpoints/epoch_0300.ckpt)
- stick-push 完成400轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/stick-push/attempt_01/checkpoints/epoch_0400.ckpt)
- stick-push 完成500轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/stick-push/attempt_01/checkpoints/final.ckpt)
- door-open 完成300轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/door-open/attempt_01/checkpoints/epoch_0300.ckpt)
- door-open 完成400轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/door-open/attempt_01/checkpoints/epoch_0400.ckpt)
- door-open 完成500轮：[checkpoint](/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch500_ddim10_20261020/runs/door-open/attempt_01/checkpoints/final.ckpt)
