# SimpleDP3 DDIM5：独立敏感性实验

仅去噪步数10→5。相同terminal EMA模型、horizon16、obs2、action8、训练seed0。
五任务各20局；统一评测seeds2026102000–2026102019，每局200步。
保留DDIM10原始基线；这是观察主表后的参数敏感性实验，不替换主表，也不声称独立未见测试。

状态：complete。

| 任务 | DDIM10基线 | DDIM5当前 | 阶段 |
|---|---:|---:|---|
| shelf-place | 14/20 | 14/20 | complete |
| pick-place-wall | 19/20 | 19/20 | complete |
| pick-place | 20/20 | 20/20 | complete |
| stick-push | 20/20 | 20/20 | complete |
| door-open | 20/20 | 20/20 | complete |
