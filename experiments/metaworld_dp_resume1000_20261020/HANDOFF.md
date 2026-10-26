# DP五任务：500轮完整状态续训到1000轮，每100轮保存

用户授权：在已完成500轮的DP训练上增加到累计1000轮并重新评测；随后补充要求每100轮保存快照。

## 固定协议

- 五任务全部从各自准确完成500轮的快照继续，新增500轮，累计1000轮。原500轮结果、权重、日志和所有原始评测数据保留。
- 继承原训练seed0、50示教（40训练/10验证）、batch128、horizon16、n_obs_steps2、action_horizon8、DDIM10/sample、EMA、官方模型/优化器/损失。
- 原余弦学习率计划本来就是1000轮。本轮不重启warmup、不延长成1500轮；恢复学习率调度到30500次更新，累计61000次更新后LR为0。
- 累计600、700、800、900、1000轮各保存独立模型/EMA/优化器checkpoint，以及学习率、EMA计数、随机数、采样诊断批次和runner等续训辅助状态。1000轮final.ckpt是epoch_1000.ckpt的硬链接，不重复写盘。
- 本次不补造旧阶段100/200轮文件；“每100轮”应用于从500轮开始的本次续训。
- 固定1000轮终点EMA评测各20局，共100局；DDIM10、同一20个seed2026102000–2026102019、每局200步。原生info.success曾为真视为成功。
- 与500轮结果逐seed配对，保留所有结果，不依据成功率挑选轮次、参数或seed。该配对seed池此前已经观察过，不声称是新的独立测试池或DP3论文完整评测协议复现。

## 完整状态恢复

- `targets/*.json`冻结准确完成500轮的模型快照、resume.pt、收据及原500轮结果SHA。
- 恢复model、ema_model、optimizer、epoch500、global_step30500；加载时明确排除旧_output_dir，防止写回原实验目录。
- 重建官方原始训练循环，加载LR scheduler.state_dict并恢复优化器当前LR；恢复EMA optimization_step=30500，避免重新计数。
- 恢复训练采样诊断批次、开发评测runner round_id和scores，以及Python、NumPy、Torch CPU和CUDA RNG。
- 原persistent dataloader在500轮时已经初始化；新的loader若直接使用默认generator，会额外从全局RNG抽取一次worker base seed，打乱采样。故仅给新loader的worker base seed分配独立generator，保持RandomSampler.generator=None，最后恢复全局RNG，沿用原采样顺序。dataset.__getitem__没有随机增强。
- `resume_preflight.json`已验证真实运行环境下：persistent loader的下一epoch索引及全局Torch RNG在重建后匹配；真实保存的学习率/EMA/RNG状态恢复；下一更新LR符合原1000轮余弦计划。
- 不能声称跨进程、GPU数值计算逐比特复现；恢复计数器、采样和调度状态的证据写`restore_audit.json`。

## 运行与审计

- 入口`run.py`，每个worker为`resume.py <task> <attempt>`。原500轮执行文件不修改。
- 当前GPU0无其他训练、GPU1有其他用户训练；最多三个本轮任务，GPU0最多两个、GPU1最多一个，每个预算10000MiB并留4096MiB，按实际剩余显存派发。不终止其他用户或实验的任务。
- 错误写alert，失败attempt完整保留；基础设施错误最多三次重试，重试仍从原500轮完整状态开始。日志/heartbeat超1200秒或GPU空闲显存低于3500MiB时只停止本轮自己的worker。
- `plan.json`启动时冻结执行源码、原500轮plan、目标文件、原worker、官方训练源码与基础协议SHA。运行中不得修改冻结代码，修复前先停本轮supervisor及workers，保留版本和错误记录。
- 每项续训完成后核验五份100轮快照SHA、序列化epoch/global_step、模型/EMA/optimizer字段；终点核验61000更新、EMA61000、scheduler61000、LR0。
- 最终100局核验种子、500轮与1000轮初始环境一致、原生success、有限动作、25个8步chunk和250次实际去噪调用/局，保存视频及NPZ/JSON。
- 实时状态：progress.md、status.json；已存快照：checkpoint_index.json；最终输出：summary.md/csv/json、audit.json、successful_episodes.md/json、episode_manifest.json。
- 到累计1000轮后停止训练，不自动继续。

## 时间估计

初始按上一阶段每轮约14秒估计约5–6小时（含额外500轮/任务及100局评测）。本轮并发和当前GPU负载不同，实际启动后依据日志测速更新估计；不保证新增训练会提高成功率。


## 启动确认及配置修复

首次启动在训练前因保存配置缺少Hydra now resolver退出，五项attempt_01均没有训练更新，原500轮数据不受影响。已停止旧supervisor，归档冻结源码和plan于audits/hydra_resolver_repair_v2，按Hydra官方setup_globals注册标准resolver；五项完整配置解析与mmap检查回归通过。plan修订2，旧alerts保留并在resolved_alerts.json标记修复。当前supervisor3054516，shelf-place3054519/GPU0、pick-place-wall3054520/GPU0、pick-place3054521/GPU1均已从500轮、30500更新恢复并开始训练；restore_audit均通过。其余两项排队。累计600/700/800/900/1000轮快照规则已写入实际训练循环。

## 完成记录（2026-10-21）

北京时间2026-10-21 01:02:55全部完成，status.json为complete，complete.json确认100局审计通过。上述启动段落为历史记录，五项现已全部完成1000轮训练及各20局评测。本轮累计600/700/800/900/1000轮各五项，共25份模型快照及配套续训状态均已保存。

固定1000轮EMA结果：shelf-place 3/20（15%），pick-place-wall 6/20（30%），pick-place 4/20（20%），stick-push 5/20（25%），door-open 19/20（95%）；五项平均37%，原500轮平均36%。全部100局原始JSON/NPZ/视频和37个成功episode索引保留；详细结果见summary.md/csv、audit.json和successful_episodes.md。不自动训练超过1000轮。
