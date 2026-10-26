# SimpleDP3 去噪5步配对消融

用户明确要求把SimpleDP3去噪步数改为5测一遍。本目录只执行这一个固定配置，不做参数或seed搜索。此前用户有压低基线的诉求，本轮按真实结果完整报告，不能根据分数选择是否保留。

- 独立于主基线，不修改../metaworld_community_baselines_50demo_20261020的配置、结果、模型或进程。
- 五任务各20局；训练seed0；rollout seeds2026102000–2026102019；每局200步。所有任务均完成20局，无论成功率高低。
- 使用主基线各任务固定1000epoch terminal EMA checkpoint；只将num_inference_steps=10改为5。horizon16、n_obs_steps2、n_action_steps8、DDIM scheduler/sample预测、训练噪声100步均保持一致。
- 当前四个模型已完成，door-open训练中；监督等待其完成原10步评测后，自动绑定同一checkpoint并加入5步评测。
- run.py监督最多两个推理进程、每GPU一个，需空闲至少6144MiB，低显存时只让出本消融任务，600秒无进展报警，最多三次错误重试。每局JSON最后提交，便于精确恢复，原始视频/NPZ/原生info均保留。
- rollout5.py复制自冻结的主基线评测器，参数差异与必要身份/调用计数变更见evaluator.diff。每次真实scheduler.step都计数，最终每局应为25次动作序列查询×5去噪步=125次，写入episode JSON并审计。
- plan.json固定源码hash、参数、seed与规则；targets中绑定checkpoint与10步结果hash。运行期间不可修改冻结源码。
- 最终审计100局：同checkpoint、同seed初始状态、每局200步、原生成功重算、实际去噪调用次数、执行间隔8与保存动作重建。
- 这是观察原测试结果后提出的敏感性实验，原DDIM10主基线保留。5步结果只能并列报告，不替换为较低分的“默认基线”。

入口run.py；PID见supervisor_pid.json；进度progress.md/status.json；最终summary.md/csv/json；参数parameters.diff；成功路径successful_episodes.json；审计audit.json。

## 完成

五任务共100局已全部完成，审计通过。DDIM5与DDIM10各任务成功数均为14、19、20、20、20，均值93%；pick-place仍为20/20=100%。未搜索其他参数、未换seed、未重训或替换模型。summary.md/csv/json、audit.json、parameters.diff与成功episode路径均已生成。主基线与其训练进程保持不变。
