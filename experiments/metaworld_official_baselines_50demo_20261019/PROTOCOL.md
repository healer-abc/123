# MetaWorld 50 条示教 baseline

用户于 2026-10-19 明确要求每场景改为 50 条示教，并允许灵活安排。该指令替代旧轮的 10 条示教及严格保留所有运行控制设置的限制。原版模型、不增加改进模块的要求继续有效。

## 数据与训练

任务固定为 shelf-place、pick-place-wall、pick-place、stick-push、door-open。每场景 50 条**成功**示教，共 250 条；三种方法共用同一份数据和同一个 40 train / 10 validation 的 episode 划分。划分与 ACT 原生的 `set_seed(1)` + `np.random.permutation(50)` 一致，冻结于 `manifests/dataset_split.json`。

每场景复用已核验的旧 10 条，逐文件检查原采集凭证哈希；用同一官方 MetaWorld 专家、相机、物理环境、软件渲染和成功筛选规则再采 40 条，新增采集 seed43。合并时保留全部原始数值、动作和 episode 边界，检查 50 个初始状态互异。原始 expert action 不修改，不做额外平滑或筛选。ACT HDF5 是同一份 zarr 的无损视图。旧轮模型与结果不混入本轮。

全部 15 个策略重新训练。DP/SimpleDP3 任务网络随机初始化；ACT 按官方实现使用 ImageNet ResNet18 初始化视觉骨干，任务策略新训练，不加载旧轮任务权重。

| 方法 | 官方模型/训练主体 | epoch / batch | 训练数据 |
|---|---|---:|---|
| DP | 官方 image U-Net、ResNet18（weights=null）、DDPM100、EMA、AdamW/cosine | 8000 / 64 | 固定40条 |
| SimpleDP3 | 官方 simple_dp3、点云编码器、DDIM10、EMA、AdamW/cosine | 3000 / 128 | 固定40条 |
| ACT | 官方ACT、预训练ResNet18、KL10、chunk100、hidden512、FF3200、lr1e-5 | 2000 / 8 | 固定40条 |

训练 seed0。不增添模型模块、改变损失或专家辅助控制。ACT 40条训练/batch8，每epoch5个optimizer step，2000epochs共10000步，恢复到官方50示教仿真示例的更新数量。ACT保持本轮预注册的 `temporal_agg=False`，按官方规则每100环境步更新一次动作序列；不因最终成绩更改。

## 灵活调整的范围

本轮调整数据划分和运行控制，保留上表训练主体与完整 epoch 预算：

- DP/SimpleDP3 中途开发 rollout 改为每500epochs一次，每次5个固定开发seed；checkpoint周期与其对齐。旧轮每次20个、DP每50epochs一次，导致大量时间耗在开发评测上。
- DP validation 每20epochs一次；ACT仍按官方训练循环每epoch验证并选最小validation loss模型。DP/SimpleDP3 的训练动作采样诊断每100epochs一次。
- DP/SimpleDP3 使用2个持久数据加载worker，ACT使用1个持久worker，减少每epoch重新创建进程开销。采样分布、batch size、模型和优化器保留；不声称与旧轮逐mini-batch随机序列一致。
- 开发环境评测保存/恢复主进程Python、NumPy、PyTorch CPU/CUDA随机状态，避免评测播种重置训练随机流。
- 同时最多4个worker、每GPU最多2个。按实际空闲显存及本任务当前显存与预留峰值进行准入，保留4GiB左右余量。调度按活动方法数平衡，避免ACT后续场景被长训练持续挤占。

这些都是在新一轮最终评测前固定的协议调整，不能将新旧轮差异解释成“仅示教数量”的单变量效果，也不能冒称逐项照搬官方仓库所有默认运行设置。

## 最终评测

仍为每方法×场景20个固定最终seed，完整200环境步，官方 `info.success` 曾为真即成功。与开发5个seed分离；最终共15组、300个episode。DP/SimpleDP3用完整训练结束的terminal EMA，ACT用官方最小validation loss checkpoint；不通过最终成绩选模型或重复刷失败episode。

每次最终记录绑定 checkpoint、50条数据凭证、共同划分和协议的SHA256。跨方法验证相同task/seed的模拟器、观测与目标初态hash一致。保存全部episode记录与视频。新结果写到本目录 `report/results.md` 和 `report/results.csv`。

## 范围与来源

DP、ACT需要MetaWorld I/O适配，本轮仍称为“官方实现的MetaWorld适配baseline”。ACT仅9维robot state/4维action接口调整，补丁位于 `act_metaworld_dimensions.patch`。官方源码通过软链接复用旧轮冻结归档，新执行源码和配置有独立 `source_manifest.json`。

官方来源：[DP](https://github.com/real-stanford/diffusion_policy)、[SimpleDP3](https://github.com/YanjieZe/3D-Diffusion-Policy)、[ACT](https://github.com/tonyzhaozh/act)。一训练seed/20次rollout仍不等于多训练种子的论文级统计；截图中旧改进方法的数据与评测协议未匹配，不直接计算提升。

## 后台检测

看门狗 `baseline_watchdog.py` 每15秒检查GPU余量、进程和日志/心跳。显存低于3500MiB时只让出本轮同GPU的一个进程，优先释放预算较大的进程；不触碰其他实验。1200秒无进展告警。基础设施故障最多两次独立尝试；OOM降低并发，不降低batch或改模型。未完成数据隔离保存，失败尝试和旧数据不覆盖。

状态 `logs/status.json`，事件 `logs/events.jsonl`，告警 `logs/latest_alert.json` / `logs/alerts.jsonl`。未知异常或非有限值会停止并保存诊断，不宣称能够自动修复任意代码错误。停止本轮时核对 `logs/watchdog_pid.json` 与命令行后向本目录看门狗发送SIGTERM，由其清理自己启动的进程组。

### 2026-10-19 18:35 显存准入修订

五场景数据完成后，共享GPU仅余8–9GiB，初始8GiB预留和保守预算使所有训练等待。依据旧轮和本轮真实full-batch峰值，将固定空闲余量改为4096MiB，ACT预算3584MiB、SimpleDP3预算4096MiB，DP继续保留12000MiB预算。低于3500MiB释放本实验进程的规则保持。变更前保存状态/源码/manifest于`audits/resource_admission_revision_20261019_1833/`，10项回归测试通过后重新冻结；科学训练/数据/评测协议JSON内容逐项对比完全相同。未改模型、batch、epoch或loss。
