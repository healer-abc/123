# π0.5 MetaWorld SFT 五任务评测交接

用户要求对之前五任务各20次使用MetaWorld微调过的π0.5重跑，选择更快下载线路。选RLinf发布的`RLinf/RLinf-Pi05-MetaWorld-SFT`，固定HF commit `60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e`、完整权重SHA256 `6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221`、长度7473091464字节。官方MetaWorld训练配置为`Pi0Config(pi05=True,action_horizon=5,discrete_state_input=False)`。使用发布者`lerobot/metaworld_mt50`统计和PI05分位数归一化；动作是原生MetaWorld四维xyz+夹爪。

2026-10-16 06:28 CST启动`download_checkpoint.py`，日志`logs/download.log`，分片保存在`checkpoints/RLinf-Pi05-MetaWorld-SFT/parts`，可中断续传。实测8MiB range：直连`hf-mirror.com`约2.46MB/s、代理Hugging Face约0.60MB/s、直连Hugging Face18秒超时；选直连镜像12路并行，每片HTTP Content-Range校验，完整下载后SHA256校验。若进程退出，直接重跑同命令即可续传，不应删除parts。仓库主页和发布者统计在HF源页面公开可核对。

更新2026-10-16 06:34 CST：下载已经完成，`verified.json`对重量/大小/metadata/统计文件的哈希均通过。模型严格加载通过，`logs/weight_load.json`的missing和unexpected均为空。五任务各一轮smoke在`smoke/`，全部正常执行；peg-unplug-side第82步成功，其余四轮未成功，smoke不计入正式成绩。

2026-10-16 06:46 CST最终：100/100完成，29成功。assembly 1/20、peg-insert-side 2/20、disassemble 8/20、peg-unplug-side 11/20、stick-pull 7/20。两worker均正常退出，GPU显存回落。`results/summary.json`、`summary.csv`、`episodes.csv`和`report.md`已生成；`verify_results.py`无`--partial`完整通过，结果在`results/validation.json`。总计39737环境步，权重完整SHA、严格模型键、同任务身份/种子、动作重算、首次成功停止及100个NPZ/MP4哈希全部通过。与未微调基础模型0/100相比有29个百分点的观测差异，但两版动作接口不同，不能仅归因于权重微调。

正式运行使用双GPU：GPU0负责assembly和peg-insert-side，日志`logs/worker_gpu0.log`；GPU1负责disassemble、peg-unplug-side和stick-pull，日志`logs/worker_gpu1.log`。两进程已退出，以下命令仅用于同协议补跑：

```bash
env CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 projects/openpi/.venv/bin/python -u experiments/pi05_metaworld_sft_20261016/launch.py --tasks assembly-v2 peg-insert-side-v2
env CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 projects/openpi/.venv/bin/python -u experiments/pi05_metaworld_sft_20261016/launch.py --tasks disassemble-v2 peg-unplug-side-v2 stick-pull-v2
```

`run.py`会跳过已有episode JSON；需要补跑时，先确认该JSON及其轨迹/视频是否完整，勿覆盖或混合smoke。补跑完成后重新运行`run.aggregate()`和`verify_results.py`刷新报表与全量审计。

未微调的`../pi05_metaworld_base_fixed_20261015/results_droid_units/`已100/100完成并独立校验，0成功。原始与中间接口版本保留，不合并成绩。新协议在`protocol.json`：同五任务、同task身份和seed42–61、500环境步上限、首次`info.success`立即停止。新结果单独放本目录`results/`，对应视频在`videos/results/`。

保留本机其他用户的GPU任务，LIBERO后续评测仍暂停。旧`../pi0_metaworld_sft_hf_20261016/`是π0模型、含shelf-place，仅借用经过验证的依赖和环境读数方法；不混用权重与成绩。
