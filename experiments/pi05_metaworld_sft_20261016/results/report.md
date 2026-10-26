# MetaWorld SFT π0.5 五任务评测

| 任务 | 成功/20 | 成功率 |
|---|---:|---:|
| assembly-v2 | 1/20 | 5.0% |
| peg-insert-side-v2 | 2/20 | 10.0% |
| disassemble-v2 | 8/20 | 40.0% |
| peg-unplug-side-v2 | 11/20 | 55.0% |
| stick-pull-v2 | 7/20 | 35.0% |

完成100/100，成功29，整体成功率29%。20次/任务；每轮最多500个环境步，首次`info.success`立即停止。
权重：RLinf/RLinf-Pi05-MetaWorld-SFT，冻结推理，使用发布者的MetaWorld归一化统计和原生4D动作。
下载：固定HF提交`60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e`，使用实测更快的hf-mirror直连并行分片；完整权重SHA256`6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221`，经独立校验。
对照：同任务身份、seed的未微调π0.5基础模型是0/100。这里的29/100说明MetaWorld SFT模型加其原生动作接口能在这些v2任务中成功；基础模型使用DROID/Sawyer代理接口，因此不能把29个百分点的差异单独归因于微调权重。
核验：`validation.json`覆盖100轮，任务身份/种子/首次成功停机、动作及轨迹/视频哈希全部通过，总计39737个环境步；没有将环境或运行错误计为失败。
该模型原训练/参考评测基于MetaWorld v3；这里评测用户指定的v2任务。
