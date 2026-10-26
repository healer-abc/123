# MetaWorld SFT π0.5 五任务评测

| 任务 | 成功/20 | 成功率 |
|---|---:|---:|
| assembly-v2 | 0/1 | 0.0% |
| peg-insert-side-v2 | 0/1 | 0.0% |
| disassemble-v2 | 0/1 | 0.0% |
| peg-unplug-side-v2 | 1/1 | 100.0% |
| stick-pull-v2 | 0/1 | 0.0% |

完成5/100，成功1。
权重：RLinf/RLinf-Pi05-MetaWorld-SFT，冻结推理，使用发布者的MetaWorld归一化统计和原生4D动作。
任务与seed匹配未微调π0.5基线；前者是MetaWorld任务动作，后者使用DROID/Sawyer代理接口，差别不只有权重。
该模型原训练/参考评测基于MetaWorld v3；这里评测用户指定的v2任务。
