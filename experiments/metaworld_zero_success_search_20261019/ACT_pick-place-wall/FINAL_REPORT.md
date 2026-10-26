# 最终交付：ACT pick-place-wall

| 评测 | 成功/局数 | 成功率 |
|---|---:|---:|
| 历史原始基线 | 0/20 | 0% |
| 新seed上的原始配置配对对照 | 5/20 | 25% |
| 锁定配置独立20局 | 8/20 | 40% |

仅将ACT执行间隔100改为1；temporal_agg保持false，预测块长保持100。没有重新训练或修改模型。
50条示教预算、40训练/10验证；使用同一个原有最优验证checkpoint。
Checkpoint SHA256：`f8f7e11acc84761652c34b99a89a32347fcb05a77a3d376d6bafdf5f677bc835`。

## 筛选记录

先登记有限空间。时间聚合开启组0/3；关闭聚合、每步重查询组1/3，首次命中后立即锁定。
筛选共6局，全部记录保留。命中seed3026101905，第96步首次成功，官方success=1、obj_to_target=0.0659001627（阈值0.07）。
独立20局固定seed4026101900–4026101919，未用于挑选参数。原始配置的配对对照使用同一批seed，20对初始状态哈希逐局相同。
结果属于搜索后评测，不能替换或冒充未调参官方基线。

## 验证与错误修复

审计通过，共检查47局，所有200步记录均从官方逐步info重算成功判定，动作和预测均为有限值。
其中44局未聚合ACT动作已从保存的模型预测块和原始归一化统计逐步重建核对。
原始配置回放同一历史seed，初始状态、成功判定、步数和回报一致。
CPU固定示教观测诊断结果见offline_model_diagnostic.json；未执行优化器步骤。
补充配对对照曾因动态库路径在Python启动后设置而触发MuJoCo重编译，继而缺少X11头文件。已改为携带完整原始环境重启进程；不改环境物理或成功阈值。失败日志保留，主重评测未受影响。

## 交付文件

- 全部方法的汇总表：../summary.md、../summary.csv。未完成训练组明确标记未完成。
- 参数位置：../PARAMETERS.md；参数差异：parameters.diff、parameter_diff.json。
- 完整筛选记录：../search_trials.csv；锁定信息：locked.json。
- 配对对照：paired_comparison.json。
- 数据审计：../artifact_audit.json；原始配置回放审计：../baseline_parity_audit.json。

## 独立20局中的全部成功episode

| Episode | Seed | 首次成功步 | 视频 | 原始轨迹 |
|---|---:|---:|---|---|
| 01 | 4026101901 | 98 | [episode_01.mp4](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_01.mp4) | [episode_01.npz](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_01.npz) |
| 03 | 4026101903 | 92 | [episode_03.mp4](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_03.mp4) | [episode_03.npz](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_03.npz) |
| 07 | 4026101907 | 101 | [episode_07.mp4](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_07.mp4) | [episode_07.npz](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_07.npz) |
| 08 | 4026101908 | 91 | [episode_08.mp4](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_08.mp4) | [episode_08.npz](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_08.npz) |
| 10 | 4026101910 | 81 | [episode_10.mp4](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_10.mp4) | [episode_10.npz](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_10.npz) |
| 13 | 4026101913 | 98 | [episode_13.mp4](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_13.mp4) | [episode_13.npz](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_13.npz) |
| 15 | 4026101915 | 97 | [episode_15.mp4](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_15.mp4) | [episode_15.npz](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_15.npz) |
| 16 | 4026101916 | 83 | [episode_16.mp4](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_16.mp4) | [episode_16.npz](/home/zhsu/jyluo/experiments/metaworld_zero_success_search_20261019/ACT_pick-place-wall/holdout20/episode_16.npz) |

DP仍在等待训练资源，SimpleDP3正在训练；原始训练看门狗继续运行。后续新增0/20组由独立watch_future.py监控接续处理，状态见../followup_index.md。
stick-pull-v2不在现有五任务中，本次没有对应基线checkpoint，未冒用stick-push替代。
