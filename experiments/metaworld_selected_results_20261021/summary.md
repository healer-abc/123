# MetaWorld 五任务：用户指定版本汇总

每种算法每任务20局；训练seed0；每任务50条示教（40训练/10验证）。

| 任务 | DP（1000轮） | ACT（最后完成版本） | SimpleDP3（50轮） |
|---|---:|---:|---:|
| shelf-place | 15%（3/20） | 55%（11/20） | 40%（8/20） |
| pick-place-wall | 30%（6/20） | 95%（19/20） | 85%（17/20） |
| pick-place | 20%（4/20） | 70%（14/20） | 90%（18/20） |
| stick-push | 25%（5/20） | 100%（20/20） | 80%（16/20） |
| door-open | 95%（19/20） | 95%（19/20） | 100%（20/20） |
| **五项平均** | **37%** | **83%** | **79%** |

配置：DP累计1000轮EMA、DDIM10、horizon16、obs2、action8；SimpleDP3第50轮EMA、DDIM5、horizon16、obs2、action8；ACT训练2000轮，使用该次训练按验证损失选取的权重，chunk10、时间聚合开启、每步查询、KL10。

口径说明：此表按用户指定的版本汇总，训练预算和去噪步数不同；SimpleDP3的50轮属于查看筛选结果后指定的探索评测，pick-place复用此前同一配置的20局。此表不替代原始基线或声称是统一预算、预先固定协议的比较。全部原始结果保留。

来源：
- ACT：/home/zhsu/jyluo/experiments/metaworld_community_baselines_50demo_20261020
- SimpleDP3：/home/zhsu/jyluo/experiments/metaworld_simpledp3_epoch50_ddim5_all5_20261020
- DP：/home/zhsu/jyluo/experiments/metaworld_dp_resume1000_20261020
