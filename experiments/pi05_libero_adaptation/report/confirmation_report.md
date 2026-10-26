# E0/E1/E2 固定清单确认评测

完成时间：2026-10-22T01:15:23+08:00。每组 1600 次，训练 seed=42。

| 模型 | Spatial | Object | Goal | Long | 四套等权环境成功率 | 严格边界成功率 |
|---|---:|---:|---:|---:|---:|---:|
| E0-s42 | 97.67% | 98.40% | 95.80% | 91.67% | 95.88% | 0.00% |
| E1-s42 | 94.33% | 99.20% | 98.40% | 93.33% | 96.32% | 0.00% |
| E2-s42 | 98.00% | 98.20% | 97.40% | 89.67% | 95.82% | 0.00% |

主要比较 E2−E1：-0.50 个百分点，配对分层 bootstrap 95% CI [-1.85, 0.83]。
环境成功口径预登记非劣规则：满足。

原始动作边界口径独立保留；环境成功不能替代原严格指标。

每任务成功数和 Wilson 区间、各比较及两种计分见 confirmation_comparison.json。

限制：
- one training seed (42), no training-seed uncertainty claim
- E0 historical all-suite results already exposed; this is fixed-benchmark confirmation, not a pristine independent test
- development is used for selection and excluded from confirmation estimates
- E0 EMA and E1/E2 non-EMA recipe comparison is not a single-variable ablation
- no candidate or execution changes after viewing confirmation outcomes

完整 2000 次开发+确认的描述性汇总（不替代上方独立列出的确认估计）：

- E0-s42: 1914/2000，四套等权 95.70%
- E1-s42: 1928/2000，四套等权 96.40%
- E2-s42: 1918/2000，四套等权 95.90%
