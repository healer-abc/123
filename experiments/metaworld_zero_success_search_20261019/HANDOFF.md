# 零成功率推理参数搜索

此任务来自用户 2026-10-19 的明确授权：只调整推理参数、逐组3局筛选、首个命中后固定20局复评；不删除或伪造数据，有限空间穷尽时如实报告。

## 当前范围

- 基线：`../metaworld_official_baselines_50demo_20261019`；50条示教预算，共同40训练/10验证划分。
- 冻结时已完成5个ACT组，只有ACT/pick-place-wall为0/20。DP未训练，SimpleDP3训练中，不能把未完成当成0%。
- stick-pull-v2不在该五任务清单。此项目评测stick-push，没有替换任务。
- 冻结的完整有限空间、seed与checkpoint身份见 `plan.json`。保持 `search.py` 原始哈希不变。

## 已验证事实

- trial_000：官方temporal_agg=true、k=0.01，每步查询，0/3。
- trial_001：temporal_agg=false、执行1步后重新查询，1/3；成功seed3026101905，第96步首次成功。
- 模型权重及训练配置未改。预测块长仍100。
- 锁定配置见 `ACT_pick-place-wall/locked.json`。
- 独立20局seed4026101900–4026101919预先登记，与筛选分开。必须完整跑20局，不根据中途成功提前停，不根据holdout结果改配置。
- 至少已在独立seed4026101901、4026101903取得真实成功。最终以holdout20/summary.json为准，不把中途数字当最终结果。

## 进程与结果

- 当前搜索监督进程身份：`supervisor_pid.json`；状态：`status.json`；每10步心跳：`heartbeat.json`。
- 每局JSON是提交完成标记，写入前同名NPZ和MP4已保存。
- 原始动作、模型预测块和查询时刻：每局NPZ；完整官方info：每局JSON；所有失败也保留。
- `check_baseline_parity.py` 已后台启动：等待20局结束后，使用原始配置和原始首个seed回放，核对初始状态、回报、成功判定。
- `diagnose_model.py` 已完成CPU诊断：固定5个训练初始观测与全部10个验证初始观测，结果在 `ACT_pick-place-wall/offline_model_diagnostic.json`。未执行优化器步骤。
- 当前20局和parity完成后运行：`/home/zhsu/dp3_ablation_v52/env/bin/python <此目录>/audit.py`。它重新从每步info与NPZ验证分数、长度、有限值、checkpoint身份，并生成summary、search_trials.csv和参数diff。

## 后续新增零组

- `watch_future.py` 已后台启动，身份见 `followup_pid.json`、状态见 `followup_status.json`、索引见 `followup_index.md`。
- 每30秒扫描基线结果，只处理新完成且0/20的组。
- 为每个组建立独立的同级followup目录，首局之前冻结相同的预登记有限网格，再搜索与独立20局、自动审计。
- 后续使用 `search_followup.py` 副本，只补充官方OmegaConf eval resolver注册，以支持DP/SimpleDP3 checkpoint中的配置插值；当前ACT运行源码不改。
- GPU推理至少留4GiB物理余量，GPU显存不足则排队。故障原地记录，最多3次重启，已完成episode不丢弃；600秒无进展报警。
- 不触碰原基线训练进程、其他用户任务或Libero实验。

## 报告口径

`summary.md` / `summary.csv` 保留原始基线列和独立搜索后复评列。调参后的结果明确称“搜索后评测”，不能冒充未调参官方基线。筛选局不混入20局分母。原始0/20也不覆盖。

## 本轮最终结果（已审计）

- 锁定配置独立20局：8/20，40%。
- 原始配置在同一批20个新seed上的配对对照：5/20，25%；所有初始状态哈希相同。
- 历史原始基线0/20保留。因此不能把40%与历史0%之差全部归因于参数。
- 完成47局数据审计，含6局筛选、20局锁定配置、20局配对对照及1局历史配置回放。checkpoint未变，44局未聚合ACT动作可从保存的模型输出重建。
- 回放核验通过；交付见 `ACT_pick-place-wall/FINAL_REPORT.md`、`summary.md`、`summary.csv`、`artifact_audit.json`。
- 补充对照启动曾遇到动态库路径在Python启动后设置导致MuJoCo导入失败，已通过携带完整环境重启辅助进程修复，失败日志保留。
- 当前搜索、配对对照、回放、CPU诊断及finalizer均已结束；仅原始训练看门狗和后续零组监控继续运行。
