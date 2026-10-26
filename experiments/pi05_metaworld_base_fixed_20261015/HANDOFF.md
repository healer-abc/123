# 接口修复与重跑交接

最后更新：2026-10-16 06:34 CST。DROID指令单位修正后，数值物理检查、两项rollout规则测试及五任务smoke均通过；正式100次已完成并通过`verify_results.py`独立审计，五任务各0/20。结果：results_droid_units/report.md、summary.json、validation.json。现用户要求使用MetaWorld SFT π0.5，见`../pi05_metaworld_sft_20261016/HANDOFF.md`。

用户授权：排查未微调基础模型＋动作接口为何0成功，修复实际问题，再跑相同五任务各20次。

- 原始100次保留：../pi05_metaworld_base_20261015/。
- 排查结论和限制：DIAGNOSIS.md。已修动作维数、夹爪索引、量纲/反归一化、真实关节state先编码后补齐及控制时基。
- 基础模型仍为原始pi05_base；不使用MetaWorld SFT，不训练或按成功率调参。
- 数值物理验证：diagnostics/interface_validation.json，五任务全部通过。
- 五任务各一个基础模型smoke已经完成，全部执行有效、0成功，日志logs/smoke.log。
- 补充发现连续夹爪位置问题：`2*closure-1`缺少中间闭合量跟踪，已修成position actuator分段逆映射＋闭合量反馈；25%、50%、75%跟踪误差最大0.001512（归一化闭合量）。
- 中间候选PID1886069已停止；已完成71条留在results/，排除正式成绩。旧协议在diagnostics/protocol_before_gripper_servo.json，旧视频/NPZ未覆盖。
- 连续夹爪版旧协议SHA256：5c848b117665523c4a415ca50ee455cefb13a4e1a0af646a7701215570d42753。results_servo/完成100次0成功，但DROID命令单位仍错误，现排除出最终成绩。
- 官方DROID执行器证实joint_velocity是归一化指令，先按无穷范数统一限幅，再乘0.2rad/周期×15Hz；代码已补齐。旧协议在diagnostics/protocol_before_droid_units.json。
- 最终100次结果必须使用results_droid_units/和videos/results_droid_units/；新smoke为smoke_droid_units/。不得与results/或results_servo/拼接。新协议SHA256：f04241af4c4ce51b2b2669d937d9dadaf8438d30c83a2a82d3a73dd5afc949a9。
- DROID统计移植Sawyer有5/8初始状态维超出训练分位范围；这是未消除的跨机器人限制，不得报告“完全匹配”。
- 其他用户GPU任务保持原状，LIBERO后续评测仍暂停。
- 环境独立诊断：五任务各首个相同task/seed42，用MetaWorld自带脚本策略均成功（92/65/138/98/90步），详见diagnostics/environment_oracles.json。它读取目标信息，仅验证环境，不计入模型成绩。

正式运行命令：

```bash
cd /home/zhsu/jyluo
env CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 PYTHONUNBUFFERED=1 projects/openpi/.venv/bin/python -u experiments/pi05_metaworld_base_fixed_20261015/run.py
```

状态：logs/status.json；最终结果：results_droid_units/report.md、results_droid_units/summary.json；轨迹/视频逐轮保存。正式运行前必须通过check_interface.py（stamp绑定协议）、test_rollout_rules.py和五任务smoke；完成后运行verify_results.py，不带--partial。
