# 原始基础模型接口失败排查

2026-10-15。排查对象是 `../pi05_metaworld_base_20261015` 的100次0成功结果。

## 已证实的问题

1. **动作语义错误。** 上一版直接把32维归一化输出的前4维映射为Cartesian xyz和夹爪，没有建立与基础模型预训练动作定义的对应。官方DROID接口保留前8维：7个关节量和1个夹爪。第四维并不是夹爪。
2. **没有反归一化。** 上一版把网络归一化输出直接截断至[-1,1]执行。基于统计量的物理单位恢复缺失。正式100次有16.14%的执行动作分量发生限幅。
3. **状态含义及维数不符。** 上一版将末端XYZ及开度放在关节状态的位置，并先补齐32维再生成state文本。标准数据管线先对真实状态归一化、tokenize，再补齐状态张量。修正版使用七关节角＋夹爪闭合量，共8维。
4. **控制时基不明。** 环境的真实控制步长为0.0125秒，动作xyz对应每步0.01米的mocap位移；网络输出不能绕过单位转换。本次明确用DROID式15Hz关节速度，每一步以实际dt和Sawyer雅可比计算位移。

旧轨迹证据：五任务平均动作均持续向+x/-y/-z，末端最终约97%–100%的xyz分量接近工作空间边界。这与错误的四维直连一致，但不是对模型能力的独立证明。

## 实施的修复

- 使用原始pi05_base，自带assets/droid统计；pi05按官方DataConfigFactory选择quantile归一化/反归一化。
- 读取真实Sawyer七关节角和两个指爪滑动关节；闭合量按指爪物理行程定义。
- 第八维控制夹爪；正闭负开经过实际MuJoCo控制验证。第四维只影响关节运动。
- 连续夹爪采用实际指爪position actuator的分段逆映射＋闭合量反馈。简单`2*closure-1`只验证端点不充分：25%、50%、75%的实测跟踪误差现在均小于0.16个百分点。
- 七维关节速度先投影到固定工具朝向允许的零空间，再通过当前TCP雅可比转换为世界系末端速度，乘env.dt后除env.action_scale。
- 执行过程中每个环境步重新计算雅可比；策略动作按15Hz保持，五个策略动作后重规划。
- 相机180°旋转与RLinf公开评测代码一致，检查中没有证据证明它是本次错误来源，因此保留。
- 未微调、不使用任务目标坐标/物体状态作为策略输入、不换成SFT权重。

## 验证与未消除的限制

五任务雅可比与数值差分的最大误差约2e-10；归一化往返、零速度零位移、第四维不控夹爪、第八维控夹爪、开合方向全部通过。五任务各一个真实基础模型smoke完成，无执行错误，未计入正式100次。

补充环境可达性诊断：`check_environment_oracles.py`使用MetaWorld自带脚本控制器，在五任务各自相同首个task身份和seed42上，分别于92、65、138、98、90步触发`info.success`。仅该独立诊断允许控制器读取目标位置，证明这五个场景的成功判定可以触发；不代表其他95个初始状态都经过oracle验证，也不计入基础模型成绩。记录为`diagnostics/environment_oracles.json`，正式模型输入仍只有图像、关节状态和文字任务。

**DROID使用Franka，MetaWorld使用Sawyer。二者关节轴、零点、运动学和视觉外观不相同，基础权重没有提供Sawyer专用统计或动作头。当前按七关节顺序移植是明确标记的代理方案，不能被这些数值单测证明为模型语义完全匹配。** 初始状态有5/8维超出DROID的1%–99%分位范围，是实际分布差异的证据；不得通过任意改offset/选择最佳seed掩盖。

另外，当前仅提供单外部相机，缺失腕部相机以mask处理；每轮500个环境步对应6.25秒模拟时间，与原基线保持一致。DROID官方部署示例还选择了二值夹爪和逐维限幅；本方案明确使用连续夹爪位置伺服及底层执行器的统一限幅规则，并非复刻原生DROID硬件评测。报告只能说明指定跨机器人接口下的效果，不能独立证明基础模型本身不具备这些任务能力。

本次按接口证据修复后冻结协议，正式重跑100次，不按成功率搜索映射、不替换有任务训练的模型。无成功也按原样交付。

在补齐连续夹爪验证前启动的71条中间候选记录保留于`results/`，随后已停止，**排除出最终成绩**。当时协议留在`diagnostics/protocol_before_gripper_servo.json`。

随后连续夹爪版`results_servo/`完成100次、0成功，但继续核对官方执行器时发现DROID的`joint_velocity`是归一化控制指令，不能直接解释成rad/s。该轮同样保留为诊断记录，不计入最终结果；协议备份在`diagnostics/protocol_before_droid_units.json`。修复依据是[DROID官方RobotIKSolver](https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py)：七维命令统一除以max(1,无穷范数)，再乘0.2 rad/周期；15Hz下应乘3才得到rad/s，然后做固定朝向投影和雅可比转换。新增小命令单位换算及超范围命令统一限幅测试。

最终完整重跑单独使用`results_droid_units/`和`videos/results_droid_units/`。两次排除均由独立的接口正确性检查决定，不按成功率选择候选。

你打开的 `../pi0_metaworld_baseline_20261015/results/report.md` 是另一个MetaWorld SFT π0实验（含shelf-place），不能与本次未微调π0.5的结果混用。

依据：本机 `projects/openpi/docs/norm_stats.md`、`src/openpi/policies/droid_policy.py`、`src/openpi/training/config.py`、`src/openpi/transforms.py`、`src/openpi/models/tokenizer.py` 和固定版本MetaWorld的 `set_xyz_action`/`step`/机器人XML。
