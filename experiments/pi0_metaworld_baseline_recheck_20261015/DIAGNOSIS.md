# 原始 π0 MetaWorld SFT baseline 排查

范围：用户指定的 assembly-v2、shelf-place-v2、disassemble-v2、peg-unplug-side-v2、stick-pull-v2，各20次。模型为原实验的冻结 RLinf π0 MetaWorld SFT；未训练、未换权重、未按成绩调整参数。

## 已证实的发现

1. **旧表统计正确。** 独立读取100个NPZ，与JSON逐项核对 success_once、最后一步、步数、数值合法性，没有错项。随后使用保存的动作在同一环境中重放全部100回合：逐步四维状态和奖励最大绝对差均为0，成功标志完全相同，未观察到模拟异常。旧31/100不能认定为统计错误。证据：`../pi0_metaworld_baseline_20261015/audit/old_results_integrity.json`、`../pi0_metaworld_baseline_20261015/audit/replay_audit.json`。
2. **种子API调用有误。** 此固定版本的 `SawyerXYZEnv.reset(seed=...)` 明确忽略seed参数，原脚本仅调用该接口。现在显式调用 `env.seed(seed)` 再reset。测试验证原实现仍保持旧RNG状态，修复后与指定种子的RNG状态一致。由于任务位置来自固定的MT1配置，五任务第0回合的初始四维状态修复前后相同；不能把该问题直接认定为低成功率的原因。证据：`diagnostics/interface_validation.json`。
3. **原来缺少内部仿真错误拦截。** MetaWorld可能捕获模拟异常后返回上一次稳定观测与false成功标志。新脚本检查该内部标记及qpos/qvel/qacc有限性，异常立刻退出，不计作策略失败。这是防止以后误计的修复；原100条动作重放未触发此问题。
4. **原汇总仅检查文件存在。** 新版校验逐步success、首个成功步、末步、return、完整观测衔接、任务/种子唯一性、轨迹与视频SHA256和执行脚本SHA256。完整39维观测只落盘供审计，不输入模型。

## 未发现接口错误的部分

- 官方 `pi0_metaworld` 配置使用 action_horizon=5，非pi05；Normalize为均值/标准差，非分位数；不加DeltaActions。当前保持一致。
- 输入使用RGB相机2、位置[0.75,0.075,0.7]、180度旋转、480×480图像，state为observation[:4]；缺失腕部相机置零并掩码。动作取反归一化后的前4维；规划块5，去噪5，稳定等待15步，策略步160，与所参照的官方独立评测脚本设置一致。
- 模型输出xyz有超出[-1,1]的值并不单独构成错误：官方训练统计本身包含超范围专家动作，原生环境在`set_xyz_action`内部限幅。没有额外调比例或翻转夹爪来提升成绩。
- 7,011,543,960字节权重重新计算SHA256，与原实验一致；所有参数冻结。`diagnostics/checkpoint_reverified.json`。

## 比较限制

环境健康检查额外运行了MetaWorld自带的脚本控制器，使用同样20个初始化和160步。assembly 20/20、shelf-place 20/20、disassemble 14/20、peg-unplug-side 20/20、stick-pull 13/20；这些仅证明环境能够触发成功，特别是stick-pull并非恒为False。该控制器可以访问完整观测，因此不可与π0性能公平比较，也绝不计入正式成绩。证据：`diagnostics/expert_environment_check.json`。

- 此处按用户原始任务保留MetaWorld **v2**。RLinf安装脚本使用MetaWorld 3.0.0，其当前独立评测脚本调用v3任务。因此这是“官方SFT权重在原生v2任务上的适配评测”，不能声称复现了官方MT50 v3成绩。修复过程中没有暗中替换环境版本。
- 主指标是160个策略步中**曾经成功一次**。最后一步仍成功是另一个指标，两者都展示；不会事后挑选更高的口径。
- 每任务仅20次，单次结果对应5个百分点。低成绩本身不证明实现错误。保持原20组场景、种子和模型参数进行配对重跑，不挑选更容易的重置场景。

参考：
- https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/metaworld_eval.py
- https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/dataconfig/__init__.py
- https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/dataconfig/metaworld_dataconfig.py
- https://github.com/Farama-Foundation/Metaworld/blob/83ac03ca3207c0060112bfc101393ca794ebf1bd/metaworld/envs/mujoco/sawyer_xyz/sawyer_xyz_env.py

所有正式重跑结果保存于本目录的`results/`。`smoke_results/`及`diagnostics/`仅作验证，永不进入100回合成绩。
