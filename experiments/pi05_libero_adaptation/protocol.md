# π0.5 LIBERO 参数高效适配实验协议

状态更新时间：2026-10-14 21:38 CST。

本目录执行用户提供的《π0.5 在 LIBERO 上的参数高效任务适配：最终实验设计》。主线固定为：接收 E0/E1 → 审计 → E2 冻结视觉单变量对照 → 开发集评测 → 候选确认。未经 `decision_log.md` 预登记，不扩展 rank 网格或多个 E3。

## 核心实验

- E0：`pi05_libero_full_gpu1`，全量微调，30k，LR 5e-5，EMA 0.999。
- E1：`pi05_libero_lora_gpu0`，两分支 LoRA，同时训练视觉编码器及动作/时间投影，30k，LoRA schedule，EMA 关闭。
- E2：拟新增 `pi05_libero_lora_frozen_vision_s42`。完全复制 E1，仅额外冻结 `PaliGemma/img/**`；必须从 `pi05_base` 开始。

E0/E1 是多因素配方比较，不是单变量 LoRA 消融。E1/E2 才用于回答冻结视觉编码器的影响。

## 固定训练协议

- 数据：`physical-intelligence/libero`，1693 episodes、273465 frames、40 tasks。
- batch 32，30,000 updates，seed 42，action horizon 10，action dim 32。
- `pi05=True`，`discrete_state_input=False`，`extra_delta_transform=False`。
- E2 继承 E1 optimizer、LR schedule、EMA、rank、数据顺序及增强规则。
- norm stats SHA-256：`c0ee3c1a6ca5f1414d31180af3bc333dca1d78cf912f341943191b393fe68707`。
- 主结果只认最终可加载 checkpoint；中间 checkpoint 不按评测成绩择优。

## 固定评测协议

- suites：Spatial、Object、Goal、Long。
- 每任务 50 个预设初始状态；开发集为 Spatial/Long 的 0–19，共 400；确认为 Spatial/Long 的 20–49 加 Object/Goal 的 0–49，共 1600。
- 新评测必须显式记录 task、init state、env seed、policy seed、checkpoint、异常及视频路径。
- 旧 E0 评测共 2000 episode，但缺 init-state/seed/checkpoint 列，保留为历史完整基准；除非能从日志可靠重建，否则不与新协议数据拼成严格配对样本。
- 基础设施失败保留原始尝试，同一 episode 最多重试一次；模型 NaN、越界或任务内超时计策略失败并单列。

### 2026-10-18 动作计分审计补充

原始动作边界合规成功率继续保留，任何实际发送的原始动作超出 [-1,1] 都按原规则单列失败；E1/E2 此口径均为 0/400。实际 robosuite 对运动输入裁剪、对夹爪取符号，因此另行报告环境成功率（两组均 377/400），并以此开展原方案动作侧 E3 的探索性配方诊断。不得把环境成功率冒充原严格口径结论，不覆盖原始 episode。后续 E3 同时报告两项，并保存运动裁剪输入和夹爪符号代理；这些代理不是实际关节力矩。详见 `report/development_review/review_notes.md`。

## 停止条件

冻结集合不符、非有限 loss/action、checkpoint 损坏时暂停。训练 loss 正常不能替代闭环评测。2026-10-18 用户授权根据实际空闲显存安排并发：按模型预算加安全余量准入；明确基础设施错误保留原始尝试并限次重试，耗尽后暂停。具体调度参数见 `decision_log.md` 的当日记录。

## 2026-10-21 候选终审与确认登记

开发终审支持选择 E2，E3 不入选；依据为 `report/final_development_review/review_notes.md`。确认启动前保存 `manifests/candidate_freeze.json` 与 SHA-256 sidecar，检查三组最终推理权重与资产的内容哈希、环境版本、评测代码及证据哈希。后续不根据确认结果改候选、参数、状态或样本量。

- 共同比较 E0/E1/E2，各 1600 次。E2−E1 是主要比较，E0 为历史已暴露、单训练种子的参考。
- 环境成功口径为此次预先固定的实用确认指标；原始严格动作边界口径独立保留，不能由环境成功推导原严格指标已通过。
- 单任务给成功数及 Wilson 95% CI；四套分别按任务等权，总体四套等权，不能将 1600 个 episode 直接混池计分。
- 固定 10000 次任务内成对 bootstrap，随机种子 20261021；E2−E1 的环境成功差 95% CI 下界严格高于 -5pp 才满足该口径工程非劣规则。任一套下降超过 10pp 必须单列。
- 三组全量最终 checkpoint 固定为 29999；E0 使用 EMA 导出，E1/E2 无 EMA。每组确认阶段基础设施故障同 episode 最多两次尝试，模型异常单列并留证。
- 为补全 E0 的新协议状态身份，在确认后额外评测 E0 开发 400 次；共同完整 2000 次表只作描述性汇总，不混入确认估计。新增总计 5200 次，其中确认 4800 次。
- 最低交付保持训练 seed=42；不据单种子 rollout 区间声称跨训练种子稳定，不把历史 E0 重跑称为全新独立测试集。

