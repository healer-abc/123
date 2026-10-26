# π0 MetaWorld SFT baseline 重跑

用户当前明确对原 `pi0_metaworld_baseline_20261015/results/report.md` 提出排查并重跑要求。本目录专门处理该SFT评测，与相邻pi05基础模型实验无关。

原100条已逐步重放一致。种子API误用已修复；增加仿真错误拦截与逐回合完整性校验。细节见DIAGNOSIS.md。模型、5个v2场景、20个固定场景配置、种子42–61、160策略步、action chunk=5、去噪5保持原计划。

依赖、权重、官方源码符号链接至原目录，只读使用。原结果与视频不覆盖。

systemd用户服务：
- pi0-mw-recheck-bootstrap：新协议下单回合策略验证；通过后写与协议SHA绑定的policy_smoke_passed.json。
- pi0-mw-recheck-watchdog：两GPU各一个进程，每任务20回合。完成100次后自动退出。

状态：logs/watchdog_state.json；错误：logs/alerts.jsonl。结果：results/summary.json、report.md、summary.csv、episodes.csv。脚本：rollout.py、aggregate.py、watchdog.py。

不要中途更改protocol或rollout.py，会使不同回合的协议/代码SHA不一致。不要停止服务器已有的其他训练或推理进程。

diagnostics/expert_env_audit.py是官方脚本控制器的环境健康检查，完全排除正式π0成绩，不能把它当成模型rollout或模型提升。

完成：正式100/100、独立物理重放100/100、完整性校验100/100；新旧对照见results/comparison.md。看门狗已完成并自动退出。
