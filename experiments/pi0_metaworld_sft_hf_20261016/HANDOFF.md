# π0 MetaWorld SFT 10k 权重评测

用户要求换用 MetaWorld 微调过的π0，对相同五个MetaWorld v2场景各做20次rollout。

之前两轮用的是 RLinf ModelScope 镜像 e63a613a 的 `RLinf/RLinf-Pi0-MetaWorld`；其公开 metadata.pt 显示 `pi0_metaworld`、训练 global_step=4999、目标5000步，模型SHA256为 bef6bacc...。这确实是MetaWorld SFT 5k权重，之前不应把它与 Hugging Face 最新 SFT 仓库视作同一文件。

本轮切换到RLinf Hugging Face `RLinf/RLinf-Pi0-MetaWorld-SFT` 主分支 d0b8fbe41877a8105e2fd91e750e44542e6855bd；公开 metadata.pt 显示 global_step=9999、目标10000步。HF LFS权重SHA256应为 c606538b48414b00af1966dd8cf27fe0a87893f390619cac6bf0920a7cf06e2d，长度7,011,543,960字节。实际下载必须做整文件SHA校验再评测。模型文件开头8MiB与旧文件不同；元数据对照在diagnostics/model_metadata_comparison.json。

输入/环境/参数沿用上轮经过审计的脚本：五任务各20；相同MT1 task SHA、seed42–61、160步；4维state和RGB图像；冻结权重。完整配置在protocol.json。新旧模型要做配对结果对照，绝不将上一轮的成绩混入本轮。

后台user systemd服务：`pi0-mw-hf-sft-download-proxy.service` 有网络代理，断点续传和重启；`pi0-mw-hf-sft-bootstrap.service` 等权重校验并做排除正式成绩的单回合策略验证；`pi0-mw-hf-sft-watchdog.service` 验证通过后调度最多两个GPU worker，统计结果。检查 `logs/watchdog_state.json`、`logs/download.log`、`logs/alerts.jsonl`。当前下载和评测过程不应因聊天结束而失去服务。

运行完毕后，结果在 `results/report.md`、`results/summary.csv` 和 `results/episodes.csv`。独立物理回放脚本 `diagnostics/replay_new.py` 应对正式100次进行交叉验证。`compare_runs.py` 对照上轮审计结果（29/100）；首轮原始结果31/100也保留在旧目录。

注意：RLinf 当前独立评测脚本基于MetaWorld v3；本用户指定的场景是v2，因此仍需标注跨版本适配限制。
