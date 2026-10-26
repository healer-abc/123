"""Export the completed A-D results without relaxing the final five-group gate."""
from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

import summarize_results as report

ROOT, P = report.ROOT, report.P


def partial_page(*args, **kwargs):
    report.page(*args, **kwargs, data_source="report/results_ABCD_per_seed_20261023.csv")


def main():
    records, curves = report.collect(groups="ABCD")
    # Validate checkpoint completion and evaluation identity as well as row counts.
    for group in "ABCD":
        for seed in P["training_seeds"]:
            run = ROOT / "runs" / P["run_prefix"] / f"{group}_seed{seed}"
            marker = json.loads((run / "final/complete.json").read_text())
            summary = json.loads((run / "summary.json").read_text())
            evaluation = ROOT / "evaluations" / P["run_prefix"] / run.name
            identity = json.loads((evaluation / "summary.json").read_text())
            assert marker["steps"] == summary["timesteps"] < P["formal_steps"] + 2048
            assert (run / "final/model.zip").is_file()
            assert identity["checkpoint"] == str((run / "final").resolve())
            assert identity["seed_base"] == P["test"]["seed_base"]
    aggregate = []
    for group in "ABCD":
        rows = [r for r in records if r["group"] == group and r["gain"] == 1.0]
        mean, sd = report.stats([r["success_rate"] for r in rows])
        aggregate.append({"group": group, "method": P["groups"][group],
                          "success_mean": mean, "success_sd": sd,
                          "distance_mean_m": float(np.mean([r["distance_m"] for r in rows])),
                          "hours_mean": float(np.mean([r["seconds"] for r in rows])) / 3600,
                          "seed_success_rates": "/".join(f"{r['success_rate']:.2f}" for r in rows)})
    report.write_csv(ROOT / "report/results_ABCD_20261023.csv", aggregate)
    report.write_csv(ROOT / "report/results_ABCD_per_seed_20261023.csv", records)
    report.figures(records, curves, include_robustness=False)

    gains = P["test"]["robust_gains"]
    values = np.array([[next(r["success_rate"] for r in records if r["group"] == "D" and r["training_seed"] == seed and r["gain"] == gain) * 100 for gain in gains] for seed in P["training_seeds"]])
    means, sds = values.mean(axis=0), values.std(axis=0, ddof=1)
    robustness = [{"gain": g, "success_mean": mean / 100, "success_sd": sd / 100,
                   "seed0": values[0, i] / 100, "seed1": values[1, i] / 100,
                   "seed2": values[2, i] / 100} for i, (g, mean, sd) in enumerate(zip(gains, means, sds))]
    report.write_csv(ROOT / "report/robustness_D_only_20261023.csv", robustness)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for seed, row in zip(P["training_seeds"], values):
        ax.plot(gains, row, alpha=0.55, lw=1, ls=":", marker=".", label=f"D seed {seed}")
    ax.plot(gains, means, color=report.COLORS["D"], lw=2, marker="o", label="D 均值")
    ax.fill_between(gains, np.maximum(0, means-sds), np.minimum(100, means+sds), color=report.COLORS["D"], alpha=0.12)
    report.axes(ax, "执行端动作增益 g")
    ax.set_xticks(gains)
    ax.set_title("HER 基线的控制误差测试（E 组待完成）")
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.1, 0.015, "每模型每增益 100 回合；阴影为三训练种子样本标准差，截于 0–100%；不推断随机化收益。", fontsize=8)
    report.save(fig, "robustness_D_only_20261023")

    table_rows = [[r["group"], r["seed_success_rates"], f"{100*r['success_mean']:.1f}±{100*r['success_sd']:.1f}",
                   f"{100*r['distance_mean_m']:.2f}", f"{r['hours_mean']:.2f}"] for r in aggregate]
    table = ["| 组 | seed0/1/2 成功率（比例） | 成功率 %（均值±SD） | 最终距离 cm | 单次训练 h |",
             "|---|---|---:|---:|---:|"] + ["| " + " | ".join(row) + " |" for row in table_rows]
    d, c = aggregate[3], aggregate[2]
    delta = 100 * (d["success_mean"] - c["success_mean"])
    method = "A PPO 密集、B SAC 密集、C SAC 稀疏、D SAC+HER 稀疏；每组三个训练种子 0/1/2，各训练 100 万步（PPO 实际 1,001,472 步）。每模型正常条件测试 100 回合，独立场景 seed 20000–20099；成功以第 50 步物体距目标小于 5 cm 为准。D 另测五档增益，已完成 12 个模型、共 2,400 个最终测试回合。"
    finding = f"固定配置与预算下，HER 使稀疏奖励成功率由 6.0% 提升至 {100*d['success_mean']:.1f}%，差值 {delta:+.1f} 个百分点；D 平均最终距离为 {100*d['distance_mean_m']:.2f} cm。B 的三个训练种子分别为 53%、6%、6%，均值高于 A，但波动很大，不能宣称 SAC 稳定优于 PPO。"
    pending = "E（SAC+HER+回合级增益随机化）尚无最终结果。三个进程于 10 月 23 日 00:12 左右中断，原因未确认；因未到首次检查点，07:30 已从零重启相同种子和参数，旧数据完整归档。仅将 E 检查点间隔缩至 10 万步。完整 D/E 对照、最终六页报告、简历定稿和 90 秒案例视频仍待 E 完成；不将训练中验证结果作为最终测试。"
    limitation = "统计为三个训练种子的均值±样本标准差（ddof=1），不是置信区间，也不构成显著性检验。测试场景固定配对；结论仅适用于本任务、参数和预算。共享 CPU 耗时包含训练内验证且受其他负载影响，不能作公平硬件速度基准。预实验与短训练不纳入正式结果。"
    contribution = "工程内容包括 conda 环境与版本锁定、统一训练和模型/回放保存恢复、独立配对评估、执行端动作增益包装、HER 重标记练习、统计图表与交付自动化。算法核心使用 Stable-Baselines3；输入是仿真状态，输出为末端位移指令。"
    diagnostic = json.loads((ROOT / "checks/zero_action_diagnostic_20261023.json").read_text())
    diagnosis = f"事后零动作诊断：相同 100 个测试场景不施加动作，也得到 {100*diagnostic['zero_action_success_rate']:.0f}% 成功率；A seed0 的逐回合最终距离与零动作结果最大差值为 {diagnostic['A_seed0_max_distance_difference_from_zero_action_m']:.2g} m。因此 6% 不足以表明学会推物。该诊断未用于调参或选择模型，原始 CSV 保留在 checks/zero_action_diagnostic_20261023.csv。"
    resume = f"基于 MuJoCo FetchPush 搭建机械臂强化学习实验，集成 PPO、SAC 与 HER，完成 4 组×3 种子、每模型百万步对照及 2,400 回合独立评估；SAC+HER 成功率达 {100*d['success_mean']:.1f}±{100*d['success_sd']:.1f}%，较无 HER 稀疏奖励基线提升 {delta:.1f} 个百分点；实现模型与回放恢复、动作增益误差测试及统计可视化。"
    lines = ["# MuJoCo 机械臂推物项目结果汇总（A–D 已完成，E 补跑中）", "", f"生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}", "", method, "", "## 正常条件结果", "", *table, "", finding, "", diagnosis,
             "", "![A–D 学习曲线](../figures/learning_steps.png)", "", "曲线为每 2 万步、20 回合验证；最终表格为独立 100 回合测试。细线为各种子，粗线及阴影为均值±样本标准差。",
             "", "![PPO/SAC 实际耗时](../figures/learning_time.png)", "", "## 控制误差测试：目前仅 D 组", "", "| 动作增益 | D 成功率 %（均值±SD） |", "|---:|---:|"]
    lines += [f"| {r['gain']:.1f} | {100*r['success_mean']:.1f}±{100*r['success_sd']:.1f} |" for r in robustness]
    lines += ["", "![D 组增益测试](../figures/robustness_D_only_20261023.png)", "", "D 在 g=0.6 时成功率下降至 85.0%；当前数据只能描述 HER 基线对执行误差的表现，E 完成前不能判断增益随机化是否有效。", "", "## 完成范围和待完成部分", "", contribution, "", pending, "", limitation,
              "", "## 当前可用简历表述", "", resume, "", "当前条目仅陈述 A–D 已完成工作；完整五组及随机化收益待 E 最终评估后补充。",
              "", "## 数据和复现", "", "- 汇总表：`results_ABCD_20261023.csv`；分种子/增益统计：`results_ABCD_per_seed_20261023.csv`。",
              "- 逐回合原始测试：`../evaluations/formal_20261022/{A,B,C,D}_seed{0,1,2}/episodes.csv`。",
              "- 配置与来源：`../configs/experiment.json`、冻结 runner、各 run 的 config/system/summary 和 final/complete.json。",
              "- 中断与恢复记录：`recovery_20261023.json`；旧 E 输出：`../archive/E_interruption_20261023_073019/`。",
              "- 激活：`source env.sh`；重建本汇总：`python tools/summarize_completed_abcd.py`。完整报告生成器仍要求五组结果齐全。"]
    (ROOT / "report/PROJECT_RESULTS_20261023.md").write_text("\n".join(lines) + "\n")
    (ROOT / "report/RESUME_ABCD_20261023.md").write_text("# 当前已完成部分的简历条目\n\n" + resume + "\n\n增益随机化扩展仍在补跑，暂不声称其提高鲁棒性。\n")
    with PdfPages(ROOT / "report/PROJECT_RESULTS_20261023.pdf") as pdf:
        partial_page(pdf, "1 / 4  已完成结果：HER 消融效果明确", [method, finding, "本文件为 A–D 阶段汇总，E 尚未完成；全项目最终报告另行生成。"], table=(["组", "各种子成功比例", "成功率 %±SD", "距离 cm", "耗时 h"], table_rows))
        partial_page(pdf, "2 / 4  固定预算下的样本效率", [finding, diagnosis, limitation], image=ROOT / "figures/learning_steps.png")
        partial_page(pdf, "3 / 4  PPO/SAC 的实测训练耗时", ["两组均用密集奖励。分别展示每个种子，避免对不一致的评估时刻做插值平均。", limitation, contribution], image=ROOT / "figures/learning_time.png")
        partial_page(pdf, "4 / 4  控制误差基线与待完成范围", ["D 在 g=0.6/0.8/1.0/1.2/1.4 的三种子平均成功率分别为 85.0/98.0/98.7/98.0/94.7%。这不是随机化训练的收益对照。", pending, "当前简历条目见 RESUME_ABCD_20261023.md；原始测试与模型均保留。"], image=ROOT / "figures/robustness_D_only_20261023.png")
    assert sum(r["test_episodes"] for r in records) == 2400
    evidence = {"completed_models": 12, "test_episodes": 2400, "groups": "ABCD", "E_final_result_available": False,
                "runner_sha256": hashlib.sha256((ROOT / P["runner"]).read_bytes()).hexdigest(), "rows": aggregate}
    (ROOT / "checks/result_summary_20261023.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
