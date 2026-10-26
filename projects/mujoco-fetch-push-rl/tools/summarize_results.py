"""Produce final tables, three scientific figures, a six-page PDF and resume text."""
import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import textwrap
import unicodedata
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "configs/experiment.json").read_text())
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": FONT.get_name(), "axes.unicode_minus": False,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"A": "#555555", "B": "#D55E00", "C": "#A251A0", "D": "#0072B2", "E": "#00865C"}
STYLES = {"A": ":", "B": "--", "C": "-.", "D": "-", "E": "--"}


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    assert len(values) == 3
    return float(np.mean(values)), float(np.std(values, ddof=1))


def collect(groups="ABCDE"):
    records, curves = [], {}
    for group in groups:
        for seed in P["training_seeds"]:
            run = ROOT / "runs" / P["run_prefix"] / f"{group}_seed{seed}"
            evaluation = ROOT / "evaluations" / P["run_prefix"] / f"{group}_seed{seed}"
            summary = json.loads((run / "summary.json").read_text())
            system = json.loads((run / "system.json").read_text())
            config = json.loads((run / "config.json").read_text())
            assert summary["status"] == "final" and summary["timesteps"] >= P["formal_steps"]
            assert system["runner_sha256"] == P["runner_sha256"]
            assert config["group"] == group and config["seed"] == seed
            assert config["target_steps"] == P["formal_steps"] and config["task"] == "Push"
            rows = read_csv(evaluation / "episodes.csv")
            gains = P["test"]["robust_gains"] if group in "DE" else [1.0]
            assert len(rows) == 100 * len(gains)
            for gain in gains:
                subset = [row for row in rows if float(row["gain"]) == gain]
                assert len(subset) == 100
                assert {int(row["episode_seed"]) for row in subset} == set(range(20000, 20100))
                assert all(int(row["steps"]) == 50 for row in subset)
                distances = [float(row["distance_m"]) for row in subset]
                assert np.isfinite(distances).all()
                records.append({"group": group, "training_seed": seed, "gain": gain,
                                "success_rate": float(np.mean([int(row["success"]) for row in subset])),
                                "distance_m": float(np.mean(distances)), "test_episodes": 100,
                                "actual_steps": summary["timesteps"], "seconds": summary["session_seconds"]})
            curves[group, seed] = read_csv(run / "validation.csv")
    return records, curves


def axes(ax, xlabel):
    ax.set(xlabel=xlabel, ylabel="成功率 (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.2)


def save(fig, name):
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def figures(records, curves, include_robustness=True):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    curve_rows = []
    for group in "ABCD":
        maps = [{int(r["timesteps"]): float(r["success_rate"]) for r in curves[group, seed]} for seed in P["training_seeds"]]
        common = sorted(set.intersection(*(set(m) for m in maps)))
        assert common, group
        values = np.array([[m[t] * 100 for t in common] for m in maps])
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        for seed in P["training_seeds"]:
            rows = curves[group, seed]
            ax.plot([int(r["timesteps"]) for r in rows], [100 * float(r["success_rate"]) for r in rows],
                    color=COLORS[group], lw=0.7, alpha=0.2)
            curve_rows.extend({"group": group, "training_seed": seed, **r} for r in rows)
        ax.plot(common, mean, color=COLORS[group], ls=STYLES[group], lw=2, label=P["groups"][group])
        ax.fill_between(common, np.maximum(0, mean - sd), np.minimum(100, mean + sd), color=COLORS[group], alpha=0.1)
    axes(ax, "实际环境交互步数")
    ax.set_title("样本效率：算法、奖励与 HER 对照")
    ax.legend(fontsize=8)
    fig.text(0.1, 0.015, "细线：各训练种子；粗线：均值；阴影：±1 样本标准差（截于 0–100%），不是置信区间。", fontsize=8)
    save(fig, "learning_steps")
    write_csv(ROOT / "report/learning_curve_data.csv", curve_rows)

    fig, panels = plt.subplots(1, 3, figsize=(8.5, 3.7), sharey=True)
    for ax, seed in zip(panels, P["training_seeds"]):
        for group in "AB":
            rows = curves[group, seed]
            ax.plot([float(r["session_seconds"]) / 3600 for r in rows],
                    [100 * float(r["success_rate"]) for r in rows],
                    color=COLORS[group], ls=STYLES[group], lw=1.5, label=P["groups"][group])
        axes(ax, "单次运行小时")
        ax.set_title(f"训练 seed {seed}")
    panels[-1].legend(fontsize=7)
    fig.suptitle("实际耗时：PPO 与 SAC（相同密集奖励）")
    fig.text(0.07, 0.015, "共享 CPU，每进程两线程；按种子展示，未在不同采样时刻插值或平均。", fontsize=8)
    save(fig, "learning_time")

    if not include_robustness:
        return

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for group in "DE":
        gains = P["test"]["robust_gains"]
        values = np.array([[next(r["success_rate"] for r in records if r["group"] == group and r["training_seed"] == seed and r["gain"] == g) * 100 for g in gains] for seed in P["training_seeds"]])
        for row in values:
            ax.plot(gains, row, color=COLORS[group], alpha=0.22, lw=0.7)
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.plot(gains, mean, marker="o" if group == "D" else "s", ls=STYLES[group], color=COLORS[group], lw=2, label=P["groups"][group])
        ax.fill_between(gains, np.maximum(0, mean - sd), np.minimum(100, mean + sd), color=COLORS[group], alpha=0.12)
    axes(ax, "执行端动作增益 g")
    ax.set_xticks(P["test"]["robust_gains"])
    ax.axvspan(0.8, 1.2, color="#999999", alpha=0.08, label="E 组训练范围")
    ax.set_title("控制误差鲁棒性：HER 与增益随机化")
    ax.legend(fontsize=8)
    fig.text(0.1, 0.015, "每点：3 个训练种子×每模型 100 回合；阴影：±1 样本标准差（截于 0–100%）。", fontsize=8)
    save(fig, "robustness")


def wrap(text, width=94):
    lines, current, count = [], "", 0
    for char in text:
        weight = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if char == "\n" or count + weight > width:
            lines.append(current)
            current, count = "", 0
        if char != "\n":
            current += char
            count += weight
    if current:
        lines.append(current)
    return lines


def page(pdf, title, paragraphs, image=None, table=None, data_source="report/results_per_seed.csv"):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=17, weight="bold")
    y = 0.89
    if image:
        ax = fig.add_axes([0.07, 0.46, 0.86, 0.4])
        ax.imshow(plt.imread(image))
        ax.axis("off")
        y = 0.42
    if table:
        ax = fig.add_axes([0.07, 0.49, 0.86, 0.25])
        ax.axis("off")
        artist = ax.table(cellText=table[1], colLabels=table[0], loc="center", cellLoc="center")
        artist.auto_set_font_size(False)
        artist.set_fontsize(9)
        artist.scale(1, 1.7)
        y = 0.44
    for paragraph in paragraphs:
        for line in wrap(paragraph):
            fig.text(0.08, y, line, fontsize=10)
            y -= 0.021
        y -= 0.018
    assert y > 0.06, (title, y)
    fig.text(0.08, 0.035, f"MuJoCo FetchPush | 3 training seeds | data: {data_source}", fontsize=7, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    records, curves = collect()  # Refuse to create final claims from missing runs.
    write_csv(ROOT / "report/results_per_seed.csv", records)
    aggregate = []
    for group in "ABCDE":
        subset = [r for r in records if r["group"] == group and r["gain"] == 1.0]
        success, success_sd = stats([r["success_rate"] for r in subset])
        distance, distance_sd = stats([r["distance_m"] for r in subset])
        seconds, seconds_sd = stats([r["seconds"] for r in subset])
        aggregate.append({"group": group, "method": P["groups"][group], "n_training_seeds": 3,
                          "success_mean": success, "success_sd": success_sd,
                          "distance_mean_m": distance, "distance_sd_m": distance_sd,
                          "seconds_mean": seconds, "seconds_sd": seconds_sd,
                          "actual_steps_min": min(r["actual_steps"] for r in subset),
                          "actual_steps_max": max(r["actual_steps"] for r in subset)})
    write_csv(ROOT / "report/results.csv", aggregate)
    figures(records, curves)
    by_group = {r["group"]: r for r in aggregate}
    deltas = {f"{right}-{left}": 100 * (by_group[right]["success_mean"] - by_group[left]["success_mean"]) for left, right in (('A','B'),('B','C'),('C','D'),('D','E'))}
    ood = {}
    for group in "DE":
        values = [np.mean([r["success_rate"] for r in records if r["group"] == group and r["training_seed"] == s and r["gain"] in (0.6, 1.4)]) for s in P["training_seeds"]]
        ood[group] = stats(values)
    ood_delta = 100 * (ood['E'][0] - ood['D'][0])
    rows = [[r["group"], f"{100*r['success_mean']:.1f}±{100*r['success_sd']:.1f}",
             f"{r['distance_mean_m']:.3f}±{r['distance_sd_m']:.3f}", f"{r['seconds_mean']/3600:.2f}",
             f"{r['actual_steps_min']:,}"] for r in aggregate]
    table = ["| 组 | 正常成功率 %（均值±SD） | 最终距离 m | 平均耗时 h | 实际步数 |",
             "|---|---:|---:|---:|---:|"] + ["| " + " | ".join(row) + " |" for row in rows]
    cases = json.loads((ROOT / "videos/cases.json").read_text())
    assert (ROOT / "videos/project_demo_90s.mp4").is_file()
    analysis = f"在固定预算下，B−A 为 {deltas['B-A']:+.1f} 个百分点，C−B 为 {deltas['C-B']:+.1f}，D−C 为 {deltas['D-C']:+.1f}，E−D 为 {deltas['E-D']:+.1f}。差值来自三种子均值，不构成显著性检验；解释仅沿预定单因素对照展开。"
    robustness = f"训练范围外 g=0.6/1.4 的每种子平均成功率：D={100*ood['D'][0]:.1f}±{100*ood['D'][1]:.1f}%，E={100*ood['E'][0]:.1f}±{100*ood['E'][1]:.1f}%，E−D={ood_delta:+.1f} 个百分点。无论差值正负，都保留实际结果；不能据此宣称实机迁移能力。"
    interpretation = "本次最明确的对照是 C/D：HER 在固定配置和预算下提高了稀疏奖励成功率。B 的各种子为 53%/6%/6%，波动较大；D/E 未观察到增益随机化提高整体范围外成功率，不能写成鲁棒性提升。"
    diagnostic = ROOT / "checks/zero_action_diagnostic_20261023.json"
    diagnosis = ""
    if diagnostic.exists():
        control = json.loads(diagnostic.read_text())
        diagnosis = f"事后零动作诊断：相同测试场景不施加动作也有 {100*control['zero_action_success_rate']:.0f}% 成功率，A seed0 的逐回合最终距离与其一致；6% 本身不能证明学会推物。该诊断未用于调参或选模型。"
    gain_table = ["| 增益 g | D 成功率 %±SD | E 成功率 %±SD |", "|---:|---:|---:|"]
    for gain in P["test"]["robust_gains"]:
        cells = []
        for group in "DE":
            mean, sd = stats([r["success_rate"] for r in records if r["group"] == group and r["gain"] == gain])
            cells.append(f"{100*mean:.1f}±{100*sd:.1f}")
        gain_table.append(f"| {gain:.1f} | {cells[0]} | {cells[1]} |")
    method = "五组×三个训练种子，seed 0/1/2；每模型 100 万步，PPO 按 rollout 允许轻微超出。A/B/C 分别为 PPO 密集、SAC 密集、SAC 稀疏；D/E 为 SAC+HER 稀疏、再加入增益随机化。每 2 万步验证 20 回合，最终测试 seed 20000–20099，共 3,900 回合；测试集不用于调参或选模型。报告固定预算结束模型，不补做最优验证模型选择。"
    contribution = "工程贡献是统一训练与恢复、独立配对评估、动作增益模块、统计与可视化、可复现交付；PPO/SAC/HER 核心使用 Stable-Baselines3，不宣称从零实现算法。状态来自仿真器，动作是末端位移指令，不涉及视觉、关节力矩控制或实机部署。"
    limitation = "仅一个仿真任务、三个训练种子、固定参数预算。标准差描述训练种子的波动，不是置信区间。实际耗时在共享 CPU 上测量，可能受其他负载影响。增益实验是受限的执行误差模型，不覆盖延迟、传感噪声和全部动力学变化。"
    failure = "失败类型依据轨迹启发式划分为未形成有效推动、方向偏差、曾成功后离开目标和未完成精确到位；不是人工确认的接触检测。视频从测试轨迹选取案例，完整成功率仍以全部回合统计为准。"
    categories = Counter(case["category"] for case in cases)
    failure += "本次示例分类（只描述所选案例）：" + "、".join(f"{key} {count} 例" for key, count in categories.items()) + "。"
    markdown = ["# 机械臂推物策略学习实验报告", "", f"生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}", "", "## 任务与实验设计", "", method, "", "## 正常条件结果", "", *table, "", analysis, "", interpretation, "", diagnosis,
                "", "![A–D 学习曲线](../figures/learning_steps.png)", "", "![A/B 实际耗时](../figures/learning_time.png)", "", "## 控制误差鲁棒性", "", robustness,
                "", *gain_table, "", "![D/E 鲁棒性](../figures/robustness.png)", "", "## 方法与复现", "", contribution,
                "", "激活 `env.sh`；训练协议见 `configs/experiment.json`，原始逐回合结果见 `evaluations/`，原始源代码 SHA-256 和依赖见各训练目录。",
                "", "## 失败案例与局限", "", failure, "", limitation,
                "", f"示例视频共 {len(cases)} 个案例，选择与轨迹标签见 `videos/cases.json`；90 秒组合视频位于 `videos/project_demo_90s.mp4`。",
                "", "## 上游", "", "- https://robotics.farama.org/envs/fetch/push/", "- https://stable-baselines3.readthedocs.io/en/master/modules/her.html"]
    (ROOT / "report/REPORT.md").write_text("\n".join(markdown) + "\n")
    with (ROOT / "report/REPORT.md").open("a") as stream:
        stream.write("\n### 示例轨迹\n\n")
        for case in cases:
            stream.write(f"- {case['group']} seed={case['training_seed']}，g={case['gain']}，场景 {case['episode_seed']}：{case['category']}，最终距离 {case['distance_m']:.3f} m；轨迹 `{case['trace']}`。\n")
    with PdfPages(ROOT / "report/REPORT.pdf") as pdf:
        page(pdf, "1 / 6  机械臂推物：实验概览", [method, "表中成功率和距离按三个训练种子汇总；耗时为平均单次运行耗时。", contribution], table=(["组", "成功率 %±SD", "距离 m±SD", "耗时 h", "实际步数"], rows))
        page(pdf, "2 / 6  样本效率与奖励机制", [analysis, interpretation, diagnosis, "曲线以真实交互步数对齐。细线保留训练种子差异，阴影是样本标准差，不能用于显著性判断。跨奖励设置不比较累计回报。"], image=ROOT / "figures/learning_steps.png")
        page(pdf, "3 / 6  PPO 与 SAC 的实际耗时", ["两组都使用密集奖励，各隐藏层 256×256，学习率 3e-4，每进程两个 CPU 线程。不同算法更新规则按原协议保留。", "按训练种子分别展示耗时曲线，避免在不一致的评估时刻插值。横轴为各运行的真实训练时长，含训练内验证；共享服务器负载仍是限制。"], image=ROOT / "figures/learning_time.png")
        page(pdf, "4 / 6  控制误差鲁棒性", [robustness, "E 每回合抽取一次 g∈[0.8,1.2]，只缩放前三维动作并裁剪，目标和奖励定义不变；增益不输入策略。0.6/1.4 为范围外测试。D/E 使用相同测试场景种子。"], image=ROOT / "figures/robustness.png")
        page(pdf, "5 / 6  方法、实现与复现", [method, "SAC 更新：采样回放→计算目标 Q→更新两个 Q 网络→更新策略→更新熵系数与目标网络。HER 修改回放目标并重算奖励，不改变物理轨迹。", "恢复包含 SAC 回放池；时间截断保留 SB3 处理。恢复会重新初始化仿真，不是逐位确定性续跑。正式模型从零开始，调参种子与正式种子分开。", contribution, "配置：configs/experiment.json；冻结 runner、实际参数、源文件哈希、依赖清单和系统版本均保留。HER 实例练习：learning/her_relabel_samples.csv。"])
        page(pdf, "6 / 6  案例、局限与面试要点", [failure, f"视频包含 {len(cases)} 个真实案例。每个片段标明算法、增益、场景 seed 和最终距离；重复播放明确标记，不作为新增独立成功回合。案例清单和轨迹保存在 videos/。", limitation, "面试说明：比较因果关系只沿 A↔B、B↔C、C↔D、D↔E；HER 不能保证探索成功；正常性能和范围外鲁棒性的变化需要同时报告。", "实现与使用边界：集成成熟算法，自己完成实验对照、执行误差包装、评估、恢复和分析。最终量化结论以原始三种子统计为准。"])
    d = by_group['D']
    resume = f"基于 MuJoCo FetchPush 搭建机械臂强化学习实验，集成 PPO、SAC 与 HER，完成五组×三种子、每模型百万步对照及 3,900 回合独立评估；SAC+HER 正常成功率 {100*d['success_mean']:.1f}±{100*d['success_sd']:.1f}%，相对无 HER 稀疏奖励组差值 {deltas['D-C']:+.1f} 个百分点；实现动作增益随机化与配对评估，范围外（g=0.6/1.4）成功率 E−D 差值 {ood_delta:+.1f} 个百分点，交付可复现代码、统计图表与案例视频。"
    (ROOT / "report/RESUME.md").write_text("# 简历条目（仅使用已完成实验数值）\n\n" + resume + "\n\n边界：状态输入、仿真末端位移控制；不写视觉感知、力矩控制、实机迁移或从零实现算法。\n")
    print("Final report, six-page PDF, figures, tables and resume generated.")


if __name__ == "__main__":
    main()
