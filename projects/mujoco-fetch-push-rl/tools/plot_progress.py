"""Plot a real pilot run, explicitly separated from formal three-seed figures."""
import csv
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/push_pilot_s99_300k_20261022")
with (directory / "validation.csv").open() as file:
    rows = list(csv.DictReader(file))
assert rows, "No real validation observations to plot yet"
plt.rcParams.update({"font.family": FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name(),
                     "axes.unicode_minus": False, "font.size": 10})
fig, panels = plt.subplots(1, 2, figsize=(8, 3.6))
x = [int(r["timesteps"]) for r in rows]
panels[0].plot(x, [100 * float(r["success_rate"]) for r in rows], "o-", color="#0072B2")
panels[0].set(ylabel="最终成功率 (%)", ylim=(0, 100))
panels[1].plot(x, [float(r["distance_m"]) for r in rows], "s-", color="#00865C")
panels[1].set(ylabel="最终距离 (m)", ylim=(0, None))
for ax in panels:
    ax.set_xlabel("累计训练步数")
    ax.grid(alpha=0.2)
fig.suptitle("Push 预实验 seed99：进行中，非正式三种子结果")
fig.text(0.07, 0.025, f"每点 20 个固定验证场景；数据来源：{directory.name}/validation.csv", fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(ROOT / "figures/pilot_progress.png", dpi=180)
plt.close(fig)
print(f"Pilot progress updated through {x[-1]} steps")
