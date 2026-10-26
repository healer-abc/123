"""Select documented test cases and record a 90-second software-rendered demo."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for key, value in {
    "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    "LIBGL_ALWAYS_SOFTWARE": "1", "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe", "LP_NUM_THREADS": "2",
}.items():
    os.environ[key] = value

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import run

torch.set_num_threads(2)
P = json.loads((ROOT / "configs/experiment.json").read_text())
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT = ImageFont.truetype(FONT_PATH, 24)
SMALL = ImageFont.truetype(FONT_PATH, 21)
TITLE = ImageFont.truetype(FONT_PATH, 34)


def all_rows():
    rows = []
    for group in "ABCDE":
        for seed in P["training_seeds"]:
            with (ROOT / "evaluations" / P["run_prefix"] / f"{group}_seed{seed}/episodes.csv").open() as file:
                rows.extend({**r, "group": group, "training_seed": seed} for r in csv.DictReader(file))
    return rows


def select(rows):
    selected, used = [], set()
    # Demonstrate actual pushing: omit already-near-goal successes from the video.
    # This changes illustrative case selection only, never final evaluation statistics.
    initial_distances = {}
    env = run.make_env({"group": "D", "task": P["task"]})
    try:
        for seed in sorted({int(r["episode_seed"]) for r in rows}):
            obs, _ = env.reset(seed=seed)
            initial_distances[seed] = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
    finally:
        env.close()
    rows = [r for r in rows if not int(r["success"]) or initial_distances[int(r["episode_seed"])] > 0.08]
    targets = [lambda r: r["group"] == "D" and float(r["gain"]) == 1 and int(r["success"]),
               lambda r: r["group"] == "E" and float(r["gain"]) == 1 and int(r["success"]),
               lambda r: r["group"] == "E" and float(r["gain"]) <= 0.8 and int(r["success"]),
               lambda r: r["group"] == "E" and float(r["gain"]) >= 1.2 and int(r["success"]),
               lambda r: r["group"] == "D" and float(r["gain"]) != 1 and not int(r["success"]),
               lambda r: r["group"] == "E" and not int(r["success"])]
    def key(r):
        return r["group"], r["training_seed"], float(r["gain"]), int(r["episode_seed"])
    for predicate in targets:
        candidate = next((r for r in rows if predicate(r) and key(r) not in used), None)
        if candidate:
            selected.append(candidate)
            used.add(key(candidate))
    # An absent category is never fabricated. Fill with real remaining cases.
    for candidate in rows:
        if len(selected) == 6:
            break
        if key(candidate) not in used:
            selected.append(candidate)
            used.add(key(candidate))
    assert len(selected) == 6
    return selected


def canvas(frame, headline, subtitle):
    image = Image.new("RGB", (960, 720), "#13191f")
    image.paste(Image.fromarray(frame).resize((600, 600)), (180, 40))
    draw = ImageDraw.Draw(image)
    draw.text((24, 5), headline, font=FONT, fill="white")
    draw.text((24, 652), subtitle, font=SMALL, fill="white")
    return np.array(image)


def card(title, lines):
    image = Image.new("RGB", (960, 720), "#13191f")
    draw = ImageDraw.Draw(image)
    draw.text((45, 100), title, font=TITLE, fill="#8bc7ea")
    for i, line in enumerate(lines):
        draw.text((45, 210 + i * 64), line, font=FONT, fill="white")
    return np.array(image)


def record(case, index):
    directory = ROOT / "videos/cases"
    checkpoint = ROOT / "runs" / P["run_prefix"] / f"{case['group']}_seed{case['training_seed']}/final"
    model, config, bound = run.load_bundle(checkpoint, "cpu")
    gain, seed = float(case["gain"]), int(case["episode_seed"])
    env = run.make_env(config, gain=gain, render=True)
    frames, trace = [], []
    try:
        obs, _ = env.reset(seed=seed)
        initial, goal = obs["achieved_goal"].copy(), obs["desired_goal"].copy()
        headline = f"{case['group']} 训练 seed={case['training_seed']} | g={gain:.1f} | 场景 seed={seed}"
        frames.append(canvas(env.render(), headline, "真实测试案例；末端位移控制，25 Hz"))
        ever = False
        positions = []
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ever |= bool(info["is_success"])
            position = obs["achieved_goal"].copy()
            positions.append(position)
            distance = float(np.linalg.norm(position - goal))
            row = {"step": step + 1, "distance_m": distance, "reward": float(reward),
                   "success": int(bool(info["is_success"])), "gain": gain}
            for label, vector in (("object", position), ("goal", goal), ("gripper", obs["observation"][:3]), ("policy_action", action)):
                row.update({f"{label}_{i}": float(v) for i, v in enumerate(vector)})
            trace.append(row)
            frames.append(canvas(env.render(), headline, f"step={step+1}/50 | 当前距离={distance:.3f} m | 成功阈值 <0.05 m"))
            if terminated or truncated:
                break
        success = int(bool(info["is_success"]))
        assert len(trace) == 50 and success == int(case["success"])
        assert abs(distance - float(case["distance_m"])) < 5e-5
        moved = max(float(np.linalg.norm(p - initial)) for p in positions)
        initial_distance = float(np.linalg.norm(initial - goal))
        if success:
            assert initial_distance > 0.08 and moved > 0.01
        direction = goal - initial
        projection = float(np.dot(positions[-1] - initial, direction) / max(np.dot(direction, direction), 1e-12))
        if success:
            category = "成功"
        elif moved < 0.01:
            category = "未形成有效推动"
        elif ever and projection > 1:
            category = "推过目标"
        elif ever:
            category = "到达后离开目标"
        elif projection < 0:
            category = "方向偏离"
        else:
            category = "未完成精确到位"
        name = f"case_{index}_{case['group']}_s{case['training_seed']}_g{gain:.1f}_scene{seed}"
        with imageio.get_writer(directory / f"{name}.mp4", fps=25) as writer:
            for frame in frames:
                writer.append_data(frame)
        imageio.imwrite(directory / f"{name}.png", frames[-1])
        with (directory / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=trace[0])
            writer.writeheader()
            writer.writerows(trace)
        metadata = {**case, "gain": gain, "episode_seed": seed, "success": success,
                    "distance_m": distance, "initial_distance_m": initial_distance,
                    "max_object_displacement_m": moved, "category": category, "frames": 51,
                    "checkpoint": str(checkpoint), "model_sha256": hashlib.sha256((checkpoint / "model.zip").read_bytes()).hexdigest(),
                    "video": f"videos/cases/{name}.mp4", "still": f"videos/cases/{name}.png",
                    "trace": f"videos/cases/{name}.csv", "selection": "illustrative case from held-out evaluation; not an unbiased sample"}
        return frames, metadata
    finally:
        env.close()
        bound.close()


def main():
    output = ROOT / "videos/project_demo_90s.mp4"
    if output.exists():
        cases = json.loads((ROOT / "videos/cases.json").read_text())
        assert len(cases) == 6
        for case in cases:
            assert hashlib.sha256((Path(case["checkpoint"]) / "model.zip").read_bytes()).hexdigest() == case["model_sha256"]
            if int(case["success"]):
                assert case["initial_distance_m"] > 0.08 and case["max_object_displacement_m"] > 0.01
        with imageio.get_reader(output) as reader:
            assert reader.count_frames() == 2250 and reader.get_meta_data()["fps"] == 25
        print("Existing complete demo verified; nothing overwritten.")
        return
    directory = ROOT / "videos/cases"
    directory.mkdir(exist_ok=False)
    rows = all_rows()
    candidates = select(rows)
    metadata = []
    frame_count = 0
    with imageio.get_writer(output, fps=25) as writer:
        intro = card("机械臂推物：稀疏奖励与控制误差", ["MuJoCo FetchPush | 五组 × 三个训练种子", "输入：机器人、物体、目标状态；输出：末端位移", "视频是测试案例；完整表现以三种子统计为准", "每个案例先按 25 Hz 回放，再冻结末帧；重复标记明确"])
        for _ in range(250):
            writer.append_data(intro)
            frame_count += 1
        for i, case in enumerate(candidates, 1):
            frames, detail = record(case, i)
            metadata.append(detail)
            for repeat in range(3):
                for raw in frames:
                    image = Image.fromarray(raw.copy())
                    draw = ImageDraw.Draw(image)
                    draw.text((24, 686), f"同一回合回放 {repeat+1}/3；不是额外独立回合", font=SMALL, fill="#8bc7ea")
                    writer.append_data(np.array(image))
                    frame_count += 1
            frozen = Image.fromarray(frames[-1].copy())
            ImageDraw.Draw(frozen).text((24, 686), f"冻结末帧 | {detail['category']} | 最终距离 {detail['distance_m']:.3f} m", font=SMALL, fill="#8bc7ea")
            for _ in range(250 - len(frames) * 3):
                writer.append_data(np.array(frozen))
                frame_count += 1
        rates = {}
        for group in "DE":
            rates[group] = []
            for seed in P["training_seeds"]:
                nominal = [int(r["success"]) for r in rows if r["group"] == group and r["training_seed"] == seed and float(r["gain"]) == 1.0]
                rates[group].append(100 * float(np.mean(nominal)))
        summary = card("正常条件：全部测试回合统计", [f"D: {np.mean(rates['D']):.1f} ± {np.std(rates['D'], ddof=1):.1f}%", f"E: {np.mean(rates['E']):.1f} ± {np.std(rates['E'], ddof=1):.1f}%", "均值 ± 训练种子样本标准差；每模型 100 回合", "不把案例成功率或 300 回合当作独立训练重复"])
        closing = card("项目贡献与边界", ["统一训练恢复、独立评估与动作增益模块", "PPO / SAC / HER 核心采用 Stable-Baselines3", "状态输入、仿真位移控制；未做视觉或实机部署", "原始结果、图表、报告与代码均保留以便复现"])
        for image in (summary, closing):
            for _ in range(250):
                writer.append_data(image)
                frame_count += 1
    assert frame_count == 2250
    with imageio.get_reader(output) as reader:
        assert reader.count_frames() == 2250 and reader.get_meta_data()["fps"] == 25
    (ROOT / "videos/cases.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Generated 90 seconds / 2250 frames and {len(metadata)} documented cases")


if __name__ == "__main__":
    main()
