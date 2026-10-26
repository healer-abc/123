"""Audit existing actions and prepare reproducible failure-video contact sheets."""
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report/development_review"


def identity(record):
    return record["suite"], record["task_id"], record["init_state_index"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, failures, bounds = {}, [], []
    for run in ("E1-s42", "E2-s42"):
        paths = sorted((ROOT / "evaluations" / run / "development/episodes").glob("*/*/*.json"))
        values = [json.loads(p.read_text()) for p in paths]
        runs[run] = {identity(v): v for v in values}
        for value in values:
            with np.load(value["trajectory"]) as data:
                actions = data["actions"]
            assert np.isfinite(actions).all()
            motion = np.abs(actions[:, :6]) > 1.000001
            gripper = np.abs(actions[:, 6]) > 1.000001
            bounds.append({"run_id": run, "suite": value["suite"], "task_id": value["task_id"],
                           "init_state_index": value["init_state_index"], "environment_success": value["success"],
                           "raw_bound_strict_success": bool(value["success"] and not (motion.any() or gripper.any())),
                           "motion_exceeding_steps": int(motion.any(axis=1).sum()),
                           "gripper_exceeding_steps": int(gripper.sum()), "steps": len(actions),
                           "max_abs_motion": float(np.abs(actions[:, :6]).max()),
                           "max_abs_gripper": float(np.abs(actions[:, 6]).max())})
            if not value["success"]:
                failures.append({"run_id": run, **value})
    with (OUT / "action_bounds.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(bounds[0]))
        writer.writeheader()
        writer.writerows(bounds)
    summary = {}
    for run in runs:
        subset = [r for r in bounds if r["run_id"] == run]
        summary[run] = {"episodes": len(subset),
                        "environment_successes": sum(r["environment_success"] for r in subset),
                        "raw_bound_strict_successes": sum(r["raw_bound_strict_success"] for r in subset),
                        "motion_exceeding_episodes": sum(r["motion_exceeding_steps"] > 0 for r in subset),
                        "gripper_exceeding_episodes": sum(r["gripper_exceeding_steps"] > 0 for r in subset),
                        "max_abs_motion": max(r["max_abs_motion"] for r in subset),
                        "max_abs_gripper": max(r["max_abs_gripper"] for r in subset)}
    pairs = []
    for key, e1 in runs["E1-s42"].items():
        e2 = runs["E2-s42"][key]
        if e1["success"] != e2["success"]:
            pairs.append({"suite": key[0], "task_id": key[1], "init_state_index": key[2],
                          "task_description": e1["task_description"], "e1_success": e1["success"],
                          "e2_success": e2["success"], "failure_video": e2["video"] if e1["success"] else e1["video"],
                          "e1_trajectory": e1["trajectory"], "e2_trajectory": e2["trajectory"]})
    (OUT / "action_bounds_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "discordant_pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    # A fixed sample of at most 40 failures, as specified in the original protocol.
    failures.sort(key=lambda r: (r["run_id"], *identity(r)))
    rng = np.random.default_rng(20261018)
    selected = sorted(rng.choice(len(failures), min(40, len(failures)), replace=False).tolist())
    sample = [failures[i] for i in selected]
    index = []
    for page in range((len(sample) + 4) // 5):
        page_rows = sample[page * 5:(page + 1) * 5]
        sheet = Image.new("RGB", (224 * 5, len(page_rows) * 252), "white")
        draw = ImageDraw.Draw(sheet)
        for row_idx, record in enumerate(page_rows):
            video = Path(record["video"])
            assert hashlib.sha256(video.read_bytes()).hexdigest() == record["video_sha256"]
            reader = imageio.get_reader(video)
            count = reader.count_frames()
            frame_ids = np.linspace(0, count - 1, 5, dtype=int)
            label = f"{record['run_id']} {record['suite']} task {record['task_id']} state {record['init_state_index']}"
            draw.text((3, row_idx * 252 + 2), label, fill="black")
            for column, frame in enumerate(frame_ids):
                sheet.paste(Image.fromarray(reader.get_data(int(frame))).resize((224, 224)),
                            (column * 224, row_idx * 252 + 23))
            reader.close()
            index.append({"index": page * 5 + row_idx, "page": page, "row": row_idx, "label": label,
                          "task_description": record["task_description"], "video": str(video),
                          "run_id": record["run_id"], "suite": record["suite"],
                          "task_id": record["task_id"], "init_state_index": record["init_state_index"]})
        sheet.save(OUT / f"failures_{page:02d}.jpg")
    (OUT / "failure_sample.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"bounds": summary, "discordant_pairs": len(pairs), "failure_sample": len(index)}, indent=2))


if __name__ == "__main__":
    main()
