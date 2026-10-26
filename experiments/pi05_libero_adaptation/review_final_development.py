"""Audit frozen development artifacts and sample E3 videos before candidate selection."""
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import development_watchdog as helpers

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report/final_development_review"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with helpers.MANIFEST.open() as file:
        rows = [r for r in csv.DictReader(file) if r["phase"] == "development"]
    runs = json.loads((ROOT / "manifests/runs.json").read_text())["runs"]
    audit, records = {}, {}
    for run_id in ("E1-s42", "E2-s42", "E3-s42"):
        run = next(r for r in runs if r["run_id"] == run_id)
        values, motion_episodes, strict, failures = {}, 0, 0, []
        max_motion = max_gripper = 0.0
        for row in rows:
            key = row["suite"], int(row["task_id"]), int(row["init_state_index"])
            path = ROOT / "evaluations" / run_id / "development/episodes" / key[0] / f"task_{key[1]:02d}" / f"state_{key[2]:02d}.json"
            value = json.loads(path.read_text())
            assert value["status"] == "valid" and value["config"] == run["config"]
            assert value["checkpoint"] == run["final_checkpoint"]
            assert value["manifest_sha256"] == helpers.EXPECTED_MANIFEST
            assert all(str(value[k]) == str(v) for k, v in row.items())
            assert sha(value["trajectory"]) == value["trajectory_sha256"]
            with np.load(value["trajectory"]) as data:
                actions = data["actions"]
                assert actions.shape == (value["steps"], 7) and np.isfinite(actions).all()
                if run_id == "E3-s42":
                    np.testing.assert_array_equal(data["motion_controller_inputs"], np.clip(actions[:, :6], -1, 1))
                    np.testing.assert_array_equal(data["gripper_sign"], np.sign(actions[:, 6]))
            assert value["steps"] <= int(row["max_policy_steps"])
            motion_episodes += bool((np.abs(actions[:, :6]) > 1.000001).any())
            strict += bool(value["success"] and (np.abs(actions) <= 1.000001).all())
            max_motion = max(max_motion, float(np.abs(actions[:, :6]).max()))
            max_gripper = max(max_gripper, float(np.abs(actions[:, 6]).max()))
            if not value["success"]:
                assert value["steps"] == int(row["max_policy_steps"])
                assert sha(value["video"]) == value["video_sha256"]
                failures.append(value)
            values[key] = value
        records[run_id] = values
        audit[run_id] = {"episodes": len(values), "environment_successes": sum(v["success"] for v in values.values()),
                         "strict_successes": strict, "failures": len(failures),
                         "motion_exceeding_episodes": motion_episodes, "max_abs_motion": max_motion,
                         "max_abs_gripper": max_gripper, "manifest_and_trajectory_hashes_verified": True,
                         "failure_video_hashes_verified": True, "all_actions_finite": True}
    pairs = []
    for reference in ("E1-s42", "E2-s42"):
        discordant = []
        for key, value in records["E3-s42"].items():
            other = records[reference][key]
            if value["success"] != other["success"]:
                discordant.append({"suite": key[0], "task_id": key[1], "init_state_index": key[2],
                                   "e3_success": value["success"], "reference_success": other["success"],
                                   "failure_video": value["video"] or other["video"]})
        pairs.append({"reference": reference, "e3_only_successes": sum(p["e3_success"] for p in discordant),
                      "reference_only_successes": sum(p["reference_success"] for p in discordant), "pairs": discordant})
    helpers.atomic_json(OUT / "artifact_audit.json", {"created_at": helpers.now(), "runs": audit, "passed": True})
    helpers.atomic_json(OUT / "discordant_pairs.json", pairs)
    failures = [v for _, v in sorted(records["E3-s42"].items()) if not v["success"]]
    rng = np.random.default_rng(20261021)
    indices = sorted(rng.choice(len(failures), min(40, len(failures)), replace=False).tolist())
    sample = []
    for index, selected in enumerate(indices):
        value = failures[selected]
        key = value["suite"], value["task_id"], value["init_state_index"]
        sample.append({"index": index, "suite": key[0], "task_id": key[1], "init_state_index": key[2],
                       "task_description": value["task_description"], "video": value["video"],
                       "video_sha256": value["video_sha256"],
                       "e1_success": records["E1-s42"][key]["success"],
                       "e2_success": records["E2-s42"][key]["success"]})
    helpers.atomic_json(OUT / "failure_sample.json", {"seed": 20261021, "population": len(failures), "sample": sample})
    for page in range((len(sample) + 4) // 5):
        subset = sample[page * 5:(page + 1) * 5]
        sheet = Image.new("RGB", (8 * 160, len(subset) * 190), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, entry in enumerate(subset):
            reader = imageio.get_reader(entry["video"])
            frame_ids = np.linspace(0, reader.count_frames() - 1, 8, dtype=int)
            draw.text((3, row_index * 190 + 2),
                      f"#{entry['index']} {entry['suite']} t{entry['task_id']} s{entry['init_state_index']} E1={entry['e1_success']} E2={entry['e2_success']}", fill="black")
            for column, frame_id in enumerate(frame_ids):
                sheet.paste(Image.fromarray(reader.get_data(int(frame_id))).resize((160, 160)),
                            (column * 160, row_index * 190 + 24))
            reader.close()
        sheet.save(OUT / f"e3_failures_{page:02d}.jpg")
    print(json.dumps({"audit": audit, "sample": len(sample), "pairs": [{k: v for k, v in p.items() if k != "pairs"} for p in pairs]}, indent=2))


if __name__ == "__main__":
    main()
