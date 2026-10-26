"""Compare LeRobot values and decoded images with their actual source episodes."""
import argparse
import json
import os
from pathlib import Path
import sys
import h5py
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
sys.path.insert(0, str(root / "source_materials"))
from decode_image_bit import decode_image_bit

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", default="local/robotwin_pi0_interface")
parser.add_argument("--report", default="converted-dataset-check.json")
args = parser.parse_args()
folder = Path(os.environ["HF_LEROBOT_HOME"]) / args.repo_id
manifest = [json.loads(line) for line in (folder / "meta/source_episodes.jsonl").read_text().splitlines()]
dataset = LeRobotDataset(args.repo_id)
offset = 0
results = []
for row in manifest:
    index = row["episode_index"]
    table = pq.read_table(folder / f"data/chunk-{index//1000:03d}/episode_{index:06d}.parquet", columns=["observation.state", "action"]).to_pydict()
    with h5py.File(row["source"], "r") as original:
        recorded = np.asarray(original["joint_action/vector"], dtype=np.float32)
        state = recorded[:-1]
        np.testing.assert_array_equal(np.asarray(table["observation.state"]), state)
        np.testing.assert_array_equal(np.asarray(table["action"]), recorded[1:])
        frames = sorted({0, len(state)//2, len(state)-1})
        for frame in frames:
            loaded = dataset[offset + frame]
            for raw, camera in [("head_camera", "cam_high"), ("left_camera", "cam_left_wrist"), ("right_camera", "cam_right_wrist")]:
                expected = decode_image_bit(original[f"observation/{raw}/rgb"][frame])
                actual = loaded[f"observation.images.{camera}"].permute(1, 2, 0).numpy()
                np.testing.assert_allclose(actual, expected.astype(np.float32)/255, rtol=0, atol=1e-6)
    results.append({"episode_index": index, "seed": row["seed"], "variant": row["variant"], "frames": len(state), "checked_image_frames": frames})
    offset += len(state)
assert len(dataset) == offset
result = {"status": "passed", "repo_id": args.repo_id, "episodes": results, "total_frames": offset, "nominal_fps": dataset.fps}
(root / "report" / args.report).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
