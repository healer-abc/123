"""Convert retained RLinf_support episodes to the Pi0 LeRobot v2.1 contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
sys.path.insert(0, str(ROOT / "source_materials"))
from decode_image_bit import decode_image_bit

CAMERAS = {"head_camera": "cam_high", "left_camera": "cam_left_wrist", "right_camera": "cam_right_wrist"}
JOINTS = [f"{side}_{name}" for side in ["left", "right"] for name in ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="local/robotwin_pi0_interface")
    parser.add_argument("--max-episodes-per-split", type=int, default=1)
    parser.add_argument("--source-root", type=Path, default=Path(os.environ["ROBOTWIN_WORKSPACE"]) / "data/robotwin_pi0/pilot")
    parser.add_argument("--splits", nargs="+", choices=["clean", "randomized"], default=["clean", "randomized"])
    parser.add_argument("--config-suffix", default="")
    parser.add_argument("--episode-selection", type=Path, help="JSON mapping each split to selected raw episode indices")
    args = parser.parse_args()
    selection = json.loads(args.episode_selection.read_text()) if args.episode_selection else None
    output = Path(os.environ["HF_LEROBOT_HOME"]) / args.repo_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    sources = []
    for split in args.splits:
        folder = args.source_root / split / "adjust_bottle" / f"demo_{split}_pi0{args.config_suffix}"
        seeds = [int(s) for s in (folder / "seed.txt").read_text().split()]
        indices = selection[split] if selection else range(args.max_episodes_per_split)
        assert len(set(indices)) == len(indices), f"Duplicate source episode in {split}"
        for index in indices:
            path = folder / "data" / f"episode{index}.hdf5"
            instruction_path = folder / "instructions" / f"episode{index}.json"
            assert path.is_file() and instruction_path.is_file(), f"Incomplete source episode: {path}"
            descriptions = json.loads(instruction_path.read_text())
            assert descriptions.get("seen"), instruction_path
            snapshot = folder / "collection_config.yml"
            config_path = snapshot if snapshot.exists() else ROOT / "configs" / f"demo_{split}_pi0{args.config_suffix}.yml"
            config = yaml.safe_load(config_path.read_text())
            sources.append((split, path, descriptions["seen"][0], seeds[index], config_path, config))
    staging = output.with_name(output.name + ".building")
    dataset = None
    manifest = []
    for episode_index, (split, path, instruction, seed, config_path, config) in enumerate(sources):
        with h5py.File(path, "r") as source:
            recorded = np.asarray(source["joint_action/vector"], dtype=np.float32)
            assert recorded.ndim == 2 and recorded.shape[1] == 14 and len(recorded) > 1, recorded.shape
            assert np.isfinite(recorded).all(), path
            state, action = recorded[:-1], recorded[1:]
            images = {dest: decode_image_bit(source[f"observation/{src}/rgb"][:-1]) for src, dest in CAMERAS.items()}
        for camera, frames in images.items():
            assert len(frames) == len(state) and frames.dtype == np.uint8 and frames.shape[-1] == 3, (camera, frames.shape)
            assert frames.std() > 1, f"Constant image stream: {path}, {camera}"
        if dataset is None:
            features = {
                "observation.state": {"dtype": "float32", "shape": (14,), "names": [JOINTS]},
                "action": {"dtype": "float32", "shape": (14,), "names": [JOINTS]},
            }
            for camera, frames in images.items():
                h, w = frames.shape[1:3]
                features[f"observation.images.{camera}"] = {"dtype": "image", "shape": (3, h, w), "names": ["channels", "height", "width"]}
            # Match the official dataset's nominal metadata clock. This does
            # not assert that legacy physics samples were uniformly 50 Hz.
            dataset = LeRobotDataset.create(args.repo_id, fps=50, features=features, root=staging, robot_type="aloha", use_videos=False, image_writer_threads=4)
        for frame_index in range(len(state)):
            frame = {"observation.state": state[frame_index], "action": action[frame_index]}
            frame.update({f"observation.images.{camera}": frames[frame_index] for camera, frames in images.items()})
            dataset.add_frame(frame, task=instruction)
        dataset.save_episode()
        manifest.append({"episode_index": episode_index, "source": str(path.resolve()), "source_sha256": sha256(path), "variant": split, "seed": seed, "source_frames": len(recorded), "frames": len(state), "instruction": instruction, "action_alignment": "state=recorded[:-1]; action=recorded[1:]; no invented terminal target", "nominal_fps": 50, "config": str(config_path), "config_sha256": sha256(config_path), "physics_timestep": config.get("timestep", 1/250), "save_freq": config["save_freq"], "denoiser": os.environ["ROBOTWIN_DENOISER"], "planner": config["planner_backend"]})
        print(f"converted {split} seed={seed} frames={len(state)}", flush=True)
    assert dataset is not None
    dataset.stop_image_writer()
    (staging / "meta/source_episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    staging.rename(output)
    print(output, flush=True)

if __name__ == "__main__":
    main()
