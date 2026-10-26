"""Retain each pool's collection protocol and reject changes during continuation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--gpu")
parser.add_argument("--count")
parser.add_argument("--record-attempt", action="store_true")
args = parser.parse_args()
robo = Path(os.environ["ROBOTWIN_PATH"])
path = robo / "task_config" / f"{args.config}.yml"
content = path.read_bytes()
cfg = yaml.safe_load(content)
folder = (robo / cfg["save_path"] / "adjust_bottle" / args.config).resolve()
folder.mkdir(parents=True, exist_ok=True)
snapshot = folder / "collection_config.yml"
if snapshot.exists():
    assert snapshot.read_bytes() == content, f"Collection config changed in existing pool: {folder}"
else:
    snapshot.write_bytes(content)
files = ["envs/_base_task.py", "envs/adjust_bottle.py", "envs/robot/robot.py", "envs/robot/planner.py", "assets/embodiments/aloha-agilex/config.yml", "task_config/_camera_config.yml"]
protocol = {"config_sha256": hashlib.sha256(content).hexdigest(), "denoiser": os.environ["ROBOTWIN_DENOISER"], "packages": {p: importlib.metadata.version(p) for p in ["sapien", "nvidia-curobo", "numpy"]}, "code_sha256": {f: hashlib.sha256((robo/f).read_bytes()).hexdigest() for f in files if (robo/f).exists()}}
meta = folder / "collection_protocol.json"
if meta.exists():
    assert json.loads(meta.read_text())["protocol"] == protocol, f"Collection code/runtime changed in existing pool: {folder}"
else:
    meta.write_text(json.dumps({"recorded_utc": datetime.now(timezone.utc).isoformat(), "raw_data_preexists_snapshot": any((folder/'data').glob('episode*.hdf5')), "protocol": protocol}, indent=2) + "\n")
if args.record_attempt:
    with (folder / "collection_attempts.jsonl").open("a") as stream:
        stream.write(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(), "gpu": args.gpu, "target": int(args.count) if args.count else cfg["episode_num"], "config_sha256": protocol["config_sha256"]}) + "\n")
print(folder)
