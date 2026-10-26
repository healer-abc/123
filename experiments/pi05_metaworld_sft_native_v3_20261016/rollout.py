"""Run the verified RLinf Pi05 SFT policy through the Farama v3 evaluator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
V3 = BASE / "pi0_metaworld_native_v3_20261016" / "rollout.py"
PI05 = BASE / "pi05_metaworld_sft_20261016" / "run.py"
CHECKPOINT = BASE / "pi05_metaworld_sft_20261016" / "checkpoints/RLinf-Pi05-MetaWorld-SFT"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

reference = load("metaworld_v3_reference", V3)
pi05 = load("metaworld_pi05_policy", PI05)
sys.path.insert(0, str(V3.parent / "deps"))
sys.path.insert(0, str(V3.parent / "sources"))
sys.path.insert(0, str(V3.parent / "runtime"))
protocol = json.loads((ROOT / "protocol.json").read_text())
protocol_hash = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
reference.ROOT = ROOT
reference.PROTOCOL = protocol
reference.PROTOCOL_HASH = protocol_hash
reference.CHECKPOINT = CHECKPOINT
reference.build_policy = pi05.build_policy
pi05.ROOT = ROOT
pi05.P = protocol
pi05.PHASH = protocol_hash
pi05.CHECKPOINT = CHECKPOINT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=protocol["tasks"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-group", default="results")
    args = parser.parse_args()
    assert CHECKPOINT.joinpath("verified.json").is_file()
    reference.evaluate(args.task, args.limit, args.output_group)

if __name__ == "__main__":
    main()
