"""Execute the official collector for an incremental target in one config."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("count", type=int)
args = parser.parse_args()
robotwin = Path(os.environ["ROBOTWIN_PATH"])
os.chdir(robotwin)
sys.path.insert(0, str(robotwin / "script"))
from test_render import Sapien_TEST
Sapien_TEST()
mp.set_start_method("spawn", force=True)
spec = importlib.util.spec_from_file_location("official_collect", robotwin / "script/collect_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_run = module.run

def run(task, config):
    assert 0 < args.count <= config["episode_num"], (args.count, config["episode_num"])
    config["episode_num"] = args.count
    return original_run(task, config)

module.run = run
module.main(task_name="adjust_bottle", task_config=args.config)
