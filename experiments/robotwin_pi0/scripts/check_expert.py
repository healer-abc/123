"""Run one official expert episode with a bounded external timeout."""
import argparse
import faulthandler
import importlib.util
import os
from pathlib import Path
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="demo_clean_pi0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--backend", choices=["mplib", "curobo"], default="curobo")
options = parser.parse_args()
root = Path(os.environ["ROBOTWIN_PATH"])
os.chdir(root)
spec = importlib.util.spec_from_file_location("official_collect", root / "script/collect_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
faulthandler.dump_traceback_later(60, repeat=True)

def probe(task, args):
    args.update(need_plan=True, save_data=False, render_freq=0, planner_backend=options.backend)
    start = time.monotonic()
    try:
        print(f"expert setup seed={options.seed}", flush=True)
        task.setup_demo(now_ep_num=0, seed=options.seed, **args)
        print(f"setup complete {time.monotonic()-start:.2f}s", flush=True)
        task.play_once()
        assert task.plan_success and task.check_success(), "Expert episode failed"
        print(f"expert complete {time.monotonic()-start:.2f}s plan={task.plan_success} success={task.check_success()}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        task.close_env()
        faulthandler.cancel_dump_traceback_later()

module.run = probe
module.main(task_name="adjust_bottle", task_config=options.config)
