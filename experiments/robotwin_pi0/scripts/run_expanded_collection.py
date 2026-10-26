"""Collect and validate two 200-episode pools after the interface gate passes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
WORKSPACE = Path(os.environ["ROBOTWIN_WORKSPACE"])
gate = json.loads((ROOT / "report/data-interface-check.json").read_text())
assert gate["status"] == "passed", "Interface validation must complete first"
lock = threading.Lock()
stop = threading.Event()
state = {"status": "running", "target_per_variant": 200, "variants": {}, "interface_gate": gate}

def record(variant=None, **fields):
    with lock:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        if variant is None:
            state.update(fields)
        else:
            state["variants"].setdefault(variant, {}).update(fields)
        path = ROOT / "report/expanded-collection-state.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(path)

def collect(variant, gpu):
    name = f"demo_{variant}_pi0_train200"
    folder = WORKSPACE / "data/robotwin_pi0/raw" / variant / "adjust_bottle" / name
    log_path = ROOT / "logs/collection" / f"{name}-driver.log"
    existing = 0
    while (folder / "data" / f"episode{existing}.hdf5").is_file() and (folder / "instructions" / f"episode{existing}.json").is_file():
        existing += 1
    record(variant, completed_episodes=existing, gpu=gpu)
    with log_path.open("a") as log:
        for count in [10, 20, 30, 40, 50, 100, 150, 200]:
            if existing >= count:
                continue
            if stop.is_set():
                raise RuntimeError("Other variant failed; no further batch started")
            child = subprocess.Popen(["bash", str(ROOT / "scripts/run_collection.sh"), name, str(gpu), str(count)], cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
            record(variant, phase="collecting", current_target=count, gpu=gpu, pid=child.pid, log=str(log_path))
            code = child.wait()
            if code:
                stop.set()
                raise RuntimeError(f"{name} target {count} failed with exit {code}")
            record(variant, completed_episodes=count, phase="batch_validated", current_target=count)
            existing = count
        repo_id = f"local/robotwin_pi0_{variant}200"
        record(variant, phase="converting", repo_id=repo_id)
        output = Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
        if not output.exists():
            staging = output.with_name(output.name + ".building")
            if staging.exists():
                staging.rename(staging.with_name(staging.name + ".interrupted-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")))
            subprocess.run([sys.executable, str(ROOT / "scripts/convert_legacy_lerobot.py"), "--source-root", str(WORKSPACE / "data/robotwin_pi0/raw"), "--config-suffix", "_train200", "--splits", variant, "--max-episodes-per-split", "200", "--repo-id", repo_id], check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, str(ROOT / "scripts/check_converted_dataset.py"), "--repo-id", repo_id, "--report", f"{variant}200-converted-check.json"], check=True, stdout=log, stderr=subprocess.STDOUT)
        norm_dir = ROOT / "report" / f"{variant}200_norm_stats"
        record(variant, phase="computing_training_stats")
        subprocess.run([sys.executable, str(ROOT / "scripts/compute_norm_stats.py"), "--repo-id", repo_id, "--output", str(norm_dir)], check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, str(ROOT / "scripts/check_sft_loader.py"), "--dataset", repo_id, "--norm-stats", str(norm_dir / "norm_stats.json"), "--report", f"{variant}200-loader-check.json"], check=True, stdout=log, stderr=subprocess.STDOUT)
        record(variant, phase="complete", completed_episodes=200, norm_stats=str(norm_dir / "norm_stats.json"), raw_dataset=str(folder))

(ROOT / "logs/collection").mkdir(parents=True, exist_ok=True)
record()
errors = []
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = {pool.submit(collect, variant, gpu): variant for variant, gpu in [("clean", 0), ("randomized", 1)]}
    for future in as_completed(futures):
        variant = futures[future]
        try:
            future.result()
        except Exception as error:
            stop.set()
            errors.append(f"{variant}: {error}")
            record(variant, phase="failed", error=str(error))
record(status="failed" if errors else "complete", errors=errors)
raise SystemExit(1 if errors else 0)
