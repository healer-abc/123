"""Run the matched S1/S2 experiments in isolated processes and export adapters."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
report = root / "report"
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
state_path = report / "sft-experiments-state.json"
lock = threading.Lock()
export_lock = threading.Lock()
state = {"status": "running", "budget_steps": 10000, "experiments": {}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(label=None, **fields):
    with lock:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        if label:
            state["experiments"].setdefault(label, {}).update(fields)
        else:
            state.update(fields)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)


def metrics(run):
    result = {}
    for path in run.rglob("events.out.tfevents.*"):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        for tag in ["train/loss", "train/grad_norm", "time/training"]:
            if tag in events.Tags()["scalars"]:
                rows = events.Scalars(tag)
                if rows:
                    assert all(math.isfinite(row.value) for row in rows), f"Nonfinite {tag} in {run}"
                    result[tag] = rows[-1].value
                    result["completed_steps"] = max(result.get("completed_steps", 0), rows[-1].step + 1)
    return result


def run_one(label):
    run = root / "runs" / f"{label}_{stamp}"
    run.mkdir()
    config = root / "configs" / f"{label}.yaml"
    cfg = OmegaConf.load(config)
    dataset = Path(cfg.data.train_data_paths)
    norm = Path(cfg.actor.model.openpi_data.norm_stats_path)
    base = Path(cfg.actor.model.model_path)
    provenance = {
        "dataset": str(dataset), "dataset_info": json.loads((dataset / "meta/info.json").read_text()),
        "dataset_manifest_sha256": sha(dataset / "meta/source_episodes.jsonl"),
        "config_sha256": sha(config), "norm_stats": str(norm), "norm_stats_sha256": sha(norm),
        "base_model": str(base), "base_source": json.loads((base / "source.json").read_text()),
    }
    (run / "data-source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (run / "driver.log").open("w") as log:
        child = subprocess.Popen(["bash", str(root / "scripts/run_sft.sh"), label, str(run)],
                                 cwd=workspace, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record(label, phase="training", pid=child.pid, run_dir=str(run), completed_steps=0)
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    record(label, **metrics(run))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        record(label, exit_code=code, **metrics(run))
        if code:
            raise RuntimeError(f"{label} training failed with exit {code}; see {run}/run.log")
        assert metrics(run)["completed_steps"] == 10000, f"Incomplete training: {label}"
        checkpoint = run / label / "checkpoints/global_step_10000/actor/dcp_checkpoint"
        assert (checkpoint / ".metadata").is_file() and len(list(checkpoint.glob("*.distcp"))) == 2
        # Serialize CPU export to limit host RAM and temporary disk demand.
        with export_lock:
            output = workspace / "models/robotwin_pi0" / f"{label}_{stamp}"
            record(label, phase="exporting", output=str(output))
            subprocess.run([sys.executable, str(root / "scripts/merge_sft_lora.py"),
                            str(checkpoint), str(output), "--dcp", "--norm-stats", str(norm),
                            "--base-model", str(base / "model.safetensors")],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
            assert json.loads((output / "merge-check.json").read_text())["tensor_count"] == 632
            assert sha(output / "norm_stats.json") == sha(norm)
            shutil.copy2(run / "data-source.json", output / "training-source.json")
        record(label, phase="complete", completed_steps=10000, output=str(output))


assert json.loads((report / "expanded-data-check.json").read_text())["status"] == "passed"
assert json.loads((report / "mixed200-converted-check.json").read_text())["status"] == "passed"
assert json.loads((report / "mixed200-loader-check.json").read_text())["status"] == "passed"
configs = [OmegaConf.load(root / "configs" / f"{label}.yaml") for label in ["s1_clean", "s2_mixed"]]
assert OmegaConf.to_container(configs[0].actor) == OmegaConf.to_container(configs[1].actor), "S1/S2 actor settings differ"
assert configs[0].runner.max_steps == configs[1].runner.max_steps == 10000
assert configs[0].actor.micro_batch_size == configs[1].actor.micro_batch_size == 4
assert configs[0].actor.global_batch_size == configs[1].actor.global_batch_size == 16
for cfg, counts in zip(configs, [{"clean": 200}, {"clean": 100, "randomized": 100}]):
    rows = [json.loads(line) for line in (Path(cfg.data.train_data_paths) / "meta/source_episodes.jsonl").read_text().splitlines()]
    assert {variant: sum(row["variant"] == variant for row in rows) for variant in counts} == counts
    assert len(rows) == len({row["source"] for row in rows}) == 200
assert json.loads((report / "mixed200-loader-check.json").read_text())["norm_stats_sha256"] == sha(Path(configs[0].actor.model.openpi_data.norm_stats_path))
profile = json.loads((report / "sft-memory-profile.json").read_text())
assert profile["status"] == "passed" and profile["micro_batch_size"] == 4
assert shutil.disk_usage(workspace).free > 80 * 1024**3, "Need 80 GiB free for four DCP checkpoints and two exports"
record(resource_profile=profile)
errors = []
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = {pool.submit(run_one, label): label for label in ["s1_clean", "s2_mixed"]}
    for future in as_completed(futures):
        label = futures[future]
        try:
            future.result()
        except Exception as error:
            errors.append(f"{label}: {error}")
            record(label, phase="failed", error=str(error))
record(status="failed" if errors else "complete", errors=errors)
raise SystemExit(1 if errors else 0)
