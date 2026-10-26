"""Evaluate S1/S2 on fixed Clean/Randomized seed batches with bounded output."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
report = root / "report"
state_path = report / "eval-experiments-state.json"
training = json.loads((report / "sft-experiments-state.json").read_text())
assert training["status"] == "complete" and not training["errors"]
source_seeds = Path(os.environ["RLINF_ROOT"]) / "rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
seeds = json.loads(source_seeds.read_text())["adjust_bottle"]["success_seeds"]
assert len(seeds) == len(set(seeds)) == 150
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.resume:
    state = json.loads(state_path.read_text())
    assert state["status"] == "failed", "Resume requires a failed evaluation state"
    output = Path(state["output_dir"])
    assert output.parent == root / "runs" and output.is_dir()
    shutil.copy2(state_path, report / f"eval-experiments-state-before-resume-{stamp}.json")
    state.update(status="running", resumed_utc=datetime.now(timezone.utc).isoformat(),
                 resumed_free_GiB=shutil.disk_usage(workspace).free / 1024**3,
                 previous_error=state.pop("error", None), completed_batches=0, formal_episodes=0)
else:
    output = root / "runs" / f"eval_sft_{stamp}"
    output.mkdir()
    state = {"status": "running", "output_dir": str(output), "target_per_condition": 150,
             "phase": "pilot", "completed_batches": 0, "formal_episodes": 0,
             "conditions": {}, "initial_free_GiB": shutil.disk_usage(workspace).free / 1024**3}


def record(**fields):
    state.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
    temp = state_path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(state_path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes, total):
    if not total:
        return None
    z = 1.95996398454
    p = successes / total
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-radius, center+radius]


def budget_check():
    free = shutil.disk_usage(workspace).free
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    assert free >= 3 * 1024**3, "Free disk below 3 GiB; preserve results and stop before next batch"
    assert size < 1024**3, "Evaluation output exceeded the 1 GiB budget"
    record(free_disk_GiB=free / 1024**3, output_MiB=size / 1024**2)


def summarize_batch(job, targets):
    rows = [json.loads(line) for path in sorted((job / "audit").glob("episodes-*.jsonl"))
            for line in path.read_text().splitlines() if line]
    episodes = [row for row in rows if row["event"] == "episode"]
    assert len(episodes) == len(targets), f"Expected {len(targets)} episodes, found {len(episodes)}"
    assert sorted(row["requested_seed"] for row in episodes) == sorted(targets), "Seed coverage mismatch"
    assert all(row["episode_len"] == 200 and math.isfinite(row["return"]) for row in episodes)
    scalars = {}
    for path in job.rglob("events.out.tfevents.*"):
        events = EventAccumulator(str(path))
        events.Reload()
        for tag in ["eval/success_once", "eval/num_trajectories"]:
            if tag in events.Tags()["scalars"]:
                scalars[tag] = events.Scalars(tag)[-1].value
    native_rate = sum(row["success_once"] for row in episodes) / len(episodes)
    assert abs(scalars["eval/success_once"] - native_rate) < 1e-6, "Native success metric disagrees with per-seed audit"
    (job / "batch-results.json").write_text(json.dumps({"status": "passed", "episodes": episodes, "native_metrics": scalars}, indent=2) + "\n")
    return episodes


def run_batch(model, variant, index, targets, phase):
    job = output / f"{phase}_{model}_{variant}_{index:02d}"
    if job.exists():
        assert args.resume, f"Existing job requires explicit resume: {job}"
        exit_file = job / "exit-code.txt"
        if exit_file.exists() and exit_file.read_text().strip() == "0" and (job / "batch-results.json").exists():
            source = json.loads((job / "provenance.json").read_text())
            assert source["requested_seeds"] == targets and source["phase"] == phase and source["variant"] == variant
            assert source["audit_extension_sha256"] == sha(root / "compat/robotwin_pi0_eval_ext.py")
            assert source["norm_stats_sha256"] == sha(Path(source["model"]) / "norm_stats.json")
            return summarize_batch(job, targets)
        job.rename(job.with_name(job.name + ".interrupted-" + stamp))
    budget_check()
    job.mkdir()
    cfg = OmegaConf.load(root / "configs/eval_smoke_2gpu.yaml")
    cfg.runner.logger.log_path = str(job)
    cfg.runner.logger.experiment_name = f"{model}_{variant}"
    cfg.env.eval.total_num_envs = len(targets)
    cfg.env.eval.auto_reset = False
    cfg.env.eval.use_fixed_reset_state_ids = True
    cfg.env.eval.video_cfg.save_video = phase == "pilot"
    cfg.env.eval.task_config.collect_data = False
    cfg.env.eval.task_config.save_data = False
    cfg.env.eval.task_config.eval_video_log = False
    cfg.env.eval.task_config.save_path = str(job / "environment")
    cfg.env.eval.task_config.clear_cache_freq = 1
    cfg.env.eval.task_config.domain_randomization = OmegaConf.load(root / f"configs/demo_{variant}_pi0_train200.yml").domain_randomization
    cfg.env.eval.seeds_path = str(job / "seeds.json")
    cfg.rollout.model.model_path = training["experiments"][model]["output"]
    cfg.rollout.model.openpi_data.norm_stats_path = str(Path(cfg.rollout.model.model_path) / "norm_stats.json")
    (job / "seeds.json").write_text(json.dumps({"adjust_bottle": {"task_name": "adjust_bottle", "success_seeds": targets}}, indent=2) + "\n")
    OmegaConf.save(cfg, job / "config.yaml")
    provenance = {"model": cfg.rollout.model.model_path, "training_model_source_sha256": sha(Path(cfg.rollout.model.model_path) / "training-source.json"),
                  "norm_stats_sha256": sha(Path(cfg.rollout.model.openpi_data.norm_stats_path)),
                  "config_sha256": sha(job / "config.yaml"), "requested_seeds": targets,
                  "phase": phase, "variant": variant, "renderer": os.environ["ROBOTWIN_DENOISER"],
                  "policy_noise_seed": "20261026 + rollout rank; reset for each paired static batch",
                  "audit_extension_sha256": sha(root / "compat/robotwin_pi0_eval_ext.py")}
    (job / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    child = subprocess.Popen(["bash", str(root / "scripts/run_eval_batch.sh"), str(job)], cwd=workspace, start_new_session=True)
    record(phase=phase, active_model=model, active_variant=variant, active_batch=index,
           active_job=str(job), pid=child.pid)
    started = time.monotonic()
    while child.poll() is None:
        time.sleep(10)
        record(active_elapsed_seconds=round(time.monotonic()-started))
        if shutil.disk_usage(workspace).free < 2*1024**3:
            import signal
            os.killpg(child.pid, signal.SIGTERM)
            raise RuntimeError("Disk below 2 GiB during evaluation; stopped this job")
    assert child.returncode == 0, f"Evaluation failed with exit {child.returncode}: {job}/run.log"
    result = summarize_batch(job, targets)
    # Delete only this completed job's disposable Ray files on tmpfs.
    temp = Path((job / "ray-tmp-dir.txt").read_text().strip())
    assert temp.parent == Path("/dev/shm") and temp.name.startswith("rw-eval-")
    shutil.rmtree(temp, ignore_errors=True)
    return result


def update_results(all_rows):
    summary = {}
    for model in ["s1_clean", "s2_mixed"]:
        for variant in ["clean", "randomized"]:
            key = f"{model}/{variant}"
            rows = [row for row in all_rows if row["model"] == model and row["variant"] == variant]
            valid = [row for row in rows if row["valid_scene"]]
            successes = sum(row["success_once"] for row in valid)
            summary[key] = {"attempted": len(rows), "valid_scenes": len(valid), "invalid_scenes": len(rows)-len(valid),
                            "successes": successes, "success_rate": successes/len(valid) if valid else None,
                            "wilson_95": wilson(successes, len(valid))}
    (output / "results.json").write_text(json.dumps({"conditions": summary, "episodes": all_rows}, indent=2) + "\n")
    record(conditions=summary, formal_episodes=len(all_rows))


record()
all_rows = []
try:
    for variant in ["clean", "randomized"]:
        for model in ["s1_clean", "s2_mixed"]:
            rows = run_batch(model, variant, 0, seeds[:2], "pilot")
            assert all(row["valid_scene"] for row in rows), "Pilot scene/seed audit failed; inspect audit before formal evaluation"
    record(pilot_status="passed")
    for variant in ["clean", "randomized"]:
        for index, start in enumerate(range(0, 150, 10)):
            for model in ["s1_clean", "s2_mixed"]:
                rows = run_batch(model, variant, index, seeds[start:start+10], "formal")
                all_rows.extend({**row, "model": model, "variant": variant} for row in rows)
                record(completed_batches=state["completed_batches"]+1)
                update_results(all_rows)
    assert len(all_rows) == 600
    for variant in ["clean", "randomized"]:
        valid_sets = [{row["requested_seed"] for row in all_rows if row["model"] == model and row["variant"] == variant and row["valid_scene"]} for model in ["s1_clean", "s2_mixed"]]
        assert valid_sets[0] == valid_sets[1], f"Different valid scenes for paired models in {variant}"
    record(status="complete", phase="complete", results=str(output / "results.json"))
except Exception as error:
    record(status="failed", error=str(error))
    raise
