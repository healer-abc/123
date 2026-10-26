"""Supervise E3 training/evaluation or sequential isolated training cost profiles."""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
import traceback
from zoneinfo import ZoneInfo

import development_watchdog as helpers

ROOT, OPENPI, LOGS = helpers.ROOT, helpers.OPENPI, helpers.LOGS
STOP = False
ACTIVE = None


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def stop(*_args):
    global STOP
    STOP = True


def event(mode, kind, **details):
    helpers.event(kind, supervisor=mode, **details)


def sample(gpu, pid=None):
    raw = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    used, free, utilization = map(int, next(csv.reader(raw.splitlines())))
    process_mib = 0
    if pid is not None:
        raw = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
            "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=15)
        for row in csv.reader(raw.splitlines()):
            if row and row[0].strip() == str(pid) and row[1].strip().isdigit():
                process_mib += int(row[1])
    return {"gpu_used_mib": used, "gpu_free_mib": free, "gpu_utilization": utilization,
            "process_mib": process_mib}


def environment(gpu):
    env = helpers.environment(gpu)
    env.update(OPENPI_DATA_HOME="/home/zhsu/jyluo/myopenpi/dataset/ckpt",
               HF_LEROBOT_HOME="/home/zhsu/jyluo/dataset_root", HF_HUB_OFFLINE="1",
               CC="/home/zhsu/jyluo/miniforge3/bin/x86_64-conda-linux-gnu-gcc",
               CXX="/home/zhsu/jyluo/miniforge3/bin/x86_64-conda-linux-gnu-g++")
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    return env


def state(mode, **details):
    helpers.atomic_json(LOGS / f"{mode}_status.json", {"updated_at": stamp(), "pid": os.getpid(), **details})


def run_child(mode, stage, command, gpu, required_free, resource_file, stall_seconds=1800,
              stable_free_seconds=0):
    global ACTIVE
    free_since = None
    while not STOP:
        memory = sample(gpu)
        if memory["gpu_free_mib"] >= required_free:
            if free_since is None:
                free_since = time.time()
            if time.time() - free_since >= stable_free_seconds:
                break
        else:
            free_since = None
        state(mode, state="waiting_for_memory", stage=stage, gpu=gpu, required_free_mib=required_free,
              stable_free_seconds=stable_free_seconds,
              stable_free_elapsed_seconds=0 if free_since is None else time.time() - free_since, **memory)
        for _ in range(15):
            if STOP:
                raise InterruptedError("stopped while waiting for memory")
            time.sleep(1)
    if STOP:
        raise InterruptedError("stopped before launch")
    logfile = LOGS / f"{stage}.log"
    env = environment(gpu)
    record = {"command": command, "gpu": gpu, "started_at": stamp(),
              "config_sha256": hashlib.sha256((OPENPI / "src/openpi/training/config.py").read_bytes()).hexdigest(),
              "prelaunch_memory": memory,
              "environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "OPENPI_DATA_HOME",
                  "HF_LEROBOT_HOME", "XLA_PYTHON_CLIENT_PREALLOCATE", "OMP_NUM_THREADS", "WANDB_MODE")}}
    helpers.atomic_json(LOGS / f"{stage}.launch.json", record)
    ACTIVE = helpers.spawn(command, env, logfile)
    event(mode, "stage_start", stage=stage, pid=ACTIVE.pid, gpu=gpu, log=str(logfile))
    started, last_change, last_size = time.time(), time.time(), -1
    last_sample = memory
    resource_file.parent.mkdir(parents=True, exist_ok=True)
    with resource_file.open("a", newline="") as file:
        fields = ["timestamp", "stage", "pid", "gpu", "process_mib", "gpu_used_mib", "gpu_free_mib", "gpu_utilization", "last_logged_step"]
        writer = csv.DictWriter(file, fieldnames=fields)
        if file.tell() == 0:
            writer.writeheader()
        cycle, step = 0, None
        while ACTIVE.poll() is None:
            if STOP:
                raise InterruptedError("user stopped supervisor")
            if cycle % 15 == 0:
                size = logfile.stat().st_size
                if size != last_size:
                    last_change, last_size = time.time(), size
                text = helpers.log_tail(logfile)
                steps = re.findall(r"Step\s+(\d+):", text)
                if steps:
                    step = int(steps[-1])
                if re.search(r"(?:loss|grad_norm|param_norm)=[+-]?(?:nan|inf)\b", text, re.IGNORECASE):
                    raise RuntimeError(f"non-finite training metric in {logfile}")
                if "Traceback (most recent call last):" in text:
                    raise RuntimeError(f"training traceback in {logfile}: {text}")
                if time.time() - last_change > stall_seconds:
                    raise RuntimeError(f"no log progress for {stall_seconds}s in {logfile}")
            try:
                last_sample = sample(gpu, ACTIVE.pid)
            except (subprocess.SubprocessError, ValueError) as error:
                event(mode, "error", stage=stage, reason=f"transient GPU sampling failure: {error}")
                time.sleep(1)
                cycle += 1
                continue
            writer.writerow({"timestamp": time.time(), "stage": stage, "pid": ACTIVE.pid, "gpu": gpu,
                             "last_logged_step": step, **last_sample})
            file.flush()
            if last_sample["gpu_free_mib"] < 4096:
                raise RuntimeError(f"memory pressure: less than 4096 MiB free on GPU {gpu}")
            if cycle % 15 == 0:
                state(mode, state="running", stage=stage, child_pid=ACTIVE.pid, gpu=gpu,
                      elapsed_seconds=time.time() - started, last_logged_step=step,
                      log=str(logfile), resource_samples=str(resource_file), **last_sample)
            time.sleep(1)
            cycle += 1
    code = ACTIVE.returncode
    ACTIVE = None
    if code:
        raise RuntimeError(f"{stage} exited {code}: {helpers.log_tail(logfile)}")
    event(mode, "stage_complete", stage=stage, elapsed_seconds=time.time() - started)


def complete_checkpoint(path):
    return (path / "_CHECKPOINT_METADATA").is_file() and (path / "params").is_dir() and (path / "train_state").is_dir()


def next_e3_attempt_number():
    """Return a fresh attempt number without reusing evidence from an earlier supervisor run."""
    numbers = []
    for path in LOGS.glob("E3_train_attempt_*.log"):
        match = re.fullmatch(r"E3_train_attempt_(\d+)\.log", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def train_e3(args):
    mode = "e3_training"
    verification = json.loads((ROOT / "audits/e3_smoke_verification.json").read_text())
    if not verification.get("passed") or verification.get("config") != "pi05_libero_action_only":
        raise RuntimeError("E3 smoke verification gate not passed")
    if not (ROOT / "report/development_review/action_bounds_summary.json").is_file():
        raise RuntimeError("action scoring audit not present")
    checkpoint_root = OPENPI / "checkpoints/pi05_libero_action_only/pi05_libero_action_only_s42"
    command = [str(OPENPI / ".venv/bin/python"), "-u", "scripts/train.py", "pi05_libero_action_only",
               "--exp-name", "pi05_libero_action_only_s42", "--batch-size", "32", "--num-train-steps", "30000"]
    if complete_checkpoint(checkpoint_root / "29999"):
        event(mode, "training_already_complete")
    else:
        first_attempt = next_e3_attempt_number()
        for retry_index in range(2):
            attempt = first_attempt + retry_index
            child_command = list(command)
            checkpoints = [p for p in checkpoint_root.glob("*") if p.name.isdigit() and complete_checkpoint(p)]
            if checkpoints:
                child_command.append("--resume")
                event(mode, "retry", reason="resume finalized E3 checkpoint", checkpoint=max(map(lambda p: int(p.name), checkpoints)),
                      note="stock loader does not restore iterator order; record resume limitation")
            elif checkpoint_root.exists():
                raise RuntimeError("E3 output exists without a finalized checkpoint; refusing overwrite")
            try:
                run_child(mode, f"E3_train_attempt_{attempt}", child_command, args.gpu, 81920,
                          ROOT / f"metrics/e3/resources_attempt_{attempt}.csv", stable_free_seconds=120)
                break
            except RuntimeError as error:
                helpers.terminate(ACTIVE)
                event(mode, "error", reason=str(error))
                text = str(error)
                if retry_index or any(term in text for term in ("non-finite", "Traceback", "AssertionError", "ModuleNotFoundError")):
                    raise
                if not any(p.name.isdigit() and complete_checkpoint(p) for p in checkpoint_root.glob("*")):
                    raise
        if not complete_checkpoint(checkpoint_root / "29999"):
            raise RuntimeError("training exited without finalized checkpoint 29999")
    state(mode, state="development_evaluation_starting", checkpoint=str(checkpoint_root / "29999"))
    run_child(mode, "E3_development_watchdog", [str(OPENPI / ".venv/bin/python"), "-u",
              str(ROOT / "development_watchdog.py"), "--run-ids", "E3-s42", "--skip-comparison"],
              args.gpu, 32768, ROOT / "metrics/e3/evaluation_resources.csv", stall_seconds=5400)
    summary = json.loads((ROOT / "evaluations/E3-s42/development/summary.json").read_text())
    if not summary.get("complete") or summary.get("valid") != 400:
        raise RuntimeError("E3 evaluation incomplete; inspect watchdog alerts")
    state(mode, state="completed_awaiting_candidate_review", development_summary=summary)


def profile(args):
    mode = "training_profiles"
    configurations = ["pi05_libero_lora", "pi05_libero_lora_frozen_vision", "pi05_libero_action_only"]
    summaries = []
    for run, config in zip(("E1", "E2", "E3"), configurations):
        stage = f"{run}_profile_s42"
        output = ROOT / f"metrics/profiles/{run}_steps.jsonl"
        resources = ROOT / f"metrics/profiles/{run}_resources.csv"
        summary_path = ROOT / f"metrics/profiles/{run}_summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text()))
            continue
        checkpoint = OPENPI / "checkpoints" / config / f"{run.lower()}_profile_220_s42"
        if checkpoint.exists() or output.exists():
            raise RuntimeError(f"profile partial outputs already exist: {checkpoint}; refusing overwrite")
        run_child(mode, stage, [str(OPENPI / ".venv/bin/python"), "-u", str(ROOT / "profile_training.py"),
                  "--config", config, "--exp-name", f"{run.lower()}_profile_220_s42", "--output", str(output)],
                  args.gpu, 65536, resources)
        values = [json.loads(line) for line in output.read_text().splitlines()]
        measured = [v for v in values if v["event"] == "step" and v["measured"]]
        assert len(measured) == 200 and values[-1]["event"] == "profile_complete"
        with resources.open() as file:
            samples = list(csv.DictReader(file))
        start, end = measured[0]["timestamp"], measured[-1]["timestamp"]
        steady = [v for v in samples if start <= float(v["timestamp"]) <= end]
        summary = {"config": config, "gpu": args.gpu, "warmup_steps": 20, "measured_steps": 200,
                   "mean_step_seconds": statistics.mean(v["elapsed_since_previous_step"] for v in measured),
                   "sampled_process_peak_mib": max(int(v["process_mib"]) for v in samples),
                   "sampled_steady_process_peak_mib": max(int(v["process_mib"]) for v in steady),
                   "resource_sample_frequency_hz_approx": 1, "save_included": True,
                   "note": "observed process memory incl allocator caches; shared-GPU background samples retained; not a tensor-live-byte measurement"}
        helpers.atomic_json(summary_path, summary)
        summaries.append(summary)
    helpers.atomic_json(ROOT / "metrics/profiles/summary.json", {"profiles": summaries})
    state(mode, state="completed", profiles=summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("e3", "profiles"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    mode = "e3_training" if args.mode == "e3" else "training_profiles"
    lock = (LOGS / f"{mode}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (LOGS / f"{mode}.pid").write_text(str(os.getpid()) + "\n")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        (train_e3 if args.mode == "e3" else profile)(args)
    except Exception:
        text = traceback.format_exc()
        event(mode, "blocked", reason=text)
        state(mode, state="stopped_by_signal" if STOP else "blocked", error=text)
        raise
    finally:
        helpers.terminate(ACTIVE)


if __name__ == "__main__":
    main()
