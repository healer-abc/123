"""Resume cost profiles, recovering synchronized per-step W&B history without reruns."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import traceback

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

import training_watchdog as supervisor

ROOT, OPENPI = supervisor.ROOT, supervisor.OPENPI
MODE = "training_profiles"


def read_history(path):
    store = DataStore()
    store.open_for_scan(str(path))
    rows = []
    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField("history"):
                value = {".".join(x.nested_key) if x.nested_key else x.key: json.loads(x.value_json)
                         for x in record.history.item}
                if "loss" in value:
                    rows.append(value)
    finally:
        store.close()
    return rows


def validate_history(rows):
    if len(rows) != 220 or [r["_step"] for r in rows] != list(range(220)):
        raise ValueError("profile history must contain exactly steps 0..219")
    if any(not math.isfinite(float(r[key])) for r in rows for key in ("loss", "grad_norm", "_timestamp")):
        raise ValueError("non-finite metric or timestamp in profile history")
    intervals = [rows[i]["_timestamp"] - rows[i-1]["_timestamp"] for i in range(20, 220)]
    if any(v <= 0 for v in intervals):
        raise ValueError("nonmonotonic profile history timestamps")
    return intervals


def recover(run, config, gpu, checkpoint, resources, original_output):
    run_id = (checkpoint / "wandb_id.txt").read_text().strip()
    files = list((OPENPI / "wandb").glob(f"offline-run-*-{run_id}/run-{run_id}.wandb"))
    if len(files) != 1:
        raise ValueError(f"cannot uniquely locate offline W&B history for {run_id}")
    source = files[0]
    rows = read_history(source)
    intervals = validate_history(rows)
    lifecycle = [json.loads(line) for line in original_output.read_text().splitlines()]
    if lifecycle[-1]["event"] != "profile_complete" or not supervisor.complete_checkpoint(checkpoint / "219"):
        raise ValueError("profile lifecycle/checkpoint not complete")
    recovered = ROOT / f"metrics/profiles/{run}_steps.recovered.jsonl"
    values = [{"event": "step", "step": r["_step"], "timestamp": r["_timestamp"],
               "measured": r["_step"] >= 20, "loss": r["loss"], "grad_norm": r["grad_norm"],
               "source": str(source), "timing_source": "wandb frontend history timestamp after synchronized stock metric read"}
              for r in rows]
    recovered.write_text("".join(json.dumps(v) + "\n" for v in values))
    with resources.open() as file:
        samples = list(csv.DictReader(file))
    steady = [r for r in samples if rows[20]["_timestamp"] <= float(r["timestamp"]) <= rows[-1]["_timestamp"]]
    if not samples or not steady:
        raise ValueError("missing GPU samples for measured profile window")
    native = [v for v in lifecycle if v["event"] == "step" and v["measured"]]
    if native and len(native) != 200:
        raise ValueError("partial observer history; refusing incomplete timing")
    result = {"config": config, "gpu": gpu, "warmup_steps": 20, "measured_steps": 200,
              "mean_step_seconds": statistics.mean(intervals),
              "timing_source": "same frontend W&B wall-clock timestamps after synchronized stock metric reads for every configuration",
              "observer_measured_steps": len(native),
              "observer_mean_step_seconds": statistics.mean(v["elapsed_since_previous_step"] for v in native) if native else None,
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "recovered_steps": str(recovered),
              "sampled_process_peak_mib": max(int(v["process_mib"]) for v in samples),
              "sampled_steady_process_peak_mib": max(int(v["process_mib"]) for v in steady),
              "resource_sample_frequency_hz_approx": 1, "save_included": True,
              "note": "wall-clock interval timing, not monotonic observer timing; shared-GPU and CPU/I/O contention limitations remain"}
    supervisor.helpers.atomic_json(ROOT / f"metrics/profiles/{run}_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    lock = (ROOT / "logs/training_profiles.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / "logs/training_profiles.pid").write_text(str(os.getpid()) + "\n")
    signal.signal(signal.SIGTERM, supervisor.stop)
    signal.signal(signal.SIGINT, supervisor.stop)
    try:
        summaries = []
        for run, config in (("E1", "pi05_libero_lora"), ("E2", "pi05_libero_lora_frozen_vision"),
                            ("E3", "pi05_libero_action_only")):
            checkpoint = OPENPI / "checkpoints" / config / f"{run.lower()}_profile_220_s42"
            output = ROOT / f"metrics/profiles/{run}_steps.jsonl"
            resources = ROOT / f"metrics/profiles/{run}_resources.csv"
            if supervisor.complete_checkpoint(checkpoint / "219"):
                summaries.append(recover(run, config, args.gpu, checkpoint, resources, output))
                supervisor.event(MODE, "profile_recovered", run_id=run, action="reuse finalized training and original offline history")
                continue
            if checkpoint.exists() or output.exists():
                raise RuntimeError(f"partial outputs for {run}; refusing overwrite")
            supervisor.run_child(MODE, f"{run}_profile_s42", [str(OPENPI / ".venv/bin/python"), "-u",
                str(ROOT / "profile_training.py"), "--config", config, "--exp-name", f"{run.lower()}_profile_220_s42",
                "--output", str(output)], args.gpu, 65536, resources)
            summaries.append(recover(run, config, args.gpu, checkpoint, resources, output))
        supervisor.helpers.atomic_json(ROOT / "metrics/profiles/summary.json", {"profiles": summaries})
        supervisor.state(MODE, state="completed", profiles=summaries)
    except Exception:
        error = traceback.format_exc()
        supervisor.event(MODE, "blocked", reason=error)
        supervisor.state(MODE, state="blocked", error=error)
        raise
    finally:
        supervisor.helpers.terminate(supervisor.ACTIVE)


if __name__ == "__main__":
    main()
