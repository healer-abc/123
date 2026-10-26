"""Summarize loss, sampled device memory and GPU time without double counting."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(os.environ.get("ROBOTWIN_EXPERIMENT_ROOT", Path(__file__).resolve().parents[1]))
state = json.loads((root / "report/sft-experiments-state.json").read_text())
experiments = {}
intervals = {}
peaks = {}
for label, entry in state["experiments"].items():
    run = Path(entry["run_dir"])
    scalars = {}
    for path in sorted(run.rglob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        for tag in ["train/loss", "train/grad_norm", "time/training"]:
            if tag in events.Tags()["scalars"]:
                scalars.setdefault(tag, []).extend(events.Scalars(tag))
    loss = sorted(scalars.get("train/loss", []), key=lambda x: x.step)
    grad = scalars.get("train/grad_norm", [])
    times = scalars.get("time/training", [])
    spans = {}
    with (run / "gpu.csv").open() as stream:
        for raw in csv.DictReader(stream):
            row = {key.strip(): value.strip() if value else "" for key, value in raw.items() if key}
            try:
                timestamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                gpu = int(row["index"])
                memory = int(row["memory.used [MiB]"].split()[0])
            except (KeyError, ValueError, IndexError):
                continue  # A concurrent writer can leave an incomplete final row.
            spans.setdefault(gpu, [timestamp, timestamp])[1] = timestamp
            peaks[gpu] = max(peaks.get(gpu, 0), memory)
    for gpu, (start, end) in spans.items():
        intervals.setdefault(gpu, []).append((start, end, label))
    experiments[label] = {
        "phase": entry["phase"], "run_dir": str(run),
        "completed_steps": loss[-1].step + 1 if loss else 0,
        "loss_latest": loss[-1].value if loss else None,
        "loss_mean_last_100": statistics.mean(x.value for x in loss[-100:]) if loss else None,
        "loss_average_samples": min(len(loss), 100),
        "grad_norm_latest": grad[-1].value if grad else None,
        "training_seconds_mean_last_100": statistics.mean(x.value for x in times[-100:]) if times else None,
        "elapsed_gpu_monitor_hours": max((end - start for start, end in spans.values()), default=0) / 3600,
        "gpu_hours_equal_share_estimate": 0.0,
    }

# Integrate each physical GPU once; divide overlapping time evenly among jobs.
physical_gpu_seconds = 0.0
for gpu, spans in intervals.items():
    boundaries = sorted({value for start, end, _ in spans for value in [start, end]})
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2
        active = [label for left, right, label in spans if left <= midpoint < right]
        if active:
            physical_gpu_seconds += end - start
            for label in active:
                experiments[label]["gpu_hours_equal_share_estimate"] += (end - start) / len(active) / 3600

result = {
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "experiments": experiments,
    "physical_gpu_hours_total": physical_gpu_seconds / 3600,
    "peak_device_memory_MiB_combined_jobs": peaks,
    "accounting": "Union of gpu.csv monitoring intervals per physical GPU; includes initialization and checkpoint saves, excludes CPU export after monitoring stops. Equal-share per-job values are estimates, not measured per-process GPU execution time. Memory is device-wide sampled once per second, not per-job or exact allocator peak.",
}
output = root / "report/sft-metrics-latest.json"
temporary = output.with_suffix(".tmp")
temporary.write_text(json.dumps(result, indent=2) + "\n")
temporary.replace(output)
print(f"{'Experiment':<12} {'Steps':>7} {'Loss':>10} {'Mean(100)':>10} {'Elapsed(h)':>11} {'GPU-h*':>9}")
for label, value in experiments.items():
    latest = f"{value['loss_latest']:.5f}" if value["loss_latest"] is not None else "pending"
    mean = f"{value['loss_mean_last_100']:.5f}" if value["loss_mean_last_100"] is not None else "pending"
    print(f"{label:<12} {value['completed_steps']:>7} {latest:>10} {mean:>10} {value['elapsed_gpu_monitor_hours']:>11.3f} {value['gpu_hours_equal_share_estimate']:>9.3f}")
print(f"Physical GPU-hours total: {result['physical_gpu_hours_total']:.3f}")
print("Peak device memory, both jobs combined: " + ", ".join(f"GPU {gpu}: {memory} MiB ({memory/1024:.2f} GiB)" for gpu, memory in sorted(peaks.items())))
print("* Per-job GPU-hours use equal sharing during overlap; do not multiply by two again.")
print(output)
