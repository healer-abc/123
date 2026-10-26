"""Memory-aware E1/E2 evaluation supervisor. All child processes are owned sessions."""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import ProxyHandler, build_opener
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
OPENPI = ROOT.parents[1] / "projects/openpi"
LOGS = ROOT / "logs"
MANIFEST = ROOT / "manifests/eval_manifest.csv"
EXPECTED_MANIFEST = "6f52e79c4e194e04a57d40a283812078db54bef4117a86a5f02c52e31c17311b"
HTTP = build_opener(ProxyHandler({}))
STOP = False
STATUS_NAME = "watchdog_status.json"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def event(kind, **details):
    record = {"at": now(), "event": kind, **details}
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    with (LOGS / "watchdog_events.jsonl").open("a") as file:
        file.write(line + "\n")
    if kind in {"error", "blocked", "retry", "memory_pressure"}:
        with (LOGS / "alerts.jsonl").open("a") as file:
            file.write(line + "\n")
        atomic_json(LOGS / "latest_alert.json", record)


def gpu_memory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    return {int(parts[0]): {"free_mib": int(parts[1]), "used_mib": int(parts[2]),
                           "total_mib": int(parts[3])}
            for parts in csv.reader(output.splitlines())}


def select_gpu(memory, jobs, budget, reserve):
    # Starting servers have not yet allocated their full memory. Reserve it up front.
    candidates = []
    for gpu, sample in memory.items():
        active = [j for j in jobs if j.get("gpu") == gpu and j["status"] in {"starting", "running"}]
        pending = sum(budget for j in active if j["status"] == "starting")
        available = sample["free_mib"] - pending
        if available >= budget + reserve:
            candidates.append((len(active), -available, gpu))
    return min(candidates)[2] if candidates else None


def healthy(port):
    try:
        with HTTP.open(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def free_port(port):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def spawn(command, environment, log):
    with log.open("a") as file:
        return subprocess.Popen(command, cwd=OPENPI, env=environment, stdout=file,
                                stderr=subprocess.STDOUT, start_new_session=True)


def terminate(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def stop_job(job):
    for worker in job.get("workers", {}).values():
        terminate(worker["process"])
    terminate(job.get("server"))
    job["workers"] = {}
    job["server"] = None


def environment(gpu):
    result = os.environ.copy()
    result.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                  MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", WANDB_MODE="offline",
                  XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="2",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                  PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    result["PYTHONPATH"] = str(OPENPI / "third_party/libero") + os.pathsep + result.get("PYTHONPATH", "")
    return result


def log_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - 8000))
        return file.read().decode(errors="replace")


def fatal_error(text):
    return any(term in text for term in (
        "non-finite policy action", "invalid action plan shape", "invalid explicit noise",
        "wrong policy server metadata", "incompatible existing episode", "episode retry limit reached",
        "AssertionError", "FileNotFoundError", "ModuleNotFoundError", "ImportError"))


def summarize(job, rows):
    output = ROOT / "evaluations" / job["run_id"] / job.get("phase", "development")
    values = []
    for row in rows:
        path = output / "episodes" / row["suite"] / f"task_{int(row['task_id']):02d}" / f"state_{int(row['init_state_index']):02d}.json"
        if path.exists():
            value = json.loads(path.read_text())
            if (value.get("status") != "valid" or value.get("checkpoint") != job["checkpoint"]
                    or value.get("config") != job["config"]
                    or value.get("manifest_sha256") != EXPECTED_MANIFEST
                    or any(str(value.get(k)) != str(v) for k, v in row.items())):
                raise ValueError(f"incompatible existing episode: {path}")
            values.append(value)
    tasks = []
    for suite, task_id in sorted({(v["suite"], v["task_id"]) for v in values}):
        subset = [v for v in values if (v["suite"], v["task_id"]) == (suite, task_id)]
        successes = sum(v["success"] for v in subset)
        tasks.append({"suite": suite, "task_id": task_id, "completed": len(subset),
                      "successes": successes, "success_rate": successes / len(subset)})
    summary = {"complete": len(values) == len(rows), "completed": len(values), "planned": len(rows),
               "valid": len(values), "successes": sum(v["success"] for v in values),
               "raw_bound_strict_successes": sum(bool(v["success"] and not v.get("action_out_of_nominal_bounds", True)) for v in values),
               "tasks": tasks, "manifest_sha256": EXPECTED_MANIFEST,
               "suite_rates": {s: sum(t["success_rate"] for t in tasks if t["suite"] == s)
                               / sum(t["suite"] == s for t in tasks)
                               for s in sorted({v["suite"] for v in values})}}
    summary["suite_equal_success_rate"] = (sum(summary["suite_rates"].values()) / len(summary["suite_rates"])) if summary["suite_rates"] else None
    atomic_json(output / "summary.json", summary)
    job["completed"] = len(values)
    job["successes"] = summary["successes"]
    return summary


def launch_job(job, gpu, args):
    if not free_port(job["port"]):
        job["status"] = "blocked"
        event("blocked", run_id=job["run_id"], reason=f"port {job['port']} already occupied")
        return
    job["launches"] += 1
    job["gpu"] = gpu
    job["status"] = "starting"
    job["started"] = time.time()
    prefix = job["run_id"] + ("_confirmation" if job.get("phase") == "confirmation" else "")
    job["server_log"] = LOGS / f"{prefix}_server.attempt_{job['launches']}.log"
    job["server"] = spawn([
        str(OPENPI / ".venv/bin/python"), "-u", str(ROOT / "seeded_policy_server.py"),
        "--config", job["config"], "--checkpoint", job["checkpoint"], "--port", str(job["port"])],
        environment(gpu), job["server_log"])
    job["workers"] = {}
    event("server_start", run_id=job["run_id"], gpu=gpu, port=job["port"],
          pid=job["server"].pid, attempt=job["launches"], log=str(job["server_log"]))


def launch_worker(job, shard, args, restart=0):
    phase = job.get("phase", "development")
    prefix = job["run_id"] + ("_confirmation" if phase == "confirmation" else "")
    log = LOGS / f"{prefix}_shard_{shard}.server_{job['launches']}.retry_{restart}.log"
    suites = ["libero_spatial", "libero_10"] if phase == "development" else ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    process = spawn([
        str(OPENPI / "examples/libero/.venv/bin/python"), "-u", str(ROOT / "eval_manifest.py"),
        "--run-id", job["run_id"], "--config", job["config"], "--checkpoint", job["checkpoint"],
        "--phase", phase, "--suites", *suites,
        "--port", str(job["port"]), "--shard-index", str(shard), "--shard-count", str(args.workers)],
        environment(job["gpu"]), log)
    job["workers"][shard] = {"process": process, "started": time.time(), "restarts": restart, "log": log}
    event("worker_start", run_id=job["run_id"], shard=shard, pid=process.pid, gpu=job["gpu"], log=str(log))


def fail_job(job, reason, fatal=False):
    event("error", run_id=job["run_id"], reason=reason, fatal=fatal)
    stop_job(job)
    if fatal or job["launches"] >= 2:
        job["status"] = "blocked"
        event("blocked", run_id=job["run_id"], reason=reason)
    else:
        job["status"] = "queued"
        job["not_before"] = time.time() + 60
        event("retry", run_id=job["run_id"], reason=reason, action="restart owned server and resume missing episodes")


def snapshot(jobs, memory, state):
    snapshot_jobs = []
    for job in jobs:
        snapshot_jobs.append({k: job.get(k) for k in (
            "run_id", "config", "checkpoint", "status", "gpu", "port", "launches", "completed", "successes")})
        snapshot_jobs[-1]["server_pid"] = job["server"].pid if job.get("server") else None
        snapshot_jobs[-1]["worker_pids"] = {str(k): w["process"].pid for k, w in job.get("workers", {}).items()
                                            if w["process"].poll() is None}
    atomic_json(LOGS / STATUS_NAME, {"updated_at": now(), "timestamp": time.time(),
                "pid": os.getpid(), "state": state, "gpus": memory, "jobs": snapshot_jobs,
                "alerts": str(LOGS / "alerts.jsonl")})


def request_stop(*_args):
    global STOP
    STOP = True


def main():
    global STATUS_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--budget-mib", type=int, default=24576)
    parser.add_argument("--reserve-mib", type=int, default=8192)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--run-ids", nargs="+", default=["E1-s42", "E2-s42"])
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--freeze", type=Path, default=ROOT / "manifests/candidate_freeze.json")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.budget_mib < 8192 or args.reserve_mib < 4096 or args.interval < 1:
        parser.error("workers=1..4, budget>=8192 MiB, reserve>=4096 MiB, interval>=1 required")
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS / f"{args.phase}_watchdog.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{args.phase} watchdog already running")
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise SystemExit("frozen evaluation manifest hash mismatch")
    with MANIFEST.open() as file:
        rows = [r for r in csv.DictReader(file) if r["phase"] == args.phase]
    assert len(rows) == (400 if args.phase == "development" else 1600)
    if args.phase == "confirmation":
        from confirmation_gate import validate_freeze
        validate_freeze(args.freeze, args.run_ids)
        STATUS_NAME = "confirmation_watchdog_status.json"
    records = json.loads((ROOT / "manifests/runs.json").read_text())["runs"]
    jobs = []
    if args.run_ids != ["E1-s42", "E2-s42"] and not args.skip_comparison:
        parser.error("custom run-ids require --skip-comparison")
    for position, run_id in enumerate(args.run_ids):
        port = {"E1-s42": 8101, "E2-s42": 8102, "E3-s42": 8103}.get(run_id, 8110 + position)
        run = next(r for r in records if r["run_id"] == run_id)
        if not (Path(run["final_checkpoint"]) / "params").is_dir():
            raise SystemExit(f"missing checkpoint for {run_id}")
        jobs.append({"run_id": run_id, "config": run["config"], "checkpoint": run["final_checkpoint"],
                     "port": port, "phase": args.phase, "status": "queued", "launches": 0, "workers": {}, "completed": 0})
    (LOGS / f"{args.phase}_watchdog.pid").write_text(str(os.getpid()) + "\n")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    event("watchdog_start", pid=os.getpid(), workers_per_model=args.workers,
          model_budget_mib=args.budget_mib, reserve_mib=args.reserve_mib, phase=args.phase)
    state, memory = "running", {}
    try:
        while not STOP:
            memory = gpu_memory()
            if args.gpus is not None:
                memory = {gpu: value for gpu, value in memory.items() if gpu in args.gpus}
            for job in jobs:
                summary = summarize(job, rows)
                if summary["complete"]:
                    if job["status"] != "completed":
                        stop_job(job)
                        job["status"] = "completed"
                        event("evaluation_complete", run_id=job["run_id"], successes=summary["successes"], episodes=len(rows), phase=args.phase)
                    continue
                if job["status"] == "queued" and time.time() >= job.get("not_before", 0):
                    gpu = select_gpu(memory, jobs, args.budget_mib, args.reserve_mib)
                    if gpu is not None:
                        launch_job(job, gpu, args)
                    continue
                if job["status"] not in {"starting", "running"}:
                    continue
                if memory[job["gpu"]]["free_mib"] < 4096:
                    event("memory_pressure", run_id=job["run_id"], gpu=job["gpu"], action="release owned processes")
                    fail_job(job, "GPU free memory below 4096 MiB")
                    continue
                if job["server"].poll() is not None:
                    text = log_tail(job["server_log"])
                    fail_job(job, f"server exited {job['server'].returncode}: {text}", fatal_error(text))
                    continue
                if job["status"] == "starting":
                    if healthy(job["port"]):
                        job["status"] = "running"
                        for shard in range(args.workers):
                            launch_worker(job, shard, args)
                    elif time.time() - job["started"] > 1200:
                        fail_job(job, "server startup timeout after 1200 seconds")
                    continue
                for shard, worker in list(job["workers"].items()):
                    code = worker["process"].poll()
                    if code == 0:
                        shard_summary = ROOT / "evaluations" / job["run_id"] / args.phase / f"summary.shard_{shard}.json"
                        if args.workers == 1:
                            shard_summary = shard_summary.with_name("summary.json")
                        if not shard_summary.exists() or not json.loads(shard_summary.read_text())["complete"]:
                            fail_job(job, f"shard {shard} exited without complete summary", fatal=True)
                            break
                        continue
                    heartbeat_path = ROOT / "evaluations" / job["run_id"] / args.phase / f"heartbeat.shard_{shard}.json"
                    heartbeat = json.loads(heartbeat_path.read_text()) if heartbeat_path.exists() else {}
                    last_active = max(worker["started"], heartbeat.get("timestamp", 0))
                    stalled = time.time() - last_active > args.stall_seconds
                    if code is not None or stalled:
                        text = log_tail(worker["log"])
                        reason = f"shard {shard} {'stalled' if stalled else 'exited ' + str(code)}: {text}"
                        event("error", run_id=job["run_id"], reason=reason)
                        if fatal_error(text) or worker["restarts"] >= 1:
                            fail_job(job, reason, fatal=True)
                            break
                        terminate(worker["process"])
                        event("retry", run_id=job["run_id"], shard=shard, action="resume missing episodes; maximum two attempts per episode")
                        launch_worker(job, shard, args, restart=worker["restarts"] + 1)
            if all(j["status"] == "completed" for j in jobs):
                if args.phase == "confirmation":
                    result = subprocess.run([str(OPENPI / ".venv/bin/python"), str(ROOT / "analyze_confirmation.py"),
                                             "--freeze", str(args.freeze)], cwd=OPENPI, capture_output=True, text=True, timeout=600)
                    with (LOGS / "confirmation_analysis.log").open("a") as file:
                        file.write(result.stdout + result.stderr)
                    if result.returncode:
                        event("blocked", reason="confirmation analysis failed", details=result.stderr)
                        state = "blocked"
                    else:
                        state = "completed"
                        event("confirmation_complete", run_ids=args.run_ids)
                    break
                if args.skip_comparison:
                    state = "completed_awaiting_candidate_review"
                    event("development_complete", run_ids=args.run_ids)
                    break
                event("comparison_start")
                result = subprocess.run([str(OPENPI / ".venv/bin/python"), str(ROOT / "compare_development.py")],
                                        cwd=OPENPI, capture_output=True, text=True, timeout=300)
                (LOGS / "development_comparison.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    event("blocked", reason="comparison failed", details=result.stderr)
                    state = "blocked"
                else:
                    state = "completed_awaiting_e3_review"
                    event("comparison_complete", output=str(ROOT / "evaluations/development_comparison.json"),
                          next_step="review failed videos and register exactly one E3 branch")
                break
            snapshot(jobs, memory, "blocked" if any(j["status"] == "blocked" for j in jobs) else state)
            if all(j["status"] in {"blocked", "completed"} for j in jobs):
                state = "blocked"
                break
            for _ in range(args.interval):
                if STOP:
                    break
                time.sleep(1)
    except Exception:
        state = "blocked"
        event("blocked", reason="unexpected watchdog error", details=traceback.format_exc())
        raise
    finally:
        for job in jobs:
            stop_job(job)
        if STOP:
            state = "stopped_by_signal"
        snapshot(jobs, memory, state)
        event("watchdog_stop", state=state)


if __name__ == "__main__":
    main()
