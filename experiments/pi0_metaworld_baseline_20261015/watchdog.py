"""Persistent supervisor; never modifies policy weights or experiment settings."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from aggregate import aggregate

ROOT = Path(__file__).resolve().parent
PYTHON = "/home/zhsu/jyluo/projects/openpi/.venv/bin/python"
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())


def save_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    os.replace(temp, path)


def alert(event, **details):
    message = {"time": time.time(), "event": event, **details}
    with (ROOT / "logs/alerts.jsonl").open("a") as file:
        file.write(json.dumps(message) + "\n")
    print(json.dumps(message), flush=True)


def gpu_status():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True)
    return [{"index": int(x[0]), "total_mb": int(x[1]), "used_mb": int(x[2]), "utilization": int(x[3])}
            for line in result.stdout.splitlines() if (x := line.split(","))]


def downloader_pid():
    try:
        pid = int((ROOT / "download.pid").read_text())
        proc = Path("/proc") / str(pid)
        state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = (proc / "cmdline").read_bytes()
        if state == "Z" or str(ROOT / "download_checkpoint.py").encode() not in command:
            return None
        return pid
    except (OSError, ValueError, IndexError):
        return None


def resume_download():
    with (ROOT / "logs/download.log").open("ab") as file:
        process = subprocess.Popen([PYTHON, "-u", str(ROOT / "download_checkpoint.py")],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=file,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT / "download.pid").write_text(str(process.pid))
    return process


def launch(task, gpu, attempt):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               JAX_PLATFORMS="cpu", TORCH_COMPILE_DISABLE="1", OMP_NUM_THREADS="4",
               OPENPI_DATA_HOME="/home/zhsu/jyluo/myopenpi/dataset/ckpt")
    cuda_libs = [str(path) for path in (ROOT / "deps/nvidia").glob("*/lib")]
    cuda_libs.append("/home/zhsu/jyluo/projects/openpi/.venv/lib/python3.11/site-packages/cusparselt/lib")
    env["LD_LIBRARY_PATH"] = ":".join(cuda_libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    log = ROOT / "logs" / f"{task}_attempt{attempt}.log"
    with log.open("ab") as file:
        process = subprocess.Popen([PYTHON, "-u", str(ROOT / "rollout.py"), "--task", task],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=file, stderr=subprocess.STDOUT, start_new_session=True)
    alert("worker_started", task=task, gpu=gpu, pid=process.pid, attempt=attempt)
    return {"process": process, "gpu": gpu, "start": time.time(), "attempt": attempt, "log": log}


def main():
    lock = (ROOT / "watchdog.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A watchdog is already running")
    (ROOT / "watchdog.pid").write_text(str(os.getpid()))
    running, attempts, blocked = {}, {}, {}
    last_download_bytes, last_download_progress = 0, time.time()
    download_alerted = False
    ready_alerted = False
    recovery_file = ROOT / "logs/download_recovery.json"
    recovery = json.loads(recovery_file.read_text()) if recovery_file.exists() else {"restarts": 0, "last_restart": 0}
    children = []
    progress_samples = []
    while True:
        now = time.time()
        summary = aggregate()
        try:
            gpus = gpu_status()
        except Exception as error:
            alert("gpu_telemetry_error", error=str(error))
            gpus = []
        parts = list((ROOT / "checkpoints/RLinf-Pi0-MetaWorld-SFT/parts").glob("*"))
        download_bytes = sum(p.stat().st_size for p in parts)
        if download_bytes > last_download_bytes:
            last_download_bytes, last_download_progress = download_bytes, now
            download_alerted = False
        progress_samples.append((now, download_bytes))
        progress_samples = [(t, b) for t, b in progress_samples if now - t <= 120]
        elapsed = now - progress_samples[0][0]
        bytes_per_second = (download_bytes - progress_samples[0][1]) / elapsed if elapsed > 0 else 0
        checkpoint_ready = (ROOT / "checkpoints/RLinf-Pi0-MetaWorld-SFT/verified.json").exists()
        download_pid = downloader_pid()
        if not checkpoint_ready and now - last_download_progress > 600 and not download_alerted:
            alert("checkpoint_download_stalled", bytes=download_bytes)
            download_alerted = True
            if download_pid is not None and recovery["restarts"] < 3:
                os.killpg(download_pid, signal.SIGTERM)
                alert("stalled_download_terminated", pid=download_pid)
        if not checkpoint_ready and download_pid is None and recovery["restarts"] < 3 and now - recovery["last_restart"] >= 60:
            child = resume_download()
            children.append(child)
            recovery["restarts"] += 1
            recovery["last_restart"] = now
            save_json(recovery_file, recovery)
            last_download_progress = now
            alert("checkpoint_download_restarted", pid=child.pid, restart=recovery["restarts"], preserved_bytes=download_bytes)
        children[:] = [child for child in children if child.poll() is None]
        ready = checkpoint_ready and (ROOT / "logs/policy_smoke_passed.json").exists()
        if ready and not ready_alerted:
            alert("evaluation_ready")
            ready_alerted = True
        for task, worker in list(running.items()):
            process = worker["process"]
            code = process.poll()
            heartbeat_file = ROOT / "logs" / f"heartbeat_{task}.json"
            heartbeat = json.loads(heartbeat_file.read_text()) if heartbeat_file.exists() else {}
            if code is None:
                last = heartbeat.get("time", worker["start"])
                if heartbeat.get("pid") != process.pid:
                    last = worker["start"]
                timeout = 900 if heartbeat.get("phase") in ["loading_policy", "constructing_environment", None] else 300
                if now - last > timeout:
                    alert("worker_stalled", task=task, pid=process.pid, phase=heartbeat.get("phase"))
                    os.killpg(process.pid, signal.SIGTERM)
                continue
            del running[task]
            completed = len(list((ROOT / "results" / task).glob("episode_*.json")))
            if code == 0 and completed == PROTOCOL["trials_per_task"]:
                alert("task_complete", task=task, completed=completed)
                continue
            error = worker["log"].read_text(errors="replace")[-5000:]
            retryable = any(x in error.lower() for x in ["out of memory", "connection reset", "timed out"])
            alert("worker_error", task=task, exit_code=code, completed=completed,
                  retryable=retryable, log=str(worker["log"]), error=error)
            if not retryable or worker["attempt"] >= 3:
                blocked[task] = str(worker["log"])
            else:
                # Requeue exactly the same seeds; completed episode files are kept.
                time.sleep(2)
        if ready:
            summary = aggregate()
            busy = {w["gpu"] for w in running.values()}
            for row in summary["tasks"]:
                task = row["task"]
                if row["status"] == "complete" or task in running or task in blocked:
                    continue
                choices = [g for g in gpus if g["index"] not in busy and g["total_mb"] - g["used_mb"] >= 16000]
                if not choices:
                    break
                gpu = max(choices, key=lambda g: g["total_mb"] - g["used_mb"])["index"]
                attempts[task] = attempts.get(task, 0) + 1
                running[task] = launch(task, gpu, attempts[task])
                busy.add(gpu)
        state = {"time": now, "pid": os.getpid(), "phase": "complete" if summary["complete"] else
                 ("evaluating" if ready else "waiting_for_checkpoint_and_policy_validation"),
                 "download_bytes": download_bytes, "checkpoint_verified": checkpoint_ready,
                 "download_pid": downloader_pid(), "download_mib_per_second_120s": bytes_per_second / 1048576,
                 "download_restarts": recovery["restarts"],
                 "policy_validated": (ROOT / "logs/policy_smoke_passed.json").exists(),
                 "gpu": gpus, "running": {t: {"pid": w["process"].pid, "gpu": w["gpu"], "attempt": w["attempt"]} for t,w in running.items()},
                 "blocked_tasks": blocked, "completed_rollouts": summary["completed_rollouts"]}
        save_json(ROOT / "logs/watchdog_state.json", state)
        with (ROOT / "logs/telemetry.jsonl").open("a") as file:
            file.write(json.dumps(state) + "\n")
        if summary["complete"] and not running:
            alert("experiment_complete", successes=summary["successes"], total=summary["completed_rollouts"])
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
