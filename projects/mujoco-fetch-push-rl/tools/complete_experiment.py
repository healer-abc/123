"""Bounded pilot, three CPU seed workers, fixed-budget evaluation and delivery."""
import concurrent.futures
import csv
import fcntl
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
P = json.loads((ROOT / "configs/experiment.json").read_text())
assert hashlib.sha256((ROOT / P["runner"]).read_bytes()).hexdigest() == P["runner_sha256"]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = str(P["threads"])
STATUS = {"stage": "starting", "tasks": {}, "gpu_used": False}
LOCK = threading.Lock()
LAST_PILOT_PLOT = None


def update(stage=None, task=None, value=None, **extra):
    with LOCK:
        if stage:
            STATUS["stage"] = stage
        if task:
            STATUS["tasks"][task] = value
        STATUS.update(extra)
        STATUS["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        target = ROOT / "report/status.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(STATUS, ensure_ascii=False, indent=2) + "\n")
        temp.replace(target)
        rows = ["# 实验运行状态", "", f"更新时间：{STATUS['updated_at']}", "",
                f"阶段：**{STATUS['stage']}**", "", "GPU：未使用；正式最多三个 CPU 种子进程，每个两线程。", ""]
        for key, item in sorted(STATUS["tasks"].items()):
            rows.append(f"- {key}: {item}")
        if STATUS.get("error"):
            rows += ["", "错误：", "```text", STATUS["error"], "```"]
        if STATUS.get("pilot_validation"):
            rows += ["", f"预实验最近验证：{STATUS['pilot_validation']}"]
        (ROOT / "report/STATUS.md").write_text("\n".join(rows) + "\n")


def progress(directory):
    global LAST_PILOT_PLOT
    path = ROOT / directory / "validation.csv"
    if not path.exists():
        return
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("timesteps") and r.get("success_rate") and r.get("distance_m")]
    if not rows:
        return
    update(pilot_validation=rows[-2:])
    token = (str(directory), rows[-1]["timesteps"])
    if token != LAST_PILOT_PLOT:
        with (ROOT / "logs/plot_progress.log").open("a") as log:
            subprocess.run([sys.executable, "tools/plot_progress.py", str(directory)], stdout=log, stderr=subprocess.STDOUT, check=True)
        LAST_PILOT_PLOT = token


def child(arguments, log, monitor=None):
    command = [sys.executable, *map(str, arguments)]
    print(json.dumps(command, ensure_ascii=False), flush=True)
    with (ROOT / "logs" / log).open("a") as stream:
        if monitor is None:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    progress(monitor)
                    time.sleep(15)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
                progress(monitor)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=120)


def finished(directory, steps, group=None, seed=None):
    directory = ROOT / directory
    if not directory.exists():
        return False
    for name in ("summary.json", "config.json", "system.json", "final/complete.json", "final/model.zip"):
        if not (directory / name).is_file():
            raise RuntimeError(f"Partial output needs explicit recovery: {directory}")
    summary = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    system = json.loads((directory / "system.json").read_text())
    marker = json.loads((directory / "final/complete.json").read_text())
    assert summary["status"] == "final" and marker["steps"] == summary["timesteps"]
    assert summary["timesteps"] >= steps
    assert summary["timesteps"] < steps + 2048
    assert config["target_steps"] == steps and config["device"] == "cpu"
    assert system["runner_sha256"] == P["runner_sha256"]
    if group:
        assert config["group"] == group and config["seed"] == seed
        assert config["task"] == P["task"] and config["learning_rate"] == P["learning_rate"]
        assert config["buffer_size"] == P["buffer_size"]
        assert config["learning_starts"] == P["learning_starts"]
    return True


def pilot_ready(directory):
    with (ROOT / directory / "validation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    latest = rows[-2:]
    rates = [float(row["success_rate"]) for row in latest]
    update(pilot_validation=latest)
    return len(rates) == 2 and sum(rates) / 2 >= P["pilot"]["min_last_two_validation_mean"]


def pilot():
    accepted = P["pilot"].get("accepted_checkpoint")
    if accepted:
        checkpoint = ROOT / accepted
        assert all((checkpoint / name).is_file() for name in ("complete.json", "config.json", "model.zip", "replay.pkl"))
        config = json.loads((checkpoint / "config.json").read_text())
        system = json.loads((checkpoint.parent / "system.json").read_text())
        assert config["seed"] == 99 and config["group"] == "D" and config["task"] == "Push"
        assert system["runner_sha256"] == P["runner_sha256"]
        evaluation = json.loads((ROOT / P["pilot"]["admission_evaluation"] / "summary.json").read_text())
        assert evaluation["checkpoint"] == str(checkpoint.resolve())
        assert evaluation["seed_base"] == 10000 and evaluation["training_seed"] == 99
        assert len(evaluation["metrics"]) == 1
        metric = evaluation["metrics"][0]
        assert metric["gain"] == 1.0 and metric["episodes"] == 20
        assert metric["success_rate"] >= P["pilot"]["min_last_two_validation_mean"]
        assert pilot_ready(checkpoint.parent)
        update("pilot_accepted_early", pilot_selected=accepted, admission_success_rate=metric["success_rate"])
        return accepted
    directory = P["pilot"]["first"]
    update("pilot_300k")
    while not (ROOT / directory / "summary.json").is_file():
        alive = subprocess.run(["tmux", "has-session", "-t", P["pilot"]["external_tmux"]],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not alive:
            raise RuntimeError("External pilot stopped before producing a final summary; inspect its log.")
        progress(directory)
        time.sleep(15)
    assert finished(directory, 300000)
    if pilot_ready(directory):
        return directory
    extended = P["pilot"]["extended"]
    update("pilot_extend_to_1m")
    if not finished(extended, P["pilot"]["max_steps"]):
        child([P["runner"], "train", "--resume", ROOT / directory / "final",
               "--steps", P["pilot"]["max_steps"], "--eval-every", 20000,
               "--eval-episodes", 20, "--device", "cpu", "--out", extended], "pilot_extended_20261022.log", monitor=extended)
    if not pilot_ready(extended):
        update("needs_diagnosis", error="Seed99 has not met the declared learning gate after 1M steps. Formal experiments were not started.")
        return None
    return extended


def check_evaluation(directory, checkpoint, gains):
    path = ROOT / directory
    if not path.exists():
        return False
    summary = json.loads((path / "summary.json").read_text())
    assert summary["checkpoint"] == str((ROOT / checkpoint).resolve())
    assert summary["seed_base"] == 20000
    with (path / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for gain in gains:
        subset = [row for row in rows if float(row["gain"]) == gain]
        assert len(subset) == 100
        assert {int(row["episode_seed"]) for row in subset} == set(range(20000, 20100))
    assert len(rows) == len(gains) * 100
    return True


def seed_worker(seed):
    for group in P["order"]:
        directory = f"runs/{P['run_prefix']}/{group}_seed{seed}"
        key = f"{group}/seed{seed}"
        update(task=key, value="training")
        if not finished(directory, P["formal_steps"], group, seed):
            child([P["runner"], "train", "--task", P["task"], "--group", group,
                   "--seed", seed, "--steps", P["formal_steps"], "--lr", P["learning_rate"],
                   "--buffer-size", P["buffer_size"], "--learning-starts", P["learning_starts"],
                   "--eval-every", P["validation"]["every"], "--eval-episodes", P["validation"]["episodes"],
                   "--checkpoint-every", P.get("checkpoint_overrides", {}).get(group, P["checkpoint_every"]), "--threads", P["threads"],
                   "--device", "cpu", "--out", directory], f"formal_{group}_seed{seed}.log")
        assert finished(directory, P["formal_steps"], group, seed)
        update(task=key, value="evaluating")
        gains = P["test"]["robust_gains"] if group in ("D", "E") else [1.0]
        evaluation = f"evaluations/{P['run_prefix']}/{group}_seed{seed}"
        checkpoint = f"{directory}/final"
        if not check_evaluation(evaluation, checkpoint, gains):
            child([P["runner"], "eval", "--checkpoint", checkpoint, "--episodes", 100,
                   "--seed-base", 20000, "--gains", *gains, "--device", "cpu", "--out", evaluation],
                  f"final_eval_{group}_seed{seed}.log")
        assert check_evaluation(evaluation, checkpoint, gains)
        update(task=key, value="complete")


def main():
    formal = ROOT / "runs" / P["run_prefix"]
    formal.mkdir(parents=True, exist_ok=True)
    manifest = formal / "protocol.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == P, "Protocol changed; use a new run prefix."
    else:
        manifest.write_text(json.dumps(P, ensure_ascii=False, indent=2) + "\n")
    update(protocol=str(manifest.relative_to(ROOT)))
    source = pilot()
    if source is None:
        return 2
    update("formal_matrix", pilot_selected=source)
    with concurrent.futures.ThreadPoolExecutor(max_workers=P["max_workers"]) as pool:
        futures = {pool.submit(seed_worker, seed): seed for seed in P["training_seeds"]}
        errors = []
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(f"seed {futures[future]}: {error}")
                update(task=f"seed{futures[future]}", value=f"failed: {error}")
        if errors:
            raise RuntimeError("\n".join(errors))
    update("case_videos")
    child(["tools/make_case_video.py"], "case_video.log")
    update("figures_report_resume")
    child(["tools/summarize_results.py"], "summarize_results.log")
    update("complete", deliverables=["report/REPORT.md", "report/REPORT.pdf", "report/RESUME.md",
                                    "report/results.csv", "figures/", "videos/project_demo_90s.mp4"])
    return 0


if __name__ == "__main__":
    process_lock = (ROOT / "report/experiment.lock").open("a")
    fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        raise SystemExit(main())
    except Exception:
        update("failed", error=traceback.format_exc())
        raise
