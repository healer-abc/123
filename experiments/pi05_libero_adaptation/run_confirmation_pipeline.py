"""Run frozen confirmation, supplement E0 development identities, and publish the report."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import time

import development_watchdog as helpers
from confirmation_gate import validate_freeze

ROOT, OPENPI, LOGS = helpers.ROOT, helpers.OPENPI, helpers.LOGS
STOP = False
ACTIVE = None


def stop(*_):
    global STOP
    STOP = True


def status(**details):
    helpers.atomic_json(LOGS / "confirmation_pipeline_status.json", {"updated_at": helpers.now(), "pid": os.getpid(), **details})


def stage(name, command):
    global ACTIVE
    ACTIVE = helpers.spawn(command, helpers.environment(0), LOGS / f"{name}.log")
    helpers.event("confirmation_stage_start", stage=name, child_pid=ACTIVE.pid)
    try:
        while ACTIVE.poll() is None:
            if STOP:
                raise InterruptedError("confirmation pipeline stopped by user")
            status(state="running", stage=name, child_pid=ACTIVE.pid, log=str(LOGS / f"{name}.log"))
            for _ in range(10):
                if STOP:
                    break
                time.sleep(1)
        if ACTIVE.returncode:
            raise RuntimeError(f"{name} exited {ACTIVE.returncode}: {helpers.log_tail(LOGS / f'{name}.log')}")
    finally:
        helpers.terminate(ACTIVE)
        ACTIVE = None


def main():
    lock = (LOGS / "confirmation_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (LOGS / "confirmation_pipeline.pid").write_text(str(os.getpid()) + "\n")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        status(state="validating_frozen_artifacts")
        freeze = validate_freeze(ROOT / "manifests/candidate_freeze.json", ["E0-s42", "E1-s42", "E2-s42"])
        with (ROOT / "manifests/exposure_log.csv").open("a", newline="") as file:
            for run in freeze["runs"]:
                csv.writer(file).writerow([helpers.now(), run["run_id"], "fixed confirmation 1600", "confirmation_started_after_freeze",
                                          "frozen E2 versus E1; E0 reference", "E0 historical benchmark already exposed; preserve both metrics"])
        base = [str(OPENPI / ".venv/bin/python"), "-u", str(ROOT / "development_watchdog.py")]
        stage("confirmation_watchdog", base + ["--phase", "confirmation", "--run-ids", "E0-s42", "E1-s42", "E2-s42",
                                               "--skip-comparison", "--gpus", "0"])
        if json.loads((LOGS / "confirmation_watchdog_status.json").read_text())["state"] != "completed":
            raise RuntimeError("confirmation watchdog did not complete; inspect alerts")
        # Historical E0 results lack per-state identities. Fill its 400 dev rows for the common 2000-row table.
        stage("E0_development_supplement", base + ["--run-ids", "E0-s42", "--skip-comparison", "--gpus", "0"])
        e0 = json.loads((ROOT / "evaluations/E0-s42/development/summary.json").read_text())
        if not e0.get("complete") or e0.get("valid") != 400:
            raise RuntimeError("E0 development supplement is incomplete")
        stage("confirmation_final_analysis", [str(OPENPI / ".venv/bin/python"), "-u", str(ROOT / "analyze_confirmation.py")])
        status(state="completed", report=str(ROOT / "report/confirmation_report.md"), confirmation_episodes=4800,
               e0_development_supplement=400)
        helpers.event("confirmation_pipeline_complete", report=str(ROOT / "report/confirmation_report.md"))
    except Exception as error:
        status(state="stopped_by_signal" if STOP else "blocked", error=str(error))
        helpers.event("blocked", supervisor="confirmation_pipeline", reason=str(error))
        raise
    finally:
        helpers.terminate(ACTIVE)


if __name__ == "__main__":
    main()
