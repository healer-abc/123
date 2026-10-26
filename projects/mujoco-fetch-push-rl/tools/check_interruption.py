"""Stop this test's own runner after its first validation, then resume on CPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
out = ROOT / "runs/interruption_probe"
resumed = ROOT / "runs/interruption_resume"
assert not out.exists() and not resumed.exists(), "Refusing to overwrite a prior test"
command = [sys.executable, "-u", "run.py", "train", "--task", "Reach", "--group", "D",
           "--steps", "100000", "--buffer-size", "20000", "--learning-starts", "100",
           "--eval-every", "200", "--eval-episodes", "1", "--checkpoint-every", "200",
           "--device", "cpu", "--out", str(out)]
with (ROOT / "logs/interruption_probe.log").open("x") as log:
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while not (out / "validation.csv").exists():
            assert process.poll() is None, "Runner exited before interruption test"
            if time.monotonic() > deadline:
                raise TimeoutError("First validation did not arrive within 120 seconds")
            time.sleep(0.2)
        process.send_signal(signal.SIGTERM)
        returncode = process.wait(timeout=120)
        assert returncode == 130, returncode
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=120)

first = json.loads((out / "summary.json").read_text())
assert first["status"] == "interrupted"
assert (out / "interrupted/complete.json").is_file()
target = first["timesteps"] + 300
with (ROOT / "logs/interruption_resume.log").open("x") as log:
    subprocess.run([
        sys.executable, "-u", "run.py", "train", "--resume", str(out / "interrupted"),
        "--steps", str(target), "--eval-every", "200", "--eval-episodes", "1",
        "--device", "cpu", "--out", str(resumed),
    ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
last = json.loads((resumed / "summary.json").read_text())
assert last["status"] == "final" and last["timesteps"] == target
assert last["session_steps"] == 300
assert (resumed / "final/complete.json").is_file()
result = {"status": "passed", "signal": "SIGTERM", "interruption_returncode": returncode,
          "interrupted_steps": first["timesteps"], "resumed_steps": last["timesteps"],
          "added_steps": last["session_steps"], "device": "cpu"}
(ROOT / "checks/interruption.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
