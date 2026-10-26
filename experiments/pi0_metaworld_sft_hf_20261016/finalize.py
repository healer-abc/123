"""Run the independent replay and paired comparison after all 100 rollouts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = "/home/zhsu/jyluo/projects/openpi/.venv/bin/python"


def main():
    state_path = ROOT / "logs/watchdog_state.json"
    while True:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("phase") == "complete":
            break
        if state.get("phase") == "blocked_by_worker_errors":
            raise RuntimeError(f"Worker blocked: {state.get('blocked_tasks')}")
        time.sleep(10)
    summary = json.loads((ROOT / "results/summary.json").read_text())
    assert summary["complete"] and summary["completed_rollouts"] == 100
    for script in ("diagnostics/replay_new.py", "compare_runs.py"):
        subprocess.run([PYTHON, "-u", str(ROOT / script)], cwd=ROOT, check=True)
    replay = json.loads((ROOT / "diagnostics/new_trajectory_replay.json").read_text())
    pairs = json.loads((ROOT / "diagnostics/paired_comparison.json").read_text())
    manifest = json.loads((ROOT / "checkpoints/RLinf-Pi0-MetaWorld-SFT/verified.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    smoke = json.loads((ROOT / "logs/policy_smoke_passed.json").read_text())
    assert replay["complete"] and replay["passed"] and len(replay["episodes"]) == 100
    assert len(pairs) == 100
    assert manifest["sha256"] == protocol["checkpoint_sha256"] == smoke["checkpoint_sha256"]
    assert smoke["success"]
    weights = [json.loads(p.read_text()) for p in (ROOT / "logs").glob("weight_load_*.json")]
    assert len(weights) >= 1 and all(not x["missing"] and not x["unexpected"] for x in weights)
    result = {"complete": True, "checkpoint_sha256": manifest["sha256"],
              "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
              "episodes": 100, "successes": summary["successes"],
              "independent_replays_passed": len(replay["episodes"]),
              "paired_scenes_verified": len(pairs), "strict_weight_loads": len(weights)}
    (ROOT / "diagnostics/final_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    report = ROOT / "results/report.md"
    with report.open("a") as file:
        file.write("\n检查点训练元数据：旧 ModelScope 5,000 步；新 Hugging Face 10,000 步；均为 MetaWorld SFT。\n")
        file.write("逐回合独立模拟器重放及同场景配对检查均为 100/100，详见 `diagnostics/final_validation.json` 与 `results/comparison.md`。\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
