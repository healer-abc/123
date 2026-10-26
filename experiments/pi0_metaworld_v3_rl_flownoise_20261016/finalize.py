"""Audit RL-fine-tuned pi0 v3 rollout and publish paired SFT comparison."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = "/home/zhsu/jyluo/projects/openpi/.venv/bin/python"


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    state_file = ROOT / "logs/watchdog_state.json"
    while True:
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        if state.get("phase") == "complete":
            break
        if state.get("phase") == "blocked_by_worker_errors":
            raise RuntimeError(f"Worker blocked: {state.get('blocked_tasks')}")
        time.sleep(10)
    summary = json.loads((ROOT / "results/summary.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert summary["complete"] and summary["completed_rollouts"] == 100
    assert sha(ROOT / "sources/metaworld-3.1.1-py3-none-any.whl") == protocol["metaworld_wheel_sha256"]
    for script in ("diagnostics/replay_new.py", "compare_runs.py"):
        subprocess.run([PYTHON, "-u", str(ROOT / script)], cwd=ROOT, check=True)
    replay = json.loads((ROOT / "diagnostics/new_trajectory_replay.json").read_text())
    manifest = json.loads((ROOT / "checkpoints/RLinf-Pi0-MetaWorld-RL-FlowNoise/verified.json").read_text())
    smoke = json.loads((ROOT / "logs/policy_smoke_passed.json").read_text())
    assert replay["complete"] and replay["passed"] and len(replay["episodes"]) == 100
    assert manifest["sha256"] == protocol["checkpoint_sha256"] == smoke["checkpoint_sha256"]
    assert smoke["success"]
    weights = [json.loads(p.read_text()) for p in (ROOT / "logs").glob("weight_load_*.json")]
    from rollout import EXPECTED_UNUSED_TENSORS
    assert len(weights) >= 1 and all(not x["missing"] and set(x["unexpected"]) == EXPECTED_UNUSED_TENSORS for x in weights)
    result = {"complete": True, "environment": "Farama MetaWorld 3.1.1 v3",
              "metaworld_wheel_sha256": protocol["metaworld_wheel_sha256"],
              "checkpoint_sha256": manifest["sha256"],
              "protocol_sha256": sha(ROOT / "protocol.json"),
              "episodes": 100, "successes": summary["successes"],
              "independent_replays_passed": len(replay["episodes"]),
              "policy_weight_loads_verified": len(weights),
              "all_policy_tensors_loaded": True,
              "unused_training_tensors_per_load": len(EXPECTED_UNUSED_TENSORS)}
    (ROOT / "diagnostics/final_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "results/report.md").open("a") as file:
        file.write("\n100/100 回合通过独立模拟器逐步重放；与 SFT 的配对对照见 `results/comparison.md`。\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
