"""Wait for verified assets, then validate one real policy rollout before the full run."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    protocol_hash = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    validation = json.loads((ROOT / "diagnostics/interface_validation.json").read_text())
    assert validation["passed"] and validation["protocol_sha256"] == protocol_hash
    while not ((ROOT / "checkpoints/RLinf-Pi0-MetaWorld-RL-FlowNoise/verified.json").exists()
               and (ROOT / "logs/runtime_ready.json").exists()):
        time.sleep(10)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", JAX_PLATFORMS="cpu", TORCH_COMPILE_DISABLE="1",
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="4")
    cuda_libs = [str(path) for path in (ROOT / "deps/nvidia").glob("*/lib")]
    cuda_libs.append("/home/zhsu/jyluo/projects/openpi/.venv/lib/python3.11/site-packages/cusparselt/lib")
    env["LD_LIBRARY_PATH"] = ":".join(cuda_libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    with (ROOT / "logs/policy_smoke.log").open("ab") as file:
        process = subprocess.Popen(["/home/zhsu/jyluo/projects/openpi/.venv/bin/python", "-u",
                                    str(ROOT / "rollout.py"), "--task", "assembly-v3", "--limit", "1",
                                    "--output-group", "smoke_results"], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=file, stderr=subprocess.STDOUT)
        print(json.dumps({"event": "policy_smoke_started", "pid": process.pid}), flush=True)
        code = process.wait()
    if code:
        with (ROOT / "logs/alerts.jsonl").open("a") as file:
            file.write(json.dumps({"time": time.time(), "event": "policy_smoke_failed", "exit_code": code,
                                   "log": str(ROOT / "logs/policy_smoke.log")}) + "\n")
        raise SystemExit(code)
    result = json.loads((ROOT / "smoke_results/assembly-v3/episode_00.json").read_text())
    artifact = {"time": time.time(), "success": True, "smoke_task_success": result["success"],
                "protocol_sha256": protocol_hash,
                "policy_steps": result["steps"], "peak_cuda_memory_mb": result["peak_cuda_memory_mb"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "note": "Smoke excluded from the 100 experimental trials; success means execution validity."}
    (ROOT / "logs/policy_smoke_passed.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps({"event": "policy_smoke_passed", **artifact}), flush=True)


if __name__ == "__main__":
    main()
