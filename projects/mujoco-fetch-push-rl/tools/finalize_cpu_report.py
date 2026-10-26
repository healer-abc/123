"""Check the final conda environment and summarize completed CPU acceptance."""
import csv
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3.common.logger import configure
import run

torch.set_num_threads(2)
assert Path(sys.prefix) == ROOT / "envs/fetch-rl"
assert (Path(sys.prefix) / "conda-meta/history").is_file()
assert torch.version.cuda is None


def read(path):
    return json.loads((ROOT / path).read_text())


report = {"status": "passed", "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
          "root": str(ROOT), "environment_manager": "conda", "python_executable": sys.executable,
          "gpu_used": False, "training_quality_validated": False}
for name in ("cpu_environment", "acceptance", "interruption", "cpu_rendering"):
    value = read(f"checks/{name}.json")
    assert value["status"] == "passed"
    report[name] = value

before, _, before_env = run.load_bundle(ROOT / "runs/smoke_A_push/final", "cpu")
after, _, after_env = run.load_bundle(ROOT / "runs/smoke_A_resume/final", "cpu")
try:
    assert before.num_timesteps == 2048 and after.num_timesteps == 4096
    assert after._n_updates > before._n_updates
    assert all(torch.isfinite(p).all() for p in after.policy.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before.policy.parameters(), after.policy.parameters()))
    report["ppo_resume"] = {"status": "passed", "initial_steps": 2048, "final_steps": 4096,
                            "initial_updates": before._n_updates, "final_updates": after._n_updates,
                            "policy_parameters_updated": True}
finally:
    before_env.close()
    after_env.close()

baseline = read("runs/speed_cpu/summary.json")
assert baseline["status"] == "final" and baseline["timesteps"] == 50000
packages = read("runs/speed_cpu/system.json")["packages"]
assert all(version(name) == value for name, value in packages.items())
model, _, env = run.load_bundle(ROOT / "runs/speed_cpu/final", "cpu", training=True)
try:
    assert model.num_timesteps == 50000 and model.replay_buffer.size() == 50000
    assert model._n_updates == 40000
    model.set_logger(configure(format_strings=[]))
    model.train(gradient_steps=1, batch_size=256)
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
finally:
    env.close()
with (ROOT / "runs/speed_cpu/progress.csv").open() as file:
    for row in csv.DictReader(file):
        assert all(np.isfinite(float(value)) for value in row.values() if value)
report["cpu_benchmark"] = {**baseline, "original_environment_manager": "uv",
                           "core_package_versions_match_final_conda": True,
                           "checkpoint_loaded_and_updated_in_conda": True}

with imageio.get_reader(ROOT / "videos/reach_smoke_cpu.mp4") as reader:
    assert reader.count_frames() == 51
    assert reader.get_meta_data()["fps"] == 25
report["video"] = {"status": "passed", "frames": 51, "fps": 25,
                   "path": "videos/reach_smoke_cpu.mp4", "renderer": "llvmpipe"}
(ROOT / "checks/cpu_video.json").write_text(json.dumps(report["video"], indent=2) + "\n")
report["tmux"] = subprocess.check_output(["tmux", "-V"], text=True).strip()
socket = "fetch-rl-conda-preflight"
try:
    subprocess.run(["tmux", "-L", socket, "-f", "/dev/null", "new-session", "-d", "-s", "check", "sleep 30"], check=True)
    subprocess.run(["tmux", "-L", socket, "list-sessions"], check=True, stdout=subprocess.DEVNULL)
finally:
    subprocess.run(["tmux", "-L", socket, "kill-server"], check=True)
report["tmux_session_lifecycle"] = "passed"
(ROOT / "checks/cpu_stage_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "environment_manager": "conda",
                  "cpu_steps_per_second": baseline["session_steps_per_second"],
                  "ppo_resume": report["ppo_resume"], "gpu_used": False}, indent=2))
