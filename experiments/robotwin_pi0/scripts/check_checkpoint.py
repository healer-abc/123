"""Check downloaded shard headers and create a local, small official eval recipe."""
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from safetensors import safe_open

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
model_path = workspace / "models/robotwin_pi0/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
index = json.loads((model_path / "model.safetensors.index.json").read_text())
report = {"model_path": str(model_path), "shards": {}, "norm_stats_path": str(model_path / "physical-intelligence/robotwin/norm_stats.json")}
for filename in sorted(set(index["weight_map"].values())):
    with safe_open(model_path / filename, framework="pt", device="cpu") as tensors:
        actual = set(tensors.keys())
        expected = {key for key, shard in index["weight_map"].items() if shard == filename}
        assert actual == expected, filename
        report["shards"][filename] = {"size_bytes": (model_path / filename).stat().st_size, "tensor_count": len(actual)}
stats = json.loads(Path(report["norm_stats_path"]).read_text())["norm_stats"]
assert {"state", "actions"} <= stats.keys(), stats.keys()
for key in ["state", "actions"]:
    assert len(stats[key]["mean"]) == 32 and len(stats[key]["std"]) == 32, key
report["status"] = "headers_and_norm_stats_verified"
(root / "report/checkpoint-check.json").write_text(json.dumps(report, indent=2) + "\n")
source = Path(os.environ["RLINF_ROOT"]) / "evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml"
cfg = OmegaConf.load(source)
cfg.defaults.append("_self_")
cfg.cluster.component_placement = {"env, rollout": "0-1"}
cfg.runner.logger.log_path = str(root / "runs/eval_smoke")
cfg.runner.logger.experiment_name = "robotwin_pi0_eval_smoke_2gpu"
cfg.runner.max_epochs = 1
cfg.env.eval.total_num_envs = 2
cfg.env.eval.assets_path = os.environ["ROBOTWIN_PATH"]
cfg.env.eval.task_config.save_path = str(workspace / "data/robotwin_pi0/smoke")
cfg.rollout.model.model_path = str(model_path)
cfg.rollout.model.openpi_data.norm_stats_path = report["norm_stats_path"]
cfg.rollout.model.openpi.torch_compile = False
OmegaConf.save(cfg, root / "configs/eval_smoke_2gpu.yaml")
print(json.dumps(report, indent=2))
print(f"Prepared: {root / 'configs/eval_smoke_2gpu.yaml'}")
