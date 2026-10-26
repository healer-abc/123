"""Require actual training, resume, export and online update evidence before expansion."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
for name in ["sft_run", "sft_resume_run", "eval_run", "ppo_run", "ppo_resume_run"]:
    parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
args = parser.parse_args()
root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
records = {}
for name, run in vars(args).items():
    assert int((run / "exit-code.txt").read_text()) == 0, run
    accumulator = EventAccumulator(str(run / "tensorboard")).Reload()
    scalars = {tag: [{"step": e.step, "value": e.value} for e in accumulator.Scalars(tag)] for tag in accumulator.Tags()["scalars"]}
    assert scalars and all(math.isfinite(e["value"]) for series in scalars.values() for e in series), run
    records[name] = {"run_dir": str(run.resolve()), "exit_code": 0, "metrics": {k:v for k,v in scalars.items() if any(x in k for x in ["loss", "grad_norm", "success", "num_trajectories"])}}
for name, experiment, step in [("sft_run", "robotwin_pi0_lora_sft_smoke", 2), ("sft_resume_run", "robotwin_pi0_lora_sft_smoke", 3), ("ppo_run", "robotwin_pi0_ppo_smoke_2gpu", 1), ("ppo_resume_run", "robotwin_pi0_ppo_smoke_2gpu", 2)]:
    checkpoint = getattr(args, name) / experiment / f"checkpoints/global_step_{step}/actor"
    assert (checkpoint / "dcp_checkpoint/.metadata").is_file(), checkpoint
    assert len(list((checkpoint / "dcp_checkpoint").glob("*.distcp"))) == 2, checkpoint
    assert (checkpoint / "model_state_dict/full_weights.pt").stat().st_size > 6_000_000_000, checkpoint
    records[name]["checkpoint"] = str(checkpoint.resolve())
    records[name]["step"] = step
initial = torch.load(Path(records["sft_run"]["checkpoint"]) / "model_state_dict/full_weights.pt", mmap=True, map_location="cpu", weights_only=True)
resumed = torch.load(Path(records["sft_resume_run"]["checkpoint"]) / "model_state_dict/full_weights.pt", mmap=True, map_location="cpu", weights_only=True)
adapter = "llm.layers.0.attn.q_proj.0.lora_B.default.weight"
base = "llm.layers.0.attn.q_proj.0.base_layer.weight"
assert not torch.equal(initial[adapter], resumed[adapter]), "Resume did not update adapter"
assert torch.equal(initial[base], resumed[base]), "Frozen base changed during LoRA SFT"
ppo_initial = torch.load(Path(records["ppo_run"]["checkpoint"]) / "model_state_dict/full_weights.pt", mmap=True, map_location="cpu", weights_only=True)
ppo_resumed = torch.load(Path(records["ppo_resume_run"]["checkpoint"]) / "model_state_dict/full_weights.pt", mmap=True, map_location="cpu", weights_only=True)
assert not torch.equal(ppo_initial["value_head.mlp.0.weight"], ppo_resumed["value_head.mlp.0.weight"])
assert not torch.equal(ppo_initial["llm.layers.0.attn.q_proj.1.weight"], ppo_resumed["llm.layers.0.attn.q_proj.1.weight"])
assert torch.equal(ppo_initial["llm.layers.0.attn.q_proj.0.weight"], ppo_resumed["llm.layers.0.attn.q_proj.0.weight"])
reports = {name: json.loads((root / "report" / name).read_text()) for name in ["converted-dataset-check.json", "sft-loader-check.json", "pilot-own-stats-loader-check.json", "lora-action-parity.json", "official-action-alignment-check.json"]}
assert all(record["status"] == "passed" for record in reports.values())
merged = workspace / "models/robotwin_pi0/pilot_lora_merged"
for name, run in vars(args).items():
    config = OmegaConf.load(run / "tensorboard/config.yaml")
    if name.startswith("sft"):
        assert Path(config.data.train_data_paths).resolve() == (workspace / "data/robotwin_pi0/lerobot/local/robotwin_pi0_interface").resolve()
        assert config.actor.model.is_lora
        stamp = json.loads((run / "data-source.json").read_text())
        manifest = workspace / "data/robotwin_pi0/lerobot/local/robotwin_pi0_interface/meta/source_episodes.jsonl"
        assert stamp["total_frames"] == 292
        assert stamp["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    else:
        model = config.rollout.model if name == "eval_run" else config.actor.model
        assert Path(model.model_path).resolve() == merged.resolve()
        assert not model.is_lora
merge_check = json.loads((merged / "merge-check.json").read_text())
expected_source = Path(records["sft_run"]["checkpoint"]) / "model_state_dict/full_weights.pt"
assert Path(merge_check["source_checkpoint"]).resolve() == expected_source.resolve()
assert Path(reports["lora-action-parity.json"]["checkpoint"]).resolve() == expected_source.resolve()
assert reports["converted-dataset-check.json"]["total_frames"] == 292
assert all(c["merge_float32_error"] < 1e-4 for c in merge_check["checks"])
official_norm = workspace / "models/robotwin_pi0/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json"
assert hashlib.sha256(official_norm.read_bytes()).digest() == hashlib.sha256((merged / "norm_stats.json").read_bytes()).digest()
result = {"status": "passed", "runs": records, "data_checks": reports, "merge": merge_check, "frozen_base_unchanged": True, "resumed_adapter_updated": True, "render_denoiser": os.environ["ROBOTWIN_DENOISER"], "normalization": "same pinned official statistics for SFT/export/eval/PPO; pilot-derived statistics separately verified"}
(root / "report/data-interface-check.json").write_text(json.dumps(result, indent=2) + "\n")
print("Data interface passed; expansion may start")
