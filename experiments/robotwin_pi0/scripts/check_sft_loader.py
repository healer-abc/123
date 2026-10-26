"""Check the actual RLinf/OpenPI SFT batch contract without loading a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from rlinf.data.datasets.openpi.official_sft_data_loader import build_official_openpi_sft_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--report", default="sft-loader-check.json")
parser.add_argument("--norm-stats")
args = parser.parse_args()
root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.1"):
    cfg = compose(config_name="sft_smoke_2gpu")
if args.norm_stats:
    cfg.actor.model.openpi_data.norm_stats_path = args.norm_stats
loader, data_cfg = build_official_openpi_sft_dataloader(cfg, 1, 0, args.dataset)
observation, action = next(iter(loader))
assert observation.state.shape == (1, 32), observation.state.shape
assert action.shape == (1, 50, 32), action.shape
assert torch.isfinite(observation.state).all() and torch.isfinite(action).all()
images = {key: list(value.shape) for key, value in observation.images.items()}
assert len(images) == 3, images
assert all(torch.isfinite(value).all() for value in observation.images.values())
norm_path = Path(cfg.actor.model.openpi_data.norm_stats_path)
result = {"status": "passed", "dataset": args.dataset, "state": list(observation.state.shape), "action": list(action.shape), "images": images, "norm_stats_keys": list(data_cfg.norm_stats), "norm_stats_path": str(norm_path.resolve()), "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(), "prompt_tokens": list(observation.tokenized_prompt.shape)}
(root / "report" / args.report).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
