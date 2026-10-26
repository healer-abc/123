"""Prepare matched S1/S2 LoRA experiments and a two-job memory profile."""
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
common = OmegaConf.load(root / "configs/sft_smoke_2gpu.yaml")
common.runner.max_steps = 10000
common.runner.save_interval = 5000
common.runner.resume_dir = None
common.actor.micro_batch_size = 4
common.actor.global_batch_size = 16
common.actor.optim.total_training_steps = 10000
common.actor.optim.lr_warmup_steps = 500
common.actor.fsdp_config.save_full_model_weights = False
common.actor.model.model_path = str(workspace / "models/robotwin_pi0/pi0_base_openpi")
common.actor.model.openpi_data.norm_stats_path = str(root / "report/clean200_norm_stats/norm_stats.json")
common.data.num_workers = 2

for label, repo in [("s1_clean", "robotwin_pi0_clean200"), ("s2_mixed", "robotwin_pi0_mixed200")]:
    cfg = OmegaConf.create(OmegaConf.to_container(common))
    cfg.data.train_data_paths = str(workspace / "data/robotwin_pi0/lerobot/local" / repo)
    cfg.runner.logger.experiment_name = label
    cfg.runner.logger.log_path = str(root / "runs" / label)
    OmegaConf.save(cfg, root / "configs" / f"{label}.yaml")

for label in ["profile_a", "profile_b"]:
    cfg = OmegaConf.create(OmegaConf.to_container(common))
    cfg.runner.max_steps = 5
    cfg.actor.micro_batch_size = 2
    cfg.runner.save_interval = -1
    cfg.runner.logger.experiment_name = label
    # These profiles measure resource use and DCP saving; they are not S1/S2.
    cfg.actor.model.model_path = str(workspace / "models/robotwin_pi0/RLinf-Pi0-RoboTwin-SFT-adjust_bottle")
    cfg.data.train_data_paths = str(workspace / "data/robotwin_pi0/lerobot/local/robotwin_pi0_clean200")
    OmegaConf.save(cfg, root / "configs" / f"sft_{label}.yaml")

for name in ["s1_clean", "s2_mixed"]:
    path = root / "configs" / f"{name}.yaml"
    print(name, hashlib.sha256(path.read_bytes()).hexdigest())
