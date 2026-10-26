"""Copy the pinned official PPO recipe into a local two-GPU, one-epoch candidate."""
import os
from pathlib import Path

from omegaconf import OmegaConf

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
cfg = OmegaConf.load(Path(os.environ["RLINF_ROOT"]) / "examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml")
cfg.defaults.append("_self_")
cfg.cluster.component_placement = {"actor, env, rollout": "0-1"}
cfg.runner.logger.log_path = str(root / "runs/ppo_smoke")
cfg.runner.logger.experiment_name = "robotwin_pi0_ppo_smoke_2gpu"
cfg.runner.max_epochs = 1
cfg.runner.max_steps = -1
cfg.runner.val_check_interval = -1
cfg.runner.save_interval = 1
cfg.algorithm.update_epoch = 1
cfg.env.train.rollout_epoch = 1
cfg.env.train.total_num_envs = 8
cfg.env.eval.total_num_envs = 2
for split in ["train", "eval"]:
    cfg.env[split].assets_path = os.environ["ROBOTWIN_PATH"]
    cfg.env[split].task_config.save_path = str(workspace / "data/robotwin_pi0/smoke")
cfg.actor.micro_batch_size = 2
cfg.actor.global_batch_size = 32
cfg.actor.model.model_path = str(workspace / "models/robotwin_pi0/RLinf-Pi0-RoboTwin-SFT-adjust_bottle")
cfg.actor.model.openpi_data = {"norm_stats_path": str(Path(cfg.actor.model.model_path) / "physical-intelligence/robotwin/norm_stats.json")}
cfg.actor.model.openpi.torch_compile = False
cfg.actor.fsdp_config.gradient_checkpointing = True
OmegaConf.save(cfg, root / "configs/ppo_smoke_2gpu.yaml")
print(f"Prepared candidate: {root / 'configs/ppo_smoke_2gpu.yaml'}")
