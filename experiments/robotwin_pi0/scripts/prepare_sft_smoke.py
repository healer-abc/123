"""Prepare a small LoRA SFT run through the official RLinf entry point."""
import os
from pathlib import Path
from omegaconf import OmegaConf

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
cfg = OmegaConf.load(Path(os.environ["RLINF_ROOT"]) / "examples/sft/config/robotwin_adjust_bottle_sft_openpi.yaml")
cfg.hydra.searchpath = ["file://${oc.env:RLINF_ROOT}/examples/sft/config/", "file://${oc.env:EMBODIED_PATH}/config/"]
cfg.cluster.component_placement = {"actor": "0-1"}
cfg.runner.logger.log_path = str(root / "runs/sft_smoke")
cfg.runner.logger.experiment_name = "robotwin_pi0_lora_sft_smoke"
cfg.runner.max_epochs = -1
cfg.runner.max_steps = 2
cfg.runner.save_interval = 1
cfg.runner.val_check_interval = -1
cfg.runner.resume_dir = None
cfg.data.train_data_paths = str(workspace / "data/robotwin_pi0/lerobot/local/robotwin_pi0_interface")
cfg.data.num_workers = 0
cfg.actor.micro_batch_size = 1
cfg.actor.global_batch_size = 2
cfg.actor.model.model_path = str(workspace / "models/robotwin_pi0/RLinf-Pi0-RoboTwin-SFT-adjust_bottle")
cfg.actor.model.is_lora = True
cfg.actor.model.lora_rank = 8
cfg.actor.model.lora_target_scope = "all_linear"
cfg.actor.model.openpi.torch_compile = False
cfg.actor.model.openpi_data.norm_stats_path = str(Path(cfg.actor.model.model_path) / "physical-intelligence/robotwin/norm_stats.json")
cfg.actor.fsdp_config.sharding_strategy = "full_shard"
cfg.actor.fsdp_config.wrappers.projections = {"module_names": ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]}
cfg.actor.fsdp_config.wrappers.lora_adapters = {"module_names": ["*.lora_A.default", "*.lora_B.default"]}
cfg.actor.fsdp_config.mixed_precision_rules.lora_adapters = "fp32"
cfg.actor.fsdp_config.wrappers.projection_bases = {"module_names": [f"{name}.base_layer" for name in ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]]}
cfg.actor.fsdp_config.mixed_precision_rules.projection_bases = "fp32"
cfg.actor.optim.total_training_steps = 10
cfg.actor.optim.lr_warmup_steps = 1
OmegaConf.save(cfg, root / "configs/sft_smoke_2gpu.yaml")
print(root / "configs/sft_smoke_2gpu.yaml")
