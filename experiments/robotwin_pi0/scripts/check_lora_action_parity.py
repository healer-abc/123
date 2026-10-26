"""Compare a real Pi0 action window before and after exporting trained LoRA."""
import argparse
import json
import os
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from peft.tuners.tuners_utils import BaseTunerLayer
from rlinf.data.datasets.openpi.official_sft_data_loader import build_official_openpi_sft_dataloader
from rlinf.models import get_model
import robotwin_pi0_ext

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--merged", required=True, type=Path)
args = parser.parse_args()
root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
robotwin_pi0_ext.register()
with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.1"):
    cfg = compose(config_name="sft_smoke_2gpu")
cfg.actor.model.openpi.embed_dtype = "fp32"
loader, _ = build_official_openpi_sft_dataloader(cfg, 1, 0, cfg.data.train_data_paths)
observation, _ = next(iter(loader))
model = get_model(cfg.actor.model)
state = torch.load(args.checkpoint, mmap=True, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model = model.cuda().eval()
# Pi0's constructor selects TF32; disable it for an FP32 export equivalence test.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
observation = model._observation_to_device(observation)
generator = torch.Generator(device="cuda").manual_seed(0)
noise = torch.randn(1, 50, 32, device="cuda", generator=generator)
with torch.inference_mode():
    before = model.sample_actions(observation, num_steps=5, noise=noise.clone())
    for name, module in list(model.named_modules()):
        if isinstance(module, BaseTunerLayer):
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, module.get_base_layer())
    merged = torch.load(args.merged, mmap=True, map_location="cpu", weights_only=True)
    model.load_state_dict(merged, strict=True)
    after = model.sample_actions(observation, num_steps=5, noise=noise.clone())
assert torch.isfinite(before).all() and torch.isfinite(after).all()
torch.testing.assert_close(after, before, rtol=5e-4, atol=5e-4)
result = {"status": "passed", "precision": "fp32", "tf32": False, "shape": list(after.shape), "max_abs_action_error": float((after-before).abs().max()), "mean_abs_action_error": float((after-before).abs().mean()), "checkpoint": str(args.checkpoint.resolve()), "merged": str(args.merged.resolve())}
(root / "report/lora-action-parity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
