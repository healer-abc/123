"""Fetch the official adjust_bottle Pi0 SFT checkpoint, without starting training."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

destination = Path(os.environ["ROBOTWIN_WORKSPACE"]) / "models/robotwin_pi0/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
snapshot_download("RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle", local_dir=destination, max_workers=4)
print(destination)
