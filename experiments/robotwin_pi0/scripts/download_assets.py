"""Download official RoboTwin assets into this workspace and configure robot paths."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--skip-backgrounds", action="store_true", help="Download robots and objects only; randomized backgrounds remain unavailable")
args = parser.parse_args()
root = Path(os.environ["ROBOTWIN_PATH"])
patterns = ["embodiments.zip", "objects.zip"]
if not args.skip_backgrounds:
    patterns.append("background_texture.zip")
snapshot_download("TianxingChen/RoboTwin2.0", repo_type="dataset", allow_patterns=patterns, local_dir=root / "assets", max_workers=3)
for name in patterns:
    archive = root / "assets" / name
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(root / "assets")
subprocess.run([sys.executable, "script/update_embodiment_config_path.py"], cwd=root, check=True)
print(f"Assets configured in {root / 'assets'}")
