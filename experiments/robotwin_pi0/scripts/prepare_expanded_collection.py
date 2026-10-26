"""Keep formal collection separate from pilot episodes, using the same protocol."""
import os
from pathlib import Path
import yaml

root = Path(os.environ["ROBOTWIN_EXPERIMENT_ROOT"])
workspace = Path(os.environ["ROBOTWIN_WORKSPACE"])
robotwin = Path(os.environ["ROBOTWIN_PATH"])
for split in ["clean", "randomized"]:
    cfg = yaml.safe_load((root / "configs" / f"demo_{split}_pi0.yml").read_text())
    cfg["episode_num"] = 200
    cfg["save_path"] = f"../../data/robotwin_pi0/raw/{split}"
    name = f"demo_{split}_pi0_train200"
    path = root / "configs" / f"{name}.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    for link, target in [
        (robotwin / "task_config" / f"{name}.yml", path),
        (robotwin / "data/adjust_bottle" / name, workspace / "data/robotwin_pi0/raw" / split / "adjust_bottle" / name),
    ]:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            assert link.resolve() == target.resolve(), link
        elif link.exists():
            raise FileExistsError(link)
        else:
            if target == path:
                assert target.exists()
            else:
                target.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target, target_is_directory=target != path)
    print(f"Prepared {name}: 200 episodes, {cfg['planner_backend']}")
