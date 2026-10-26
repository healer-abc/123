"""Run RLinf's official normalization pipeline with a small pilot batch."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
from openpi.shared import normalize
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from toolkits.lerobot.calculate_norm_stats import create_torch_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
config = get_openpi_config("pi0_aloha_robotwin", repo_id=args.repo_id, batch_size=2)
data_cfg = config.data.create(config.assets_dirs, config.model)
loader, batches = create_torch_dataloader(data_cfg, config.model.action_horizon, 2, config.model, num_workers=0)
assert batches > 0, batches
stats = {key: normalize.RunningStats() for key in ["state", "actions"]}
for batch in loader:
    for key, accumulator in stats.items():
        accumulator.update(np.asarray(batch[key]))
result = {key: accumulator.get_statistics() for key, accumulator in stats.items()}
for record in result.values():
    assert all(np.isfinite(v).all() for v in dataclasses.asdict(record).values() if v is not None)
normalize.save(args.output, result)
(args.output / "source.json").write_text(json.dumps({"repo_id": args.repo_id, "config_name": "pi0_aloha_robotwin", "batches": batches, "pipeline": "official repack and data transforms; before model padding"}, indent=2) + "\n")
print(args.output / "norm_stats.json")
