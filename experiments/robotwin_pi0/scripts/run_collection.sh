#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/env.sh"
CONFIG="${1:?task config name required}"
GPU="${2:-0}"
COUNT="${3:-}"
cd "$ROBOTWIN_PATH"
mkdir -p "$PROJECT_DIR/logs/collection"
export CUDA_VISIBLE_DEVICES="$GPU"
python "$PROJECT_DIR/scripts/snapshot_collection_config.py" "$CONFIG" --gpu "$GPU" --count "$COUNT" --record-attempt
if [[ -n "$COUNT" ]]; then
  python -u "$PROJECT_DIR/scripts/collect_config.py" "$CONFIG" "$COUNT" 2>&1 | tee -a "$PROJECT_DIR/logs/collection/$CONFIG.log"
else
  python -u script/collect_data.py adjust_bottle "$CONFIG" 2>&1 | tee -a "$PROJECT_DIR/logs/collection/$CONFIG.log"
fi
python - "$CONFIG" "$COUNT" <<'PY'
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import yaml
name=sys.argv[1]
cfg=yaml.safe_load(Path(f'task_config/{name}.yml').read_text())
folder=Path(cfg['save_path'])/'adjust_bottle'/name
count=int(sys.argv[2]) if sys.argv[2] else cfg['episode_num']
seeds=[int(s) for s in (folder/'seed.txt').read_text().split()]
assert len(seeds)>=count,len(seeds)
for i in range(count):
    with h5py.File(folder/'data'/f'episode{i}.hdf5','r') as source:
        assert source['joint_action/vector'].shape[1]==14
        assert np.isfinite(source['joint_action/vector'][:]).all()
    instruction=json.loads((folder/'instructions'/f'episode{i}.json').read_text())
    assert instruction.get('seen'),i
print(f'Collection complete: {name}, {count} episodes with instructions')
PY
