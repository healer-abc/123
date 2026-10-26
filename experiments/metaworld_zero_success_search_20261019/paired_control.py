"""Original settings on the locked policy's same 20 holdout seeds; never selects parameters."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('zero_search', HERE / 'search.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)
plan = search.get_plan()
assert len(plan['targets']) == 1 and plan['targets'][0]['method'] == 'ACT'
assert (HERE / 'ACT_pick-place-wall/locked.json').is_file()
# Separate progress files; score identity refers to an identical copy of the frozen plan.
progress_root = HERE / 'paired_control_state'
progress_root.mkdir(exist_ok=True)
(progress_root / 'plan.json').write_bytes((HERE / 'plan.json').read_bytes())
search.HERE = progress_root
while True:
    current_gpu = json.loads((HERE / 'status.json').read_text()).get('gpu', 0)
    out = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    candidates = [(int(f),int(g)) for g,f in (line.split(',') for line in out.splitlines()) if int(g) != current_gpu]
    free, gpu = max(candidates)
    if free >= 6144:
        break
    time.sleep(15)
child_env = search.environment(gpu)
if os.environ.get('METAWORLD_PAIRED_RUNTIME_READY') != '1':
    child_env['METAWORLD_PAIRED_RUNTIME_READY'] = '1'
    os.execve(str(search.PYTHON), [str(search.PYTHON), str(Path(__file__).resolve())], child_env)
os.environ.update(child_env)
search.bootstrap()
row = plan['targets'][0]
policy, stats, checkpoint = search.load_model(row)
result = search.rollout(policy, stats, row, {'temporal_agg': False, 'action_horizon': 100},
    plan['holdout_seeds'], HERE / 'ACT_pick-place-wall/paired_original20', 'paired_original20')
search.atomic_json(HERE / 'paired_control_complete.json', {'result': result, 'checkpoint_unchanged': search.sha256(checkpoint) == row['checkpoint_sha256']})
