"""Replay one original seed with original inference settings, after holdout ends."""
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
while True:
    state = json.loads((HERE / 'status.json').read_text())
    if state['state'] == 'complete':
        break
    if state['state'] == 'infrastructure_error':
        raise RuntimeError(state)
    time.sleep(15)
while True:
    out = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free, gpu = max((int(f),int(g)) for g,f in (line.split(',') for line in out.splitlines()))
    if free >= 6144:
        break
    time.sleep(15)
child_env = search.environment(gpu)
if os.environ.get('METAWORLD_PARITY_RUNTIME_READY') != '1':
    child_env['METAWORLD_PARITY_RUNTIME_READY'] = '1'
    os.execve(str(search.PYTHON), [str(search.PYTHON), str(Path(__file__).resolve())], child_env)
os.environ.update(child_env)
search.bootstrap()
row = search.get_plan()['targets'][0]
assert row['method'] == 'ACT'
policy, stats, checkpoint = search.load_model(row)
old_path = Path(row['output']) / 'final_rollouts/episode_00.json'
old = json.loads(old_path.read_text())
dest = HERE / 'ACT_pick-place-wall/baseline_replay'
search.rollout(policy, stats, row, {'temporal_agg': False, 'action_horizon': 100},
               [old['seed']], dest, 'baseline_parity_replay')
new = json.loads((dest / 'episode_00.json').read_text())
result = {'original': str(old_path), 'replayed': str(dest / 'episode_00.json'),
    'seed': old['seed'], 'initial_state_equal': old['initial_state_sha256'] == new['initial_state_sha256'],
    'success_equal': old['success'] == new['success'], 'steps_equal': old['steps'] == new['steps'],
    'original_return': old['return'], 'replayed_return': new['return'],
    'return_abs_difference': abs(old['return'] - new['return'])}
result['passed'] = result['initial_state_equal'] and result['success_equal'] and result['steps_equal'] and result['return_abs_difference'] < 1e-6
search.atomic_json(HERE / 'baseline_parity_audit.json', result)
assert result['passed'], result
print(json.dumps(result), flush=True)
