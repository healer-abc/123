"""Audit protocol, seeds, success-stop semantics and recorded raw/executed actions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z*z / count
    center = (p + z*z / (2*count)) / scale
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / scale
    return [max(0., center-radius), min(1., center+radius)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    protocol_hash = hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
    manifest = {(r['task'],r['episode']):r for r in json.loads((ROOT/'manifests/episodes.json').read_text())}
    records = []
    for task in protocol['tasks']:
        task_records = []
        for path in sorted((ROOT/'results'/task).glob('episode_*.json')):
            record = json.loads(path.read_text())
            assert record['protocol_sha256'] == protocol_hash, path
            identity = manifest[task, record['episode']]
            assert record['task'] == task and record['seed'] == identity['seed'], path
            assert record['task_index'] == identity['task_index'], path
            if record['status'] == 'valid':
                assert record['task_data_sha256'] == identity['task_data_sha256'], path
                assert Path(record['video']).is_file() and Path(record['video']).stat().st_size > 0, path
                with np.load(record['trajectory']) as trajectory:
                    actions, raw = trajectory['actions'], trajectory['raw_plans']
                    success_flags = trajectory['successes']
                    count = len(actions)
                    assert count == record['steps'] == len(success_flags) == len(trajectory['rewards']), path
                    assert 0 < count <= protocol['max_policy_steps'] and actions.shape == (count,4), path
                    assert np.isfinite(actions).all() and np.isfinite(raw).all(), path
                    selected = raw[:,:protocol['replan_steps'],:4].reshape(-1,4)[:count]
                    assert np.array_equal(actions, np.clip(selected,-1,1)), path
                    assert record['clipped_action_components'] == np.count_nonzero(actions != selected), path
                    assert record['success'] == bool(success_flags.any()), path
                    if record['success']:
                        assert success_flags[-1] and not success_flags[:-1].any(), path
                        assert record['first_success_step'] == count, path
                    else:
                        assert record['first_success_step'] is None, path
            task_records.append(record)
        assert len({r['episode'] for r in task_records}) == len(task_records), task
        assert len({r['task_index'] for r in task_records}) == len(task_records), task
        if not args.partial:
            assert len(task_records) == protocol['trials_per_task'], (task,len(task_records))
        records.extend(task_records)
    valid = [r for r in records if r['status']=='valid']
    task_metrics = []
    for task in protocol['tasks']:
        items = [r for r in valid if r['task']==task]
        count = len(items)
        successes = sum(r['success'] for r in items)
        task_metrics.append(dict(task=task, valid=count, successes=successes,
                                 success_rate=successes/count if count else None,
                                 wilson_95=wilson(successes,count)))
    validation = dict(complete=len(records)==100, records=len(records), valid=len(valid),
                      missing=len(records)-len(valid), verified_protocol_sha256=protocol_hash,
                      success_stop_rule='all successful trajectories end at first true success flag',
                      raw_to_executed_action_consistency='passed',
                      seed_and_reset_manifest_consistency='passed',
                      tasks=task_metrics,
                      total_policy_steps=sum(r['steps'] for r in valid),
                      total_episode_seconds=sum(r['elapsed_seconds'] for r in valid),
                      clipped_action_components=sum(r['clipped_action_components'] for r in valid),
                      total_action_components=sum(r['action_components'] for r in valid))
    for filename, expected in {
        'manifest.ocdbt':'78b6898ea1f8897a0225022b0ab799455cff0e15158a9ca349d8efbbe044aa55',
        '_METADATA':'250d0ad0f6540a5e4bdd70442d6864cef729724f2d23e2dd2cc669563a668d70',
    }.items():
        actual=hashlib.sha256((Path(protocol['checkpoint'])/'params'/filename).read_bytes()).hexdigest()
        assert actual==expected, filename
    validation['base_checkpoint_metadata_unchanged'] = True
    if not args.partial:
        (ROOT/'results/validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        paths=[ROOT/name for name in ('adapter.py','run.py','protocol.json','verify_results.py')]
        sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        (ROOT/'manifests/source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
