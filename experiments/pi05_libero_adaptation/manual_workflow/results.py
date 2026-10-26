"""New manual runs only: validate records, freeze candidate, and compare paired results.

No automatic model execution, retries, or candidate selection. Reuses the historical
analysis math without executing its hardcoded main routine or modifying its outputs.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OPENPI = PROJECT.parents[1] / 'projects/openpi'
sys.path.insert(0, str(PROJECT))
from analyze_confirmation import compare, summarize
from confirmation_gate import inventory, sha256

CONFIGS = {'E0':'pi05_libero', 'E1':'pi05_libero_lora',
           'E2':'pi05_libero_lora_frozen_vision', 'E3':'pi05_libero_action_only'}


def load(root, arm, phase):
    run = f'{root.name}_{arm}_s42'
    checkpoint = str(OPENPI/'checkpoints'/CONFIGS[arm]/run/'29999')
    paths = sorted((PROJECT/'evaluations'/run/phase/'episodes').glob('*/*/*.json'))
    import csv
    with (PROJECT/'manifests/eval_manifest.csv').open() as f:
        expected = {(r['suite'],int(r['task_id']),int(r['init_state_index'])):r
                    for r in csv.DictReader(f) if r['phase']==phase}
    records = {}
    for path in paths:
        value = json.loads(path.read_text())
        key = (value['suite'],value['task_id'],value['init_state_index'])
        if key in records or key not in expected:
            raise ValueError(f'duplicate/unexpected identity: {path}')
        assert value['status']=='valid' and value['config']==CONFIGS[arm]
        assert value['checkpoint']==checkpoint
        assert value['manifest_sha256']==sha256(PROJECT/'manifests/eval_manifest.csv')
        assert all(str(value.get(k))==str(v) for k,v in expected[key].items()), path
        assert sha256(value['trajectory'])==value['trajectory_sha256']
        with np.load(value['trajectory']) as data:
            actions = data['actions']
        assert actions.shape==(value['steps'],7) and np.isfinite(actions).all()
        assert bool(np.max(np.abs(actions))>1.000001)==value['action_out_of_nominal_bounds']
        if not value['success']:
            assert value['video'] and sha256(value['video'])==value['video_sha256']
        records[key]=value
    assert records.keys()==expected.keys(), f'{arm}: incomplete {phase} ({len(records)}/{len(expected)})'
    return records


def check(root):
    target=root/'freeze.json'
    assert sha256(target)==(root/'freeze.sha256').read_text().strip()
    frozen=json.loads(target.read_text())
    assert sha256(PROJECT/'manifests/eval_manifest.csv')==frozen['manifest_sha256']
    for path,digest in frozen['sources'].items():
        assert sha256(path)==digest, path
    for checkpoint,expected in frozen['weights'].items():
        assert inventory(checkpoint)==expected, checkpoint
    print('candidate/source/manifest/weight hashes match',flush=True)
    return frozen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command', choices=('summarize','freeze','check-freeze'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('development','confirmation'),default='development')
    parser.add_argument('--arms',nargs='+',choices=tuple(CONFIGS),default=['E0','E1','E2','E3'])
    parser.add_argument('--candidate',choices=('E2','E3'),default='E2')
    parser.add_argument('--reason')
    parser.add_argument('--require-arm',choices=tuple(CONFIGS))
    args=parser.parse_args()
    root=args.root.resolve()
    assert root.parent==PROJECT/'manual_runs' and root.name.startswith('manual_'), 'manual_runs/manual_* only'
    root.mkdir(parents=True,exist_ok=True)
    if args.command=='check-freeze':
        frozen=check(root)
        if args.require_arm:
            assert args.require_arm in frozen['arms'], 'arm is not part of the frozen confirmation cohort'
        return
    if args.command=='freeze':
        assert args.reason, 'supply an actual reviewed selection reason'
        target=root/'freeze.json'
        assert not target.exists(), 'freeze already exists; never overwrite'
        arms=list(dict.fromkeys(['E0','E1',args.candidate]))
        for arm in arms:
            load(root,arm,'development')
            assert not (PROJECT/'evaluations'/f'{root.name}_{arm}_s42'/'confirmation').exists(), 'confirmation already exposed for this run'
        sources=[PROJECT/'eval_manifest.py',PROJECT/'seeded_policy_server.py',Path(__file__),
                 Path(__file__).with_name('lab.sh'),OPENPI/'pyproject.toml',OPENPI/'uv.lock']
        sources+=list((OPENPI/'src/openpi').rglob('*.py'))
        sources+=list((OPENPI/'scripts').glob('*.py'))
        sources+=list((OPENPI/'third_party/libero/libero/libero').rglob('*.py'))
        frozen={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'candidate':args.candidate,'reference':'E1','arms':arms,'reason':args.reason,
                'manifest_sha256':sha256(PROJECT/'manifests/eval_manifest.csv'),
                'sources':{str(p):sha256(p) for p in sources},
                'weights':{},'environment_margin_pp':-5,
                'limitations':['previously exposed benchmark; not independent OOD confirmation',
                               'one training seed; replay/process variation not included in bootstrap',
                               'environment and strict raw-action success reported separately']}
        for arm in arms:
            checkpoint=OPENPI/'checkpoints'/CONFIGS[arm]/f'{root.name}_{arm}_s42'/'29999'
            print(f'hashing {checkpoint}',flush=True)
            frozen['weights'][str(checkpoint)]=inventory(checkpoint)
        with target.open('x') as f:json.dump(frozen,f,indent=2)
        with (root/'freeze.sha256').open('x') as f:f.write(sha256(target)+'\n')
        print(target);return
    if args.phase=='confirmation':
        frozen=check(root)
        assert set(args.arms)==set(frozen['arms'])
    records={arm:load(root,arm,args.phase) for arm in args.arms}
    result={'phase':args.phase,'runs':{a:summarize(r) for a,r in records.items()},'comparisons':{}}
    if 'E1' in records:
        for arm in records:
            if arm=='E1':continue
            result['comparisons'][arm+'-E1']={
                'environment':compare(records[arm],records['E1']),
                'strict':compare(records[arm],records['E1'],strict=True)}
    output=root/f'{args.phase}_comparison.json'
    with output.open('x') as f:json.dump(result,f,indent=2)
    for arm,r in result['runs'].items():
        print(arm,r['episodes'],r['successes'],r['suite_equal_environment_success_rate'])
    print(output)


if __name__=='__main__':main()
