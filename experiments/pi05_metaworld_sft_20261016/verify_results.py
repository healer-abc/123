"""Independent artifact and protocol audit of the frozen SFT evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    p=json.loads((ROOT/'protocol.json').read_text())
    phash=digest(ROOT/'protocol.json')
    checkpoint=ROOT/'checkpoints/RLinf-Pi05-MetaWorld-SFT'
    verified=json.loads((checkpoint/'verified.json').read_text())
    assert verified['sha256']==p['checkpoint_sha256'] and verified['revision']==p['checkpoint_revision']
    if not args.partial:assert digest(checkpoint/'model.safetensors')==p['checkpoint_sha256']
    load=json.loads((ROOT/'logs/weight_load.json').read_text())
    assert not load['missing'] and not load['unexpected']
    old=ROOT/'../pi05_metaworld_base_20261015/manifests/episodes.json'
    manifest={(r['task'],r['episode']):r for r in json.loads(old.read_text())}
    rows=[];records=[];max_action_error=0.;total_steps=0
    for task in p['tasks']:
        items=[]
        for file in sorted((ROOT/'results'/task).glob('episode_*.json')):
            r=json.loads(file.read_text())
            expected=manifest[task,r['episode']]
            assert r['status']=='valid' and r['task']==task and r['protocol_sha256']==phash
            assert r['checkpoint_sha256']==p['checkpoint_sha256']
            assert all(r[key]==expected[key] for key in ('seed','task_index','task_data_sha256'))
            video=Path(r['video']);trajectory=Path(r['trajectory'])
            assert video.is_file() and trajectory.is_file()
            assert digest(video)==r['video_sha256'] and digest(trajectory)==r['trajectory_sha256']
            with np.load(trajectory) as t:
                n=r['steps'];assert 0<n<=p['max_policy_steps']
                assert all(len(t[key])==n for key in ('states','actions','raw_actions','rewards','successes'))
                assert t['states'].shape==(n,4) and t['actions'].shape==(n,4)
                assert len(t['inference_seconds'])==(n+p['action_chunk']-1)//p['action_chunk']
                assert len(t['inference_image_sha256'])==len(t['inference_seconds'])
                for key in ('states','actions','raw_actions','rewards','inference_seconds'):
                    assert np.isfinite(t[key]).all(),(file,key)
                assert np.array_equal(t['successes'],np.asarray(t['successes'],dtype=bool))
                assert bool(t['successes'].any())==r['success']
                if r['success']:
                    assert t['successes'][-1] and not t['successes'][:-1].any()
                    assert r['first_success_step']==n
                else:assert r['first_success_step'] is None
                error=float(np.max(np.abs(t['actions']-np.clip(t['raw_actions'],-1,1))))
                assert error<1e-6,(file,error)
                max_action_error=max(max_action_error,error)
                assert np.isclose(float(np.sum(t['rewards'])),r['return'],atol=1e-4)
                total_steps+=n
            items.append(r)
        assert len({r['episode'] for r in items})==len(items)
        assert len({r['task_index'] for r in items})==len(items)
        if not args.partial:assert len(items)==p['trials_per_task'],(task,len(items))
        rows.append(dict(task=task,episodes=len(items),successes=sum(r['success'] for r in items)))
        records.extend(items)
    if not args.partial:
        summary=json.loads((ROOT/'results/summary.json').read_text())
        assert summary['completed']==len(records) and summary['successes']==sum(r['success'] for r in records)
    result=dict(complete=len(records)==100,records=len(records),
                successes=sum(r['success'] for r in records),
                tasks=rows,total_environment_steps=total_steps,
                max_action_reconstruction_error=max_action_error,
                protocol_sha256=phash,checkpoint_sha256=p['checkpoint_sha256'],
                checks=['publisher weight SHA256','strict model load','same task identity and seed',
                        'effective 4D action','first success stops','per-episode NPZ and MP4 SHA256'])
    if not args.partial:(ROOT/'results/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
