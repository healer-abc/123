"""Independent numeric replay audit of normalized outputs through the kinematic bridge."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
RESULT_GROUP='results_droid_units'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    p=json.loads((ROOT/'protocol.json').read_text())
    phash=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
    old=Path(p['previous_run'])
    manifest={(r['task'],r['episode']):r for r in json.loads((old/'manifests/episodes.json').read_text())}
    stats=json.loads((Path(p['checkpoint'])/'assets'/p['normalization_asset']/'norm_stats.json').read_text())['norm_stats']
    sa0,sa1=np.asarray(stats['actions']['q01']),np.asarray(stats['actions']['q99'])
    sq0,sq1=np.asarray(stats['state']['q01']),np.asarray(stats['state']['q99'])
    rows=[];records=[];max_rotation=0.;max_action_error=0.;eef_displacements=[]
    for task in p['tasks']:
        items=[]
        for file in sorted((ROOT/RESULT_GROUP/task).glob('episode_*.json')):
            r=json.loads(file.read_text());assert r['protocol_sha256']==phash
            expected=manifest[task,r['episode']]
            assert r['task']==task and r['seed']==expected['seed'] and r['task_index']==expected['task_index']
            assert r['task_data_sha256']==expected['task_data_sha256']
            assert r['status']=='valid'
            assert Path(r['video']).is_file() and Path(r['video']).stat().st_size>0
            with np.load(r['trajectory']) as t:
                for key in t.files:assert np.isfinite(t[key]).all(),(file,key)
                n=r['steps'];assert n==len(t['actions'])==len(t['successes'])
                assert 0<n<=p['max_policy_steps']
                physical=(t['raw_plans'][...,:8]+1)/2*(sa1-sa0+1e-6)+sa0
                assert np.allclose(physical,t['physical_plans'],atol=2e-6)
                all_commands=physical[:,:p['replan_steps']].reshape(-1,8)
                ids=t['command_ids']
                expected_ids=np.floor(np.arange(n)*r['env_dt']*p['policy_hz']+1e-8).astype(int)
                assert np.array_equal(ids,expected_ids),(file,'control timebase')
                assert np.allclose(t['physical_commands'],all_commands[ids],atol=2e-6)
                normalized=2*(t['physical_states']-sq0)/(sq1-sq0+1e-6)-1
                assert np.allclose(t['states'],normalized,atol=2e-6)
                jr,jp=t['jacr'],t['jacp'];commands=t['physical_commands'][:,:7]
                commands=commands/np.maximum(1,np.max(np.abs(commands),axis=1,keepdims=True))
                commands=commands*p['droid_max_joint_delta_rad']*p['policy_hz']
                matrices=jr@np.swapaxes(jr,1,2)+1e-6*np.eye(3)
                correction=np.linalg.solve(matrices,(jr@commands[...,None]))
                projected=commands-(np.swapaxes(jr,1,2)@correction)[...,0]
                assert np.allclose(projected,t['projected_velocities'],atol=2e-6)
                translation=(jp@projected[...,None])[...,0]*r['env_dt']/r['action_scale']
                closure=np.clip(t['physical_commands'][:,7],0,1)
                equilibrium=np.where(closure<=6/7,.035*closure,.07*closure-.03)
                gripper=equilibrium+p['gripper_servo_gain']*(closure-t['physical_states'][:,7])
                expected_preclip=np.c_[translation,gripper]
                assert np.allclose(expected_preclip,t['preclip_actions'],atol=2e-6)
                action_error=float(np.max(np.abs(t['actions']-np.clip(expected_preclip,-1,1))))
                assert action_error<2e-6,(file,action_error)
                max_action_error=max(max_action_error,action_error)
                assert np.count_nonzero(np.abs(t['preclip_actions'][:,:3])>1)==r['cartesian_clipped_components']
                max_rotation=max(max_rotation,float(np.max(np.linalg.norm((jr@projected[...,None])[...,0],axis=1))))
                flags=t['successes'];assert bool(flags.any())==r['success']
                if r['success']:
                    assert flags[-1] and not flags[:-1].any() and r['first_success_step']==n
                else:assert r['first_success_step'] is None
                eef_displacements.append(float(np.linalg.norm(t['eef_positions'][-1]-t['eef_positions'][0])))
            items.append(r)
        assert len({r['episode'] for r in items})==len(items)
        assert len({r['task_index'] for r in items})==len(items)
        if not args.partial:assert len(items)==p['trials_per_task'],(task,len(items))
        rows.append(dict(task=task,episodes=len(items),successes=sum(r['success'] for r in items)))
        records.extend(items)
    result=dict(complete=len(records)==100,records=len(records),valid=len(records),tasks=rows,
                max_action_reconstruction_error=max_action_error,max_unrepresentable_rotation_rad_s=max_rotation,
                max_tcp_displacement_m=max(eef_displacements,default=0),
                mean_tcp_displacement_m=float(np.mean(eef_displacements)) if eef_displacements else None,
                total_environment_steps=sum(r['steps'] for r in records),
                cartesian_clipped_components=sum(r['cartesian_clipped_components'] for r in records),
                mean_state_ood_fraction=float(np.mean([r['state_ood_fraction'] for r in records])) if records else None,
                policy_inferences=sum(r['policy_inferences'] for r in records),protocol_sha256=phash,
                checks=['base metadata','same task identities and seeds','quantile normalization roundtrip',
                        '7D DROID normalized command to rad/s with uniform limit/8th gripper','Jacobian conversion','15 Hz resampling',
                        'first success stops episode','all trajectory and video files'])
    for name,expected in {
        'manifest.ocdbt':'78b6898ea1f8897a0225022b0ab799455cff0e15158a9ca349d8efbbe044aa55',
        '_METADATA':'250d0ad0f6540a5e4bdd70442d6864cef729724f2d23e2dd2cc669563a668d70',
    }.items():
        assert hashlib.sha256((Path(p['checkpoint'])/'params'/name).read_bytes()).hexdigest()==expected
    if not args.partial:
        (ROOT/RESULT_GROUP/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
        sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                 for name in ('adapter.py','run.py','protocol.json','check_interface.py','verify_results.py')}
        sources['inherited_base_adapter']=hashlib.sha256((old/'adapter.py').read_bytes()).hexdigest()
        sources['normalization_asset']=hashlib.sha256((Path(p['checkpoint'])/'assets'/p['normalization_asset']/'norm_stats.json').read_bytes()).hexdigest()
        (ROOT/'manifests/source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
