"""Validate completed local diagnostic evidence; no model execution or parameter updates."""
import csv
import json

import numpy as np

from common import ROOT, PROJECT, events_and_mask, sha, write_json


def main():
    prep = json.loads((ROOT/'artifacts/preparation_complete.json').read_text())
    for relative, key in [('PROTOCOL.md','protocol_sha256'), ('artifacts/samples.json','sample_sha256'),
                          ('artifacts/rollout_sample.json','rollout_sample_sha256')]:
        assert sha(ROOT/relative) == prep[key], relative
    frozen = json.loads((PROJECT/'manifests/candidate_freeze.json').read_text())
    source_checks = {name: sha(name)==expected for name, expected in frozen['source_and_evidence_hashes'].items()}
    assert all(source_checks.values()), [p for p,v in source_checks.items() if not v]
    replay_audit = json.loads((ROOT/'artifacts/replay_audit.json').read_text())
    run_summaries = {}
    failures = []
    for run in ('E1-s42','E2-s42'):
        summary = json.loads((ROOT/'rollouts'/run/'summary.json').read_text())
        assert summary['complete'] and len(summary['records'])==12
        for record in summary['records']:
            assert record['status']=='valid'
            assert record['manifest_sha256']==frozen['manifest_sha256']
            assert sha(record['trajectory'])==record['trajectory_sha256']
            assert sha(record['all_video'])==record['all_video_sha256']
            with np.load(record['trajectory']) as data:
                actions = data['actions']
            assert actions.shape==(record['steps'],7) and np.isfinite(actions).all()
            if not record['success']:
                failures.append((run,record['suite'],record['task_id'],record['init_state_index']))
        run_summaries[run] = {'episodes':12,'successes':summary['successes'],
                              'raw_bound_strict_successes':sum(r['raw_bound_strict_success'] for r in summary['records'])}
    with (ROOT/'artifacts/failure_visual_labels.csv').open() as stream:
        labels = list(csv.DictReader(stream))
    assert {(r['run_id'],r['suite'],int(r['task_id']),int(r['init_state_index'])) for r in labels}==set(failures)
    distances=[]
    for label in labels:
        record=next(r for r in replay_audit['records'] if r['run_id']==label['run_id'] and r['pair_index']==int(label['pair_index']))
        result={**label,'persistent_command_steps':record['events']}
        if label['first_deviation_start_step']:
            lo,hi=int(label['first_deviation_start_step']),int(label['first_deviation_end_step'])
            d=np.array([min(abs(k-e) for e in record['events']) for k in range(lo,hi+1)])
            result['distance_range_policy_steps']=[int(d.min()),int(d.max())]
            for radius in (5,10):
                result[f'within_{radius}_steps']={'possible':bool((d<=radius).any()),'entire_interval':bool((d<=radius).all())}
        else:
            result['distance_range_policy_steps']=None
        distances.append(result)
    reloads=[json.loads((ROOT/f'artifacts/fixed_reload_{n}.json').read_text()) for n in (1,2)]
    assert reloads[0]['input_bundle_sha256']==reloads[1]['input_bundle_sha256']
    result={'state':'completed','training_updates':0,'runs':run_summaries,
            'historical_outcome_disagreements':sum(not r['same_success_as_original'] for r in replay_audit['records']),
            'selection_caution':'deliberately selected historical discordant development cases; not a benchmark success estimate or overall flip-rate estimate',
            'preregistered_identity_hashes_verified':True,'frozen_source_and_evidence_files_verified':len(source_checks),
            'all_24_new_trajectories_and_videos_verified':True,'failure_annotations':distances,
            'fresh_process_first_action_max_abs_delta':float(np.max(np.abs(np.array(reloads[0]['first_action'])-reloads[1]['first_action']))),
            'decision':'do not launch A/B/R training; establish replay numerical provenance and broaden blind failure annotations first'}
    write_json(ROOT/'artifacts/diagnostic_summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
