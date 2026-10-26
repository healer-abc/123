"""Summarize matched probe windows without treating tokens/repeats as independent samples."""
import json
import numpy as np

from common import ROOT, SEED, write_json


def main():
    samples=json.loads((ROOT/'artifacts/samples.json').read_text())['samples']
    metrics=('stock_32','action_7','motion_6','gripper_1','padding_25')
    values={}
    parity={}
    for run in ('E1-s42','E2-s42'):
        data=json.loads((ROOT/'artifacts'/f'{run}_loss_probe.json').read_text())
        parity[run]=data['stock_loss_max_abs_difference']
        values[run]={}
        for index in range(48):
            values[run][index]={kind:{metric:float(np.mean([r[metric] for r in data['records'] if r['sample_index']==index and r['kind']==kind]))
                                    for metric in metrics} for kind in ('event','control')}
        for index in range(48):
            for rep in range(4):
                pair=[r for r in data['records'] if r['sample_index']==index and r['repeat']==rep]
                assert len(pair)==2 and pair[0]['flow_time']==pair[1]['flow_time']
    a=json.loads((ROOT/'artifacts/E1-s42_loss_probe.json').read_text())['records']
    b=json.loads((ROOT/'artifacts/E2-s42_loss_probe.json').read_text())['records']
    assert len(a)==len(b)==384
    assert all((x['sample_index'],x['kind'],x['repeat'],x['flow_time'])==(y['sample_index'],y['kind'],y['repeat'],y['flow_time']) for x,y in zip(a,b))
    summaries={}
    for scope,indices in {'all':list(range(48)),'long_training_tasks':[s['index'] for s in samples if s['task_index']<10],
                          'close_commands':[s['index'] for s in samples if s['event_direction']=='close_command'],
                          'open_commands':[s['index'] for s in samples if s['event_direction']=='open_command']}.items():
        summaries[scope]={'pairs':len(indices)}
        for run in values:
            group={kind:{m:float(np.mean([values[run][i][kind][m] for i in indices])) for m in metrics} for kind in ('event','control')}
            group['event_over_control']={m:group['event'][m]/group['control'][m] for m in metrics}
            summaries[scope][run]=group
    # Exploratory paired task-cluster interval, retaining all samples within each selected task.
    rng=np.random.default_rng(SEED)
    tasks=sorted({s['task_index'] for s in samples})
    cluster_delta=np.array([np.mean([values['E2-s42'][s['index']]['event']['stock_32']-values['E1-s42'][s['index']]['event']['stock_32'] for s in samples if s['task_index']==task]) for task in tasks])
    reps=cluster_delta[rng.integers(0,len(tasks),(10000,len(tasks)))].mean(axis=1)
    output={'sampling_unit':'48 paired windows from 34 selected demonstration tasks; four repetitions averaged before statistics',
            'all_common_flow_times_verified':True,'stock_loss_parity_max_abs':parity,'summaries':summaries,
            'event_E2_minus_E1_task_equal_stock32':float(cluster_delta.mean()),
            'exploratory_task_cluster_bootstrap_95':np.quantile(reps,[.025,.975]).tolist(),
            'limitations':['In-training demonstrations, not generalization validation.',
                           'Control windows match episode and approximate motion energy, not all visual or stage difficulty.',
                           'Command discontinuity can increase gripper loss intrinsically; high flow MSE does not prove loss weighting helps.',
                           'A single training seed; the cluster interval is diagnostic sampling uncertainty only.']}
    write_json(ROOT/'artifacts/loss_summary.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
