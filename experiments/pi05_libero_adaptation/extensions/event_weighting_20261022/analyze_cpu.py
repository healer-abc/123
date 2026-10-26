"""Calibrate command events against recorded gripper motion; audit existing exposed rollouts."""
import collections
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pyarrow.parquet as pq

from common import ROOT, PROJECT, DATA, events_and_mask, write_json


def demo_response(path):
    t=pq.read_table(path,columns=['actions','state','episode_index','task_index'],use_threads=False)
    a=np.asarray(t['actions'].to_pylist(),dtype=float)
    state=np.asarray(t['state'].to_pylist(),dtype=float)
    width=np.abs(state[:,-2:]).sum(axis=1)
    events,_=events_and_mask(a)
    output=[]
    for k in events:
        direction=-1 if a[k,6]>0 else 1
        # Expected signed finger-width response of at least 2mm, persistent for 2 observations.
        progress=direction*(width[k:]-width[k])
        choices=[j for j in range(1,min(30,len(progress)-1)) if progress[j]>=.002 and progress[j+1]>=.002]
        output.append({'episode':int(t['episode_index'][0].as_py()),'task_index':int(t['task_index'][0].as_py()),
                       'frame':k,'direction':'close_command' if direction==-1 else 'open_command',
                       'response_delay_frames':choices[0] if choices else None,
                       'available_followup_frames':len(progress)-1,
                       'width_delta_5':float(width[min(k+5,len(width)-1)]-width[k])})
    return output


def main():
    paths=sorted((DATA/'data').glob('*/*.parquet'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses=[v for row in pool.map(demo_response,paths) for v in row]
    summary={}
    for direction in ('close_command','open_command'):
        subset=[r for r in responses if r['direction']==direction]
        eligible5=[r for r in subset if r['available_followup_frames']>=6]
        eligible15=[r for r in subset if r['available_followup_frames']>=16]
        delays=[r['response_delay_frames'] for r in subset if r['response_delay_frames'] is not None]
        summary[direction]={'events':len(subset),'eligible_5frame':len(eligible5),'width_response_within_5':sum(r['response_delay_frames'] is not None and r['response_delay_frames']<=5 for r in eligible5),
                            'eligible_15frame':len(eligible15),'width_response_within_15':sum(r['response_delay_frames'] is not None and r['response_delay_frames']<=15 for r in eligible15),
                            'delay_quantiles_among_responders':np.quantile(delays,[0,.25,.5,.75,.95,1]).tolist()}
    write_json(ROOT/'artifacts/physical_command_calibration.json',{'note':'2mm summed absolute finger-joint position response; not object-contact or secure-grasp ground truth','by_direction':summary,'records':responses})
    rows=[]
    for run in ('E1-s42','E2-s42'):
        for p in sorted((PROJECT/'evaluations'/run/'development/episodes').glob('*/*/*.json')):
            v=json.loads(p.read_text())
            with np.load(v['trajectory']) as d:
                actions=d['actions']
            events,mask=events_and_mask(actions)
            rows.append({'run_id':run,'suite':v['suite'],'task_id':v['task_id'],'init_state_index':v['init_state_index'],
                         'success':v['success'],'steps':v['steps'],'events':events,'events_per_100_steps':len(events)/len(actions)*100,
                         'window_fraction':float(mask.mean()),'video':v['video']})
    result={}
    for run in ('E1-s42','E2-s42'):
        result[run]={}
        for success in (True,False):
            subset=[r for r in rows if r['run_id']==run and r['success']==success]
            result[run]['success' if success else 'failure']={'episodes':len(subset),'mean_events_per_100_steps':float(np.mean([r['events_per_100_steps'] for r in subset])),
                       'mean_command_window_fraction':float(np.mean([r['window_fraction'] for r in subset]))}
    write_json(ROOT/'artifacts/existing_rollout_event_statistics.json',{'note':'descriptive command counts only; trajectories have unequal lengths and command chatter may differ; no failure onset or causal inference','summary':result,'records':rows})
    print(json.dumps({'physical_response':summary,'existing_rollouts':result},indent=2))


if __name__=='__main__':
    main()
