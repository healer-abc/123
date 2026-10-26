"""Make time-aligned paired replay sheets and event-neighborhood views after both probes finish."""
import json
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from common import ROOT, PROJECT, events_and_mask, sha, write_json


def main():
    start=time.monotonic()
    files=[ROOT/'rollouts'/run/'summary.json' for run in ('E1-s42','E2-s42')]
    while not all(p.exists() and json.loads(p.read_text()).get('complete') for p in files):
        status=json.loads((ROOT/'artifacts/gpu_status.json').read_text())
        if status['state'] in ('blocked','stopped_by_signal') or time.monotonic()-start>1800:
            raise RuntimeError('replay incomplete; inspect diagnostic GPU status')
        time.sleep(2)
    runs={run:{(v['suite'],v['task_id'],v['init_state_index']):v for v in json.loads(path.read_text())['records']}
          for run,path in zip(('E1-s42','E2-s42'),files)}
    selected=json.loads((ROOT/'artifacts/rollout_sample.json').read_text())['identities']
    audits=[]
    for page in range(6):
        sheet=Image.new('RGB',(12*128,4*164),'white')
        draw=ImageDraw.Draw(sheet)
        for pair_index,s in enumerate(selected[page*2:page*2+2]):
            key=(s['suite'],s['task_id'],s['init_state_index'])
            a,b=runs['E1-s42'][key],runs['E2-s42'][key]
            success_steps=min(v['steps'] for v in (a,b) if v['success']) if a['success'] or b['success'] else min(a['steps'],b['steps'])
            requested=list(np.linspace(0,success_steps-1,10,dtype=int))+[success_steps+50,max(a['steps'],b['steps'])-1]
            for j,(run,v) in enumerate((('E1-s42',a),('E2-s42',b))):
                assert sha(v['all_video'])==v['all_video_sha256']
                with np.load(v['trajectory']) as data:
                    actions=data['actions']
                events,mask=events_and_mask(actions)
                old=json.loads((PROJECT/'evaluations'/run/'development/episodes'/s['suite']/f"task_{s['task_id']:02d}"/f"state_{s['init_state_index']:02d}.json").read_text())
                with np.load(old['trajectory']) as data:
                    previous=data['actions']
                same_shape=actions.shape==previous.shape
                audit={'pair_index':page*2+pair_index,'run_id':run,**s,'success':v['success'],'same_success_as_original':v['success']==old['success'],
                       'same_action_shape_as_original':same_shape,'max_abs_action_delta':float(np.max(np.abs(actions-previous))) if same_shape else None,
                       'steps':v['steps'],'events':events,'command_window_fraction':float(mask.mean()),'video':v['all_video']}
                audits.append(audit)
                reader=imageio.get_reader(v['all_video'])
                assert reader.count_frames()==v['steps']
                row=pair_index*2+j
                draw.text((3,row*164+2),f"pair{page*2+pair_index} {run} {s['suite']} t{s['task_id']} state{s['init_state_index']} success={v['success']} events={events}",fill='black')
                for col,step in enumerate(requested):
                    frame=min(int(step),v['steps']-1)
                    draw.text((col*128+2,row*164+15),f"step {frame}"+(' END' if step>=v['steps'] else ''),fill='black')
                    sheet.paste(Image.fromarray(reader.get_data(frame)).resize((128,128)),(col*128,row*164+32))
                reader.close()
        sheet.save(ROOT/f'artifacts/paired_replays_{page:02d}.jpg')
    write_json(ROOT/'artifacts/replay_audit.json',{'planned_pairs':12,'completed_rollouts':len(audits),'records':audits,
               'all_successes_reproduced':all(a['same_success_as_original'] for a in audits),
               'note':'columns align absolute policy steps; repeated terminal frames explicitly labelled END; video playback fps is not simulation control frequency'})
    print(json.dumps({'completed_rollouts':len(audits),'outcome_disagreements':sum(not a['same_success_as_original'] for a in audits)},indent=2))


if __name__=='__main__':
    main()
