"""Replay twelve exposed identities with the unchanged client; save success videos too."""
import argparse
import csv
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT))
import eval_manifest as original
from libero.libero import benchmark
from openpi_client import websocket_client_policy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--port',type=int,default=8125)
    args=parser.parse_args()
    run=next(r for r in json.loads((PROJECT/'manifests/runs.json').read_text())['runs'] if r['run_id']==args.run_id)
    selected=json.loads((ROOT/'artifacts/rollout_sample.json').read_text())['identities']
    keys={(r['suite'],r['task_id'],r['init_state_index']) for r in selected}
    with original.MANIFEST.open() as f:
        rows=[r for r in csv.DictReader(f) if r['phase']=='development' and (r['suite'],int(r['task_id']),int(r['init_state_index'])) in keys]
    assert len(rows)==12
    client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',args.port)
    meta=client.get_server_metadata()
    assert meta['config']==run['config'] and meta['checkpoint']==run['final_checkpoint'] and meta['explicit_noise']
    out=ROOT/'rollouts'/args.run_id
    results=[]
    for row in rows:
        suite=benchmark.get_benchmark_dict()[row['suite']]()
        env=original.make_env(suite.get_task(int(row['task_id'])),int(row['env_seed']))
        frames=[]
        base_prepare=original.prepare_image
        calls=[0]
        def capture(img):
            prepared=base_prepare(img)
            if calls[0]%2==0:
                frames.append(prepared.copy())
            calls[0]+=1
            return prepared
        original.prepare_image=capture
        try:
            value=original.run_episode(client,row,suite,env,out,run['final_checkpoint'],run['config'])
        finally:
            original.prepare_image=base_prepare
            env.close()
        video=out/'all_videos'/row['suite']/f"task_{int(row['task_id']):02d}_state_{int(row['init_state_index']):02d}.mp4"
        video.parent.mkdir(parents=True,exist_ok=True)
        imageio.mimwrite(video,frames,fps=10)
        value['all_video']=str(video)
        value['all_video_sha256']=original.sha256(video)
        value['diagnostic_note']='original run_episode and controller unchanged; extra videos saved for successes as well'
        record_path=out/'records'/row['suite']/f"task_{int(row['task_id']):02d}_state_{int(row['init_state_index']):02d}.json"
        original.atomic_json(record_path,value)
        results.append(value)
        original.atomic_json(out/'summary.json',{'complete':len(results)==12,'episodes':len(results),'successes':sum(v['success'] for v in results),'records':results})
        print(json.dumps({'run_id':args.run_id,'completed':len(results),'suite':row['suite'],'task':row['task_id'],'state':row['init_state_index'],'success':value['success']}),flush=True)


if __name__=='__main__':
    main()
