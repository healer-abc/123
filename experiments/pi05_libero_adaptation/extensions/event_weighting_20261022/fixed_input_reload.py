"""Short fresh-process inference check against an archived exact input/noise bundle."""
import argparse
import json

import numpy as np

from openpi.policies import policy_config
from openpi.training import config as train_config

from common import ROOT, PROJECT, sha, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    run=next(r for r in json.loads((PROJECT/'manifests/runs.json').read_text())['runs'] if r['run_id']=='E1-s42')
    bundle=ROOT/'artifacts/first_input_seed_11_reset_0.npz'
    with np.load(bundle) as data:
        obs={'observation/image':data['image'],'observation/wrist_image':data['wrist'],
             'observation/state':data['state'],'prompt':'pick up the black bowl on the cookie box and place it on the plate'}
        noise=data['noise']; expected=data['actions']
    # Match the exact task instruction, rather than assuming wording from the task id.
    import csv
    with (PROJECT/'manifests/eval_manifest.csv').open() as file:
        row=next(r for r in csv.DictReader(file) if r['phase']=='development' and r['suite']=='libero_spatial' and r['task_id']=='3' and r['init_state_index']=='1')
    obs['prompt']=row['task_description']
    policy=policy_config.create_trained_policy(train_config.get_config(run['config']),run['final_checkpoint'],sample_kwargs={'num_steps':10})
    outputs=[np.asarray(policy.infer(obs,noise=noise)['actions']) for _ in range(3)]
    result={'tag':args.tag,'input_bundle_sha256':sha(bundle),'checkpoint':run['final_checkpoint'],
            'max_abs_delta_vs_archived_current_server':float(np.max(np.abs(outputs[0]-expected))),
            'max_abs_delta_within_fresh_process':max(float(np.max(np.abs(v-outputs[0]))) for v in outputs),
            'first_action':outputs[0][0].tolist(),'sampling_steps':10}
    write_json(ROOT/'artifacts'/f'{args.tag}.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
