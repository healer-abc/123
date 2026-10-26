"""Read-only paired, fixed-noise flow-loss probe using the original training data pipeline."""
import argparse
import json
from pathlib import Path
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils
from openpi.training import config as train_config
from openpi.training import data_loader

from common import ROOT, PROJECT, SEED, write_json


def dimension_losses(self, rng, observation, actions):
    # Mirror stock Pi0.compute_loss without reducing action dimensions, and verify below.
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = model_lib.preprocess_observation(preprocess_rng, observation, train=False)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * .999 + .001
    time_expanded = time[...,None,None]
    x_t = time_expanded*noise + (1-time_expanded)*actions
    u_t = noise-actions
    prefix,prefix_mask,prefix_ar = self.embed_prefix(observation)
    suffix,suffix_mask,suffix_ar,cond = self.embed_suffix(observation,x_t,time)
    mask=jnp.concatenate([prefix_mask,suffix_mask],axis=1)
    ar=jnp.concatenate([prefix_ar,suffix_ar],axis=0)
    positions=jnp.cumsum(mask,axis=1)-1
    (_,out),_=self.PaliGemma.llm([prefix,suffix],mask=make_attn_mask(mask,ar),positions=positions,adarms_cond=[None,cond])
    pred=self.action_out_proj(out[:,-self.action_horizon:])
    square=jnp.square(pred-u_t)
    return {'stock_32':square.mean(axis=-1),'action_7':square[...,:7].mean(axis=-1),
            'motion_6':square[...,:6].mean(axis=-1),'gripper_1':square[...,6],
            'padding_25':square[...,7:].mean(axis=-1),'flow_time':time}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',choices=['E1-s42','E2-s42'],required=True)
    args=parser.parse_args()
    run=next(r for r in json.loads((PROJECT/'manifests/runs.json').read_text())['runs'] if r['run_id']==args.run_id)
    samples=json.loads((ROOT/'artifacts/samples.json').read_text())['samples']
    config=train_config.get_config(run['config'])
    dc=config.data.create(config.assets_dirs,config.model)
    dataset=data_loader.transform_dataset(data_loader.create_torch_dataset(dc,10,config.model),dc)
    entries=[]
    for s in samples:
        ids=pq.read_table(s['path'],columns=['index'],use_threads=False)['index']
        entries.append({kind:dataset[int(ids[s[kind+'_anchor']].as_py())] for kind in ('event','control')})
    print('Samples loaded; restoring checkpoint in original saved dtypes',flush=True)
    model=config.model.load(model_lib.restore_params(Path(run['final_checkpoint'])/'params'))
    model.eval()
    fn=nnx_utils.module_jit(types.MethodType(dimension_losses,model))
    stock=nnx_utils.module_jit(model.compute_loss)
    results=[]
    validation=None
    for start in range(0,len(entries),4):
        for rep in range(4):
            key=jax.random.fold_in(jax.random.key(SEED),start*4+rep)
            for kind in ('event','control'):
                batch=jax.tree.map(lambda *xs:np.stack(xs),*[e[kind] for e in entries[start:start+4]])
                observation=model_lib.Observation.from_dict(batch)
                actions=jnp.asarray(batch['actions'])
                values=jax.device_get(fn(key,observation,actions))
                if validation is None:
                    expected=np.asarray(stock(key,observation,actions))
                    validation=float(np.max(np.abs(expected-values['stock_32'])))
                    assert np.allclose(expected,values['stock_32'],atol=1e-5,rtol=1e-5),validation
                    print('Stock loss parity verified:',validation,flush=True)
                assert all(np.isfinite(v).all() for v in values.values())
                for j in range(len(entries[start:start+4])):
                    results.append({'sample_index':start+j,'kind':kind,'repeat':rep,
                                    'flow_time':float(values['flow_time'][j]),
                                    **{name:np.asarray(value[j],dtype=float).tolist() for name,value in values.items() if name!='flow_time'}})
        print(f'completed pairs {start+4}/{len(entries)}',flush=True)
        write_json(ROOT/'artifacts'/f'{args.run_id}_loss_progress.json',{'completed_pairs':start+4,'planned_pairs':len(entries)})
    output={'run_id':args.run_id,'checkpoint':run['final_checkpoint'],'original_saved_dtypes':True,
            'augmentation':False,'training_updates':0,'stock_loss_max_abs_difference':validation,
            'sampling':'48 event windows and same-demonstration motion-matched non-event controls; four common noise/time repeats; in-training demonstrations',
            'records':results}
    write_json(ROOT/'artifacts'/f'{args.run_id}_loss_probe.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    main()
