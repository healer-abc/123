"""Paired SimpleDP3 500-epoch DDIM10 sensitivity evaluation, copied from frozen baseline evaluator."""
import collections
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from runtime import ROOT, atomic_json, install_render_fix, sha256


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def observation(obs, method):
    if method == 'SimpleDP3':
        return {'point_cloud':obs['point_cloud'].astype(np.float32), 'agent_pos':obs['agent_pos'].astype(np.float32)}
    return {'image':obs['image'].astype(np.float32)/255., 'agent_pos':obs['agent_pos'].astype(np.float32)}


def evaluate(policy,task,method,output_dir,*,final=False,checkpoint=None,stats=None,round_id=0):
    install_render_fix()
    from diffusion_policy_3d.env import MetaWorldEnv
    p=json.loads((ROOT/'protocol.json').read_text())
    seeds=p['final_episode_seeds'] if final else p['development_episode_seeds']
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    cfg=({'chunk_size':10,'action_horizon':1,'temporal_agg':True,'aggregation_k':0.01} if method=='ACT'
         else {'horizon':16,'n_obs_steps':2,'action_horizon':8,'num_inference_steps':10,'scheduler':'DDIM','prediction_type':'sample'})
    identity={'task':task,'method':method,'config':cfg,'protocol':sha256(ROOT/'protocol.json'),
        'dataset_episodes':50,'dataset_manifest_sha256':sha256(ROOT/'datasets'/(task+'.json')),
        'split_sha256':sha256(ROOT/'manifests/dataset_split.json'),'training_seed':0,
        'stage':'ablation_epoch500_ddim10_20' if final else 'development','round':round_id,
        'ablation_plan_sha256':sha256(Path(__file__).resolve().parent/'plan.json'),
        'checkpoint_completed_epochs':500 if final else None}
    if final:
        assert checkpoint and Path(checkpoint).is_file()
        identity['checkpoint_sha256']=sha256(checkpoint)
    records=[];env=None;policy.eval()
    assert method=='SimpleDP3' and policy.num_inference_steps==10
    if not hasattr(policy,'_ablation_step_calls'):
        policy._ablation_step_calls=0
        original_step=policy.noise_scheduler.step
        def counted_step(*args,**kw):
            policy._ablation_step_calls+=1
            return original_step(*args,**kw)
        policy.noise_scheduler.step=counted_step
    try:
        for i,seed in enumerate(seeds):
            ep=output_dir/f'episode_{i:02d}'
            if ep.with_suffix('.json').exists():
                r=json.loads(ep.with_suffix('.json').read_text())
                assert r['identity']==identity and r['seed']==seed
                assert ep.with_suffix('.npz').is_file()
                if final:assert ep.with_suffix('.mp4').is_file()
                records.append(r);continue
            seed_all(seed)
            if env is None:env=MetaWorldEnv(task,device='cuda:0',num_points=512,use_point_crop=True)
            env.env.seed(seed);obs=env.reset()
            initial=np.concatenate([env.env.sim.get_state().flatten(),obs['full_state'],np.asarray(env.env._target_pos).ravel()])
            initial_hash=hashlib.sha256(np.asarray(initial,dtype=np.float64).tobytes()).hexdigest()
            history=collections.deque([observation(obs,method)]*2,maxlen=2)
            if method!='ACT':policy.reset()
            all_time=torch.zeros((200,210,4),device='cuda') if method=='ACT' else None
            frames=[];actions=[];agent_pos=[];infos=[];rewards=[];chunks=[];query_steps=[]
            chunk=None;index=0
            sampler_calls_before=policy._ablation_step_calls
            with torch.inference_mode():
                for step in range(200):
                    agent_pos.append(np.asarray(obs['agent_pos']).copy())
                    if method=='ACT':
                        q=(obs['agent_pos']-stats['qpos_mean'])/stats['qpos_std']
                        pred=policy(torch.as_tensor(q,dtype=torch.float32,device='cuda')[None],
                            torch.as_tensor(obs['image'],dtype=torch.float32,device='cuda')[None,None]/255.)
                        assert pred.shape==(1,10,4) and bool(torch.isfinite(pred).all())
                        chunks.append(pred[0].cpu().numpy());query_steps.append(step)
                        all_time[[step],step:step+10]=pred
                        current=all_time[:,step]
                        current=current[torch.all(current!=0,axis=1)]
                        assert len(current)>0
                        weights=np.exp(-0.01*np.arange(len(current)));weights/=weights.sum()
                        raw=(current*torch.from_numpy(weights).cuda().unsqueeze(1)).sum(0)
                        action=raw.cpu().numpy()*stats['action_std']+stats['action_mean']
                    else:
                        if chunk is None or index==len(chunk):
                            batch={k:torch.as_tensor(np.stack([x[k] for x in history]),dtype=torch.float32,device='cuda')[None] for k in history[0]}
                            chunk=policy.predict_action(batch)['action'][0].cpu().numpy()
                            assert chunk.shape==(8,4)
                            chunks.append(chunk.copy());query_steps.append(step);index=0
                        action=chunk[index];index+=1
                    assert np.asarray(action).shape==(4,) and np.isfinite(action).all(),'Nonfinite model action'
                    actions.append(np.asarray(action).copy())
                    obs,reward,done,info=env.step(action)
                    infos.append({k:np.asarray(v).tolist() for k,v in info.items()});rewards.append(float(reward))
                    history.append(observation(obs,method))
                    if final and step%2==0:frames.append(obs['image'].transpose(1,2,0))
                    if (step+1)%10==0:
                        atomic_json(output_dir/'heartbeat.json',{'time':time.time(),'episode':i,'steps':step+1,'final':final})
                    if done:break
            assert len(actions)==200, 'Unexpected episode truncation'
            assert policy._ablation_step_calls-sampler_calls_before==250, 'Wrong DDIM step count'
            successful_steps=[j+1 for j,x in enumerate(infos) if bool(x['success'])]
            arr=np.asarray(actions)
            r={'identity':identity,'episode':i,'seed':seed,'steps':len(actions),
                'initial_state_sha256':initial_hash,'success':bool(successful_steps),
                'first_success_step':successful_steps[0] if successful_steps else None,
                'return':float(sum(rewards)),'infos':infos,'nonfinite_actions':int((~np.isfinite(arr)).sum()),
                'fraction_components_outside_env_bounds':float(((arr<env.action_space.low)|(arr>env.action_space.high)).mean()),
                'denoising_step_calls':policy._ablation_step_calls-sampler_calls_before,
                'num_inference_steps':policy.num_inference_steps,'trajectory':str(ep.with_suffix('.npz')),'video':str(ep.with_suffix('.mp4')) if final else None,'time':time.time()}
            np.savez_compressed(ep.with_suffix('.npz'),actions=arr,agent_pos=np.asarray(agent_pos),rewards=np.asarray(rewards),
                query_steps=np.asarray(query_steps),predicted_chunks=np.asarray(chunks),initial_state=initial)
            if final:
                import imageio.v2 as imageio
                imageio.mimsave(ep.with_suffix('.mp4'),frames,fps=10)
            atomic_json(ep.with_suffix('.json'),r)  # Commit only after all artifacts exist.
            records.append(r)
            print(json.dumps({'stage':identity['stage'],'task':task,'method':method,'episode':i,'seed':seed,'success':r['success'],'time':r['time']}),flush=True)
    finally:
        if env is not None:env.env.close()
    assert len(records)==len(seeds)
    result={'identity':identity,'episodes':len(records),'seeds':seeds,'successes':sum(x['success'] for x in records),
        'success_rate':sum(x['success'] for x in records)/len(records),
        'mean_return':float(np.mean([x['return'] for x in records])),
        'successful_episodes':[x['video'] for x in records if x['success'] and x['video']]}
    atomic_json(output_dir/'summary.json',result)
    return result


class RunnerMixin:
    def configure(self,output_dir,task_name,method,**kw):
        self.output_dir=output_dir;self.task_name=task_name;self.method=method;self.round_id=0;self.scores=[]

    def run(self,policy,**kw):
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        try:
            r=evaluate(policy,self.task_name,self.method,Path(self.output_dir)/'development_rollouts'/f'round_{self.round_id:04d}',round_id=self.round_id)
        finally:
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        self.round_id+=1;self.scores.append(r['success_rate'])
        return {'test_mean_score':r['success_rate'],'test/mean_score':r['success_rate'],'mean_success_rates':r['success_rate'],
            'mean_traj_rewards':r['mean_return'],'SR_test_L3':float(np.mean(sorted(self.scores)[-3:])),
            'SR_test_L5':float(np.mean(sorted(self.scores)[-5:]))}
