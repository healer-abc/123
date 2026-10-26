"""Frozen RLinf π0.5 MetaWorld SFT evaluation on the five paired v2 tasks."""
import argparse
from collections import deque
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import traceback

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/zhsu/jyluo/experiments/pi0_metaworld_sft_hf_20261016/rollout.py')
spec=importlib.util.spec_from_file_location('pi0_reference_rollout',SOURCE)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
P=json.loads((ROOT/'protocol.json').read_text())
PHASH=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
CHECKPOINT=ROOT/'checkpoints/RLinf-Pi05-MetaWorld-SFT'
reference.ROOT=ROOT;reference.PROTOCOL=P;reference.PROTOCOL_HASH=PHASH
reference.CHECKPOINT=CHECKPOINT


def save(path,value):
    reference.atomic_json(path,value)


def build_policy():
    import safetensors.torch
    import torch
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.policy import Policy
    from openpi.shared import normalize
    from openpi.training.config import ModelTransformFactory
    from metaworld_policy import MetaworldInputs,MetaworldOutputs

    verified=json.loads((CHECKPOINT/'verified.json').read_text())
    if verified['sha256']!=P['checkpoint_sha256'] or verified['revision']!=P['checkpoint_revision']:
        raise ValueError('Wrong checkpoint or unverified revision')
    config=Pi0Config(pi05=True,action_horizon=P['action_horizon'],discrete_state_input=False)
    model=PI0Pytorch(config)
    missing,unexpected=safetensors.torch.load_model(model,str(CHECKPOINT/'model.safetensors'),strict=False)
    load=dict(missing=sorted(missing),unexpected=sorted(unexpected),config=str(config))
    save(ROOT/'logs/weight_load.json',load)
    if missing or unexpected:raise ValueError(f'Checkpoint/model mismatch: {load}')
    model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    model.requires_grad_(False);model.eval()
    stats=normalize.load(CHECKPOINT/'lerobot/metaworld_mt50')
    model_transforms=ModelTransformFactory()(config)
    return Policy(model,
                  transforms=[MetaworldInputs(config.model_type),
                              transforms.Normalize(stats,use_quantiles=True),
                              *model_transforms.inputs],
                  output_transforms=[*model_transforms.outputs,
                                     transforms.Unnormalize(stats,use_quantiles=True),
                                     MetaworldOutputs()],
                  sample_kwargs={'num_steps':P['denoising_steps']},
                  is_pytorch=True,pytorch_device='cuda:0')


def aggregate(group='results'):
    rows=[];all_records=[]
    for task in P['tasks']:
        files=sorted((ROOT/group/task).glob('episode_*.json'))
        records=[json.loads(f.read_text()) for f in files]
        assert all(r['protocol_sha256']==PHASH for r in records)
        successes=sum(r['success'] for r in records)
        rows.append(dict(task=task,completed=len(records),successes=successes,
                         success_rate=successes/len(records) if records else None))
        all_records+=records
    complete=len(all_records)==P['trials_per_task']*len(P['tasks'])
    summary=dict(complete=complete,completed=len(all_records),planned=100,
                 successes=sum(r['success'] for r in all_records),tasks=rows,
                 protocol_sha256=PHASH,checkpoint_sha256=P['checkpoint_sha256'])
    save(ROOT/group/'summary.json',summary)
    with (ROOT/group/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (ROOT/group/'episodes.csv').open('w',newline='') as file:
        fields=['task','episode','seed','task_index','success','first_success_step','steps',
                'return','policy_inferences','elapsed_seconds','video','trajectory']
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(all_records)
    lines=['# MetaWorld SFT π0.5 五任务评测','','| 任务 | 成功/20 | 成功率 |','|---|---:|---:|']
    for row in rows:
        rate=f"{row['success_rate']:.1%}" if row['success_rate'] is not None else '待评测'
        lines.append(f"| {row['task']} | {row['successes']}/{row['completed']} | {rate} |")
    lines+=['',f"完成{len(all_records)}/100，成功{summary['successes']}。",
            '权重：RLinf/RLinf-Pi05-MetaWorld-SFT，冻结推理，使用发布者的MetaWorld归一化统计和原生4D动作。',
            '任务与seed匹配未微调π0.5基线；前者是MetaWorld任务动作，后者使用DROID/Sawyer代理接口，差别不只有权重。',
            '该模型原训练/参考评测基于MetaWorld v3；这里评测用户指定的v2任务。']
    (ROOT/group/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def episode(env,tasks,indices,task,index,policy,group):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    seed=P['episode_seeds'][index]
    chosen=tasks[indices[index]]
    obs,_=reference.reset_episode(env,chosen,seed)
    expected=json.loads((Path(P['comparison_run']).parent.parent/'pi05_metaworld_base_20261015/manifests/episodes.json').read_text())
    match=next(r for r in expected if r['task']==task and r['episode']==index)
    task_hash=hashlib.sha256(chosen.data).hexdigest()
    assert match['seed']==seed and match['task_index']==indices[index] and match['task_data_sha256']==task_hash
    base_record=Path(P['comparison_run'])/task/f'episode_{index:02d}_attempt1.npz'
    if base_record.is_file():
        with np.load(base_record) as prior:
            if not np.allclose(obs[:3],prior['eef_positions'][0],atol=1e-5):
                raise ValueError('Initial hand position differs from base run')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    prompt=json.loads((ROOT/'../pi0_metaworld_sft_hf_20261016/sources/rlinf_metaworld_config.json').read_text())['TASK_DESCRIPTIONS'][task.replace('-v2','-v3')]
    path=ROOT/group/task;path.mkdir(parents=True,exist_ok=True)
    video=ROOT/'videos'/group/task/f'episode_{index:02d}.mp4';video.parent.mkdir(parents=True,exist_ok=True)
    states=[];actions=[];rewards=[];successes=[];raw_actions=[];latencies=[];images=[]
    queue=deque();start=time.monotonic();success=False
    with imageio.get_writer(video,fps=10,codec='libx264',quality=6) as writer:
        for step in range(P['max_policy_steps']):
            image=reference.render_image(env)
            if step%5==0:writer.append_data(image)
            state=np.asarray(obs[:4],dtype=np.float32).copy()
            states.append(state)
            if not queue:
                batch={'observation/image':image,'observation/state':state,'prompt':prompt}
                t0=time.monotonic()
                with torch.inference_mode():result=policy.infer(batch)
                latencies.append(time.monotonic()-t0)
                plan=np.asarray(result['actions'],dtype=np.float32)
                if plan.shape!=(P['action_horizon'],4) or not np.isfinite(plan).all():
                    raise ValueError(f'Invalid action plan: {plan.shape}')
                queue.extend(plan[:P['action_chunk']])
                images.append(hashlib.sha256(image.tobytes()).hexdigest())
            raw=queue.popleft();action=np.clip(raw,env.action_space.low,env.action_space.high)
            obs,reward,terminated,truncated,info=env.step(action)
            reference.check_simulation(env)
            if not np.isfinite(obs).all() or not np.isfinite(reward):raise FloatingPointError('Nonfinite environment')
            actions.append(action);raw_actions.append(raw);rewards.append(float(reward))
            success=bool(info['success']);successes.append(success)
            if success:
                writer.append_data(reference.render_image(env));break
            if terminated or truncated:break
    trajectory=path/f'episode_{index:02d}.npz'
    np.savez_compressed(trajectory,states=states,actions=actions,raw_actions=raw_actions,
                        rewards=rewards,successes=successes,inference_seconds=latencies,
                        inference_image_sha256=images,initial_observation=states[0])
    record=dict(task=task,episode=index,seed=seed,task_index=indices[index],task_data_sha256=task_hash,
                status='valid',success=success,first_success_step=len(actions) if success else None,
                steps=len(actions),**{'return':sum(rewards)},policy_inferences=len(latencies),
                elapsed_seconds=time.monotonic()-start,prompt=prompt,
                video=str(video),trajectory=str(trajectory),
                video_sha256=reference.file_sha256(video),trajectory_sha256=reference.file_sha256(trajectory),
                checkpoint_sha256=P['checkpoint_sha256'],protocol_sha256=PHASH)
    save(path/f'episode_{index:02d}.json',record)
    print(json.dumps({'event':'episode_complete',**record}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--tasks',nargs='+',choices=P['tasks'])
    args=parser.parse_args()
    group='smoke' if args.smoke else 'results'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    print(json.dumps({'event':'loading_policy','group':group,'time':time.time()}),flush=True)
    policy=build_policy()
    selected=args.tasks or P['tasks']
    for task in selected:
        env,tasks,indices=reference.make_env(task)
        try:
            for i in range(1 if args.smoke else P['trials_per_task']):
                if (ROOT/group/task/f'episode_{i:02d}.json').is_file():continue
                episode(env,tasks,indices,task,i,policy,group)
                if not args.smoke and args.tasks is None:aggregate()
        finally:env.close()
    summary=aggregate(group) if args.tasks is None else None
    print(json.dumps({'event':'complete','group':group,'tasks':selected,'summary':summary}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        print(traceback.format_exc(),flush=True)
        raise
