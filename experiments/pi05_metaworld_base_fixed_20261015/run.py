"""Rerun with fixed physical semantics, identical task identities, frozen base weights."""
import argparse
from collections import deque
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('OPENPI_DATA_HOME','/home/zhsu/jyluo/myopenpi/dataset/ckpt')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
PHASH=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
RESULT_GROUP='results_droid_units'


def save(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def status(phase,**extra):
    record=dict(time=time.time(),pid=os.getpid(),phase=phase,**extra)
    save(ROOT/'logs/status.json',record);print(json.dumps(record),flush=True)


def summarize():
    rows=[];all_records=[]
    for task in P['tasks']:
        records=[json.loads(p.read_text()) for p in sorted((ROOT/RESULT_GROUP/task).glob('episode_*.json'))]
        assert all(r['protocol_sha256']==PHASH for r in records)
        valid=[r for r in records if r['status']=='valid']
        successes=sum(r['success'] for r in valid)
        rows.append(dict(task=task,completed=len(records),valid=len(valid),successes=successes,
                         success_rate=successes/len(valid) if valid else None))
        all_records.extend(records)
    summary=dict(tasks=rows,completed=len(all_records),planned=100,valid=sum(r['valid'] for r in rows),
                 successes=sum(r['successes'] for r in rows),protocol_sha256=PHASH)
    summary['complete']=summary['completed']==100
    summary['overall_success_rate']=summary['successes']/summary['valid'] if summary['valid'] else None
    save(ROOT/RESULT_GROUP/'summary.json',summary)
    with (ROOT/RESULT_GROUP/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fields=['task','episode','seed','task_index','status','success','first_success_step','steps',
            'elapsed_seconds','policy_inferences','cartesian_clipped_components','state_ood_fraction','video']
    with (ROOT/RESULT_GROUP/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(all_records)
    lines=['# 基础模型动作接口修复后重跑','','| 任务 | 有效轮次 | 成功 | 成功率 |','|---|---:|---:|---:|']
    for row in rows:
        rate=f"{row['success_rate']:.1%}" if row['success_rate'] is not None else '待评测'
        lines.append(f"| {row['task']} | {row['valid']}/20 | {row['successes']} | {rate} |")
    lines += ['',f"已完成{summary['completed']}/100，有效{summary['valid']}，成功{summary['successes']}。",
              '权重仍为原始pi05_base，没有微调。使用DROID式七关节速度＋第八维夹爪、官方分位数归一化、Sawyer运动学转换。',
              '这修复了量纲/索引/预处理错误，但DROID统计与关节索引移植到Sawyer仍是未经训练验证的跨机器人代理。',
              '与上轮使用相同五任务、20个task身份、seed42–61、每轮500个环境控制步；首次成功立即结束。',
              '旧结果保留于../pi05_metaworld_base_20261015/results/report.md。']
    (ROOT/RESULT_GROUP/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def episode(env,tasks,indices,name,index,policy,bridge,smoke=False,attempt=1):
    import imageio.v2 as imageio
    import numpy as np
    chosen=tasks[indices[index]];seed=P['episode_seeds'][index]
    env.set_task(chosen);obs,_=env.reset(seed=seed);policy.reset(seed)
    manifest=json.loads((Path(P['previous_run'])/'manifests/episodes.json').read_text())
    expected=next(r for r in manifest if r['task']==name and r['episode']==index)
    task_hash=hashlib.sha256(chosen.data).hexdigest()
    assert expected['task_data_sha256']==task_hash and expected['task_index']==indices[index]
    group='smoke_droid_units' if smoke else RESULT_GROUP
    directory=ROOT/group/name;directory.mkdir(parents=True,exist_ok=True)
    video=ROOT/'videos'/group/name/f'ep{index:02d}_attempt{attempt}.mp4'
    video.parent.mkdir(parents=True,exist_ok=True)
    raw_plans=[];physical_plans=[];states=[];physical_states=[];actions=[];physical_commands=[]
    jacp=[];jacr=[];projected=[];preclip=[];successes=[];rewards=[];command_ids=[];latencies=[]
    eef_positions=[]
    queue=deque();active=None;next_tick=0.;command_id=-1;start=time.monotonic()
    success=False;clipped=0
    try:
        with imageio.get_writer(video,fps=10,codec='libx264',quality=6) as writer:
            for step in range(P['max_policy_steps']):
                now=step*env.dt
                if active is None or now+1e-9>=next_tick:
                    if not queue:
                        image=bridge.render(env)
                        norm_state=bridge.encode_state(env,obs)
                        status('rollout',task=name,episode=index,step=step)
                        t0=time.monotonic();raw=policy.infer(image,norm_state,P['prompts'][name])
                        latencies.append(time.monotonic()-t0)
                        physical=bridge.unnormalize(raw)
                        raw_plans.append(raw);physical_plans.append(physical)
                        queue.extend(physical[:P['replan_steps']])
                    active=queue.popleft();command_id+=1;next_tick+=1/P['policy_hz']
                if step%5==0:writer.append_data(bridge.render(env))
                state=bridge.physical_state(env)
                eef_positions.append(np.asarray(obs[:3]).copy())
                norm_state=bridge.encode_state(env,obs)
                jp,jr=bridge.jacobians(env)
                action,detail=bridge.decode_action(env,active)
                obs,reward,terminated,truncated,info=env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise FloatingPointError('nonfinite_environment')
                states.append(norm_state);physical_states.append(state);actions.append(action)
                physical_commands.append(active.copy());jacp.append(jp);jacr.append(jr)
                projected.append(detail['projected_velocity']);preclip.append(detail['preclip_action'])
                clipped+=detail['clipped_components'];command_ids.append(command_id)
                rewards.append(float(reward));successes.append(bool(info['success']))
                if info['success']:
                    success=True;writer.append_data(bridge.render(env));break
                if terminated or truncated:break
        trajectory=directory/f'episode_{index:02d}_attempt{attempt}.npz'
        np.savez_compressed(trajectory,states=states,physical_states=physical_states,actions=actions,
                            physical_commands=physical_commands,jacp=jacp,jacr=jacr,
                            projected_velocities=projected,preclip_actions=preclip,raw_plans=raw_plans,
                            physical_plans=physical_plans,successes=successes,rewards=rewards,
                            command_ids=command_ids,inference_seconds=latencies,eef_positions=eef_positions)
        result=dict(task=name,episode=index,seed=seed,task_index=indices[index],task_data_sha256=task_hash,
                    status='valid',success=success,first_success_step=len(actions) if success else None,
                    steps=len(actions),elapsed_seconds=time.monotonic()-start,policy_inferences=len(raw_plans),
                    cartesian_clipped_components=clipped,state_ood_fraction=float(np.mean(np.abs(states)>1)),
                    env_dt=env.dt,action_scale=env.action_scale,video=str(video),trajectory=str(trajectory),
                    protocol_sha256=PHASH,attempt=attempt)
        save(directory/f'episode_{index:02d}.json',result)
        status('episode_complete',**result)
        return result
    except Exception:
        save(ROOT/'logs'/name/f'ep{index:02d}_attempt{attempt}_error.json',
             dict(time=time.time(),task=name,episode=index,attempt=attempt,error=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (ROOT/'diagnostics/interface_validation.json').is_file():
        raise RuntimeError('Run check_interface.py successfully before loading the model')
    stamp=json.loads((ROOT/'diagnostics/interface_validation_stamp.json').read_text())
    assert stamp['passed'] and stamp['protocol_sha256']==PHASH,'Interface validation must match current protocol'
    from adapter import JointBridge,FrozenBasePolicy
    bridge=JointBridge(P);status('loading_base_model');policy=FrozenBasePolicy(P);status('model_loaded')
    for task in P['tasks']:
        env,tasks,indices=bridge.make_env(task)
        try:
            for i in range(1 if args.smoke else P['trials_per_task']):
                if not args.smoke and (ROOT/RESULT_GROUP/task/f'episode_{i:02d}.json').exists():continue
                for attempt in (1,2):
                    try:
                        episode(env,tasks,indices,task,i,policy,bridge,args.smoke,attempt)
                        break
                    except (ValueError,AssertionError,FloatingPointError):raise
                    except Exception:
                        if attempt==2:raise
                if not args.smoke:summarize()
        finally:env.close()
    status('smoke_complete' if args.smoke else 'complete',summary=None if args.smoke else summarize())


if __name__=='__main__':
    try:main()
    except Exception:status('error',error=traceback.format_exc());raise
