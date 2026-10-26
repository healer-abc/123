"""Privileged scripted-policy environment check, never part of base-model scores."""
import hashlib
import json
from pathlib import Path

from adapter import JointBridge
from metaworld import policies
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    p=json.loads((ROOT/'protocol.json').read_text())
    classes={
        'assembly-v2':policies.SawyerAssemblyV2Policy,
        'peg-insert-side-v2':policies.SawyerPegInsertionSideV2Policy,
        'disassemble-v2':policies.SawyerDisassembleV2Policy,
        'peg-unplug-side-v2':policies.SawyerPegUnplugSideV2Policy,
        'stick-pull-v2':policies.SawyerStickPullV2Policy,
    }
    bridge=JointBridge(p);records=[]
    for task,cls in classes.items():
        env,tasks,indices=bridge.make_env(task)
        try:
            env.set_task(tasks[indices[0]])
            # Official scripted controllers require privileged goal observation.
            # This switch exists ONLY in this diagnostic process, not run.py.
            env._partially_observable=False
            obs,_=env.reset(seed=p['episode_seeds'][0]);policy=cls();success=False
            for step in range(p['max_policy_steps']):
                action=np.clip(policy.get_action(obs),env.action_space.low,env.action_space.high)
                obs,reward,terminated,truncated,info=env.step(action)
                if info['success']:
                    success=True;break
                if terminated or truncated:break
            records.append(dict(task=task,seed=p['episode_seeds'][0],task_index=indices[0],
                                task_data_sha256=hashlib.sha256(tasks[indices[0]].data).hexdigest(),
                                success=success,steps=step+1))
        finally:env.close()
    result=dict(purpose='Privileged environment diagnostic ONLY; excluded from all base-model scores',
                controller='MetaWorld bundled scripted policies with visible goals',records=records)
    (ROOT/'diagnostics/environment_oracles.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
