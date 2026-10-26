"""Bounded follow-up: distinguish fixed-input determinism from environment-constructor variation."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT))
import eval_manifest as original
from libero.libero import benchmark
from openpi_client import websocket_client_policy


def main():
    with original.MANIFEST.open() as file:
        row=next(r for r in csv.DictReader(file) if r['phase']=='development' and r['suite']=='libero_spatial' and r['task_id']=='3' and r['init_state_index']=='1')
    client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',8125)
    metadata=client.get_server_metadata()
    assert metadata['config']=='pi05_libero_lora'
    suite=benchmark.get_benchmark_dict()[row['suite']]()
    cases=[]
    for constructor_seed in (11,22):
        np.random.seed(constructor_seed)
        env=original.make_env(suite.get_task(3),int(row['env_seed']))
        try:
            for reset_index in range(2 if constructor_seed==11 else 1):
                np.random.seed(int(row['env_seed']))
                env.seed(int(row['env_seed']))
                env.reset()
                state=np.asarray(suite.get_task_init_states(3))[1]
                obs=env.set_init_state(state)
                for _ in range(10):
                    obs,_,_,_=env.step(original.DUMMY_ACTION)
                image=original.prepare_image(obs['agentview_image'])
                wrist=original.prepare_image(obs['robot0_eye_in_hand_image'])
                robot=np.concatenate((obs['robot0_eef_pos'],original.axis_angle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))
                noise=np.random.default_rng(int(row['policy_seed'])).standard_normal((10,32),dtype=np.float32)
                payload={'observation/image':image,'observation/wrist_image':wrist,'observation/state':robot,'prompt':row['task_description'],'_policy_noise':noise}
                outputs=[np.asarray(client.infer(payload)['actions']) for _ in range(3)]
                path=ROOT/'artifacts'/f'first_input_seed_{constructor_seed}_reset_{reset_index}.npz'
                np.savez_compressed(path,image=image,wrist=wrist,state=robot,noise=noise,actions=outputs[0])
                cases.append({'constructor_seed':constructor_seed,'reset_index':reset_index,
                              'base_image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                              'wrist_image_sha256':hashlib.sha256(wrist.tobytes()).hexdigest(),
                              'robot_state':robot.tolist(),'max_abs_delta_fixed_input_repeats':max(float(np.max(np.abs(a-outputs[0]))) for a in outputs),
                              'first_action':outputs[0][0].tolist(),'bundle':str(path)})
        finally:
            env.close()
    old=json.loads((PROJECT/'evaluations/E1-s42/development/episodes/libero_spatial/task_03/state_01.json').read_text())
    with np.load(old['trajectory']) as data:
        old_first=data['actions'][0]
    with np.load(ROOT/'rollouts/E1-s42/episodes/libero_spatial/task_03/state_01.npz') as data:
        replay_first=data['actions'][0]
    original.atomic_json(ROOT/'artifacts/first_input_repeatability.json',{
        'scope':'single deliberately selected unstable exposed identity; constructor/reset and in-process fixed-input checks only',
        'server_metadata':metadata,'cases':cases,
        'historical_first_action':old_first.tolist(),'current_replay_first_action':replay_first.tolist(),
        'historical_to_replay_first_action_max_abs_delta':float(np.max(np.abs(old_first-replay_first))),
        'note':'historical raw model inputs were not saved; matching current inputs cannot retrospectively prove their exact equality to historical inputs'})
    print(json.dumps(cases,indent=2))


if __name__=='__main__':
    main()
