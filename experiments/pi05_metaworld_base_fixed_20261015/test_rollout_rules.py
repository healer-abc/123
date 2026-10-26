"""Exercise success stopping and policy/env clock accounting without model inference."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run


class Writer:
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def append_data(self,*_):pass


class Env:
    dt=.0125
    action_scale=.01
    def __init__(self,success_at=None):self.success_at=success_at;self.steps=0
    def set_task(self,*_):pass
    def reset(self,**_):self.steps=0;return np.zeros(4),{}
    def step(self,_):
        self.steps+=1
        return np.zeros(4),0.,False,False,{'success':self.steps==self.success_at}


class Policy:
    def reset(self,*_):pass
    def infer(self,*_):return np.zeros((10,32))


class Bridge:
    def render(self,*_):return np.zeros((16,16,3),dtype=np.uint8)
    def physical_state(self,*_):return np.zeros(8)
    def encode_state(self,*_):return np.zeros(8)
    def unnormalize(self,raw):return raw[:,:8]
    def jacobians(self,*_):return np.eye(3,7),np.zeros((3,7))
    def decode_action(self,*_):
        return np.zeros(4),dict(projected_velocity=np.zeros(7),preclip_action=np.zeros(4),clipped_components=0)


class RolloutRules(unittest.TestCase):
    def execute(self,success_at):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            previous=root/'previous';(previous/'manifests').mkdir(parents=True)
            task=SimpleNamespace(data=b'unit-test-task')
            identity=dict(task='assembly-v2',episode=0,seed=42,task_index=0,
                          task_data_sha256=hashlib.sha256(task.data).hexdigest())
            (previous/'manifests/episodes.json').write_text(json.dumps([identity]))
            env=Env(success_at)
            with patch.object(run,'ROOT',root),patch.dict(run.P,{'previous_run':str(previous)}), \
                 patch.object(run,'status'),patch('imageio.v2.get_writer',return_value=Writer()):
                result=run.episode(env,[task],[0],'assembly-v2',0,Policy(),Bridge(),smoke=True)
                with np.load(result['trajectory']) as trajectory:
                    command_ids=trajectory['command_ids'].copy()
            return result,command_ids,env.steps

    def test_first_success_stops_before_remaining_chunk(self):
        result,_,steps=self.execute(2)
        self.assertTrue(result['success']);self.assertEqual(steps,2)
        self.assertEqual(result['first_success_step'],2)
        self.assertEqual(result['policy_inferences'],1)

    def test_timeout_is_valid_failure_with_fifteen_hz_commands(self):
        result,ids,steps=self.execute(None)
        self.assertFalse(result['success']);self.assertEqual(steps,500)
        self.assertEqual(result['status'],'valid');self.assertIsNone(result['first_success_step'])
        self.assertEqual(result['policy_inferences'],19)
        np.testing.assert_array_equal(ids,np.floor(np.arange(500)*.0125*15+1e-8).astype(int))


if __name__=='__main__':unittest.main()
