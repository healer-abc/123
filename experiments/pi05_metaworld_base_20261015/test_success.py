"""Verify early success and timeout accounting without exercising model weights."""
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run


class Writer:
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def append_data(self, *_): pass


class Env:
    def __init__(self, success_at):
        self.success_at = success_at
        self.steps = 0

    def set_task(self, *_): pass
    def reset(self, **_):
        self.steps = 0
        return np.zeros(4), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(4), 0., False, False, {'success': self.steps == self.success_at}


class Policy:
    def reset(self, *_): pass
    def infer(self, *_): return np.zeros((10, 32))


class Adapter:
    def render(self, *_): return np.zeros((16,16,3), dtype=np.uint8)
    def encode_state(self, *_): return np.zeros(32)
    def decode_actions(self, _, raw): return raw[:, :4], 0


class SuccessTests(unittest.TestCase):
    def run_episode(self, success_at):
        with TemporaryDirectory() as directory, patch.object(run, 'ROOT', Path(directory)), \
             patch.object(run, 'status'), patch('imageio.v2.get_writer', return_value=Writer()):
            chosen = SimpleNamespace(data=pickle.dumps({'rand_vec': np.zeros(3)}))
            env = Env(success_at)
            record = run.execute_episode(env, [chosen], [0], 'assembly-v2', 0,
                                         Policy(), Adapter(), 1, smoke=True)
            self.assertEqual(record['status'], 'valid')
            self.assertEqual(record['steps'], env.steps)
            saved = json.loads((Path(directory)/'smoke/assembly-v2/episode_00.json').read_text())
            self.assertEqual(saved['success'], record['success'])
            return record

    def test_first_success_stops_immediately(self):
        record = self.run_episode(2)
        self.assertTrue(record['success'])
        self.assertEqual(record['steps'], 2)
        self.assertEqual(record['first_success_step'], 2)

    def test_no_success_runs_full_budget(self):
        record = self.run_episode(None)
        self.assertFalse(record['success'])
        self.assertEqual(record['steps'], run.PROTOCOL['max_policy_steps'])
        self.assertIsNone(record['first_success_step'])


if __name__ == '__main__':
    unittest.main()
