import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import baseline_watchdog as w
from runtime import atomic_json, sha256


class CampaignTests(unittest.TestCase):
    def test_coverage_and_dependencies(self):
        jobs = w.make_jobs()
        self.assertEqual(len({j['key'] for j in jobs}), len(jobs))
        scored = [j for j in jobs if j['stage'] == 'train']
        self.assertEqual(len(scored), 15)
        self.assertEqual({(j['task'], j['method']) for j in scored},
                         {(t, m) for t in w.TASKS for m in w.METHODS})
        for j in scored:
            self.assertEqual(set(j['needs']), {'collect_' + j['task'], 'smoke_' + j['method']})

    def fixture(self, root, method):
        out = root / method
        out.mkdir()
        checkpoint = out / 'checkpoint'
        checkpoint.write_text(method)
        identity = {'task': w.TASKS[0], 'method': method, 'checkpoint_sha256': sha256(checkpoint),
                    'dataset_episodes': 50, 'split_sha256': 'fixture_split',
                    'dataset_manifest_sha256': 'fixture_dataset'}
        atomic_json(out / 'trained.json', {'checkpoint': str(checkpoint), 'dataset_episodes': 50,
            'training_episodes': 40, 'split_sha256': 'fixture_split',
            'dataset_manifest_sha256': 'fixture_dataset'})
        atomic_json(out / 'complete.json', {'episodes': 20, 'successes': 7, 'success_rate': .35,
                                          'identity': identity})
        for i in range(20):
            atomic_json(out / 'final_rollouts' / f'episode_{i:02d}.json', {
                'identity': identity, 'episode': i, 'seed': 2026101900 + i, 'steps': 200,
                'success': i < 7, 'initial_state_sha256': 'same-state-' + str(i)})
        return {'stage': 'train', 'status': 'complete', 'out': str(out),
                'task': w.TASKS[0], 'method': method}

    def test_table_requires_exact_twenty_records(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(w, 'ROOT', Path(tmp)):
            job = self.fixture(Path(tmp), 'DP')
            self.assertEqual(w.aggregate([job]), 1)
            table = (Path(tmp) / 'report/results.md').read_text()
            self.assertIn('7/20（35%）', table)
            self.assertIn('待完成', table)
            (Path(job['out']) / 'final_rollouts/episode_19.json').unlink()
            with self.assertRaises(FileNotFoundError):
                w.aggregate([job])

    def test_cross_method_initial_state_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(w, 'ROOT', Path(tmp)):
            jobs = [self.fixture(Path(tmp), method) for method in ['DP', 'ACT']]
            p = Path(jobs[1]['out']) / 'final_rollouts/episode_00.json'
            data = json.loads(p.read_text())
            data['initial_state_sha256'] = 'different-state'
            atomic_json(p, data)
            with self.assertRaisesRegex(AssertionError, 'Initial state mismatch'):
                w.aggregate(jobs)

    def test_checkpoint_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(w, 'ROOT', Path(tmp)):
            job = self.fixture(Path(tmp), 'DP')
            (Path(job['out']) / 'checkpoint').write_text('changed')
            with self.assertRaises(AssertionError):
                w.aggregate([job])

    def test_gpu_query(self):
        with patch.object(w.subprocess, 'check_output', return_value='0, 50000\n1, 12000\n'):
            self.assertEqual(w.gpu_memory(), {0: 50000, 1: 12000})

    def test_alert_preserves_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.jsonl'
            w.append(path, {'time': 123, 'reason': 'fixture'})
            self.assertEqual(json.loads(path.read_text())['time'], 123)

    def test_old_ten_demo_results_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(w, 'ROOT', Path(tmp)):
            job = self.fixture(Path(tmp), 'ACT')
            p = Path(job['out']) / 'trained.json'
            value = json.loads(p.read_text())
            value['dataset_episodes'] = 10
            atomic_json(p, value)
            with self.assertRaises(AssertionError):
                w.aggregate([job])

    def test_methods_are_scheduled_without_act_starvation(self):
        jobs = w.make_jobs()
        for j in jobs:
            if j['stage'] != 'train':
                j['status'] = 'complete'
        active = []
        methods = []
        for _ in range(4):
            chosen = w.ready_jobs(jobs, active)[0]
            chosen['status'] = 'running'
            active.append(chosen)
            methods.append(chosen['method'])
        self.assertEqual(methods, ['ACT', 'SimpleDP3', 'DP', 'ACT'])
        active[0]['status'] = 'complete'
        active = active[1:]
        self.assertEqual(w.ready_jobs(jobs, active)[0]['method'], 'ACT')

    def test_shared_split_exactly_forty_and_ten(self):
        import runtime
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime, 'ROOT', Path(tmp)):
            ids = np.random.RandomState(1).permutation(50).tolist()
            atomic_json(Path(tmp) / 'manifests/dataset_split.json', {
                'episodes': 50, 'train_episode_ids': ids[:40], 'validation_episode_ids': ids[40:]})
            mask = runtime.shared_split(50)
            self.assertEqual(int(mask.sum()), 40)
            self.assertEqual(set(np.flatnonzero(mask)), set(ids[:40]))
            with self.assertRaises(AssertionError):
                runtime.shared_split(10)

    def test_admission_uses_measured_small_model_budgets(self):
        # Real shared-GPU free-memory snapshot: small models fit with 4 GiB spare;
        # DP must continue waiting. Unallocated startup peaks remain reserved.
        free = {0: 8373, 1: 8856}
        self.assertEqual(w.eligible_gpus(free, {0: 0, 1: 0}, [], w.BUDGET['SimpleDP3']), [0, 1])
        self.assertEqual(w.eligible_gpus(free, {0: 0, 1: 0}, [], w.BUDGET['DP']), [])
        self.assertEqual(w.eligible_gpus(free, {0: 2000, 1: 0}, [], w.BUDGET['ACT']), [1])
        self.assertEqual(w.eligible_gpus({0: 7000}, {0: 0}, [], w.BUDGET['ACT']), [])
        self.assertEqual(w.eligible_gpus({0: 80000}, {0: 0}, [{'gpu':0},{'gpu':0}], w.BUDGET['ACT']), [])


if __name__ == '__main__':
    unittest.main()
