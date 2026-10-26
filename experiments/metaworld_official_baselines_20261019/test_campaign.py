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
        identity = {'task': w.TASKS[0], 'method': method, 'checkpoint_sha256': sha256(checkpoint)}
        atomic_json(out / 'trained.json', {'checkpoint': str(checkpoint)})
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


if __name__ == '__main__':
    unittest.main()
