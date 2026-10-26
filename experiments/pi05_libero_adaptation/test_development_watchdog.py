"""CPU-only checks for scheduling, manifest identity and restart-safe aggregation."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("watchdog", DIRECTORY / "development_watchdog.py")
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class WatchdogTest(unittest.TestCase):
    def test_memory_headroom_and_startup_reservations(self):
        memory = {0: {"free_mib": 85000}, 1: {"free_mib": 74000}}
        self.assertEqual(watchdog.select_gpu(memory, [], 24576, 8192), 0)
        jobs = [{"gpu": 0, "status": "starting"}]
        self.assertEqual(watchdog.select_gpu(memory, jobs, 24576, 8192), 1)
        self.assertIsNone(watchdog.select_gpu({0: {"free_mib": 32000}}, [], 24576, 8192))
        self.assertIsNone(watchdog.select_gpu({0: {"free_mib": 50000}}, jobs, 24576, 8192))

    def test_shards_are_disjoint_and_cover_frozen_development(self):
        with watchdog.MANIFEST.open() as file:
            rows = [row for row in csv.DictReader(file) if row["phase"] == "development"]
        self.assertEqual(len(rows), 400)
        shards = [{(r["suite"], r["task_id"], r["init_state_index"]) for r in rows
                   if int(r["task_id"]) % 2 == i} for i in range(2)]
        self.assertEqual([len(s) for s in shards], [200, 200])
        self.assertFalse(shards[0] & shards[1])
        self.assertEqual(len(shards[0] | shards[1]), 400)

    def test_aggregate_rejects_different_checkpoint_and_seed(self):
        row = {"suite": "libero_spatial", "task_id": "0", "init_state_index": "0", "policy_seed": "200000"}
        job = {"run_id": "E1-s42", "checkpoint": "/fixed/checkpoint", "config": "test"}
        with tempfile.TemporaryDirectory() as directory, patch.object(watchdog, "ROOT", Path(directory)):
            summary = watchdog.summarize(job, [row])
            self.assertFalse(summary["complete"])
            path = Path(directory) / "evaluations/E1-s42/development/episodes/libero_spatial/task_00/state_00.json"
            path.parent.mkdir(parents=True)
            value = {**row, "task_id": 0, "init_state_index": 0, "status": "valid", "success": True,
                     "checkpoint": job["checkpoint"], "config": "test", "manifest_sha256": watchdog.EXPECTED_MANIFEST}
            path.write_text(json.dumps(value))
            summary = watchdog.summarize(job, [row])
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["successes"], 1)
            path.write_text(json.dumps({**value, "checkpoint": "/wrong/checkpoint"}))
            with self.assertRaises(ValueError):
                watchdog.summarize(job, [row])
            path.write_text(json.dumps({**value, "policy_seed": 42}))
            with self.assertRaises(ValueError):
                watchdog.summarize(job, [row])

    def test_model_faults_stop_but_infrastructure_can_retry(self):
        self.assertTrue(watchdog.fatal_error("FloatingPointError: non-finite policy action"))
        self.assertTrue(watchdog.fatal_error("RuntimeError: episode retry limit reached"))
        self.assertFalse(watchdog.fatal_error("ConnectionClosedError: peer reset"))
        self.assertFalse(watchdog.fatal_error("RESOURCE_EXHAUSTED: out of memory"))

    def test_infrastructure_retry_is_bounded_and_releases_owned_processes(self):
        job = {"run_id": "test", "launches": 1, "status": "running"}
        with patch.object(watchdog, "event") as events, patch.object(watchdog, "stop_job") as stop:
            watchdog.fail_job(job, "connection reset")
            stop.assert_called_once_with(job)
            self.assertEqual(job["status"], "queued")
            self.assertGreater(job["not_before"], watchdog.time.time())
            self.assertEqual(events.call_args.args[0], "retry")
            job["launches"] = 2
            watchdog.fail_job(job, "connection reset again")
            self.assertEqual(job["status"], "blocked")
            self.assertEqual(events.call_args.args[0], "blocked")

    def test_model_error_is_not_retried(self):
        job = {"run_id": "test", "launches": 1, "status": "running"}
        with patch.object(watchdog, "event"), patch.object(watchdog, "stop_job"):
            watchdog.fail_job(job, "non-finite policy action", fatal=True)
        self.assertEqual(job["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
